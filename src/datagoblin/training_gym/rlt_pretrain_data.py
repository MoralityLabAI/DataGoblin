"""Freeze and verify the three data lanes for the integrated pretraining pilot."""

from __future__ import annotations

import json
import random
import secrets
import uuid
from pathlib import Path

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, load_structured

from . import rlt_revision as revision
from .rlt_lean_labels import atomic_record
from .rlt_lean_pool_export import SCHEMA as LEAN_SCHEMA

TEXT_LENGTH = 32
TEXT_TRAIN_COUNT = 512
TEXT_HELD_PER_CORPUS = 64


def json_lines(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def write_lines(path, rows):
    pend = path.with_suffix(".part.jsonl")
    with pend.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
    pend.replace(path)


def checked_path(root, name, digest):
    if Path(name).name != name:
        raise ValueError("a manifest member must sit directly under its pack root")
    path = root / name
    if sha256_file(path) != digest:
        raise ValueError(f"content changed after freezing: {path}")
    return path


def stream_text(path):
    """Read the prior array-of-token-arrays fixture with bounded temporary RAM."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8-sig") as stream:
        if stream.read(1) != "[":
            raise ValueError("expected a JSON array of training windows")
        buffer = ""
        while True:
            buffer = buffer.lstrip(" \t\r\n,")
            if buffer.startswith("]"):
                return
            try:
                row, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                block = stream.read(32 * 1024)
                if not block or len(buffer) > 256 * 1024:
                    raise ValueError("truncated or oversize JSON token row") from exc
                buffer += block
                continue
            if not isinstance(row, list) or any(type(t) is not int for t in row):
                raise ValueError("training text rows must contain integer tokens")
            yield row
            buffer = buffer[offset:]


def lean_rows(root, manifest, split):
    if manifest["schema"] != LEAN_SCHEMA:
        raise ValueError("unexpected checked Lean pack version")
    registered = manifest["splits"][split]
    public = checked_path(root, registered["public"], registered["public_sha256"])
    receipts = checked_path(root, registered["receipts"], registered["receipts_sha256"])
    ids = set()
    for row, grades in zip(json_lines(public), json_lines(receipts), strict=True):
        if (
            row["id"] != grades["id"] or row["epoch"] != grades["epoch"]
            or row["id"] in ids or row["task"] != "lean_candidate_proof"
        ):
            raise ValueError("checked Lean labels disagree with the public goal")
        ids.add(row["id"])
        lookup = {item["id"]: item for item in row["candidates"]}
        if lookup.keys() != grades["outcomes"].keys() or len(lookup) != len(row["candidates"]):
            raise ValueError("Lean labels disagree with the checked actions")
        yield row, grades
    if len(ids) != registered["goals"]:
        raise ValueError("checked Lean pack lost a goal")


def rule_rows(root, split):
    manifest = load_structured(root / "manifest.json")
    if manifest["version"] != revision.VERSION:
        raise ValueError("unexpected rule-revision pack version")
    spec = manifest["splits"][split]
    tasks = checked_path(root, f"{split}.tasks.jsonl", spec["tasks_sha256"])
    targets = checked_path(root, f"{split}.targets.jsonl", spec["targets_sha256"])
    for task, target in zip(json_lines(tasks), json_lines(targets), strict=True):
        if task["id"] != target["id"] or task["lineage"] != target["lineage"]:
            raise ValueError("rule-revision targets belong to a different task")
        yield task, target


def select_text(train_path, held_path, guard):
    corpus = load_structured(held_path)
    held = []
    excluded = set()
    for source, rows in sorted(corpus.items()):
        picked = 0
        for row in rows:
            tokens = tuple(row["bytes"][:TEXT_LENGTH + 1])
            if len(tokens) != TEXT_LENGTH + 1 or tokens in excluded:
                continue
            excluded.add(tokens)
            held.append({"id": sha256_json(tokens), "corpus": source, "tokens": tokens,
                         "source_window_id": row["id"], "source_ordinal": row["ordinal"]})
            picked += 1
            if picked == TEXT_HELD_PER_CORPUS:
                break
        if picked != TEXT_HELD_PER_CORPUS:
            raise ValueError(f"insufficient frozen held-out windows in {source}")
    if len(held) < 2 * TEXT_HELD_PER_CORPUS:
        raise ValueError("expected two held-out text corpora")
    rng = random.Random(9177334)
    train = []
    considered = 0
    for index, row in enumerate(stream_text(train_path)):
        if index % 256 == 0:
            guard()
        tokens = tuple(t for t in row if 0 <= t < 256)[:TEXT_LENGTH + 1]
        if len(tokens) != TEXT_LENGTH + 1 or tokens in excluded:
            continue
        considered += 1
        j = considered - 1 if considered <= TEXT_TRAIN_COUNT else rng.randrange(considered)
        if j < TEXT_TRAIN_COUNT:
            item = {"id": sha256_json([index, tokens]), "source_index": index, "tokens": tokens}
            if j == len(train):
                train.append(item)
            else:
                train[j] = item
    if len(train) != TEXT_TRAIN_COUNT:
        raise ValueError("insufficient pretrain text windows")
    return train, held


def prepare(root, spec_path, guard):
    spec = load_structured(spec_path)
    pack = Path(spec["checked_lean_pack"]).resolve()
    lean_source = load_structured(pack / "manifest.json")
    text_source = Path(spec["text_fixture"]).resolve()
    held_source = Path(spec["text_holdout"]).resolve()
    train_manifest = load_structured(text_source / "manifest.json")
    held_manifest = load_structured(held_source / "manifest.json")
    train_path = checked_path(text_source, "text_train.json", train_manifest["train_text_sha256"])
    held_path = checked_path(held_source, "windows.json", held_manifest["windows_sha256"])
    manifest = root / "dataset.json"
    sources = {
        "spec_sha256": sha256_file(spec_path), "original_lean_manifest_sha256": sha256_file(pack / "manifest.json"),
        "text_train_manifest_sha256": sha256_file(text_source / "manifest.json"),
        "text_held_manifest_sha256": sha256_file(held_source / "manifest.json"),
        "original_text_train_sha256": sha256_file(train_path), "original_text_held_sha256": sha256_file(held_path),
        "revision_generator_sha256": sha256_file(Path(revision.__file__)),
    }
    if manifest.exists():
        if load_structured(manifest)["sources"] != sources:
            raise ValueError("the integrated dataset already has different sources")
        return
    train_ids, train_lineages = set(), set()
    for split in ("train", "held_structures"):
        rows = list(lean_rows(pack, lean_source, split))
        ids = {public["id"] for public, _ in rows}
        lineages = {public["lineage"] for public, _ in rows}
        if ids & train_ids or lineages & train_lineages:
            raise ValueError("Lean lineage leaked across splits")
        train_ids.update(ids)
        train_lineages.update(lineages)
        write_lines(root / f"lean-{split}.public.jsonl", (public for public, _ in rows))
        write_lines(root / f"lean-{split}.receipts.jsonl", (checked for _, checked in rows))
    text_train, text_held = select_text(train_path, held_path, guard)
    write_lines(root / "text-train.jsonl", text_train)
    write_lines(root / "text-held.jsonl", text_held)
    rules = root / "revision"
    if not rules.exists():
        staging = root / f"revision-part-{uuid.uuid4().hex}"
        revision.build(staging, seed=secrets.randbelow(2**63), train=128, per_split=32)
        staging.replace(rules)
    else:
        for split in revision.SPLITS:
            list(rule_rows(rules, split))
    members = [
        "lean-train.public.jsonl", "lean-train.receipts.jsonl", "lean-held_structures.public.jsonl",
        "lean-held_structures.receipts.jsonl", "text-train.jsonl", "text-held.jsonl",
        "revision/manifest.json",
    ]
    atomic_record(manifest, {
        "schema": "datagoblin.mixed-pretrain-pilot.v1", "sources": sources,
        "files": {name: sha256_file(root / name) for name in members},
        "lean_environment": lean_source["environment"], "text_length": TEXT_LENGTH,
    })
