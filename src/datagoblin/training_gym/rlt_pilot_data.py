"""Freeze local benchmark slices and encode the RLT pilot's two tasks."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import dump_structured, load_structured

from .rlt_revision import NODES, build

TEXT = ("fineweb_edu", "openwebtext")
PUBLIC = (*TEXT, "lambada", "hellaswag", "arc_easy", "arc_challenge")
RULE_SPLITS = ("development", "length_ood", "revision_ood")
NODE = {node: i + 256 for i, node in enumerate(NODES)}
BOS, GRAPH, QUERY, REVISION, PAD = range(288, 293)
VOCAB = 293
WINDOW = 96
CONTEXT = 128


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def freeze(root: Path, fixture_seed: int) -> None:
    frozen = root / "frozen"
    frozen.mkdir(exist_ok=False)
    rules = root / "rules"
    build(rules, fixture_seed, train=2048, per_split=64)
    known: dict[str, list[dict[str, Any]]] = {}
    training_windows: list[list[int]] = []
    seen_docs: set[str] = set()
    seen_windows: set[bytes] = set()
    sources = {}
    for name in PUBLIC:
        path = root / "sources" / f"{name}.json"
        source = load_structured(path)
        sources[name] = {k: v for k, v in source.items() if k != "rows"} | {"snapshot_sha256": sha256_file(path)}
        items = []
        for entry in source["rows"]:
            ordinal, row = entry["row_idx"], entry["row"]
            key = sha256_json([name, source["observed_repo_revision"], ordinal, row])
            if name in TEXT:
                if ordinal >= 80:
                    continue
                full_hash = entry["full_text_sha256"]
                if full_hash in seen_docs:
                    continue
                seen_docs.add(full_hash)
                text = row["text"].encode()
                if len(text) < WINDOW:
                    continue
                if len(items) < 64:
                    offset = int(key[:8], 16) % (len(text) - WINDOW + 1)
                    snippet = text[offset : offset + WINDOW]
                    if snippet not in seen_windows:
                        items.append({"id": key, "ordinal": ordinal, "byte_offset": offset, "bytes": list(snippet)})
                        seen_windows.add(snippet)
                continue
            if len(items) >= 64:
                break
            item = {"id": key, "ordinal": ordinal}
            if name == "lambada":
                match = re.search(r"\s+\S+$", row["text"].rstrip())
                if match is None:
                    continue
                item.update(prompt=row["text"][: match.start()], target=match.group())
            elif name == "hellaswag":
                item.update(
                    prompt=row["ctx"],
                    choices=[" " + ending for ending in row["endings"]],
                    gold=int(row["label"]),
                )
            else:
                item.update(
                    prompt="Question: " + row["question"] + "\nAnswer:",
                    choices=[" " + choice for choice in row["choices"]["text"]],
                    gold=row["choices"]["label"].index(row["answerKey"]),
                )
            items.append(item)
        if len(items) != 64:
            raise ValueError(f"expected 64 frozen items for {name}, got {len(items)}")
        known[name] = items
    # Reserve all first-80 documents before allocating either corpus's training windows.
    # Skip exact matches to eval windows even when they cross a training window boundary.
    eval_windows = set(seen_windows)
    for name in TEXT:
        source = load_structured(root / "sources" / f"{name}.json")
        for entry in source["rows"]:
            full_hash = entry["full_text_sha256"]
            if entry["row_idx"] < 80 or full_hash in seen_docs:
                continue
            seen_docs.add(full_hash)
            text = entry["row"]["text"].encode()
            if any(window in text for window in eval_windows):
                continue
            for start in range(0, len(text) - WINDOW + 1, WINDOW):
                snippet = text[start : start + WINDOW]
                if snippet not in seen_windows:
                    training_windows.append([BOS, *snippet])
                    seen_windows.add(snippet)
    if len(training_windows) < 1000:
        raise ValueError("insufficient distinct training windows")
    dump_structured(frozen / "known.json", known)
    dump_structured(frozen / "text_train.json", training_windows)
    dump_structured(frozen / "manifest.json", {
        "benchmark_sources": sources,
        "known_items": {name: len(items) for name, items in known.items()},
        "known_sha256": sha256_file(frozen / "known.json"),
        "train_text_windows": len(training_windows),
        "train_text_sha256": sha256_file(frozen / "text_train.json"),
        "rules_manifest_sha256": sha256_file(rules / "manifest.json"),
        "rule_eval_splits": list(RULE_SPLITS),
        "scope": "Pinned viewer slices and graph lineages for a reduced-scale pilot",
    })


def rule_input(task: dict[str, Any], *, prior: int | None = None) -> list[int]:
    tokens = [BOS, GRAPH]
    for source, dest in task["links"]:
        tokens.extend((NODE[source], NODE[dest]))
    tokens.extend((QUERY, NODE[task["start"]], 256 + task["steps"]))
    if prior is not None:
        patch = task["patch"]
        tokens.extend((prior, REVISION, NODE[patch["source"]], NODE[patch["destination"]], QUERY))
    return tokens


def sample_rule_turns(root: Path) -> list[tuple[list[int], int]]:
    tasks = jsonl(root / "rules" / "train.tasks.jsonl")
    targets = jsonl(root / "rules" / "train.targets.jsonl")
    turns = []
    for task, target in zip(tasks, targets, strict=True):
        if target["id"] != task["id"]:
            raise ValueError("training fixture/target mismatch")
        prior = NODE[target["before"]]
        turns.extend(((rule_input(task), prior), (rule_input(task, prior=prior), NODE[target["after"]])))
    return turns


def batch_plan(seed: int, count: int, text_windows: int, rule_turns: int, text_batch: int, rule_batch: int):
    rng = random.Random(seed)
    for _ in range(count):
        yield rng.sample(range(text_windows), text_batch), rng.sample(range(rule_turns), rule_batch)


def initialization_digest(weights) -> str:
    digest = hashlib.sha256()
    for name, tensor in weights.items():
        digest.update(name.encode())
        digest.update(tensor.tobytes())
    return digest.hexdigest()
