"""STORY-02: pinned, multi-world Storyworld gym with source-world holdout custody.

Native builder/Mini/HG-02 and SWMD-MICRO streams are collected in separate
world archives. A held-out world's labels cannot enter train SFT exports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .storyworld_micro import export_micro, replay_micro
from .storyworld_native import SPLITS, _json, _sha_file, collect, connect, replay_archive

SCHEMA = "datagoblin.storyworld_cohort.v1"
REGISTRY_SCHEMA = "datagoblin.storyworld_cohort.registry.v1"
DEFAULT_REGISTRY = Path("training/rlt_gym/storyworld_native/world_cohort_v1.json")


def _registry(path: Path) -> dict:
    contents = json.loads(path.read_text(encoding="utf-8"))
    if contents.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("unsupported Storyworld cohort registry schema")
    worlds = contents.get("worlds", [])
    if not isinstance(worlds, list) or not 2 <= len(worlds) <= 12:
        raise ValueError("world cohort requires 2..12 registered sources")
    names, paths = set(), set()
    splits = []
    for row in worlds:
        name, world, split = row["id"], row["path"], row["split"]
        parts = Path(world)
        if (not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]{2,40}", name) is None
                or not isinstance(world, str) or parts.is_absolute() or ".." in parts.parts
                or not world.endswith(".json") or split not in SPLITS):
            raise ValueError("invalid source-world registration")
        if name in names or world in paths:
            raise ValueError("duplicate world name or source path")
        names.add(name)
        paths.add(world)
        splits.append(split)
    if "train" not in splits or not any(s in ("ordinary_test", "ood_test") for s in splits):
        raise ValueError("cohort needs a training world and a held-out test world")
    return contents


def _world_seed(seed: int, name: str, world_hash: str) -> int:
    digest = hashlib.sha256(_json([SCHEMA, seed, name, world_hash]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def collect_cohort(source: Path, registry: Path, out: Path, *, seed: int = 23,
                   per_mode: int = 3, micro_examples: int = 2, micro_max_encounters: int = 6) -> dict:
    if out.exists():
        raise FileExistsError("world cohort output exists")
    if per_mode < 1 or micro_examples < 1 or micro_max_encounters < 1:
        raise ValueError("cohort sampling counts must be positive")
    contents = _registry(registry)
    pin = contents["source_commit"]
    if contents.get("source_repository") != "MoralityLabAI/GPTStoryworld" or not re.fullmatch(r"[a-f0-9]{40}", pin):
        raise ValueError("invalid source identity in registry")
    # Validate the entire frozen source set before creating any output.
    prepared = []
    for row in contents["worlds"]:
        native = connect(source, row["path"], expect_commit=pin)
        prepared.append((row, native))
    out.mkdir(parents=True)
    records = []
    totals = {"native_tasks": 0, "native_controls": 0, "micro_tasks": 0, "train_sft_rows": 0}
    for row, native in prepared:
        name = row["id"]
        split = row["split"]
        world_seed = _world_seed(seed, name, native.hashes["world"])
        folder = out / name
        folder.mkdir()
        native_manifest = collect(native, folder / "native", seed=world_seed,
                                  per_mode=per_mode, split_override=split)
        micro_manifest = export_micro(
            source, folder / "micro", seed=world_seed,
            examples_per_encounter=micro_examples, max_encounters=micro_max_encounters,
            world=row["path"], expect_source_commit=pin, split_override=split,
        )
        native_count = native_manifest["artifacts"]["public_tasks.jsonl"]["rows"]
        controls_count = native_manifest["artifacts"]["baseline_rollouts.jsonl"]["rows"]
        micro_count = micro_manifest["artifacts"]["micro_public.jsonl"]["rows"]
        train_count = micro_manifest["artifacts"]["micro_train_sft.jsonl"]["rows"]
        if split != "train" and (train_count or native_manifest["artifacts"]["train_targets.jsonl"]["rows"]):
            raise ValueError("held-out world produced supervised train examples")
        totals["native_tasks"] += native_count
        totals["native_controls"] += controls_count
        totals["micro_tasks"] += micro_count
        totals["train_sft_rows"] += train_count
        records.append({
            "id": name, "split": split, "domain": row.get("domain", ""),
            "path": row["path"], "world_seed": world_seed,
            "source": native.descriptor(),
            "native_manifest_sha256": _sha_file(folder / "native/manifest.json"),
            "micro_manifest_sha256": _sha_file(folder / "micro/manifest.json"),
            "native_tasks": native_count, "micro_tasks": micro_count,
            "train_sft_rows": train_count, "control_rollouts": controls_count,
        })
    summary = {
        "schema": SCHEMA, "evidence": "source_native_multiworld_cpu_controls",
        "trained_model_results": False, "seed": seed, "per_mode": per_mode,
        "micro_examples": micro_examples, "micro_max_encounters": micro_max_encounters,
        "source_commit": pin, "registry_sha256": _sha_file(registry),
        "worlds": records, "totals": totals,
    }
    (out / "manifest.json").write_text(_json(summary) + "\n", encoding="utf-8")
    return summary


def replay_cohort(source: Path, registry: Path, folder: Path) -> dict:
    registered = _registry(registry)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA
            or manifest["registry_sha256"] != _sha_file(registry)
            or manifest["source_commit"] != registered["source_commit"]):
        raise ValueError("cohort registry or source pin mismatch")
    if len(manifest["worlds"]) != len(registered["worlds"]):
        raise ValueError("cohort world inventory mismatch")
    for row, archived in zip(registered["worlds"], manifest["worlds"], strict=True):
        if (row["id"] != archived["id"] or row["path"] != archived["path"]
                or row["split"] != archived["split"]):
            raise ValueError("cohort source identity mismatch")
        native = connect(source, row["path"], expect_commit=manifest["source_commit"])
        if native.descriptor() != archived["source"]:
            raise ValueError("cohort source file changed")
        base = folder / row["id"]
        if (_sha_file(base / "native/manifest.json") != archived["native_manifest_sha256"]
                or _sha_file(base / "micro/manifest.json") != archived["micro_manifest_sha256"]):
            raise ValueError("cohort child manifest digest mismatch")
        replay_archive(native, base / "native")
        replay_micro(source, base / "micro", world=row["path"],
                     expect_source_commit=manifest["source_commit"])
    # Regenerate the cohort to compare every split, child hash and exact seed.
    # This also detects a tampered summary count or reordered evaluation world.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="dg-sw-cohort-replay-") as tmp:
        fresh = collect_cohort(
            source, registry, Path(tmp) / "new", seed=manifest["seed"],
            per_mode=manifest["per_mode"], micro_examples=manifest["micro_examples"],
            micro_max_encounters=manifest["micro_max_encounters"],
        )
        if fresh != manifest:
            raise ValueError("regenerated cohort manifest mismatch")
    return {"status": "verified", "worlds": len(manifest["worlds"]), "totals": manifest["totals"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../GPTStoryworld"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--per-mode", type=int, default=3)
    parser.add_argument("--micro-examples", type=int, default=2)
    parser.add_argument("--micro-max-encounters", type=int, default=6)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path)
    action.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.out:
        result = collect_cohort(args.source, args.registry, args.out, seed=args.seed,
                                per_mode=args.per_mode, micro_examples=args.micro_examples,
                                micro_max_encounters=args.micro_max_encounters)
    else:
        result = replay_cohort(args.source, args.registry, args.replay)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
