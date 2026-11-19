"""Build and score matched rule-revision episodes for recurrent feedback studies."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, dump_structured, load_structured

VERSION = "rlt-rule-revision-v1"
SPLITS = ("train", "development", "length_ood", "revision_ood")
NODES = tuple(f"n{i:02d}" for i in range(32))


def follow(links: Mapping[str, str], start: str, steps: int) -> str:
    for _ in range(steps):
        start = links[start]
    return start


def episodes(seed: int, split: str, count: int) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Each lineage has the same starting graph and both patch variants."""
    rng = random.Random(int(sha256_json([VERSION, seed, split]), 16))
    for _ in range(count):
        cycle = rng.sample(NODES, len(NODES))
        links = dict(zip(cycle, cycle[1:] + cycle[:1], strict=True))
        start = cycle[0]
        steps = rng.randint(7, 9) if split == "length_ood" else rng.randint(3, 5)
        edit = rng.choice((0, steps - 1)) if split == "revision_ood" else rng.randrange(1, steps - 1)
        source = cycle[edit]
        replacement = cycle[rng.randrange(steps + 1, len(cycle) - steps)]
        listed = list(links.items())
        rng.shuffle(listed)
        lineage = sha256_json([links, start])
        before = follow(links, start, steps)
        for destination in rng.sample([links[source], replacement], 2):
            patch = {"source": source, "destination": destination}
            # Present the same first turn for both variants. Reveal the patch only on turn two.
            task = {"lineage": lineage, "links": listed, "start": start, "steps": steps, "patch": patch}
            task_id = sha256_json(task)
            after = follow(links | {source: destination}, start, steps)
            yield {"id": task_id, **task}, {"id": task_id, "lineage": lineage, "before": before, "after": after}


def build(out: Path, seed: int, train: int, per_split: int) -> dict[str, Any]:
    if train < 1 or per_split < 1:
        raise ValueError("requested lineage counts must be positive")
    out.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    manifest: dict[str, Any] = {"version": VERSION, "splits": {}}
    for split in SPLITS:
        count = train if split == "train" else per_split
        tasks = out / f"{split}.tasks.jsonl"
        targets = out / f"{split}.targets.jsonl"
        with (
            tasks.open("x", encoding="utf-8", newline="\n") as task_file,
            targets.open("x", encoding="utf-8", newline="\n") as target_file,
        ):
            for i, (task, target) in enumerate(episodes(seed, split, count)):
                lineage = task["lineage"]
                if i % 2 == 0:
                    if lineage in seen:
                        raise ValueError("duplicate lineage in generated pack")
                    seen.add(lineage)
                task_file.write(canonical_json(task) + "\n")
                target_file.write(canonical_json(target) + "\n")
        manifest["splits"][split] = {
            "lineages": count,
            "items": 2 * count,
            "tasks_sha256": sha256_file(tasks),
            "targets_sha256": sha256_file(targets),
        }
    dump_structured(out / "manifest.json", manifest)
    return manifest


def read_rows(path: Path, fields: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError(f"wrong fields in {path.name}")
            key = row["id"]
            if not isinstance(key, str) or key in rows:
                raise ValueError(f"non-string or duplicate item ID in {path.name}")
            rows[key] = row
    return rows


def score(pack: Path, split: str, predictions: Path) -> dict[str, Any]:
    manifest = load_structured(pack / "manifest.json")
    if manifest["version"] != VERSION or split not in SPLITS:
        raise ValueError("unknown pack version or split")
    spec = manifest["splits"][split]
    target_path = pack / f"{split}.targets.jsonl"
    if sha256_file(pack / f"{split}.tasks.jsonl") != spec["tasks_sha256"]:
        raise ValueError("task hash mismatch")
    if sha256_file(target_path) != spec["targets_sha256"]:
        raise ValueError("target hash mismatch")
    targets = read_rows(target_path, {"id", "lineage", "before", "after"})
    predicted = read_rows(predictions, {"id", "before", "after"})
    if len(targets) != spec["items"] or targets.keys() != predicted.keys():
        raise ValueError("predictions must cover every item exactly once")
    before_hits = changed_hits = unchanged_hits = changed_count = joint_hits = stale = 0
    changed_initial_correct = 0
    pair_hits: dict[str, int] = {}
    for key, target in targets.items():
        guess = predicted[key]
        before_ok = guess["before"] == target["before"]
        after_ok = guess["after"] == target["after"]
        before_hits += before_ok
        joint_hits += before_ok and after_ok
        lineage = target["lineage"]
        pair_hits[lineage] = pair_hits.get(lineage, 0) + (before_ok and after_ok)
        if target["before"] != target["after"]:
            changed_count += 1
            changed_hits += after_ok
            changed_initial_correct += before_ok
            stale += before_ok and guess["after"] == target["before"]
        else:
            unchanged_hits += after_ok
    return {
        "version": VERSION,
        "split": split,
        "tasks_sha256": spec["tasks_sha256"],
        "predictions_sha256": sha256_file(predictions),
        "before_exact": [before_hits, len(targets)],
        "changed_after_exact": [changed_hits, changed_count],
        "no_op_after_exact": [unchanged_hits, len(targets) - changed_count],
        "both_turns_exact": [joint_hits, len(targets)],
        "both_variants_exact": [sum(n == 2 for n in pair_hits.values()), len(pair_hits)],
        "stale_after_when_before_correct": [stale, changed_initial_correct],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build")
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--seed", type=int, required=True)
    make.add_argument("--train", type=int, default=256, help="number of training lineages")
    make.add_argument("--per-split", type=int, default=64, help="lineages per evaluation split")
    check = commands.add_parser("score")
    check.add_argument("--pack", type=Path, required=True)
    check.add_argument("--split", choices=SPLITS, required=True)
    check.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build(args.out, args.seed, args.train, args.per_split)
    else:
        result = score(args.pack, args.split, args.predictions)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
