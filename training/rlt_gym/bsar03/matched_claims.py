"""BSAR-03: matched positive/negative source-claim pairs on held-out Lean roots.

Unlike the BSAR-02 fixed-label question templates, each claim family here
contains a positive and a negative candidate for the *same theorem signature*.
All labels follow exact public source text, not semantic proof validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.rlt_gym.bsar02.boundary_curriculum import CATALOG, generate

SOURCE_COMMIT = "80c62433b0eb671175bd8d6169822d3ebd922b60"
SEED = 20261122
LABELS = ("PRESENT", "ABSENT")
TYPES = ("Int", "Nat", "R", "Real", "Rat")
TYPE_PATTERN = re.compile(r":\s*(Int|Nat|R|Real|Rat)\b")


def _ident(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _split(path: str) -> str:
    if "Step7" in path:
        return "test"
    if "Step6" in path:
        return "dev"
    return "train"


def _decoy(pool: set[str], actual: str, signature: str, rng: random.Random) -> str:
    choices = [item for item in pool if item != actual and item not in signature]
    if not choices:
        raise ValueError("No absent decoy available for target")
    # Similar-length decoys make constant-length shortcuts less effective.
    ranked = sorted(choices, key=lambda x: (abs(len(x) - len(actual)), x))
    window = ranked[: min(6, len(ranked))]
    return rng.choice(window)


def _pair(
    root: str, split: str, path: str, line: int, signature: str,
    family: str, actual: str, absent: str, phrase: str,
) -> list[dict[str, Any]]:
    if actual == absent or absent in signature:
        raise ValueError("Candidate pair is not a positive/negative contrast")
    pair_id = _ident(f"{root}:{family}")
    rows = []
    for truth, candidate in ((True, actual), (False, absent)):
        question = (
            "Read only this Lean theorem signature, taken from the pinned source.\n"
            f"Source: {path}:{line}\n"
            f"Signature: {signature}\n"
            f"Question: {phrase} `{candidate}`?\n"
            "Reply PRESENT or ABSENT. An answer must be supported by the displayed signature."
        )
        rows.append({
            "id": _ident(f"{pair_id}:{int(truth)}"),
            "pair_id": pair_id,
            "root": root,
            "split": split,
            "family": family,
            "source_path": path,
            "source_line": line,
            "source_commit": SOURCE_COMMIT,
            "prompt": question,
            "gold_label": "PRESENT" if truth else "ABSENT",
            "evidence_scope": "STATIC_DECLARATION_SYNTAX",
        })
    return rows


def generate_pairs(catalog: dict, seed: int = SEED) -> list[dict[str, Any]]:
    if catalog["source_commit"] != SOURCE_COMMIT or catalog["total"] != 129:
        raise ValueError("Source catalog provenance changed")
    # Inherit MathCan BSAR-02's root-disjoint split policy exactly.
    allowed_roots = {row["root"] for row in generate(catalog, seed=seed)}
    all_names = {d["name"] for round_item in catalog["rounds"]
                 for d in round_item["declarations"]}
    all_premises = {premise for round_item in catalog["rounds"]
                    for d in round_item["declarations"] for premise in d["premises"]}
    rng = random.Random(seed)
    examples = []
    for group in catalog["rounds"]:
        path = group["path"]
        split = _split(path)
        for d in group["declarations"]:
            root = f"{path}:{d['name']}"
            if root not in allowed_roots:
                continue
            name = d["name"]
            signature = d["signature"]
            line = d["line"]
            other_name = _decoy(all_names, name, signature, rng)
            examples += _pair(
                root, split, path, line, signature, "declaration_name", name, other_name,
                "Is the displayed declaration named",
            )
            if d["premises"]:
                premise = rng.choice(d["premises"])
                other_premise = _decoy(all_premises, premise, signature, rng)
                examples += _pair(
                    root, split, path, line, signature, "named_premise",
                    premise, other_premise,
                    "Is this exact hypothesis explicitly declared",
                )
            bound_types = set(TYPE_PATTERN.findall(signature))
            if bound_types:
                actual = sorted(bound_types)[0]
                other_type = _decoy(set(TYPES), actual, signature, rng)
                examples += _pair(
                    root, split, path, line, signature, "bound_type", actual, other_type,
                    "Does a parameter have exactly the bound type",
                )
    rng.shuffle(examples)
    return examples


def validate(examples: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {row["id"] for row in examples}
    if len(ids) != len(examples):
        raise ValueError("Duplicate example identifiers")
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_by_name: dict[str, str] = {}
    for row in examples:
        pairs[row["pair_id"]].append(row)
        name = row["root"].split(":")[-1]
        if name in split_by_name and split_by_name[name] != row["split"]:
            raise ValueError("Cross-split theorem-name leakage")
        split_by_name[name] = row["split"]
    if any(len(rows) != 2 or {r["gold_label"] for r in rows} != set(LABELS)
           or len({r["root"] for r in rows}) != 1 for rows in pairs.values()):
        raise ValueError("Unmatched or imbalanced positive/negative pair")
    counts = Counter(row["split"] for row in examples)
    if set(counts) != {"train", "dev", "test"}:
        raise ValueError("Empty split")
    return {
        "source_commit": SOURCE_COMMIT,
        "examples": len(examples),
        "pairs": len(pairs),
        "unique_roots": len(set(row["root"] for row in examples)),
        "split_counts": dict(sorted(counts.items())),
        "test_pairs": sum(rows[0]["split"] == "test" for rows in pairs.values()),
        "gold_scope": "exact source signature fields; no Lean kernel proof implied",
    }


def public_rows(examples: list[dict[str, Any]], split: str = "test") -> list[dict[str, Any]]:
    return [
        {"id": row["id"], "pair_id": row["pair_id"], "prompt": row["prompt"],
         "source_path": row["source_path"], "source_line": row["source_line"]}
        for row in sorted(examples, key=lambda item: item["id"])
        if row["split"] == split
    ]


def score(examples: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    test = {row["id"]: row for row in examples if row["split"] == "test"}
    if not set(predictions) <= set(test):
        raise ValueError("Predicted a non-test ID")
    if any(value not in (*LABELS, "ABSTAIN") for value in predictions.values()):
        raise ValueError("Unknown output label")
    correct = sum(predictions.get(key) == row["gold_label"] for key, row in test.items())
    pair_groups: dict[str, list[str]] = defaultdict(list)
    for key, row in test.items():
        pair_groups[row["pair_id"]].append(key)
    fully_correct = sum(
        all(predictions.get(key) == test[key]["gold_label"] for key in keys)
        for keys in pair_groups.values()
    )
    return {
        "test_examples": len(test),
        "accuracy": correct / len(test),
        "coverage": len(predictions) / len(test),
        "complete_pairs": fully_correct,
        "test_pairs": len(pair_groups),
        "report_type": "BSAR-03 source-claim matching, not a theorem prover",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--output-dir", required=True, type=Path)
    baseline = sub.add_parser("constant-baseline")
    baseline.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    examples = generate_pairs(json.loads(args.catalog.read_text(encoding="utf-8")))
    manifest = validate(examples)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tasks = public_rows(examples)
    path = out / "bsar03_step7_public.jsonl"
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in tasks),
                    encoding="utf-8")
    manifest["public_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.command == "constant-baseline":
        pred = {r["id"]: "PRESENT" for r in tasks}
        manifest["constant_baseline"] = score(examples, pred)
        (out / "constant_predictions.jsonl").write_text(
            "".join(json.dumps({"id": key, "label": val}) + "\n" for key, val in sorted(pred.items())),
            encoding="utf-8",
        )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
