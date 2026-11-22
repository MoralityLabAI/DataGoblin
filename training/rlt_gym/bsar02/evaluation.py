"""Evaluate DataGoblin's BSAR-02 Step 7 holdout without exposing gold in prompts.

A measured model must supply predictions through an independently generated
JSONL file. This script does not load a model or claim trained-model results.
Run: python -m training.rlt_gym.bsar02.evaluation prepare --output-dir /tmp/bsar02
     python -m training.rlt_gym.bsar02.evaluation score --predictions /tmp/model.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .boundary_curriculum import CATALOG, check_dataset, generate

SOURCE_COMMIT = "80c62433b0eb671175bd8d6169822d3ebd922b60"
LABELS = (
    "SOURCE_DECLARATION_PRESENT",
    "UNSUPPORTED_PDE_LIFT",
    "MISSING_ANALYTIC_TYPE_BRIDGE",
    "KERNEL_CHECK_REQUIRED",
    "NAMED_PREMISE_PRESENT",
    "MISSING_PREMISE",
)
ABSTAIN = "ABSTAIN"
EXPECTED_COUNT = 164


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_holdout(path: Path = CATALOG) -> dict[str, dict[str, Any]]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog["source_commit"] != SOURCE_COMMIT or catalog["total"] != 129:
        raise ValueError("Unexpected BSA-R catalog or source provenance")
    examples = generate(catalog)
    check_dataset(examples)
    heldout = {row["id"]: row for row in examples if row["split"] == "test"}
    if len(heldout) != EXPECTED_COUNT:
        raise ValueError(f"Holdout count changed: {len(heldout)}")
    if not all("Step7" in x["source_path"] for x in heldout.values()):
        raise ValueError("Found an unexpected source in Step 7 holdout")
    return dict(sorted(heldout.items()))


def public_tasks(heldout: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    # Never include gold_label, gold_reason, or gold provenance in model input.
    return [
        {"id": row["id"], "prompt": row["prompt"],
         "source_path": row["source_path"], "source_line": row["source_line"]}
        for row in heldout.values()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    content = "".join(_canonical_json(x) + "\n" for x in rows).encode("utf-8")
    path.write_bytes(content)
    return _sha(content)


def prepare(heldout: dict[str, dict[str, Any]], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows = public_tasks(heldout)
    digest = _write_jsonl(output / "step7_public_tasks.jsonl", rows)
    manifest = {
        "task_count": len(rows),
        "unique_roots": len({row["root"] for row in heldout.values()}),
        "source_commit": SOURCE_COMMIT,
        "source_manifest": "MathCan BSAR-02 six_round_catalog.json at commit 53351f295aaaf669857638a081a4ac22f02ba0f9",
        "public_tasks_sha256": digest,
        "heldout_ids_sha256": _sha("\n".join(heldout.keys()).encode()),
        "labels_excluded_from_model_prompts": True,
        "sealing": "PUBLIC_HELDOUT; training contamination must be audited separately",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def read_predictions(path: Path, expected: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        obj = json.loads(raw)
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
            raise ValueError(f"Malformed prediction on line {number}")
        if {"gold_label", "gold_reason", "gold", "answer"} & obj.keys():
            raise ValueError(f"Gold label field detected in model response on line {number}")
        identity = obj["id"]
        label = obj.get("label")
        if identity not in expected:
            raise ValueError(f"Unknown or non-test task id on line {number}: {identity}")
        if identity in found:
            raise ValueError(f"Duplicate task id on line {number}: {identity}")
        if label not in (*LABELS, ABSTAIN):
            raise ValueError(f"Unknown predicted label on line {number}: {label}")
        found[identity] = label
    return found


def score(heldout: dict[str, dict[str, Any]], predicted: dict[str, str]) -> dict[str, Any]:
    ids = set(heldout)
    if not set(predicted) <= ids:
        raise ValueError("Predictions contain ids outside frozen holdout")
    n = len(ids)
    assigned = {key: predicted.get(key, ABSTAIN) for key in ids}
    correct = sum(assigned[k] == heldout[k]["gold_label"] for k in ids)
    nonabstain = sum(v != ABSTAIN for v in assigned.values())
    matrix: dict[str, dict[str, int]] = {gold: {} for gold in LABELS}
    for key, row in heldout.items():
        gold = row["gold_label"]
        pred = assigned[key]
        matrix[gold][pred] = matrix[gold].get(pred, 0) + 1

    per_label: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = sum(
            assigned[k] == label and heldout[k]["gold_label"] == label
            for k in ids
        )
        fp = sum(
            assigned[k] == label and heldout[k]["gold_label"] != label
            for k in ids
        )
        fn = sum(
            assigned[k] != label and heldout[k]["gold_label"] == label
            for k in ids
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"support": tp + fn, "precision": precision,
                            "recall": recall, "f1": f1}

    by_template: dict[str, dict[str, int | float]] = {}
    for template in sorted({row["template"] for row in heldout.values()}):
        rows = {key: row for key, row in heldout.items() if row["template"] == template}
        matches = sum(assigned[key] == row["gold_label"] for key, row in rows.items())
        by_template[template] = {"total": len(rows), "correct": matches,
                                 "accuracy": matches / len(rows)}

    roots: dict[str, list[str]] = defaultdict(list)
    for key, row in heldout.items():
        roots[row["root"]].append(key)
    perfect_roots = sum(
        all(assigned[k] == heldout[k]["gold_label"] for k in keys)
        for keys in roots.values()
    )
    return {
        "evaluation": "BSAR-02 Step7 public held-out source-claim classification",
        "source_commit": SOURCE_COMMIT,
        "examples": n,
        "coverage": nonabstain / n,
        "answer_accuracy": correct / nonabstain if nonabstain else 0.0,
        "strict_accuracy": correct / n,
        "macro_f1": sum(p["f1"] for p in per_label.values()) / len(LABELS),
        "perfect_roots": perfect_roots,
        "total_roots": len(roots),
        "abstentions": n - nonabstain,
        "per_label": per_label,
        "by_template": by_template,
        "confusion": matrix,
        "scope": "String-label evaluation; no Lean kernel correctness or PDE proof implied",
    }


def template_question(row: dict[str, Any]) -> str:
    """Normalize only the model-visible question, dropping all source signature text."""
    question = row["prompt"].split("Claim to evaluate: ", 1)[1]
    return re.sub(r"\\bh[A-Za-z0-9_]+\\b", "<HYP>", question)


def template_only_predictions(catalog_path: Path, heldout: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Fit a deliberately weak template lookup on TRAIN only to detect prompt leakage."""
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    for example in generate(catalog):
        if example["split"] == "train":
            labels[template_question(example)][example["gold_label"]] += 1
    prediction = {}
    for key, example in heldout.items():
        choices = labels.get(template_question(example))
        prediction[key] = choices.most_common(1)[0][0] if choices else ABSTAIN
    return prediction


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=CATALOG)
    sub = p.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("prepare", help="Export prompt-only frozen test tasks")
    gen.add_argument("--output-dir", type=Path, required=True)
    ev = sub.add_parser("score", help="Score independent model predictions")
    ev.add_argument("--predictions", type=Path, required=True)
    ev.add_argument("--output", type=Path)
    base = sub.add_parser("majority-baseline", help="Produce a declared non-model baseline")
    base.add_argument("--output-dir", type=Path, required=True)
    tm = sub.add_parser("template-baseline", help="Measure question-template label leakage from train split")
    tm.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    holdout = load_holdout(args.catalog)
    if args.command == "prepare":
        result = prepare(holdout, args.output_dir)
    elif args.command == "majority-baseline":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "majority_predictions.jsonl"
        rows = [{"id": key, "label": "SOURCE_DECLARATION_PRESENT"} for key in holdout]
        _write_jsonl(out, rows)
        result = score(holdout, read_predictions(out, set(holdout)))
        result["producer"] = "CONSTANT_LABEL_BASELINE; no model execution"
        (args.output_dir / "majority_metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif args.command == "template-baseline":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "template_predictions.jsonl"
        predictions = template_only_predictions(args.catalog, holdout)
        _write_jsonl(out, [{"id": key, "label": value} for key, value in sorted(predictions.items())])
        result = score(holdout, read_predictions(out, set(holdout)))
        result["producer"] = "TRAIN_ONLY_QUESTION_TEMPLATE_LOOKUP_CONTROL; no model execution"
        (args.output_dir / "template_metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        result = score(holdout, read_predictions(args.predictions, set(holdout)))
        if args.output:
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
