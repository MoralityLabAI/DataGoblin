"""RLT repair environment on MathCan BSAR-09 exact Fraction windows.

A prediction must classify the chain and, if invalid, repair only its final
D_next to the exact admissible ODE frontier M*D_prev/(M-I). This preserves
the original M, I, and earlier steps, and avoids trivial zero-state repairs.

The environment evaluates model outputs. Teacher demonstrations are marked
as deterministic oracle labels and are never called trained-model results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from .mathcan_exact_oracle import (
    LABELS,
    REGIMES,
    SOURCE_COMMIT,
    check_corpus,
    encode,
    frac,
    generate,
    oracle,
    public_rows,
)

MATHCAN_MERGE_COMMIT = "0ce1a3c071142c502a41670a15c6cbca6fe14162"


def episode(row: dict[str, Any]) -> dict[str, Any]:
    """Model-visible input, with source, label and oracle trace withheld."""
    return {
        "id": row["id"],
        "pair_id": row["pair_id"],
        "prompt": row["prompt"],
    }


def tight_repair(row: dict[str, Any]) -> Fraction:
    """Maximum valid final endpoint for an otherwise admissible prefix."""
    windows = row["windows"]
    if not windows:
        raise ValueError("Empty window sequence")
    m = row["M"]
    last = windows[-1]
    im = last["I"]
    if m <= 0 or not 0 <= 2 * im <= m:
        raise ValueError("Invalid cost regime for tight repair")
    return frac(last["before"]) * Fraction(m, m - im)


def _validate_fraction_arg(value: Any) -> Fraction | None:
    if not isinstance(value, str) or not value or len(value) > 128:
        return None
    if any(ch not in "-0123456789/" for ch in value):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if max(parsed.numerator.bit_length(), parsed.denominator.bit_length()) > 1024:
        return None
    return parsed


def evaluate(
    row: dict[str, Any],
    action: dict[str, Any],
    *,
    teacher_feedback: bool = False,
) -> dict[str, Any]:
    """Single bounded verifier visit on a frozen held-out or train episode."""
    m = row["M"]
    d0 = frac(row["d0"])
    result = oracle(m, d0, row["windows"])
    actual = "SATISFIES" if result["all_local_gates"] else "VIOLATES"
    if actual != row["gold_label"]:
        raise ValueError("Corpus label and exact verifier disagree")
    verdict = action.get("verdict")
    repair = _validate_fraction_arg(action.get("repair"))
    verdict_ok = verdict == actual
    if actual == "SATISFIES":
        repair_ok = action.get("repair") is None
    else:
        repair_ok = repair is not None and repair == tight_repair(row)
        if repair_ok:
            corrected = [dict(x) for x in row["windows"]]
            corrected[-1]["after"] = encode(repair)
            if not oracle(m, d0, corrected)["all_local_gates"]:
                raise AssertionError("An exact tight repair must pass the source oracle")
    completed = verdict_ok and repair_ok
    receipt: dict[str, Any] = {
        "id": row["id"],
        "verdict_correct": verdict_ok,
        "repair_correct": repair_ok,
        "complete": completed,
        "reward": 1 if completed else 0,
        "verifier_calls": 1,
        "evidence_scope": "EXACT_FRACTION_LOCAL_ARITHMETIC_ONLY",
    }
    if teacher_feedback:
        receipt["gold_label"] = actual
        receipt["failed_gates"] = result["failed_gates"]
        receipt["tight_repair"] = encode(tight_repair(row)) if actual == "VIOLATES" else None
        receipt["producer"] = "DETERMINISTIC_EXACT_ORACLE; NO_TRAINED_MODEL"
    return receipt


def teacher_action(row: dict[str, Any]) -> dict[str, Any]:
    """Label-only deterministic teacher for train and development roots."""
    if row["split"] == "test":
        raise ValueError("Teacher actions for public test are withheld")
    actual = "SATISFIES" if oracle(row["M"], frac(row["d0"]), row["windows"])["all_local_gates"] else "VIOLATES"
    return {
        "verdict": actual,
        "repair": encode(tight_repair(row)) if actual == "VIOLATES" else None,
    }


def score_test(rows: list[dict[str, Any]], actions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    test = {row["id"]: row for row in rows if row["split"] == "test"}
    if not set(actions) <= set(test):
        raise ValueError("Prediction outside the test split")
    cases: dict[str, list[bool]] = defaultdict(list)
    complete, classified, repaired = 0, 0, 0
    for key, row in test.items():
        action = actions.get(key, {"verdict": "ABSTAIN"})
        receipt = evaluate(row, action)
        complete += int(receipt["complete"])
        classified += int(receipt["verdict_correct"])
        if row["gold_label"] == "VIOLATES":
            repaired += int(receipt["repair_correct"])
        cases[row["pair_id"]].append(receipt["complete"])
    return {
        "cases": len(test),
        "pairs": len(cases),
        "classification_accuracy": classified / len(test),
        "exact_repair_success_on_invalid": repaired / (len(test) // 2),
        "fully_completed_cases": complete,
        "fully_completed_pairs": sum(all(pair) for pair in cases.values()),
        "verifier_budget_per_case": 1,
        "evidence_scope": "LOCAL_EXACT_FRACTION_VERIFIER",
    }


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def freeze(out: Path) -> dict[str, Any]:
    rows = generate()
    source_receipt = check_corpus(rows)
    out.mkdir(parents=True, exist_ok=True)
    batches = {
        "train": [
            {
                "id": row["id"],
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": json.dumps(teacher_action(row), sort_keys=True)},
                ],
                "source_commit": SOURCE_COMMIT,
                "producer": "EXACT_FRACTION_ORACLE",
                "split": "train",
                "verifier_receipt": evaluate(row, teacher_action(row), teacher_feedback=True),
            }
            for row in rows if row["split"] == "train"
        ],
        "dev": [
            {
                "id": row["id"],
                "task": episode(row),
                "reference": teacher_action(row),
                "split": "dev",
                "producer": "EXACT_FRACTION_ORACLE",
            }
            for row in rows if row["split"] == "dev"
        ],
        "test_public": public_rows(rows),
    }
    hashes: dict[str, str] = {}
    for name, batch in batches.items():
        content = _jsonl(batch)
        file = out / f"{name}.jsonl"
        file.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    baseline = {
        row["id"]: {"verdict": "SATISFIES", "repair": None}
        for row in rows if row["split"] == "test"
    }
    report = {
        "source": "patrickdugan/MathCan",
        "mathcan_commit": MATHCAN_MERGE_COMMIT,
        "upstream_bsar_source": SOURCE_COMMIT,
        "source_corpus": source_receipt,
        "hashes": hashes,
        "constant_positive_baseline": score_test(rows, baseline),
        "custody": "Public synthetic holdout; source and labels recoverable by generation",
        "training_model_evaluated": False,
    }
    (out / "manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def read_actions(path: Path, expected: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        obj = json.loads(raw)
        key = obj.get("id")
        if not isinstance(key, str) or key not in expected:
            raise ValueError(f"Invalid held-out ID on line {number}")
        if key in found:
            raise ValueError(f"Duplicate held-out ID on line {number}")
        if {"gold_label", "oracle", "verifier_receipt", "reference"} & set(obj):
            raise ValueError(f"Gold or verifier fields leaked on line {number}")
        found[key] = {"verdict": obj.get("verdict"), "repair": obj.get("repair")}
    return found


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("freeze")
    gen.add_argument("--out", required=True, type=Path)
    ev = sub.add_parser("score")
    ev.add_argument("--predictions", required=True, type=Path)
    args = p.parse_args()
    if args.command == "freeze":
        report = freeze(args.out)
    else:
        rows = generate()
        expected = {row["id"] for row in rows if row["split"] == "test"}
        report = score_test(rows, read_actions(args.predictions, expected))
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
