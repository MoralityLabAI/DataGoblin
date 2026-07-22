from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.experiments.recipe_verdict_dataset import read_dataset_rows
from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured, load_structured

from .contract import load_gym_contract

DECISIONS = {"ACCEPT", "REVISE", "AUDIT", "ABSTAIN", "REJECT"}
EVALUATION_SPLITS = {"ordinary_test", "ood_test", "sealed_poison"}


def evaluate_candidate(
    contract_path: Path,
    dataset_manifest_path: Path,
    candidate_manifest_path: Path,
    predictions_path: Path,
    out_dir: Path,
    *,
    sealed_labels_path: Path | None = None,
) -> dict[str, Any]:
    contract = load_gym_contract(contract_path)
    candidate = load_structured(candidate_manifest_path)
    _validate_candidate_manifest(candidate, dataset_manifest_path)
    dataset_rows = read_dataset_rows(dataset_manifest_path)
    predictions = _read_predictions(predictions_path)
    expected_ids = {str(row["example_id"]) for row in dataset_rows}
    if set(predictions) != expected_ids:
        missing = sorted(expected_ids - set(predictions))
        extra = sorted(set(predictions) - expected_ids)
        raise ValueError(f"prediction identity mismatch; missing={missing[:3]} extra={extra[:3]}")
    sealed_labels = _read_sealed_labels(sealed_labels_path)
    pairs: list[dict[str, Any]] = []
    for row in dataset_rows:
        label = row.get("authoritative_label")
        if row["split"] == "sealed_poison":
            stewarded = sealed_labels.get(str(row["example_id"]))
            if stewarded is not None:
                if stewarded["recipe_hash"] != row["recipe_hash"]:
                    raise ValueError(f"sealed recipe hash mismatch for {row['example_id']}")
                label = stewarded["authoritative_label"]
        pairs.append({"dataset": row, "prediction": predictions[str(row["example_id"])], "label": label})
    split_results: dict[str, Any] = {}
    for split in ("train", "calibration", "ordinary_test", "ood_test", "sealed_poison"):
        selected = [pair for pair in pairs if pair["dataset"]["split"] == split and pair["label"] is not None]
        split_results[split] = _score_split(selected, int(candidate.get("maximum_loops", 1))) if selected else None
    resource = candidate["resource_summary"]
    report = {
        "schema_version": "1.0",
        "gym_id": contract["gym_id"],
        "candidate_id": candidate["candidate_id"],
        "architecture_family": candidate["architecture_family"],
        "evidence_status": "pipeline_smoke_only" if candidate.get("mock") else "candidate_evaluation",
        "candidate_budget": {
            "mock": bool(candidate.get("mock")),
            "trainable_parameters": int(candidate.get("trainable_parameters", 0)),
            "total_parameters": int(candidate.get("total_parameters", 0)),
            "maximum_loops": int(candidate["maximum_loops"]),
            "inference_update_budget": int(candidate.get("inference_update_budget", 0)),
        },
        "selection_results": {"calibration": split_results["calibration"]},
        "evidence_results": {name: split_results[name] for name in sorted(EVALUATION_SPLITS)},
        "diagnostic_results": {"train": split_results["train"]},
        "resource_summary": resource,
        "promotion_assessment": _promotion_assessment(contract, candidate, split_results),
        "reproducibility": {
            "contract_sha256": sha256_file(contract_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
            "predictions_sha256": sha256_file(predictions_path),
            "sealed_labels_sha256": sha256_file(sealed_labels_path) if sealed_labels_path else None,
        },
        "metric_firewall": "calibration is selection-only; ordinary, OOD, and sealed splits are evidence-only",
        "mandatory_non_claims": contract["mandatory_non_claims"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_structured(out_dir / "evaluation_report.json", report)
    _write_scored_rows(out_dir / "scored_rows.jsonl", pairs)
    (out_dir / "evaluation_report.md").write_text(_render_report(report), encoding="utf-8")
    return report


def _validate_candidate_manifest(candidate: Mapping[str, Any], dataset_manifest_path: Path) -> None:
    required = {
        "schema_version",
        "candidate_id",
        "architecture_family",
        "mock",
        "weights_loaded",
        "training_splits",
        "training_split_hashes",
        "dataset_manifest_sha256",
        "maximum_loops",
        "resource_summary",
    }
    missing = sorted(required - candidate.keys())
    if missing:
        raise ValueError(f"candidate manifest missing fields: {', '.join(missing)}")
    if candidate["schema_version"] != "1.0":
        raise ValueError("unsupported candidate manifest version")
    if candidate["dataset_manifest_sha256"] != sha256_file(dataset_manifest_path):
        raise ValueError("candidate was not bound to this dataset manifest")
    leaked = set(candidate["training_splits"]) & EVALUATION_SPLITS
    if leaked:
        raise ValueError(f"candidate declares evaluation data in training splits: {sorted(leaked)}")
    if int(candidate["maximum_loops"]) < 1:
        raise ValueError("maximum_loops must be positive")
    dataset_manifest = load_structured(dataset_manifest_path)
    declared_hashes = candidate["training_split_hashes"]
    if not isinstance(declared_hashes, Mapping) or not isinstance(candidate["resource_summary"], Mapping):
        raise ValueError("candidate training hashes and resource summary must be objects")
    for split in candidate["training_splits"]:
        expected = dataset_manifest["splits"].get(split, {}).get("sha256")
        if expected is None or declared_hashes.get(split) != expected:
            raise ValueError(f"training split hash is missing or incorrect: {split}")
    if not candidate["mock"]:
        required_resource_fields = {
            "caps_enforced",
            "peak_ram_mb",
            "avg_ram_mb",
            "peak_io_mb_s",
            "cpu_pct",
            "steps_completed",
            "cleanup_passed",
            "events_sha256",
            "cleanup_summary_sha256",
        }
        missing_resource = sorted(required_resource_fields - candidate["resource_summary"].keys())
        if missing_resource:
            raise ValueError(f"real candidate resource summary missing fields: {', '.join(missing_resource)}")


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("example_id", ""))
            if not example_id or example_id in values:
                raise ValueError(f"invalid or duplicate prediction identity at line {line_number}")
            parsed = bool(row.get("parsed", False))
            if parsed:
                if row.get("decision") not in DECISIONS:
                    raise ValueError(f"invalid parsed decision for {example_id}")
                confidence = float(row.get("confidence", -1.0))
                if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise ValueError(f"invalid confidence for {example_id}")
                if not isinstance(row.get("violations", []), list):
                    raise ValueError(f"violations must be a list for {example_id}")
            values[example_id] = row
    return values


def _read_sealed_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    values: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                values[str(row["example_id"])] = row
    return values


def _score_split(pairs: Sequence[Mapping[str, Any]], maximum_loops: int) -> dict[str, Any]:
    total = len(pairs)
    parsed_count = sum(bool(pair["prediction"].get("parsed")) for pair in pairs)
    exact_decisions: list[float] = []
    exact_records: list[float] = []
    confidences: list[float] = []
    abstained: list[bool] = []
    true_sets: list[set[str]] = []
    predicted_sets: list[set[str]] = []
    false_accepts = 0
    critical_rows = 0
    loop_counts: list[int] = []
    nonterminations = 0
    flips = 0
    flip_opportunities = 0
    for pair in pairs:
        prediction = pair["prediction"]
        label = pair["label"]
        parsed = bool(prediction.get("parsed"))
        prediction_decision = prediction.get("decision") if parsed else None
        prediction_violations = set(str(item) for item in prediction.get("violations", ())) if parsed else set()
        actual_violations = set(str(item) for item in label["violations"])
        decision_correct = prediction_decision == label["decision"]
        record_correct = decision_correct and prediction_violations == actual_violations
        exact_decisions.append(float(decision_correct))
        exact_records.append(float(record_correct))
        confidences.append(float(prediction.get("confidence", 0.0)) if parsed else 0.0)
        abstained.append(bool(prediction.get("abstained")) or prediction_decision == "ABSTAIN" or not parsed)
        true_sets.append(actual_violations)
        predicted_sets.append(prediction_violations)
        if label["decision"] == "REJECT":
            critical_rows += 1
            false_accepts += int(prediction_decision == "ACCEPT")
        trace = list(prediction.get("loop_trace", ()))
        loop_counts.append(len(trace))
        if not prediction.get("halted", False) or len(trace) > maximum_loops:
            nonterminations += 1
        trace_decisions = [item.get("decision") for item in trace]
        flips += sum(left != right for left, right in pairwise(trace_decisions))
        flip_opportunities += max(0, len(trace_decisions) - 1)
    micro_f1, macro_f1 = _violation_f1(true_sets, predicted_sets)
    risk_curve, aurc = _risk_coverage(exact_records, confidences, abstained)
    return {
        "examples": total,
        "schema_parse_rate": parsed_count / total,
        "schema_failure_rate": 1.0 - parsed_count / total,
        "exact_decision_accuracy": float(np.mean(exact_decisions)),
        "exact_record_accuracy": float(np.mean(exact_records)),
        "violation_micro_f1": micro_f1,
        "violation_macro_f1": macro_f1,
        "brier_score": float(np.mean((np.asarray(confidences) - np.asarray(exact_records)) ** 2)),
        "expected_calibration_error": _ece(exact_records, confidences),
        "abstention_rate": sum(abstained) / total,
        "critical_false_acceptance_rate": false_accepts / critical_rows if critical_rows else 0.0,
        "risk_coverage_curve": risk_curve,
        "area_under_risk_coverage_curve": aurc,
        "mean_loops": float(np.mean(loop_counts)),
        "maximum_observed_loops": max(loop_counts, default=0),
        "nontermination_rate": nonterminations / total,
        "decision_flip_rate": flips / flip_opportunities if flip_opportunities else 0.0,
    }


def _violation_f1(actual: Sequence[set[str]], predicted: Sequence[set[str]]) -> tuple[float, float]:
    tp = sum(len(left & right) for left, right in zip(actual, predicted, strict=True))
    fp = sum(len(right - left) for left, right in zip(actual, predicted, strict=True))
    fn = sum(len(left - right) for left, right in zip(actual, predicted, strict=True))
    micro = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    per_example = []
    for left, right in zip(actual, predicted, strict=True):
        numerator = 2 * len(left & right)
        denominator = len(left) + len(right)
        per_example.append(numerator / denominator if denominator else 1.0)
    return micro, float(np.mean(per_example))


def _ece(labels: Sequence[float], probabilities: Sequence[float], bins: int = 10) -> float:
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        indices = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper or (upper >= 1.0 and probability == 1.0)
        ]
        if indices:
            confidence = float(np.mean([probabilities[index] for index in indices]))
            accuracy = float(np.mean([labels[index] for index in indices]))
            result += len(indices) / len(labels) * abs(confidence - accuracy)
    return result


def _risk_coverage(
    correct: Sequence[float], confidence: Sequence[float], abstained: Sequence[bool]
) -> tuple[list[dict[str, float]], float]:
    ordered = sorted(
        ((confidence[index], correct[index]) for index in range(len(correct)) if not abstained[index]),
        reverse=True,
    )
    curve: list[dict[str, float]] = []
    for count in range(1, len(ordered) + 1):
        risk = 1.0 - float(np.mean([item[1] for item in ordered[:count]]))
        curve.append({"coverage": count / len(correct), "risk": risk})
    if not curve:
        return [], 1.0
    previous_coverage = 0.0
    area = 0.0
    for point in curve:
        area += (point["coverage"] - previous_coverage) * point["risk"]
        previous_coverage = point["coverage"]
    area += (1.0 - previous_coverage) * 1.0
    return curve, area


def _promotion_assessment(
    contract: Mapping[str, Any], candidate: Mapping[str, Any], splits: Mapping[str, Any]
) -> dict[str, Any]:
    gates = contract["promotion_gates"]
    ordinary = splits.get("ordinary_test")
    ood = splits.get("ood_test")
    sealed = splits.get("sealed_poison")
    checks = {
        "not_mock": not bool(candidate.get("mock")),
        "all_evaluation_tasks_executable": all(
            task.get("status") == "executable" for task in contract.get("evaluation_tasks", {}).values()
        ),
        "ordinary_available": ordinary is not None,
        "ood_available": ood is not None,
        "sealed_available": sealed is not None,
        "schema_parse": bool(ordinary) and ordinary["schema_parse_rate"] >= float(gates["schema_parse_rate"]),
        "critical_false_acceptance": bool(ordinary)
        and ordinary["critical_false_acceptance_rate"]
        <= float(gates["maximum_critical_false_acceptance_rate"]),
        "termination": bool(ordinary)
        and ordinary["nontermination_rate"] <= float(gates["maximum_nontermination_rate"]),
        "caps_enforced": bool(candidate["resource_summary"].get("caps_enforced")),
        "cleanup": bool(candidate["resource_summary"].get("cleanup_passed")),
        "matched_baseline_comparison_complete": False,
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "boundary": "Promotion additionally requires a matched frozen-base comparison and fresh confirmatory seeds.",
    }


def _write_scored_rows(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            dataset = pair["dataset"]
            value = {
                "example_id": dataset["example_id"],
                "split": dataset["split"],
                "category": dataset["category"],
                "prediction": pair["prediction"],
                "label_available_to_evaluator": pair["label"] is not None,
            }
            if dataset["split"] != "sealed_poison" and pair["label"] is not None:
                value["authoritative_label"] = pair["label"]
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _render_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Training-gym evaluation",
        "",
        f"Candidate: `{report['candidate_id']}` ({report['architecture_family']}); "
        f"evidence status: `{report['evidence_status']}`.",
        "",
        "| Split | Exact decision | Exact record | Violation F1 | Brier | ECE | Critical false accept | AURC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, result in report["evidence_results"].items():
        if result is None:
            unavailable = " | ".join(["unavailable"] * 7)
            lines.append(f"| {split} | {unavailable} |")
        else:
            lines.append(
                f"| {split} | {result['exact_decision_accuracy']:.3f} | {result['exact_record_accuracy']:.3f} | "
                f"{result['violation_micro_f1']:.3f} | {result['brier_score']:.3f} | "
                f"{result['expected_calibration_error']:.3f} | "
                f"{result['critical_false_acceptance_rate']:.3f} | "
                f"{result['area_under_risk_coverage_curve']:.3f} |"
            )
    lines.extend(("", "## Promotion", "", f"Eligible: `{report['promotion_assessment']['eligible']}`.", ""))
    lines.append(str(report["promotion_assessment"]["boundary"]))
    lines.extend(("", "## Claim boundary", ""))
    lines.extend(f"- {item}" for item in report["mandatory_non_claims"])
    return "\n".join(lines) + "\n"
