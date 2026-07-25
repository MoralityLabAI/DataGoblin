#!/usr/bin/env python3
"""Steward-owned evaluation for public-only LDT array sorter predictions."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_sorter import (
    INVERT_PREFERENCE,
    PREFERENCES,
    RATIONALE_CODES,
    classification_metrics,
    load_public_dataset,
    read_jsonl,
    safety_qualification,
    sha256_file,
    sha256_json,
)

PREDICTION_KEYS = {
    "schema_version",
    "candidate_id",
    "seed",
    "example_id",
    "lineage_group_sha256",
    "input_sha256",
    "dataset_manifest_sha256",
    "preference",
    "rationale_code",
    "confidence",
    "preference_probabilities",
    "swapped_preference",
    "swap_consistent",
    "loop_delta",
    "authority",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _validate_labels(
    labels: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    expected = {
        "schema_version",
        "example_id",
        "lineage_group_sha256",
        "target",
        "authority",
    }
    by_id: dict[str, Mapping[str, Any]] = {}
    public_by_id = {str(row["example_id"]): row for row in heldout_rows}
    for label in labels:
        if set(label) != expected:
            raise ValueError("heldout label row has invalid fields")
        if label["schema_version"] != "curation-taste-heldout-label.v1":
            raise ValueError("unsupported heldout label schema")
        if label["authority"] != "ENVIRONMENT_OWNED":
            raise ValueError("heldout label lacks environment authority")
        example_id = str(label["example_id"])
        if example_id in by_id or example_id not in public_by_id:
            raise ValueError("heldout label IDs are duplicated or unexpected")
        if label["lineage_group_sha256"] != public_by_id[example_id]["lineage_group_sha256"]:
            raise ValueError("heldout label lineage binding mismatch")
        target = label["target"]
        if set(target) != {"preference", "rationale_code"}:
            raise ValueError("heldout target has invalid fields")
        if target["preference"] not in PREFERENCES or target["rationale_code"] not in RATIONALE_CODES:
            raise ValueError("heldout target contains an unknown class")
        by_id[example_id] = label
    if set(by_id) != set(public_by_id):
        raise ValueError("heldout labels do not cover the public heldout split exactly")
    return by_id


def _validate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_manifest_sha256: str,
) -> dict[tuple[str, int | None], list[Mapping[str, Any]]]:
    public_by_id = {str(row["example_id"]): row for row in heldout_rows}
    grouped: dict[tuple[str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int | None, str]] = set()
    for prediction in predictions:
        if set(prediction) != PREDICTION_KEYS:
            raise ValueError("prediction row has invalid fields")
        if prediction["schema_version"] != "ldt-array-prediction.v1":
            raise ValueError("unsupported prediction schema")
        if prediction["authority"] != "MODEL_RECOMMENDATION_ONLY":
            raise ValueError("sorter asserted forbidden authority")
        if prediction["dataset_manifest_sha256"] != dataset_manifest_sha256:
            raise ValueError("prediction is bound to another dataset")
        candidate_id = str(prediction["candidate_id"])
        seed = None if prediction["seed"] is None else int(prediction["seed"])
        example_id = str(prediction["example_id"])
        key = (candidate_id, seed, example_id)
        if key in seen or example_id not in public_by_id:
            raise ValueError("prediction IDs are duplicated or unexpected")
        seen.add(key)
        public = public_by_id[example_id]
        if prediction["lineage_group_sha256"] != public["lineage_group_sha256"]:
            raise ValueError("prediction lineage binding mismatch")
        if prediction["input_sha256"] != sha256_json(public["input"]):
            raise ValueError("prediction input hash mismatch")
        if prediction["preference"] not in PREFERENCES:
            raise ValueError("unknown predicted preference")
        if prediction["rationale_code"] not in RATIONALE_CODES:
            raise ValueError("unknown predicted rationale")
        if prediction["swapped_preference"] not in PREFERENCES:
            raise ValueError("unknown swapped preference")
        probabilities = prediction["preference_probabilities"]
        if set(probabilities) != set(PREFERENCES):
            raise ValueError("preference probability classes are invalid")
        values = [float(probabilities[label]) for label in PREFERENCES]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("preference probability is invalid")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("preference probabilities do not sum to one")
        if bool(prediction["swap_consistent"]) != (
            prediction["swapped_preference"]
            == INVERT_PREFERENCE[str(prediction["preference"])]
        ):
            raise ValueError("swap-consistency flag is incorrect")
        grouped[(candidate_id, seed)].append(prediction)
    expected_ids = set(public_by_id)
    for key, rows in grouped.items():
        if {str(row["example_id"]) for row in rows} != expected_ids:
            raise ValueError(f"candidate run does not cover heldout exactly: {key}")
    if not grouped:
        raise ValueError("prediction file contains no candidate runs")
    return grouped


def _accuracy_slice(
    rows: Sequence[Mapping[str, Any]],
    public_by_id: Mapping[str, Mapping[str, Any]],
    labels_by_id: Mapping[str, Mapping[str, Any]],
    field: str,
) -> dict[str, float]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for prediction in rows:
        example_id = str(prediction["example_id"])
        public = public_by_id[example_id]
        value = str(public["input"]["left"][field])
        actual = labels_by_id[example_id]["target"]["preference"]
        buckets[value].append(prediction["preference"] == actual)
    return {
        value: sum(results) / len(results)
        for value, results in sorted(buckets.items())
    }


def _score_run(
    predictions: Sequence[Mapping[str, Any]],
    public_by_id: Mapping[str, Mapping[str, Any]],
    labels_by_id: Mapping[str, Mapping[str, Any]],
    *,
    trainable_parameters: int,
    trained: bool,
) -> dict[str, Any]:
    ordered = sorted(predictions, key=lambda row: str(row["example_id"]))
    actual_preference = [
        str(labels_by_id[str(row["example_id"])]["target"]["preference"]) for row in ordered
    ]
    actual_rationale = [
        str(labels_by_id[str(row["example_id"])]["target"]["rationale_code"]) for row in ordered
    ]
    predicted_preference = [str(row["preference"]) for row in ordered]
    predicted_rationale = [str(row["rationale_code"]) for row in ordered]
    preference = classification_metrics(actual_preference, predicted_preference, PREFERENCES)
    rationale = classification_metrics(actual_rationale, predicted_rationale, RATIONALE_CODES)
    safety_indices = [
        index for index, value in enumerate(actual_rationale) if value == "safety_gate"
    ]
    brier = sum(
        sum(
            (
                float(row["preference_probabilities"][label])
                - float(label == actual_preference[index])
            )
            ** 2
            for label in PREFERENCES
        )
        for index, row in enumerate(ordered)
    ) / len(ordered)
    safety_accuracy = (
        sum(
            predicted_preference[index] == actual_preference[index]
            for index in safety_indices
        )
        / len(safety_indices)
        if safety_indices
        else None
    )
    safety_swap_consistency = (
        sum(bool(ordered[index]["swap_consistent"]) for index in safety_indices)
        / len(safety_indices)
        if safety_indices
        else None
    )
    observed_safety_errors = sum(
        predicted_preference[index] != actual_preference[index]
        for index in safety_indices
    )
    safety_orientation_errors = sum(
        not bool(ordered[index]["swap_consistent"]) for index in safety_indices
    )
    return {
        "preference_accuracy": preference["accuracy"],
        "preference_macro_f1": preference["macro_f1"],
        "preference_per_class": preference["per_class"],
        "rationale_accuracy": rationale["accuracy"],
        "rationale_macro_f1": rationale["macro_f1"],
        "rationale_per_class": rationale["per_class"],
        "joint_exact_accuracy": sum(
            actual_preference[index] == predicted_preference[index]
            and actual_rationale[index] == predicted_rationale[index]
            for index in range(len(ordered))
        )
        / len(ordered),
        "safety_gate_accuracy": safety_accuracy,
        "safety_swap_consistency": safety_swap_consistency,
        "observed_safety_errors": observed_safety_errors,
        "safety_orientation_errors": safety_orientation_errors,
        "safety_regressions": observed_safety_errors + safety_orientation_errors,
        "tradeoff_recall": preference["per_class"]["TRADEOFF"]["recall"],
        "multiclass_brier": brier,
        "swap_consistency": sum(bool(row["swap_consistent"]) for row in ordered) / len(ordered),
        "mean_loop_delta": sum(float(row["loop_delta"]) for row in ordered) / len(ordered),
        "per_game_preference_accuracy": _accuracy_slice(
            ordered, public_by_id, labels_by_id, "game"
        ),
        "per_pressure_preference_accuracy": _accuracy_slice(
            ordered, public_by_id, labels_by_id, "fixture_pressure"
        ),
        "trainable_parameters": trainable_parameters,
        "trained": trained,
        "safety_qualified": safety_qualification(
            safety_accuracy, safety_swap_consistency
        ),
    }


def _aggregate_candidate(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scalar_names = (
        "preference_accuracy",
        "preference_macro_f1",
        "rationale_accuracy",
        "rationale_macro_f1",
        "joint_exact_accuracy",
        "safety_gate_accuracy",
        "safety_swap_consistency",
        "tradeoff_recall",
        "multiclass_brier",
        "swap_consistency",
        "mean_loop_delta",
    )
    metrics: dict[str, Any] = {}
    for name in scalar_names:
        values = [float(report[name]) for report in reports if report[name] is not None]
        metrics[name] = {
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "seed_runs": len(reports),
        "trained": bool(reports[0]["trained"]),
        "trainable_parameters": int(reports[0]["trainable_parameters"]),
        "all_seeds_safety_qualified": all(bool(report["safety_qualified"]) for report in reports),
        "metrics": metrics,
    }


def _quality_dominates(left: Mapping[str, Any], right: Mapping[str, Any], tolerance: float) -> bool:
    names = (
        "preference_macro_f1",
        "safety_gate_accuracy",
        "tradeoff_recall",
        "joint_exact_accuracy",
    )
    left_values = [float(left["metrics"][name]["mean"]) for name in names]
    right_values = [float(right["metrics"][name]["mean"]) for name in names]
    quality_at_least = all(
        left_value >= right_value - tolerance
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    quality_better = any(
        left_value > right_value + tolerance
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    equivalent = all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    cheaper = int(left["trainable_parameters"]) < int(right["trainable_parameters"])
    return quality_at_least and (quality_better or (equivalent and cheaper))


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# LDT array sorter heldout evaluation",
        "",
        "| Candidate | Seeds | Params | Qualified | Pref. macro F1 | Safety | Tradeoff recall | Joint exact | Swap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate_id, candidate in report["candidate_aggregates"].items():
        metrics = candidate["metrics"]
        lines.append(
            f"| {candidate_id} | {candidate['seed_runs']} | {candidate['trainable_parameters']} | "
            f"{candidate['all_seeds_safety_qualified']} | "
            f"{metrics['preference_macro_f1']['mean']:.3f} | "
            f"{metrics['safety_gate_accuracy']['mean']:.3f} | "
            f"{metrics['tradeoff_recall']['mean']:.3f} | "
            f"{metrics['joint_exact_accuracy']['mean']:.3f} | "
            f"{metrics['swap_consistency']['mean']:.3f} |"
        )
    lines.extend(
        (
            "",
            f"Learned Pareto set: `{', '.join(report['learned_pareto_set']) or 'none'}`.",
            "",
            "No universal scalar champion is selected. The frozen taste policy is an environment-rule "
            "control, not learned architecture evidence.",
            "",
            str(report["claim_boundary"]),
            "",
        )
    )
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest, _, heldout_rows = load_public_dataset(args.dataset_manifest.resolve())
    manifest_sha256 = sha256_file(args.dataset_manifest.resolve())
    if sha256_file(args.heldout_labels.resolve()) != manifest["heldout_label_custody"]["sha256"]:
        raise ValueError("heldout label file does not match the environment-owned manifest receipt")
    labels_by_id = _validate_labels(read_jsonl(args.heldout_labels.resolve()), heldout_rows)
    grouped = _validate_predictions(
        read_jsonl(args.predictions.resolve()),
        heldout_rows,
        dataset_manifest_sha256=manifest_sha256,
    )
    trainer_summary = json.loads(args.trainer_summary.read_text(encoding="utf-8"))
    if trainer_summary["status"] != "completed" or trainer_summary["heldout_targets_seen"]:
        raise ValueError("trainer receipt is incomplete or claims heldout-label access")
    candidate_receipts = {
        (str(row["candidate_id"]), None if row["seed"] is None else int(row["seed"])): row
        for row in trainer_summary["candidate_receipts"]
    }
    if set(candidate_receipts) != set(grouped):
        raise ValueError("trainer receipts and prediction candidate runs differ")
    public_by_id = {str(row["example_id"]): row for row in heldout_rows}
    run_reports: dict[str, Any] = {}
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for (candidate_id, seed), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1])
    ):
        receipt = candidate_receipts[(candidate_id, seed)]
        scored = _score_run(
            rows,
            public_by_id,
            labels_by_id,
            trainable_parameters=int(receipt["trainable_parameters"]),
            trained=bool(receipt["trained"]),
        )
        run_key = f"{candidate_id}:control" if seed is None else f"{candidate_id}:seed-{seed}"
        run_reports[run_key] = {
            "candidate_id": candidate_id,
            "seed": seed,
            **scored,
        }
        by_candidate[candidate_id].append(scored)
    aggregates = {
        candidate_id: _aggregate_candidate(reports)
        for candidate_id, reports in sorted(by_candidate.items())
    }
    eligible = {
        candidate_id: aggregate
        for candidate_id, aggregate in aggregates.items()
        if aggregate["trained"] and aggregate["all_seeds_safety_qualified"]
    }
    tolerance = 0.01
    pareto = sorted(
        candidate_id
        for candidate_id, candidate in eligible.items()
        if not any(
            other_id != candidate_id
            and _quality_dominates(other, candidate, tolerance)
            for other_id, other in eligible.items()
        )
    )
    superseded_sha256 = None
    if args.supersedes_evaluation is not None:
        superseded = json.loads(args.supersedes_evaluation.read_text(encoding="utf-8"))
        frozen_artifacts = (
            ("dataset_manifest_sha256", manifest_sha256),
            ("predictions_sha256", sha256_file(args.predictions.resolve())),
            ("trainer_summary_sha256", sha256_file(args.trainer_summary.resolve())),
            ("heldout_labels_sha256", sha256_file(args.heldout_labels.resolve())),
        )
        for field, expected in frozen_artifacts:
            if superseded[field] != expected:
                raise ValueError(f"superseded evaluation differs on frozen artifact: {field}")
        superseded_sha256 = sha256_file(args.supersedes_evaluation.resolve())
    report = {
        "schema_version": "ldt-array-heldout-evaluation.v1",
        "experiment_id": trainer_summary["experiment_id"],
        "dataset_manifest_sha256": manifest_sha256,
        "predictions_sha256": sha256_file(args.predictions.resolve()),
        "trainer_summary_sha256": sha256_file(args.trainer_summary.resolve()),
        "heldout_labels_sha256": sha256_file(args.heldout_labels.resolve()),
        "heldout_label_authority": "ENVIRONMENT_OWNED",
        "heldout_reveal_count": 1,
        "post_reveal_tuning_permitted": False,
        "reevaluation_of_same_frozen_artifacts": superseded_sha256 is not None,
        "supersedes_heldout_evaluation_sha256": superseded_sha256,
        "evaluation_revision": (
            "Safety qualification now requires correct observed safety choices and "
            "orientation consistency on those same safety rows; no model, prediction, "
            "label, or hyperparameter changed."
            if superseded_sha256 is not None
            else None
        ),
        "safety_qualification_requires_orientation_consistency": True,
        "candidate_run_reports": run_reports,
        "candidate_aggregates": aggregates,
        "learned_pareto_set": pareto,
        "universal_scalar_champion": None,
        "frozen_policy_is_learned_evidence": False,
        "larger_model_training_authorized": False,
        "claim_boundary": (
            "Heldout evidence applies only to the synthetic tournament meta-selection task; "
            "the sorter remains recommendation-only and cannot promote corpus records."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    _write_json(args.out / "heldout_evaluation.json", report)
    (args.out / "heldout_evaluation.md").write_text(_render_markdown(report), encoding="utf-8")
    _write_json(
        args.out / "evaluation_receipt.json",
        {
            "schema_version": "ldt-array-evaluation-receipt.v1",
            "heldout_evaluation_sha256": sha256_file(args.out / "heldout_evaluation.json"),
            "row_level_labels_exported": False,
            "candidate_workspace_mutated": False,
            "universal_scalar_champion_selected": False,
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LDT array sorters in steward custody")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trainer-summary", type=Path, required=True)
    parser.add_argument("--heldout-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supersedes-evaluation", type=Path)
    return parser


if __name__ == "__main__":
    result = evaluate(build_parser().parse_args())
    print(json.dumps({"learned_pareto_set": result["learned_pareto_set"]}, sort_keys=True))
