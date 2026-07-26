#!/usr/bin/env python3
"""Steward evaluation for v2 uncertain pair and Pareto-array sorting."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_hard_surface import (
    ARRAY_STATUSES,
    HARD_INVERT_PREFERENCE,
    HARD_PREFERENCES,
    HARD_RATIONALES,
    hard_input_sha256,
    load_hard_surface_dataset,
)
from datagoblin.experiments.ldt_array_sorter import (
    classification_metrics,
    read_jsonl,
    sha256_file,
)

PREDICTION_KEYS = {
    "schema_version",
    "candidate_id",
    "seed",
    "example_id",
    "array_id",
    "left_index",
    "right_index",
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


def _validate_pair_labels(
    labels: Sequence[Mapping[str, Any]],
    heldout_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    public = {str(row["example_id"]): row for row in heldout_pairs}
    expected = {
        "schema_version",
        "example_id",
        "array_id",
        "lineage_group_sha256",
        "target",
        "authority",
    }
    result: dict[str, Mapping[str, Any]] = {}
    for label in labels:
        if set(label) != expected:
            raise ValueError("v2 pair label has invalid fields")
        if label["schema_version"] != "curation-taste-uncertain-pair-label.v2":
            raise ValueError("unsupported v2 pair label schema")
        if label["authority"] != "ENVIRONMENT_OWNED":
            raise ValueError("v2 pair label lacks environment authority")
        example_id = str(label["example_id"])
        if example_id in result or example_id not in public:
            raise ValueError("v2 pair label IDs are duplicated or unexpected")
        row = public[example_id]
        if (
            label["array_id"] != row["array_id"]
            or label["lineage_group_sha256"] != row["lineage_group_sha256"]
        ):
            raise ValueError("v2 pair label binding mismatch")
        if label["target"]["preference"] not in HARD_PREFERENCES:
            raise ValueError("v2 pair label has unknown preference")
        if label["target"]["rationale_code"] not in HARD_RATIONALES:
            raise ValueError("v2 pair label has unknown rationale")
        result[example_id] = label
    if set(result) != set(public):
        raise ValueError("v2 pair labels do not cover heldout exactly")
    return result


def _validate_array_labels(
    labels: Sequence[Mapping[str, Any]],
    heldout_arrays: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    public = {str(row["array_id"]): row for row in heldout_arrays}
    expected = {
        "schema_version",
        "array_id",
        "lineage_group_sha256",
        "target",
        "authority",
    }
    result: dict[str, Mapping[str, Any]] = {}
    for label in labels:
        if set(label) != expected:
            raise ValueError("v2 array label has invalid fields")
        if label["schema_version"] != "curation-taste-candidate-array-label.v2":
            raise ValueError("unsupported v2 array label schema")
        if label["authority"] != "ENVIRONMENT_OWNED":
            raise ValueError("v2 array label lacks environment authority")
        array_id = str(label["array_id"])
        if array_id in result or array_id not in public:
            raise ValueError("v2 array label IDs are duplicated or unexpected")
        if label["lineage_group_sha256"] != public[array_id]["lineage_group_sha256"]:
            raise ValueError("v2 array label lineage mismatch")
        statuses = label["target"]["statuses"]
        if len(statuses) != len(public[array_id]["candidates"]):
            raise ValueError("v2 array status count mismatch")
        if any(status not in ARRAY_STATUSES for status in statuses):
            raise ValueError("v2 array label has unknown status")
        result[array_id] = label
    if set(result) != set(public):
        raise ValueError("v2 array labels do not cover heldout exactly")
    return result


def _validate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    heldout_pairs: Sequence[Mapping[str, Any]],
    *,
    manifest_sha256: str,
) -> dict[tuple[str, int | None], list[Mapping[str, Any]]]:
    public = {str(row["example_id"]): row for row in heldout_pairs}
    grouped: dict[tuple[str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int | None, str]] = set()
    for prediction in predictions:
        if set(prediction) != PREDICTION_KEYS:
            raise ValueError("v2 prediction has invalid fields")
        if prediction["schema_version"] != "ldt-array-hard-surface-prediction.v2":
            raise ValueError("unsupported v2 prediction schema")
        if prediction["authority"] != "MODEL_RECOMMENDATION_ONLY":
            raise ValueError("v2 sorter asserted forbidden authority")
        if prediction["dataset_manifest_sha256"] != manifest_sha256:
            raise ValueError("v2 prediction is bound to another dataset")
        candidate_id = str(prediction["candidate_id"])
        seed = None if prediction["seed"] is None else int(prediction["seed"])
        example_id = str(prediction["example_id"])
        key = (candidate_id, seed, example_id)
        if key in seen or example_id not in public:
            raise ValueError("v2 prediction IDs are duplicated or unexpected")
        seen.add(key)
        row = public[example_id]
        binding_fields = (
            "array_id",
            "left_index",
            "right_index",
            "lineage_group_sha256",
        )
        if any(prediction[field] != row[field] for field in binding_fields):
            raise ValueError("v2 prediction binding mismatch")
        if prediction["input_sha256"] != hard_input_sha256(row):
            raise ValueError("v2 prediction input hash mismatch")
        if prediction["preference"] not in HARD_PREFERENCES:
            raise ValueError("unknown v2 predicted preference")
        if prediction["rationale_code"] not in HARD_RATIONALES:
            raise ValueError("unknown v2 predicted rationale")
        if prediction["swapped_preference"] not in HARD_PREFERENCES:
            raise ValueError("unknown v2 swapped preference")
        probabilities = prediction["preference_probabilities"]
        if set(probabilities) != set(HARD_PREFERENCES):
            raise ValueError("v2 prediction probability classes differ")
        values = [float(probabilities[label]) for label in HARD_PREFERENCES]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("v2 prediction probability is invalid")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("v2 prediction probabilities do not sum to one")
        expected_swap = (
            prediction["swapped_preference"]
            == HARD_INVERT_PREFERENCE[str(prediction["preference"])]
        )
        if bool(prediction["swap_consistent"]) != expected_swap:
            raise ValueError("v2 swap-consistency flag is incorrect")
        grouped[(candidate_id, seed)].append(prediction)
    expected_ids = set(public)
    for key, rows in grouped.items():
        if {str(row["example_id"]) for row in rows} != expected_ids:
            raise ValueError(f"v2 candidate run does not cover heldout: {key}")
    if not grouped:
        raise ValueError("v2 prediction file is empty")
    return grouped


def _derive_predicted_array_statuses(
    pair_predictions: Sequence[Mapping[str, Any]],
    arrays_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    by_array: dict[str, dict[tuple[int, int], Mapping[str, Any]]] = defaultdict(dict)
    for prediction in pair_predictions:
        by_array[str(prediction["array_id"])][
            (int(prediction["left_index"]), int(prediction["right_index"]))
        ] = prediction
    result: dict[str, list[str]] = {}
    for array_id, array_row in arrays_by_id.items():
        candidates = array_row["candidates"]
        incoming: set[int] = set()
        uncertain: set[int] = set()
        safe = [
            index
            for index, candidate in enumerate(candidates)
            if candidate["safety_qualified"]
        ]
        pair_map = by_array[array_id]
        for offset, left_index in enumerate(safe):
            for right_index in safe[offset + 1 :]:
                forward = pair_map[(left_index, right_index)]
                reverse = pair_map[(right_index, left_index)]
                consistent = (
                    forward["preference"]
                    == HARD_INVERT_PREFERENCE[str(reverse["preference"])]
                )
                if not consistent:
                    uncertain.update((left_index, right_index))
                    continue
                preference = str(forward["preference"])
                if preference == "LEFT":
                    incoming.add(right_index)
                elif preference == "RIGHT":
                    incoming.add(left_index)
                elif preference == "ABSTAIN":
                    uncertain.update((left_index, right_index))
        statuses = []
        for index, candidate in enumerate(candidates):
            if not candidate["safety_qualified"]:
                status = "UNSAFE"
            elif index in incoming:
                status = "DOMINATED"
            elif index in uncertain:
                status = "UNRESOLVED"
            else:
                status = "SELECTED"
            statuses.append(status)
        result[array_id] = statuses
    return result


def _score_run(
    predictions: Sequence[Mapping[str, Any]],
    pair_labels: Mapping[str, Mapping[str, Any]],
    array_labels: Mapping[str, Mapping[str, Any]],
    arrays_by_id: Mapping[str, Mapping[str, Any]],
    *,
    trainable_parameters: int,
    trained: bool,
) -> dict[str, Any]:
    ordered = sorted(predictions, key=lambda row: str(row["example_id"]))
    actual_preference = [
        str(pair_labels[str(row["example_id"])]["target"]["preference"])
        for row in ordered
    ]
    actual_rationale = [
        str(pair_labels[str(row["example_id"])]["target"]["rationale_code"])
        for row in ordered
    ]
    predicted_preference = [str(row["preference"]) for row in ordered]
    predicted_rationale = [str(row["rationale_code"]) for row in ordered]
    preference = classification_metrics(
        actual_preference, predicted_preference, HARD_PREFERENCES
    )
    rationale = classification_metrics(
        actual_rationale, predicted_rationale, HARD_RATIONALES
    )
    safety_indices = [
        index for index, value in enumerate(actual_rationale) if value == "safety_gate"
    ]
    safety_accuracy = sum(
        actual_preference[index] == predicted_preference[index]
        for index in safety_indices
    ) / len(safety_indices)
    safety_swap = sum(
        bool(ordered[index]["swap_consistent"]) for index in safety_indices
    ) / len(safety_indices)
    swap_consistency = sum(bool(row["swap_consistent"]) for row in ordered) / len(
        ordered
    )
    brier = sum(
        sum(
            (
                float(row["preference_probabilities"][label])
                - float(label == actual_preference[index])
            )
            ** 2
            for label in HARD_PREFERENCES
        )
        for index, row in enumerate(ordered)
    ) / len(ordered)
    predicted_statuses = _derive_predicted_array_statuses(ordered, arrays_by_id)
    actual_status_flat: list[str] = []
    predicted_status_flat: list[str] = []
    exact_arrays = 0
    selected_set_exact = 0
    for array_id in sorted(arrays_by_id):
        actual = [str(value) for value in array_labels[array_id]["target"]["statuses"]]
        predicted = predicted_statuses[array_id]
        actual_status_flat.extend(actual)
        predicted_status_flat.extend(predicted)
        exact_arrays += actual == predicted
        selected_set_exact += {
            index for index, status in enumerate(actual) if status == "SELECTED"
        } == {
            index
            for index, status in enumerate(predicted)
            if status == "SELECTED"
        }
    status = classification_metrics(
        actual_status_flat, predicted_status_flat, ARRAY_STATUSES
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
        "abstain_recall": preference["per_class"]["ABSTAIN"]["recall"],
        "tradeoff_recall": preference["per_class"]["TRADEOFF"]["recall"],
        "safety_gate_accuracy": safety_accuracy,
        "safety_swap_consistency": safety_swap,
        "swap_consistency": swap_consistency,
        "multiclass_brier": brier,
        "status_accuracy": status["accuracy"],
        "status_macro_f1": status["macro_f1"],
        "status_per_class": status["per_class"],
        "exact_array_accuracy": exact_arrays / len(arrays_by_id),
        "selected_set_exact_accuracy": selected_set_exact / len(arrays_by_id),
        "unresolved_recall": status["per_class"]["UNRESOLVED"]["recall"],
        "unsafe_exclusion_accuracy": status["per_class"]["UNSAFE"]["recall"],
        "mean_loop_delta": sum(float(row["loop_delta"]) for row in ordered)
        / len(ordered),
        "trainable_parameters": trainable_parameters,
        "trained": trained,
        "safety_qualified": safety_accuracy == 1.0 and safety_swap == 1.0,
        "orientation_qualified": swap_consistency == 1.0,
        "uncertainty_qualified": preference["per_class"]["ABSTAIN"]["recall"]
        == 1.0,
    }


def _aggregate(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scalars = (
        "preference_accuracy",
        "preference_macro_f1",
        "rationale_accuracy",
        "rationale_macro_f1",
        "joint_exact_accuracy",
        "abstain_recall",
        "tradeoff_recall",
        "safety_gate_accuracy",
        "safety_swap_consistency",
        "swap_consistency",
        "multiclass_brier",
        "status_accuracy",
        "status_macro_f1",
        "exact_array_accuracy",
        "selected_set_exact_accuracy",
        "unresolved_recall",
        "unsafe_exclusion_accuracy",
        "mean_loop_delta",
    )
    metrics = {}
    for name in scalars:
        values = [float(report[name]) for report in reports]
        metrics[name] = {
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "seed_runs": len(reports),
        "trained": bool(reports[0]["trained"]),
        "trainable_parameters": int(reports[0]["trainable_parameters"]),
        "all_seeds_safety_qualified": all(
            bool(report["safety_qualified"]) for report in reports
        ),
        "all_seeds_orientation_qualified": all(
            bool(report["orientation_qualified"]) for report in reports
        ),
        "all_seeds_uncertainty_qualified": all(
            bool(report["uncertainty_qualified"]) for report in reports
        ),
        "metrics": metrics,
    }


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any], tolerance: float) -> bool:
    maximize = (
        "preference_macro_f1",
        "abstain_recall",
        "rationale_macro_f1",
        "status_macro_f1",
        "exact_array_accuracy",
        "selected_set_exact_accuracy",
    )
    left_values = [float(left["metrics"][name]["mean"]) for name in maximize]
    right_values = [float(right["metrics"][name]["mean"]) for name in maximize]
    at_least = all(
        left_value >= right_value - tolerance
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    better = any(
        left_value > right_value + tolerance
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    equivalent = all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    cheaper = int(left["trainable_parameters"]) < int(right["trainable_parameters"])
    return at_least and (better or (equivalent and cheaper))


def _render(report: Mapping[str, Any]) -> str:
    lines = [
        "# LDT array hard-surface heldout evaluation",
        "",
        "| Candidate | Params | Qualified | Pair F1 | Abstain | Status F1 | Exact arrays | Selected sets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate_id, candidate in report["candidate_aggregates"].items():
        metrics = candidate["metrics"]
        qualified = (
            candidate["all_seeds_safety_qualified"]
            and candidate["all_seeds_orientation_qualified"]
            and candidate["all_seeds_uncertainty_qualified"]
        )
        lines.append(
            f"| {candidate_id} | {candidate['trainable_parameters']} | {qualified} | "
            f"{metrics['preference_macro_f1']['mean']:.3f} | "
            f"{metrics['abstain_recall']['mean']:.3f} | "
            f"{metrics['status_macro_f1']['mean']:.3f} | "
            f"{metrics['exact_array_accuracy']['mean']:.3f} | "
            f"{metrics['selected_set_exact_accuracy']['mean']:.3f} |"
        )
    lines.extend(
        (
            "",
            f"Learned Pareto set: `{', '.join(report['learned_pareto_set']) or 'none'}`.",
            "",
            "No universal scalar champion is selected. Environment-rule controls are not "
            "learned architecture evidence.",
            "",
            str(report["claim_boundary"]),
            "",
        )
    )
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest, _, heldout_pairs, _, heldout_arrays = load_hard_surface_dataset(
        args.dataset_manifest.resolve()
    )
    manifest_sha256 = sha256_file(args.dataset_manifest.resolve())
    custody = manifest["heldout_label_custody"]
    if sha256_file(args.pair_labels.resolve()) != custody["pair_labels_sha256"]:
        raise ValueError("v2 pair labels differ from the custody receipt")
    if sha256_file(args.array_labels.resolve()) != custody["array_labels_sha256"]:
        raise ValueError("v2 array labels differ from the custody receipt")
    pair_labels = _validate_pair_labels(
        read_jsonl(args.pair_labels.resolve()), heldout_pairs
    )
    array_labels = _validate_array_labels(
        read_jsonl(args.array_labels.resolve()), heldout_arrays
    )
    grouped = _validate_predictions(
        read_jsonl(args.predictions.resolve()),
        heldout_pairs,
        manifest_sha256=manifest_sha256,
    )
    trainer = json.loads(args.trainer_summary.read_text(encoding="utf-8"))
    if (
        trainer["status"] != "completed"
        or trainer["heldout_pair_targets_seen"]
        or trainer["heldout_array_targets_seen"]
    ):
        raise ValueError("v2 trainer receipt is incomplete or label-contaminated")
    receipts = {
        (str(row["candidate_id"]), None if row["seed"] is None else int(row["seed"])): row
        for row in trainer["candidate_receipts"]
    }
    if set(receipts) != set(grouped):
        raise ValueError("v2 trainer receipts and prediction runs differ")
    arrays_by_id = {str(row["array_id"]): row for row in heldout_arrays}
    run_reports = {}
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for (candidate_id, seed), rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            -1 if item[0][1] is None else item[0][1],
        ),
    ):
        receipt = receipts[(candidate_id, seed)]
        scored = _score_run(
            rows,
            pair_labels,
            array_labels,
            arrays_by_id,
            trainable_parameters=int(receipt["trainable_parameters"]),
            trained=bool(receipt["trained"]),
        )
        key = f"{candidate_id}:control" if seed is None else f"{candidate_id}:seed-{seed}"
        run_reports[key] = {
            "candidate_id": candidate_id,
            "seed": seed,
            **scored,
        }
        by_candidate[candidate_id].append(scored)
    aggregates = {
        candidate_id: _aggregate(reports)
        for candidate_id, reports in sorted(by_candidate.items())
    }
    eligible = {
        candidate_id: aggregate
        for candidate_id, aggregate in aggregates.items()
        if aggregate["trained"]
        and aggregate["all_seeds_safety_qualified"]
        and aggregate["all_seeds_orientation_qualified"]
        and aggregate["all_seeds_uncertainty_qualified"]
    }
    tolerance = 0.01
    pareto = sorted(
        candidate_id
        for candidate_id, candidate in eligible.items()
        if not any(
            other_id != candidate_id
            and _dominates(other, candidate, tolerance)
            for other_id, other in eligible.items()
        )
    )
    report = {
        "schema_version": "ldt-array-hard-surface-heldout-evaluation.v2",
        "experiment_id": trainer["experiment_id"],
        "dataset_manifest_sha256": manifest_sha256,
        "predictions_sha256": sha256_file(args.predictions.resolve()),
        "trainer_summary_sha256": sha256_file(args.trainer_summary.resolve()),
        "pair_labels_sha256": sha256_file(args.pair_labels.resolve()),
        "array_labels_sha256": sha256_file(args.array_labels.resolve()),
        "heldout_label_authority": "ENVIRONMENT_OWNED",
        "heldout_reveal_count": 1,
        "post_reveal_tuning_permitted": False,
        "candidate_run_reports": run_reports,
        "candidate_aggregates": aggregates,
        "learned_pareto_set": pareto,
        "universal_scalar_champion": None,
        "environment_rule_controls_are_learned_evidence": False,
        "larger_model_training_authorized": False,
        "claim_boundary": (
            "Heldout evidence applies only to the synthetic uncertain pair and "
            "candidate-array task; the sorter cannot promote corpus records."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    _write_json(args.out / "heldout_evaluation.json", report)
    (args.out / "heldout_evaluation.md").write_text(
        _render(report), encoding="utf-8"
    )
    _write_json(
        args.out / "evaluation_receipt.json",
        {
            "schema_version": "ldt-array-hard-surface-evaluation-receipt.v2",
            "heldout_evaluation_sha256": sha256_file(
                args.out / "heldout_evaluation.json"
            ),
            "row_level_labels_exported": False,
            "candidate_workspace_mutated": False,
            "universal_scalar_champion_selected": False,
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate v2 uncertain pair and Pareto-array predictions"
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trainer-summary", type=Path, required=True)
    parser.add_argument("--pair-labels", type=Path, required=True)
    parser.add_argument("--array-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    result = evaluate(build_parser().parse_args())
    print(json.dumps({"learned_pareto_set": result["learned_pareto_set"]}, sort_keys=True))
