#!/usr/bin/env python3
"""Steward evaluation for variable-array LDT confirmation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_confirmation import (
    confirmation_input_sha256,
    load_confirmation_dataset,
    reduce_confirmation_array,
)
from datagoblin.experiments.ldt_array_hard_surface import (
    ARRAY_STATUSES,
    HARD_INVERT_PREFERENCE,
    HARD_PREFERENCES,
    HARD_RATIONALES,
)
from datagoblin.experiments.ldt_array_sorter import (
    classification_metrics,
    read_jsonl,
    sha256_file,
)
from datagoblin.serialization import load_structured

PREDICTION_KEYS = {
    "schema_version",
    "candidate_id",
    "seed",
    "example_id",
    "array_id",
    "left_index",
    "right_index",
    "left_tournament_report_sha256",
    "right_tournament_report_sha256",
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
    "calibration_temperature",
    "authority",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


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
    result = {}
    for label in labels:
        if set(label) != expected:
            raise ValueError("confirmation pair label has invalid fields")
        if (
            label["schema_version"]
            != "curation-taste-confirmation-pair-label.v3"
        ):
            raise ValueError("unsupported confirmation pair label schema")
        if label["authority"] != "ENVIRONMENT_OWNED":
            raise ValueError("confirmation pair label lacks environment authority")
        example_id = str(label["example_id"])
        if example_id in result or example_id not in public:
            raise ValueError("confirmation pair label IDs differ")
        row = public[example_id]
        if (
            label["array_id"] != row["array_id"]
            or label["lineage_group_sha256"] != row["lineage_group_sha256"]
        ):
            raise ValueError("confirmation pair label binding mismatch")
        if label["target"]["preference"] not in HARD_PREFERENCES:
            raise ValueError("confirmation pair label preference is unknown")
        if label["target"]["rationale_code"] not in HARD_RATIONALES:
            raise ValueError("confirmation pair label rationale is unknown")
        result[example_id] = label
    if set(result) != set(public):
        raise ValueError("confirmation pair labels do not cover heldout")
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
    result = {}
    for label in labels:
        if set(label) != expected:
            raise ValueError("confirmation array label has invalid fields")
        if (
            label["schema_version"]
            != "curation-taste-confirmation-array-label.v3"
        ):
            raise ValueError("unsupported confirmation array label schema")
        if label["authority"] != "ENVIRONMENT_OWNED":
            raise ValueError("confirmation array label lacks environment authority")
        array_id = str(label["array_id"])
        if array_id in result or array_id not in public:
            raise ValueError("confirmation array label IDs differ")
        row = public[array_id]
        if label["lineage_group_sha256"] != row["lineage_group_sha256"]:
            raise ValueError("confirmation array label lineage mismatch")
        statuses = label["target"]["statuses"]
        if len(statuses) != int(row["candidate_count"]):
            raise ValueError("confirmation array status count mismatch")
        if any(status not in ARRAY_STATUSES for status in statuses):
            raise ValueError("confirmation array label status is unknown")
        result[array_id] = label
    if set(result) != set(public):
        raise ValueError("confirmation array labels do not cover heldout")
    return result


def _validate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    heldout_pairs: Sequence[Mapping[str, Any]],
    *,
    manifest_sha256: str,
) -> dict[tuple[str, int | None], list[Mapping[str, Any]]]:
    public = {str(row["example_id"]): row for row in heldout_pairs}
    grouped: dict[tuple[str, int | None], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    seen = set()
    for prediction in predictions:
        if set(prediction) != PREDICTION_KEYS:
            raise ValueError("confirmation prediction has invalid fields")
        if (
            prediction["schema_version"]
            != "ldt-array-confirmation-prediction.v3"
        ):
            raise ValueError("unsupported confirmation prediction schema")
        if prediction["authority"] != "MODEL_RECOMMENDATION_ONLY":
            raise ValueError("confirmation sorter asserted forbidden authority")
        if prediction["dataset_manifest_sha256"] != manifest_sha256:
            raise ValueError("confirmation prediction is bound to another dataset")
        candidate_id = str(prediction["candidate_id"])
        seed = None if prediction["seed"] is None else int(prediction["seed"])
        example_id = str(prediction["example_id"])
        key = (candidate_id, seed, example_id)
        if key in seen or example_id not in public:
            raise ValueError("confirmation prediction IDs differ")
        seen.add(key)
        row = public[example_id]
        binding_fields = (
            "array_id",
            "left_index",
            "right_index",
            "left_tournament_report_sha256",
            "right_tournament_report_sha256",
            "lineage_group_sha256",
        )
        if any(prediction[field] != row[field] for field in binding_fields):
            raise ValueError("confirmation prediction binding mismatch")
        if prediction["input_sha256"] != confirmation_input_sha256(row):
            raise ValueError("confirmation prediction input hash mismatch")
        if prediction["preference"] not in HARD_PREFERENCES:
            raise ValueError("confirmation predicted preference is unknown")
        if prediction["rationale_code"] not in HARD_RATIONALES:
            raise ValueError("confirmation predicted rationale is unknown")
        if prediction["swapped_preference"] not in HARD_PREFERENCES:
            raise ValueError("confirmation swapped preference is unknown")
        probabilities = prediction["preference_probabilities"]
        if set(probabilities) != set(HARD_PREFERENCES):
            raise ValueError("confirmation probability classes differ")
        values = [float(probabilities[label]) for label in HARD_PREFERENCES]
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in values
        ):
            raise ValueError("confirmation probability is invalid")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("confirmation probabilities do not sum to one")
        if not math.isclose(
            float(prediction["confidence"]),
            max(values),
            abs_tol=1e-8,
        ):
            raise ValueError("confirmation confidence is not maximum probability")
        temperature = float(prediction["calibration_temperature"])
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("confirmation temperature is invalid")
        expected_swap = (
            prediction["swapped_preference"]
            == HARD_INVERT_PREFERENCE[str(prediction["preference"])]
        )
        if bool(prediction["swap_consistent"]) != expected_swap:
            raise ValueError("confirmation swap flag is incorrect")
        grouped[(candidate_id, seed)].append(prediction)
    expected_ids = set(public)
    for key, rows in grouped.items():
        if {str(row["example_id"]) for row in rows} != expected_ids:
            raise ValueError(
                f"confirmation candidate does not cover heldout: {key}"
            )
    if not grouped:
        raise ValueError("confirmation predictions are empty")
    return grouped


def _expected_calibration_error(
    actual: Sequence[str],
    predictions: Sequence[Mapping[str, Any]],
    *,
    bins: int,
) -> float:
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for target, prediction in zip(actual, predictions, strict=True):
        probabilities = prediction["preference_probabilities"]
        predicted = str(prediction["preference"])
        confidence = float(probabilities[predicted])
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((confidence, float(predicted == target)))
    result = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        confidence = sum(row[0] for row in bucket) / len(bucket)
        accuracy = sum(row[1] for row in bucket) / len(bucket)
        result += len(bucket) / len(actual) * abs(accuracy - confidence)
    return result


def _derive_predicted_array_statuses(
    pair_predictions: Sequence[Mapping[str, Any]],
    arrays_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    by_array: dict[str, dict[tuple[int, int], str]] = defaultdict(dict)
    for prediction in pair_predictions:
        by_array[str(prediction["array_id"])][
            (int(prediction["left_index"]), int(prediction["right_index"]))
        ] = str(prediction["preference"])
    result = {}
    for array_id, array_row in arrays_by_id.items():
        candidates = [
            candidate["features"] for candidate in array_row["candidates"]
        ]
        target = reduce_confirmation_array(
            candidates,
            by_array[array_id],
        )
        result[array_id] = target["statuses"]
    return result


def _score_run(
    predictions: Sequence[Mapping[str, Any]],
    pair_labels: Mapping[str, Mapping[str, Any]],
    array_labels: Mapping[str, Mapping[str, Any]],
    arrays_by_id: Mapping[str, Mapping[str, Any]],
    *,
    trainable_parameters: int,
    trained: bool,
    ece_bins: int,
    maximum_ece: float,
    minimum_selected_set_exact: float,
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
        actual_preference,
        predicted_preference,
        HARD_PREFERENCES,
    )
    rationale = classification_metrics(
        actual_rationale,
        predicted_rationale,
        HARD_RATIONALES,
    )
    safety_indices = [
        index
        for index, value in enumerate(actual_rationale)
        if value == "safety_gate"
    ]
    safety_accuracy = sum(
        actual_preference[index] == predicted_preference[index]
        for index in safety_indices
    ) / len(safety_indices)
    safety_swap = sum(
        bool(ordered[index]["swap_consistent"]) for index in safety_indices
    ) / len(safety_indices)
    swap_consistency = sum(
        bool(row["swap_consistent"]) for row in ordered
    ) / len(ordered)
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
    negative_log_likelihood = sum(
        -math.log(
            max(
                float(
                    row["preference_probabilities"][
                        actual_preference[index]
                    ]
                ),
                1e-12,
            )
        )
        for index, row in enumerate(ordered)
    ) / len(ordered)
    ece = _expected_calibration_error(
        actual_preference,
        ordered,
        bins=ece_bins,
    )
    predicted_statuses = _derive_predicted_array_statuses(
        ordered,
        arrays_by_id,
    )
    actual_status_flat = []
    predicted_status_flat = []
    exact_arrays = 0
    selected_set_exact = 0
    by_size: dict[int, list[bool]] = defaultdict(list)
    for array_id in sorted(arrays_by_id):
        actual = [
            str(value)
            for value in array_labels[array_id]["target"]["statuses"]
        ]
        predicted = predicted_statuses[array_id]
        actual_status_flat.extend(actual)
        predicted_status_flat.extend(predicted)
        exact = actual == predicted
        exact_arrays += exact
        size = int(arrays_by_id[array_id]["candidate_count"])
        by_size[size].append(exact)
        selected_set_exact += {
            index for index, status in enumerate(actual) if status == "SELECTED"
        } == {
            index
            for index, status in enumerate(predicted)
            if status == "SELECTED"
        }
    status = classification_metrics(
        actual_status_flat,
        predicted_status_flat,
        ARRAY_STATUSES,
    )
    selected_accuracy = selected_set_exact / len(arrays_by_id)
    exact_by_size = {
        str(size): sum(values) / len(values)
        for size, values in sorted(by_size.items())
    }
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
        "multiclass_negative_log_likelihood": negative_log_likelihood,
        "expected_calibration_error": ece,
        "status_accuracy": status["accuracy"],
        "status_macro_f1": status["macro_f1"],
        "status_per_class": status["per_class"],
        "exact_array_accuracy": exact_arrays / len(arrays_by_id),
        "selected_set_exact_accuracy": selected_accuracy,
        "exact_array_accuracy_by_size": exact_by_size,
        "unresolved_recall": status["per_class"]["UNRESOLVED"]["recall"],
        "unsafe_exclusion_accuracy": status["per_class"]["UNSAFE"]["recall"],
        "mean_loop_delta": sum(float(row["loop_delta"]) for row in ordered)
        / len(ordered),
        "trainable_parameters": trainable_parameters,
        "trained": trained,
        "calibration_temperature": float(ordered[0]["calibration_temperature"]),
        "safety_qualified": safety_accuracy == 1.0 and safety_swap == 1.0,
        "orientation_qualified": swap_consistency == 1.0,
        "uncertainty_qualified": (
            preference["per_class"]["ABSTAIN"]["recall"] == 1.0
        ),
        "calibration_qualified": ece <= maximum_ece,
        "selected_set_qualified": (
            selected_accuracy >= minimum_selected_set_exact
        ),
    }


def _support_counts(
    predictions: Sequence[Mapping[str, Any]],
    pair_labels: Mapping[str, Mapping[str, Any]],
    array_labels: Mapping[str, Mapping[str, Any]],
    arrays_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "pair_count": len(predictions),
        "array_count": len(arrays_by_id),
        "preference": dict(
            sorted(
                Counter(
                    str(
                        pair_labels[str(row["example_id"])]["target"][
                            "preference"
                        ]
                    )
                    for row in predictions
                ).items()
            )
        ),
        "rationale": dict(
            sorted(
                Counter(
                    str(
                        pair_labels[str(row["example_id"])]["target"][
                            "rationale_code"
                        ]
                    )
                    for row in predictions
                ).items()
            )
        ),
        "array_status": dict(
            sorted(
                Counter(
                    str(status)
                    for array_id in arrays_by_id
                    for status in array_labels[array_id]["target"][
                        "statuses"
                    ]
                ).items()
            )
        ),
    }


def _supported_macro(
    support: Mapping[str, Mapping[str, int]],
    metrics: Mapping[str, Any],
    *,
    support_key: str,
    per_class_key: str,
) -> float:
    labels = [
        label
        for label, count in support[support_key].items()
        if int(count) > 0
    ]
    return sum(
        float(metrics[per_class_key][label]["f1"])
        for label in labels
    ) / len(labels)


def _slice_reports(
    predictions: Sequence[Mapping[str, Any]],
    pair_labels: Mapping[str, Mapping[str, Any]],
    array_labels: Mapping[str, Mapping[str, Any]],
    arrays_by_id: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
    trainable_parameters: int,
    trained: bool,
    ece_bins: int,
    maximum_ece: float,
    minimum_selected_set_exact: float,
) -> dict[str, Any]:
    result = {}
    values = sorted(
        {str(row[field]) for row in arrays_by_id.values()}
    )
    for value in values:
        array_subset = {
            array_id: row
            for array_id, row in arrays_by_id.items()
            if str(row[field]) == value
        }
        pair_subset = [
            row
            for row in predictions
            if str(row["array_id"]) in array_subset
        ]
        support = _support_counts(
            pair_subset,
            pair_labels,
            array_labels,
            array_subset,
        )
        metrics = _score_run(
            pair_subset,
            pair_labels,
            array_labels,
            array_subset,
            trainable_parameters=trainable_parameters,
            trained=trained,
            ece_bins=ece_bins,
            maximum_ece=maximum_ece,
            minimum_selected_set_exact=(
                minimum_selected_set_exact
            ),
        )

        result[value] = {
            "support": support,
            "metrics": metrics,
            "support_aware_macro_f1": {
                "preference": _supported_macro(
                    support,
                    metrics,
                    support_key="preference",
                    per_class_key="preference_per_class",
                ),
                "rationale": _supported_macro(
                    support,
                    metrics,
                    support_key="rationale",
                    per_class_key="rationale_per_class",
                ),
                "array_status": _supported_macro(
                    support,
                    metrics,
                    support_key="array_status",
                    per_class_key="status_per_class",
                ),
            },
            "full_schema_macro_support_limited": {
                "preference": len(support["preference"])
                < len(HARD_PREFERENCES),
                "rationale": len(support["rationale"])
                < len(HARD_RATIONALES),
                "array_status": len(support["array_status"])
                < len(ARRAY_STATUSES),
            },
        }
    return result


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
        "multiclass_negative_log_likelihood",
        "expected_calibration_error",
        "status_accuracy",
        "status_macro_f1",
        "exact_array_accuracy",
        "selected_set_exact_accuracy",
        "unresolved_recall",
        "unsafe_exclusion_accuracy",
        "mean_loop_delta",
        "calibration_temperature",
    )
    metrics = {}
    for name in scalars:
        values = [float(report[name]) for report in reports]
        metrics[name] = {
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    sizes = sorted(
        {
            size
            for report in reports
            for size in report["exact_array_accuracy_by_size"]
        },
        key=int,
    )
    metrics["exact_array_accuracy_by_size"] = {
        size: {
            "mean": sum(
                float(report["exact_array_accuracy_by_size"][size])
                for report in reports
            )
            / len(reports),
            "minimum": min(
                float(report["exact_array_accuracy_by_size"][size])
                for report in reports
            ),
            "maximum": max(
                float(report["exact_array_accuracy_by_size"][size])
                for report in reports
            ),
        }
        for size in sizes
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
        "all_seeds_calibration_qualified": all(
            bool(report["calibration_qualified"]) for report in reports
        ),
        "all_seeds_selected_set_qualified": all(
            bool(report["selected_set_qualified"]) for report in reports
        ),
        "metrics": metrics,
    }


def _qualified(candidate: Mapping[str, Any]) -> bool:
    return all(
        bool(candidate[name])
        for name in (
            "all_seeds_safety_qualified",
            "all_seeds_orientation_qualified",
            "all_seeds_uncertainty_qualified",
            "all_seeds_calibration_qualified",
            "all_seeds_selected_set_qualified",
        )
    )


def _dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    tolerance: float,
) -> bool:
    maximize = (
        "preference_macro_f1",
        "abstain_recall",
        "rationale_macro_f1",
        "status_macro_f1",
        "exact_array_accuracy",
        "selected_set_exact_accuracy",
    )
    minimize = (
        "multiclass_brier",
        "multiclass_negative_log_likelihood",
        "expected_calibration_error",
    )
    left_max = [float(left["metrics"][name]["mean"]) for name in maximize]
    right_max = [float(right["metrics"][name]["mean"]) for name in maximize]
    left_min = [float(left["metrics"][name]["mean"]) for name in minimize]
    right_min = [float(right["metrics"][name]["mean"]) for name in minimize]
    at_least = all(
        left_value >= right_value - tolerance
        for left_value, right_value in zip(left_max, right_max, strict=True)
    ) and all(
        left_value <= right_value + tolerance
        for left_value, right_value in zip(left_min, right_min, strict=True)
    )
    better = any(
        left_value > right_value + tolerance
        for left_value, right_value in zip(left_max, right_max, strict=True)
    ) or any(
        left_value < right_value - tolerance
        for left_value, right_value in zip(left_min, right_min, strict=True)
    )
    equivalent = all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left_max, right_max, strict=True)
    ) and all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left_min, right_min, strict=True)
    )
    cheaper = int(left["trainable_parameters"]) < int(
        right["trainable_parameters"]
    )
    return at_least and (better or (equivalent and cheaper))


def _render(report: Mapping[str, Any]) -> str:
    lines = [
        "# LDT array confirmation heldout evaluation",
        "",
        "| Candidate | Params | Qualified | Pair F1 | Abstain | ECE | "
        "Status F1 | Exact arrays | Selected sets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate_id, candidate in report["candidate_aggregates"].items():
        metrics = candidate["metrics"]
        lines.append(
            f"| {candidate_id} | {candidate['trainable_parameters']} | "
            f"{_qualified(candidate)} | "
            f"{metrics['preference_macro_f1']['mean']:.3f} | "
            f"{metrics['abstain_recall']['mean']:.3f} | "
            f"{metrics['expected_calibration_error']['mean']:.3f} | "
            f"{metrics['status_macro_f1']['mean']:.3f} | "
            f"{metrics['exact_array_accuracy']['mean']:.3f} | "
            f"{metrics['selected_set_exact_accuracy']['mean']:.3f} |"
        )
    lines.extend(
        (
            "",
            f"Learned Pareto set: "
            f"`{', '.join(report['learned_pareto_set']) or 'none'}`.",
            "",
            "No universal scalar champion is selected. Environment-rule controls "
            "are not learned architecture evidence.",
            "",
            str(report["claim_boundary"]),
            "",
        )
    )
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    preregistration = load_structured(args.preregistration.resolve())
    if preregistration["status"] != "frozen_before_dataset_generation":
        raise ValueError("confirmation preregistration is not frozen")
    (
        manifest,
        _,
        heldout_pairs,
        _,
        heldout_arrays,
        _,
        _,
    ) = load_confirmation_dataset(args.dataset_manifest.resolve())
    manifest_sha256 = sha256_file(args.dataset_manifest.resolve())
    custody = manifest["heldout_label_custody"]
    if sha256_file(args.pair_labels.resolve()) != custody["pair_labels_sha256"]:
        raise ValueError("confirmation pair labels differ from custody receipt")
    if sha256_file(args.array_labels.resolve()) != custody["array_labels_sha256"]:
        raise ValueError("confirmation array labels differ from custody receipt")
    pair_labels = _validate_pair_labels(
        read_jsonl(args.pair_labels.resolve()),
        heldout_pairs,
    )
    array_labels = _validate_array_labels(
        read_jsonl(args.array_labels.resolve()),
        heldout_arrays,
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
        raise ValueError("confirmation trainer is incomplete or contaminated")
    if (
        preregistration.get("schema_version") == "8.0"
        and not trainer.get("development_selection", {}).get(
            "confirmation_join_authorized",
            False,
        )
    ):
        raise ValueError(
            "v8 development gate did not authorize confirmation"
        )
    if (
        trainer["preregistration_sha256"]
        != sha256_file(args.preregistration.resolve())
    ):
        raise ValueError("confirmation trainer used another preregistration")
    receipts = {
        (
            str(row["candidate_id"]),
            None if row["seed"] is None else int(row["seed"]),
        ): row
        for row in trainer["candidate_receipts"]
    }
    if set(receipts) != set(grouped):
        raise ValueError("confirmation trainer receipts and predictions differ")
    for key, predictions in grouped.items():
        receipt = receipts[key]
        expected_temperature = (
            float(receipt["calibration"]["selected_temperature"])
            if receipt["trained"]
            else float(receipt["calibration_temperature"])
        )
        if any(
            not math.isclose(
                float(row["calibration_temperature"]),
                expected_temperature,
                abs_tol=1e-12,
            )
            for row in predictions
        ):
            raise ValueError("confirmation prediction temperature differs")
    arrays_by_id = {
        str(row["array_id"]): row for row in heldout_arrays
    }
    calibration = preregistration["confidence_calibration"]
    evaluation = preregistration["heldout_evaluation"]
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
            ece_bins=int(calibration["expected_calibration_error_bins"]),
            maximum_ece=float(calibration["maximum_qualified_ece"]),
            minimum_selected_set_exact=float(
                evaluation["minimum_selected_set_exact_accuracy"]
            ),
        )
        key = (
            f"{candidate_id}:control"
            if seed is None
            else f"{candidate_id}:seed-{seed}"
        )
        run_reports[key] = {
            "candidate_id": candidate_id,
            "seed": seed,
            **scored,
        }
        if preregistration.get("schema_version") in {"7.0", "8.0"}:
            run_reports[key]["slices"] = {
                "by_game": _slice_reports(
                    rows,
                    pair_labels,
                    array_labels,
                    arrays_by_id,
                    field="game",
                    trainable_parameters=int(
                        receipt["trainable_parameters"]
                    ),
                    trained=bool(receipt["trained"]),
                    ece_bins=int(
                        calibration["expected_calibration_error_bins"]
                    ),
                    maximum_ece=float(
                        calibration["maximum_qualified_ece"]
                    ),
                    minimum_selected_set_exact=float(
                        evaluation[
                            "minimum_selected_set_exact_accuracy"
                        ]
                    ),
                ),
                "by_pressure": _slice_reports(
                    rows,
                    pair_labels,
                    array_labels,
                    arrays_by_id,
                    field="fixture_pressure",
                    trainable_parameters=int(
                        receipt["trainable_parameters"]
                    ),
                    trained=bool(receipt["trained"]),
                    ece_bins=int(
                        calibration["expected_calibration_error_bins"]
                    ),
                    maximum_ece=float(
                        calibration["maximum_qualified_ece"]
                    ),
                    minimum_selected_set_exact=float(
                        evaluation[
                            "minimum_selected_set_exact_accuracy"
                        ]
                    ),
                ),
            }
        by_candidate[candidate_id].append(scored)
    aggregates = {
        candidate_id: _aggregate(reports)
        for candidate_id, reports in sorted(by_candidate.items())
    }
    eligible = {
        candidate_id: aggregate
        for candidate_id, aggregate in aggregates.items()
        if aggregate["trained"] and _qualified(aggregate)
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
    retained_id = preregistration["heldout_evaluation"].get(
        "retained_candidate_id"
    )
    if retained_id is None:
        retained_id = trainer.get("development_selection", {}).get(
            "primary_candidate_id"
        )
    if (
        retained_id is None
        and preregistration.get("schema_version") not in {"7.0", "8.0"}
    ):
        retained_id = "latent-typed-h4-v3"
    retained_confirmed = (
        retained_id is not None and retained_id in pareto
    )
    target_independent_boundary = preregistration.get(
        "schema_version"
    ) in {"7.0", "8.0"}
    report = {
        "schema_version": "ldt-array-confirmation-heldout-evaluation.v3",
        "experiment_id": trainer["experiment_id"],
        "preregistration_sha256": sha256_file(
            args.preregistration.resolve()
        ),
        "dataset_manifest_sha256": manifest_sha256,
        "predictions_sha256": sha256_file(args.predictions.resolve()),
        "trainer_summary_sha256": sha256_file(
            args.trainer_summary.resolve()
        ),
        "pair_labels_sha256": sha256_file(args.pair_labels.resolve()),
        "array_labels_sha256": sha256_file(args.array_labels.resolve()),
        "heldout_label_authority": "ENVIRONMENT_OWNED",
        "heldout_reveal_count": 1,
        "post_reveal_tuning_permitted": False,
        "candidate_run_reports": run_reports,
        "candidate_aggregates": aggregates,
        "learned_pareto_set": pareto,
        "development_primary_candidate_id": retained_id,
        "retained_architecture_confirmed": retained_confirmed,
        "universal_scalar_champion": None,
        "environment_rule_controls_are_learned_evidence": False,
        "larger_model_training_authorized": False,
        "shadow_recommendation_integration_authorized": (
            retained_confirmed and not target_independent_boundary
        ),
        "claim_boundary": (
            "Heldout evidence applies only to fresh content-free tournament "
            "summaries; recommendations cannot promote corpus records and "
            "remain subordinate to the independent verifier."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    report_path = args.out / "heldout_evaluation.json"
    _write_json(report_path, report)
    (args.out / "heldout_evaluation.md").write_text(
        _render(report),
        encoding="utf-8",
    )
    _write_json(
        args.out / "evaluation_receipt.json",
        {
            "schema_version": (
                "ldt-array-confirmation-evaluation-receipt.v3"
            ),
            "heldout_evaluation_sha256": sha256_file(report_path),
            "row_level_labels_exported": False,
            "candidate_workspace_mutated": False,
            "universal_scalar_champion_selected": False,
            "post_reveal_tuning_permitted": False,
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate v3 variable-array confirmation predictions"
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trainer-summary", type=Path, required=True)
    parser.add_argument("--pair-labels", type=Path, required=True)
    parser.add_argument("--array-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    result = evaluate(build_parser().parse_args())
    print(
        json.dumps(
            {
                "learned_pareto_set": result["learned_pareto_set"],
                "retained_architecture_confirmed": result[
                    "retained_architecture_confirmed"
                ],
            },
            sort_keys=True,
        )
    )
