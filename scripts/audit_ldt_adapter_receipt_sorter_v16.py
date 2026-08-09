#!/usr/bin/env python3
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
    HARD_PREFERENCES,
)
from datagoblin.experiments.ldt_array_sorter import (
    classification_metrics,
    sha256_file,
)
from datagoblin.serialization import dump_structured, load_structured


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), abs_tol=1e-12)


def _individual_qualified(
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    mean = receipt["cv_mean_metrics"]
    folds = receipt["cv_fold_metrics"]
    calibration = receipt["calibration"]["selected_metrics"]
    return (
        _close(
            mean["safety_gate_accuracy"],
            contract["required_mean_safety_gate_accuracy"],
        )
        and _close(
            mean["swap_consistency"],
            contract["required_mean_swap_consistency"],
        )
        and float(mean["preference_macro_f1"])
        >= float(contract["minimum_mean_preference_macro_f1"])
        and float(mean["abstain_recall"])
        >= float(contract["minimum_mean_abstain_recall"])
        and float(mean["status_macro_f1"])
        >= float(contract["minimum_mean_status_macro_f1"])
        and float(mean["exact_array_accuracy"])
        >= float(contract["minimum_mean_exact_array_accuracy"])
        and float(mean["selected_set_exact_accuracy"])
        >= float(contract["minimum_mean_selected_set_exact_accuracy"])
        and min(float(row["exact_array_accuracy"]) for row in folds)
        >= float(contract["minimum_worst_fold_exact_array_accuracy"])
        and min(
            float(row["selected_set_exact_accuracy"]) for row in folds
        )
        >= float(
            contract["minimum_worst_fold_selected_set_exact_accuracy"]
        )
        and float(calibration["expected_calibration_error"])
        <= float(contract["maximum_oof_expected_calibration_error"])
    )


def _rank(receipt: Mapping[str, Any]) -> tuple[float, ...]:
    folds = receipt["cv_fold_metrics"]
    mean = receipt["cv_mean_metrics"]
    calibration = receipt["calibration"]["selected_metrics"]
    return (
        -min(float(row["selected_set_exact_accuracy"]) for row in folds),
        -min(float(row["exact_array_accuracy"]) for row in folds),
        -min(float(row["status_macro_f1"]) for row in folds),
        -min(float(row["preference_macro_f1"]) for row in folds),
        -float(mean["selected_set_exact_accuracy"]),
        -float(mean["exact_array_accuracy"]),
        -float(mean["status_macro_f1"]),
        -float(mean["preference_macro_f1"]),
        float(calibration["multiclass_negative_log_likelihood"]),
        float(receipt["trainable_parameters"]),
        float(receipt["seed"]),
    )


def _reproduce_selection(
    config: Mapping[str, Any],
    trainer: Mapping[str, Any],
) -> dict[str, Any]:
    contract = config["development_selection"]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for receipt in trainer["candidate_receipts"]:
        grouped[str(receipt["candidate_id"])].append(receipt)
    selected = {}
    qualified_seeds = {}
    for candidate_id, receipts in grouped.items():
        qualified = [
            row for row in receipts if _individual_qualified(row, contract)
        ]
        selected[candidate_id] = min(
            qualified if qualified else receipts,
            key=_rank,
        )
        qualified_seeds[candidate_id] = sorted(
            int(row["seed"]) for row in qualified
        )
    primary_id = str(contract["primary_candidate_ids"][0])
    comparison = contract["comparative_primary_gates"]
    baseline_id = str(comparison["baseline_candidate_id"])
    primary = selected[primary_id]
    baseline = selected[baseline_id]
    primary_folds = primary["cv_fold_metrics"]
    baseline_folds = baseline["cv_fold_metrics"]
    comparative = {
        "mean_exact_array_accuracy_delta": (
            float(primary["cv_mean_metrics"]["exact_array_accuracy"])
            - float(baseline["cv_mean_metrics"]["exact_array_accuracy"])
            >= float(comparison["minimum_mean_exact_array_accuracy_delta"])
        ),
        "mean_selected_set_exact_accuracy_delta": (
            float(
                primary["cv_mean_metrics"][
                    "selected_set_exact_accuracy"
                ]
            )
            - float(
                baseline["cv_mean_metrics"][
                    "selected_set_exact_accuracy"
                ]
            )
            >= float(
                comparison[
                    "minimum_mean_selected_set_exact_accuracy_delta"
                ]
            )
        ),
        "worst_fold_exact_array_accuracy_delta": (
            min(float(row["exact_array_accuracy"]) for row in primary_folds)
            - min(float(row["exact_array_accuracy"]) for row in baseline_folds)
            >= float(
                comparison[
                    "minimum_worst_fold_exact_array_accuracy_delta"
                ]
            )
        ),
        "worst_fold_selected_set_exact_accuracy_delta": (
            min(
                float(row["selected_set_exact_accuracy"])
                for row in primary_folds
            )
            - min(
                float(row["selected_set_exact_accuracy"])
                for row in baseline_folds
            )
            >= float(
                comparison[
                    "minimum_worst_fold_selected_set_exact_accuracy_delta"
                ]
            )
        ),
    }
    authorized = (
        _individual_qualified(primary, contract)
        and all(comparative.values())
    )
    return {
        "selected_seeds": {
            candidate_id: int(row["seed"])
            for candidate_id, row in selected.items()
        },
        "qualified_seeds": qualified_seeds,
        "primary_id": primary_id if authorized else None,
        "primary_seed": int(primary["seed"]) if authorized else None,
        "comparative_checks": comparative,
        "authorized": authorized,
    }


def _expected_calibration_error(
    actual: Sequence[str],
    predictions: Sequence[Mapping[str, Any]],
    *,
    bins: int,
) -> float:
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for target, prediction in zip(actual, predictions, strict=True):
        predicted = str(prediction["preference"])
        confidence = float(
            prediction["preference_probabilities"][predicted]
        )
        bucket = min(int(confidence * bins), bins - 1)
        buckets[bucket].append((confidence, float(predicted == target)))
    return sum(
        len(bucket) / len(actual)
        * abs(
            sum(value[0] for value in bucket) / len(bucket)
            - sum(value[1] for value in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    )


def _reproduce_heldout_metrics(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    pair_predictions: Sequence[Mapping[str, Any]],
    array_predictions: Sequence[Mapping[str, Any]],
    pair_labels: Sequence[Mapping[str, Any]],
    array_labels: Sequence[Mapping[str, Any]],
    public_arrays: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pair_target = {
        str(row["example_id"]): row["target"] for row in pair_labels
    }
    array_target = {
        str(row["array_id"]): row["target"] for row in array_labels
    }
    arrays = {str(row["array_id"]): row for row in public_arrays}
    ordered_pairs = sorted(pair_predictions, key=lambda row: row["example_id"])
    actual_preference = [
        str(pair_target[str(row["example_id"])]["preference"])
        for row in ordered_pairs
    ]
    predicted_preference = [
        str(row["preference"]) for row in ordered_pairs
    ]
    preference = classification_metrics(
        actual_preference,
        predicted_preference,
        HARD_PREFERENCES,
    )
    safety_indices = [
        index
        for index, row in enumerate(ordered_pairs)
        if pair_target[str(row["example_id"])]["rationale_code"]
        == "safety_gate"
    ]
    safety = sum(
        actual_preference[index] == predicted_preference[index]
        for index in safety_indices
    ) / len(safety_indices)
    safety_swap = sum(
        bool(ordered_pairs[index]["swap_consistent"])
        for index in safety_indices
    ) / len(safety_indices)
    swap = sum(bool(row["swap_consistent"]) for row in ordered_pairs) / len(
        ordered_pairs
    )
    predicted_arrays = {
        str(row["array_id"]): [str(value) for value in row["statuses"]]
        for row in array_predictions
    }
    actual_flat = []
    predicted_flat = []
    exact = 0
    selected_exact = 0
    primary_ids = {
        array_id
        for array_id, row in arrays.items()
        if str(row["fixture_pressure"])
        in set(config["surface"]["primary_domain_pressures"])
    }
    primary_exact = 0
    primary_selected = 0
    for array_id in sorted(arrays):
        actual = [str(value) for value in array_target[array_id]["statuses"]]
        predicted = predicted_arrays[array_id]
        actual_flat.extend(actual)
        predicted_flat.extend(predicted)
        exact_match = actual == predicted
        selected_match = {
            index for index, value in enumerate(actual) if value == "SELECTED"
        } == {
            index
            for index, value in enumerate(predicted)
            if value == "SELECTED"
        }
        exact += exact_match
        selected_exact += selected_match
        if array_id in primary_ids:
            primary_exact += exact_match
            primary_selected += selected_match
    status = classification_metrics(
        actual_flat,
        predicted_flat,
        ARRAY_STATUSES,
    )
    evaluation = config["heldout_evaluation"]
    metrics = {
        "safety_gate_accuracy": safety,
        "safety_swap_consistency": safety_swap,
        "swap_consistency": swap,
        "preference_macro_f1": preference["macro_f1"],
        "abstain_recall": preference["per_class"]["ABSTAIN"]["recall"],
        "expected_calibration_error": _expected_calibration_error(
            actual_preference,
            ordered_pairs,
            bins=int(
                config["confidence_calibration"][
                    "expected_calibration_error_bins"
                ]
            ),
        ),
        "status_macro_f1": status["macro_f1"],
        "exact_status_vector_accuracy": exact / len(arrays),
        "exact_selected_set_accuracy": selected_exact / len(arrays),
        "primary_domain_exact_status_vector_accuracy": (
            primary_exact / len(primary_ids)
        ),
        "primary_domain_exact_selected_set_accuracy": (
            primary_selected / len(primary_ids)
        ),
    }
    gates = {
        "safety": _close(
            metrics["safety_gate_accuracy"],
            evaluation["minimum_safety_gate_accuracy"],
        )
        and _close(
            metrics["safety_swap_consistency"],
            evaluation["minimum_swap_consistency"],
        ),
        "swap": metrics["swap_consistency"]
        >= float(evaluation["minimum_swap_consistency"]),
        "preference": metrics["preference_macro_f1"]
        >= float(evaluation["minimum_preference_macro_f1"]),
        "abstain": metrics["abstain_recall"]
        >= float(evaluation["minimum_abstain_recall"]),
        "calibration": metrics["expected_calibration_error"]
        <= float(config["confidence_calibration"]["maximum_qualified_ece"]),
        "status": metrics["status_macro_f1"]
        >= float(evaluation["minimum_status_macro_f1"]),
        "exact_status": metrics["exact_status_vector_accuracy"]
        >= float(evaluation["minimum_exact_status_vector_accuracy"]),
        "exact_selected": metrics["exact_selected_set_accuracy"]
        >= float(evaluation["minimum_selected_set_exact_accuracy"]),
        "primary_exact_status": metrics[
            "primary_domain_exact_status_vector_accuracy"
        ]
        >= float(
            evaluation[
                "minimum_primary_domain_exact_status_vector_accuracy"
            ]
        ),
        "primary_exact_selected": metrics[
            "primary_domain_exact_selected_set_accuracy"
        ]
        >= float(
            evaluation[
                "minimum_primary_domain_selected_set_exact_accuracy"
            ]
        ),
    }
    return {
        "metrics": metrics,
        "gates": gates,
        "confirmed": all(gates.values()),
        "array_count": len(arrays),
        "primary_domain_array_count": len(primary_ids),
        "manifest_sha256": sha256_file(Path(manifest["_path"])),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = load_structured(args.preregistration)
    manifest = load_structured(args.dataset_manifest)
    manifest["_path"] = str(args.dataset_manifest)
    generation = load_structured(args.generation_report)
    plan = load_structured(args.trainer_plan)
    trainer = load_structured(args.merged_trainer_summary)
    selection = load_structured(args.selection_receipt)
    primary_trainer = load_structured(args.primary_trainer_summary)
    heldout = load_structured(args.heldout_evaluation)
    evaluation_receipt = load_structured(args.evaluation_receipt)
    preregistration_sha256 = sha256_file(args.preregistration)
    manifest_sha256 = sha256_file(args.dataset_manifest)
    pair_predictions_sha256 = sha256_file(args.merged_pair_predictions)
    array_predictions_sha256 = sha256_file(args.merged_array_predictions)
    reproduced_selection = _reproduce_selection(config, trainer)

    checks: dict[str, bool] = {
        "config_bound": generation["preregistration_sha256"]
        == preregistration_sha256
        and plan["preregistration_sha256"] == preregistration_sha256
        and trainer["preregistration_sha256"] == preregistration_sha256,
        "dataset_bound": generation["dataset_manifest_sha256"]
        == manifest_sha256
        and plan["dataset_manifest_sha256"] == manifest_sha256
        and trainer["dataset_manifest_sha256"] == manifest_sha256,
        "generation_passed": generation["generation_passed"]
        and all(generation["checks"].values()),
        "lineage_contamination_zero": manifest["lineage_audit"][
            "cross_split_groups"
        ]
        == 0
        and manifest["lineage_audit"]["prior_overlap"] == 0,
        "trainer_complete": trainer["status"] == "completed"
        and len(trainer["candidate_receipts"]) == 12,
        "trainer_target_free": not trainer["heldout_pair_targets_seen"]
        and not trainer["heldout_array_targets_seen"],
        "merged_predictions_bound": trainer["predictions_sha256"]
        == pair_predictions_sha256
        and trainer["array_predictions_sha256"]
        == array_predictions_sha256,
        "selection_sources_bound": selection[
            "source_trainer_summary_sha256"
        ]
        == sha256_file(args.merged_trainer_summary)
        and selection["source_pair_predictions_sha256"]
        == pair_predictions_sha256
        and selection["source_array_predictions_sha256"]
        == array_predictions_sha256,
        "selection_reproduced": selection["confirmation_join_authorized"]
        == reproduced_selection["authorized"]
        and selection["primary_candidate_id"]
        == reproduced_selection["primary_id"]
        and selection["primary_seed"]
        == reproduced_selection["primary_seed"]
        and selection["comparative_primary_checks"]
        == reproduced_selection["comparative_checks"],
        "primary_packet_bound": selection["confirmation_packet"] is not None
        and selection["confirmation_packet"]["pair_predictions_sha256"]
        == sha256_file(args.primary_pair_predictions)
        and selection["confirmation_packet"]["array_predictions_sha256"]
        == sha256_file(args.primary_array_predictions)
        and selection["confirmation_packet"]["trainer_summary_sha256"]
        == sha256_file(args.primary_trainer_summary),
        "primary_packet_target_free": not primary_trainer[
            "heldout_pair_targets_seen"
        ]
        and not primary_trainer["heldout_array_targets_seen"]
        and primary_trainer["confirmation_packet_candidate_count"] == 1,
        "custody_bound": manifest["heldout_label_custody"][
            "pair_labels_sha256"
        ]
        == sha256_file(args.pair_labels)
        and manifest["heldout_label_custody"]["array_labels_sha256"]
        == sha256_file(args.array_labels),
        "heldout_artifacts_bound": heldout["predictions_sha256"]
        == sha256_file(args.primary_pair_predictions)
        and heldout["array_predictions_sha256"]
        == sha256_file(args.primary_array_predictions)
        and heldout["trainer_summary_sha256"]
        == sha256_file(args.primary_trainer_summary)
        and evaluation_receipt["heldout_evaluation_sha256"]
        == sha256_file(args.heldout_evaluation),
        "single_join_and_no_retune": heldout["heldout_reveal_count"] == 1
        and not heldout["post_reveal_tuning_permitted"]
        and not evaluation_receipt["post_reveal_tuning_permitted"],
    }

    expected_runs = {
        (str(model["model_id"]), int(seed))
        for model in config["models"]["candidates"]
        for seed in config["training"]["seeds"]
    }
    observed_runs = set()
    resource_passed = True
    for candidate_id, seed in expected_runs:
        run_dir = args.shard_root / f"{candidate_id}-seed-{seed}"
        receipt_path = run_dir / "run_receipt.json"
        if not receipt_path.exists():
            resource_passed = False
            continue
        receipt = load_structured(receipt_path)
        observed_runs.add((candidate_id, seed))
        resource_passed = resource_passed and bool(
            int(receipt["capped_process_exit_code"]) == 0
            and int(receipt["wsl_stage_cleanup_exit_code"]) == 0
            and receipt["caps_receipt"]["caps_enforced"]
            and receipt["cleanup_receipt"]["cleanup_passed"]
            and not receipt["heldout_pair_labels_staged"]
            and not receipt["heldout_array_labels_staged"]
            and not receipt["external_checkpoint_loaded"]
        )
    checks["resource_and_cleanup_receipts_pass"] = (
        resource_passed and observed_runs == expected_runs
    )

    public_array_path = args.dataset_manifest.parent / manifest["splits"][
        "heldout_arrays"
    ]["path"]
    reproduced_heldout = _reproduce_heldout_metrics(
        config,
        manifest,
        _read_jsonl(args.primary_pair_predictions),
        _read_jsonl(args.primary_array_predictions),
        _read_jsonl(args.pair_labels),
        _read_jsonl(args.array_labels),
        _read_jsonl(public_array_path),
    )
    aggregate = next(iter(heldout["candidate_aggregates"].values()))[
        "metrics"
    ]
    primary_domain = heldout["primary_domain_confirmation"]
    metrics_match = all(
        (
            _close(
                aggregate[report_key]["mean"],
                reproduced_heldout["metrics"][reproduced_key],
            )
            if report_key in aggregate
            else False
        )
        for report_key, reproduced_key in (
            ("safety_gate_accuracy", "safety_gate_accuracy"),
            ("swap_consistency", "swap_consistency"),
            ("preference_macro_f1", "preference_macro_f1"),
            ("abstain_recall", "abstain_recall"),
            ("expected_calibration_error", "expected_calibration_error"),
            ("status_macro_f1", "status_macro_f1"),
            ("exact_array_accuracy", "exact_status_vector_accuracy"),
            (
                "selected_set_exact_accuracy",
                "exact_selected_set_accuracy",
            ),
        )
    )
    primary_match = _close(
        primary_domain["exact_status_vector_accuracy"],
        reproduced_heldout["metrics"][
            "primary_domain_exact_status_vector_accuracy"
        ],
    ) and _close(
        primary_domain["exact_selected_set_accuracy"],
        reproduced_heldout["metrics"][
            "primary_domain_exact_selected_set_accuracy"
        ],
    )
    checks["heldout_metrics_reproduced"] = metrics_match and primary_match
    checks["heldout_conclusion_reproduced"] = (
        heldout["retained_architecture_confirmed"]
        == reproduced_heldout["confirmed"]
        and heldout["next_shadow_preregistration_authorized"]
        == reproduced_heldout["confirmed"]
    )

    passed = all(checks.values())
    confirmed = reproduced_heldout["confirmed"] and passed
    report = {
        "schema_version": "ldt-adapter-receipt-sorter-audit.v16",
        "passed": passed,
        "checks": checks,
        "reproduced_development_selection": reproduced_selection,
        "reproduced_heldout": reproduced_heldout,
        "conclusion": (
            "confirmed_for_next_shadow_preregistration_only"
            if confirmed
            else "not_confirmed_no_promotion"
        ),
        "conclusion_layers": {
            "metric_robustness": "supported_on_declared_independent_v10_holdout",
            "task_result": (
                "heldout_confirmed" if confirmed else "heldout_not_confirmed"
            ),
            "measurement_reliability": (
                "single_environment_join_independently_reproduced"
                if passed
                else "audit_integrity_failed"
            ),
            "claim_support": "supported" if confirmed else "not_supported",
            "operational_decision": (
                "next_shadow_preregistration_only"
                if confirmed
                else "no_promotion"
            ),
        },
    }
    dump_structured(args.out, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently audit the v16 adapter-receipt sorter"
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--generation-report", type=Path, required=True)
    parser.add_argument("--trainer-plan", type=Path, required=True)
    parser.add_argument("--merged-trainer-summary", type=Path, required=True)
    parser.add_argument("--merged-pair-predictions", type=Path, required=True)
    parser.add_argument("--merged-array-predictions", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--primary-trainer-summary", type=Path, required=True)
    parser.add_argument("--primary-pair-predictions", type=Path, required=True)
    parser.add_argument("--primary-array-predictions", type=Path, required=True)
    parser.add_argument("--heldout-evaluation", type=Path, required=True)
    parser.add_argument("--evaluation-receipt", type=Path, required=True)
    parser.add_argument("--pair-labels", type=Path, required=True)
    parser.add_argument("--array-labels", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    result = audit(build_parser().parse_args())
    print(json.dumps(result, sort_keys=True))
