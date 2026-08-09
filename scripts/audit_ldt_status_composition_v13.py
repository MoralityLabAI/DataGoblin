#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_sorter import read_jsonl, sha256_file
from datagoblin.serialization import dump_structured, load_structured

FACTORIZED = "composition-h4-factorized-v13"
GLOBAL_MAP = "composition-h4-global-map-v13"
SELECTED_COUNT = "selected-count-h4-global-map-v13"


def _candidate_map(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["candidate_id"]), int(row["seed"])): row
        for row in rows
    }


def _prediction_map(
    rows: list[dict[str, Any]],
    identity_field: str,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {
        (
            str(row["candidate_id"]),
            int(row["seed"]),
            str(row[identity_field]),
        ): row
        for row in rows
    }


def _pair_behavior(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["input_sha256"],
        row["preference"],
        row["rationale_code"],
        row["confidence"],
        row["preference_probabilities"],
        row["swapped_preference"],
        row["swap_consistent"],
        row["loop_delta"],
        row["calibration_temperature"],
    )


def _mean_metric(
    candidates: dict[tuple[str, int], dict[str, Any]],
    candidate_id: str,
    seeds: list[int],
    metric: str,
) -> float:
    return sum(
        float(candidates[(candidate_id, seed)]["cv_mean_metrics"][metric])
        for seed in seeds
    ) / len(seeds)


def _worst_metric(candidate: dict[str, Any], metric: str) -> float:
    return min(float(fold[metric]) for fold in candidate["cv_fold_metrics"])


def _objective_without_decoder(candidate: dict[str, Any]) -> dict[str, Any]:
    objective = dict(candidate["listwise_objective"])
    objective.pop("prediction_source")
    composition = dict(objective.pop("composition_likelihood"))
    composition.pop("decoder")
    objective["composition_likelihood"] = composition
    return objective


def _gate_failures(
    receipt: dict[str, Any],
    contract: dict[str, Any],
) -> list[str]:
    mean = receipt["cv_mean_metrics"]
    calibration = receipt["calibration"]["selected_metrics"]
    failures: list[str] = []
    exact = {
        "safety_gate_accuracy": "required_mean_safety_gate_accuracy",
        "swap_consistency": "required_mean_swap_consistency",
    }
    minimum = {
        "preference_macro_f1": "minimum_mean_preference_macro_f1",
        "abstain_recall": "minimum_mean_abstain_recall",
        "status_macro_f1": "minimum_mean_status_macro_f1",
        "exact_array_accuracy": "minimum_mean_exact_array_accuracy",
        "selected_set_exact_accuracy": (
            "minimum_mean_selected_set_exact_accuracy"
        ),
    }
    for metric, gate in exact.items():
        if not math.isclose(
            float(mean[metric]),
            float(contract[gate]),
            abs_tol=1e-12,
        ):
            failures.append(metric)
    for metric, gate in minimum.items():
        if float(mean[metric]) < float(contract[gate]):
            failures.append(metric)
    if _worst_metric(receipt, "selected_set_exact_accuracy") < float(
        contract["minimum_worst_fold_selected_set_exact_accuracy"]
    ):
        failures.append("worst_fold_selected_set_exact_accuracy")
    if float(calibration["expected_calibration_error"]) > float(
        contract["maximum_oof_expected_calibration_error"]
    ):
        failures.append("expected_calibration_error")
    return failures


def _rank(receipt: dict[str, Any]) -> tuple[float, ...]:
    mean = receipt["cv_mean_metrics"]
    calibration = receipt["calibration"]["selected_metrics"]
    return (
        -_worst_metric(receipt, "selected_set_exact_accuracy"),
        -_worst_metric(receipt, "exact_array_accuracy"),
        -_worst_metric(receipt, "status_macro_f1"),
        -_worst_metric(receipt, "preference_macro_f1"),
        -float(mean["selected_set_exact_accuracy"]),
        -float(mean["exact_array_accuracy"]),
        -float(mean["status_macro_f1"]),
        -float(mean["preference_macro_f1"]),
        float(calibration["multiclass_negative_log_likelihood"]),
        float(receipt["trainable_parameters"]),
        float(receipt["seed"]),
    )


def _resource_audit(
    run_root: Path,
    expected_keys: set[tuple[str, int]],
    source_shards: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    source_by_key = {
        (str(row["candidate_id"]), int(row["seeds"][0])): row
        for row in source_shards
    }
    receipts: list[dict[str, Any]] = []
    valid = set(source_by_key) == expected_keys
    for candidate_id, seed in sorted(expected_keys):
        receipt_path = (
            run_root / f"{candidate_id}-seed-{seed}" / "run_receipt.json"
        )
        if not receipt_path.is_file():
            valid = False
            continue
        receipt = load_structured(receipt_path)
        source = source_by_key.get((candidate_id, seed), {})
        caps = receipt["caps_receipt"]
        trainer = receipt["trainer_receipt"]
        cleanup = receipt["cleanup_receipt"]
        receipt_valid = (
            source.get("run_receipt_sha256") == sha256_file(receipt_path)
            and receipt["capped_process_exit_code"] == 0
            and receipt["wsl_stage_cleanup_exit_code"] == 0
            and bool(caps["caps_enforced"])
            and caps["exit_code"] == 0
            and caps["status"] == "completed"
            and trainer["status"] == "completed"
            and bool(trainer["hard_caps_verified"])
            and not bool(trainer["heldout_pair_targets_seen"])
            and not bool(trainer["heldout_array_targets_seen"])
            and bool(cleanup["cleanup_passed"])
            and not cleanup["lingering_owned_pids"]
            and not receipt["heldout_pair_labels_staged"]
            and not receipt["heldout_array_labels_staged"]
            and not receipt["external_checkpoint_loaded"]
        )
        valid &= receipt_valid
        receipts.append(
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "receipt_sha256": sha256_file(receipt_path),
                "elapsed_seconds": float(trainer["elapsed_seconds"]),
                "steps_completed": int(trainer["steps_completed"]),
                "peak_ram_mb": float(caps["peak_ram_mb"]),
                "cpu_pct": float(caps["cpu_pct"]),
                "peak_io_mb_s": float(caps["peak_io_mb_s"]),
                "caps_enforced": bool(caps["caps_enforced"]),
                "cleanup_passed": bool(cleanup["cleanup_passed"]),
            }
        )
    families: dict[str, dict[str, Any]] = {}
    for candidate_id in sorted({row["candidate_id"] for row in receipts}):
        rows = [row for row in receipts if row["candidate_id"] == candidate_id]
        families[candidate_id] = {
            "shards": len(rows),
            "trainer_seconds": sum(row["elapsed_seconds"] for row in rows),
            "steps_completed": sum(row["steps_completed"] for row in rows),
            "maximum_peak_ram_mb": max(row["peak_ram_mb"] for row in rows),
            "maximum_cpu_pct": max(row["cpu_pct"] for row in rows),
            "maximum_peak_io_mb_s": max(row["peak_io_mb_s"] for row in rows),
        }
    return valid, {
        "shard_count": len(receipts),
        "total_trainer_seconds": sum(row["elapsed_seconds"] for row in receipts),
        "total_steps_completed": sum(row["steps_completed"] for row in receipts),
        "families": families,
        "receipt_sha256": [row["receipt_sha256"] for row in receipts],
    }


def audit(
    preregistration_path: Path,
    review_manifest_path: Path,
    development_policy_path: Path,
    development_preflight_path: Path,
    base_preflight_path: Path,
    dataset_manifest_path: Path,
    run_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    preregistration = load_structured(preregistration_path)
    review_manifest = load_structured(review_manifest_path)
    development_preflight = load_structured(development_preflight_path)
    base_preflight = load_structured(base_preflight_path)
    manifest = load_structured(dataset_manifest_path)
    trainer_path = run_root / "merged/trainer_summary.json"
    merge_path = run_root / "merged/merge_receipt.json"
    pair_path = run_root / "merged/heldout_pair_predictions.jsonl"
    array_path = run_root / "merged/heldout_array_predictions.jsonl"
    selection_path = run_root / "selected/selection_receipt.json"
    trainer = load_structured(trainer_path)
    merge = load_structured(merge_path)
    selection = load_structured(selection_path)
    candidates = _candidate_map(trainer["candidate_receipts"])
    pair_predictions = _prediction_map(read_jsonl(pair_path), "example_id")
    array_predictions = _prediction_map(read_jsonl(array_path), "array_id")
    seeds = [int(seed) for seed in preregistration["training"]["seeds"]]
    candidate_ids = [
        str(model["model_id"])
        for model in preregistration["models"]["candidates"]
    ]
    expected_keys = {
        (candidate_id, seed)
        for candidate_id in candidate_ids
        for seed in seeds
    }

    pair_behavior_identical = True
    local_probabilities_identical = True
    decoder_change_count = 0
    for seed in seeds:
        factorized_pair_ids = {
            identity
            for candidate_id, row_seed, identity in pair_predictions
            if candidate_id == FACTORIZED and row_seed == seed
        }
        for identity in factorized_pair_ids:
            pair_behavior_identical &= (
                _pair_behavior(pair_predictions[(FACTORIZED, seed, identity)])
                == _pair_behavior(pair_predictions[(GLOBAL_MAP, seed, identity)])
            )
        factorized_array_ids = {
            identity
            for candidate_id, row_seed, identity in array_predictions
            if candidate_id == FACTORIZED and row_seed == seed
        }
        for identity in factorized_array_ids:
            factorized = array_predictions[(FACTORIZED, seed, identity)]
            global_map = array_predictions[(GLOBAL_MAP, seed, identity)]
            local_probabilities_identical &= (
                factorized["status_probabilities"]
                == global_map["status_probabilities"]
            )
            decoder_change_count += int(
                factorized["statuses"] != global_map["statuses"]
            )

    matched_objectives = all(
        candidates[(FACTORIZED, seed)]["trainable_parameters"]
        == candidates[(GLOBAL_MAP, seed)]["trainable_parameters"]
        and _objective_without_decoder(candidates[(FACTORIZED, seed)])
        == _objective_without_decoder(candidates[(GLOBAL_MAP, seed)])
        for seed in seeds
    )
    contract = preregistration["development_selection"]
    gate_failures = {
        f"{candidate_id}:{seed}": _gate_failures(receipt, contract)
        for (candidate_id, seed), receipt in sorted(candidates.items())
    }
    numerical_gate_passers = sorted(
        key for key, failures in gate_failures.items() if not failures
    )
    independently_selected: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        selected = min(
            (candidates[(candidate_id, seed)] for seed in seeds),
            key=_rank,
        )
        independently_selected[candidate_id] = selected
    selection_consistent = all(
        int(selection["candidate_selection"][candidate_id]["selected_seed"])
        == int(independently_selected[candidate_id]["seed"])
        and bool(
            selection["candidate_selection"][candidate_id][
                "development_qualified"
            ]
        )
        == (not _gate_failures(independently_selected[candidate_id], contract))
        for candidate_id in candidate_ids
    )
    selected_qualified = [
        row
        for row in independently_selected.values()
        if not _gate_failures(row, contract)
    ]
    independent_primary = (
        min(selected_qualified, key=_rank) if selected_qualified else None
    )
    primary_consistent = (
        selection["primary_candidate_id"]
        == (
            None
            if independent_primary is None
            else str(independent_primary["candidate_id"])
        )
        and selection["primary_seed"]
        == (
            None
            if independent_primary is None
            else int(independent_primary["seed"])
        )
        and bool(selection["confirmation_join_authorized"])
        == (independent_primary is not None)
    )
    resource_valid, resources = _resource_audit(
        run_root,
        expected_keys,
        trainer["source_shards"],
    )

    composition_decoder_exact_deltas = [
        float(
            candidates[(GLOBAL_MAP, seed)]["cv_mean_metrics"][
                "exact_array_accuracy"
            ]
        )
        - float(
            candidates[(FACTORIZED, seed)]["cv_mean_metrics"][
                "exact_array_accuracy"
            ]
        )
        for seed in seeds
    ]
    composition_decoder_status_deltas = [
        float(
            candidates[(GLOBAL_MAP, seed)]["cv_mean_metrics"][
                "status_macro_f1"
            ]
        )
        - float(
            candidates[(FACTORIZED, seed)]["cv_mean_metrics"][
                "status_macro_f1"
            ]
        )
        for seed in seeds
    ]
    control_exact_deltas = [
        float(
            candidates[(GLOBAL_MAP, seed)]["cv_mean_metrics"][
                "exact_array_accuracy"
            ]
        )
        - float(
            candidates[(SELECTED_COUNT, seed)]["cv_mean_metrics"][
                "exact_array_accuracy"
            ]
        )
        for seed in seeds
    ]
    control_worst_selected_deltas = [
        _worst_metric(
            candidates[(GLOBAL_MAP, seed)],
            "selected_set_exact_accuracy",
        )
        - _worst_metric(
            candidates[(SELECTED_COUNT, seed)],
            "selected_set_exact_accuracy",
        )
        for seed in seeds
    ]
    control_status_deltas = [
        float(
            candidates[(GLOBAL_MAP, seed)]["cv_mean_metrics"][
                "status_macro_f1"
            ]
        )
        - float(
            candidates[(SELECTED_COUNT, seed)]["cv_mean_metrics"][
                "status_macro_f1"
            ]
        )
        for seed in seeds
    ]

    checks = {
        "preregistration_frozen": (
            preregistration["status"] == "frozen_before_dataset_generation"
        ),
        "preregistration_bound": (
            trainer["preregistration_sha256"]
            == sha256_file(preregistration_path)
        ),
        "dataset_bound": (
            trainer["dataset_manifest_sha256"]
            == sha256_file(dataset_manifest_path)
        ),
        "fresh_lineage_firewall": (
            manifest["lineage_audit"]["forbidden_prior_groups"] == 84
            and manifest["lineage_audit"]["prior_overlap"] == 0
            and manifest["lineage_audit"]["cross_split_groups"] == 0
        ),
        "complete_candidate_seed_coverage": set(candidates) == expected_keys,
        "fifteen_capped_clean_shards": (
            resource_valid
            and len(trainer["source_shards"]) == 15
            and bool(merge["all_caps_enforced"])
            and bool(merge["all_cleanup_passed"])
        ),
        "no_confirmation_targets_seen": (
            not trainer["heldout_pair_targets_seen"]
            and not trainer["heldout_array_targets_seen"]
            and not selection["confirmation_labels_read"]
        ),
        "selection_order_independently_reproduced": selection_consistent,
        "confirmation_decision_independently_reproduced": primary_consistent,
        "composition_decoder_ablation_matched": matched_objectives,
        "matched_pair_behavior": pair_behavior_identical,
        "matched_local_status_probabilities": local_probabilities_identical,
        "global_composition_decoder_changes_at_least_one_array": (
            decoder_change_count > 0
        ),
        "development_review_manifest_bound": (
            development_preflight["provenance"]["manifest_sha256"]
            == sha256_file(review_manifest_path)
            and base_preflight["provenance"]["manifest_sha256"]
            == sha256_file(review_manifest_path)
        ),
        "development_review_claim_contract": (
            review_manifest["decision_rule"]["primary_candidate"]
            == GLOBAL_MAP
            and review_manifest["decision_rule"]["baseline_candidate"]
            == SELECTED_COUNT
            and review_manifest["dataset"]["held_out"] is False
            and not review_manifest["scoring"]["llm_judges"]
            and review_manifest["provenance"]["evidence_mode"]
            == "development"
        ),
        "scoped_development_preflight_nonblocking": (
            development_preflight["status"] in {"pass", "warn"}
            and not development_preflight["release_blocking"]
            and development_preflight["provenance"]["policy_sha256"]
            == sha256_file(development_policy_path)
        ),
        "base_preflight_release_blocking_by_design": (
            base_preflight["status"] == "fail"
            and bool(base_preflight["release_blocking"])
            and {
                "dataset_not_held_out",
                "order_randomization_missing",
                "position_swap_missing",
            }.issubset({row["id"] for row in base_preflight["findings"]})
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"v13 evidence audit failed: {failed}")

    factorized_exact = _mean_metric(
        candidates,
        FACTORIZED,
        seeds,
        "exact_array_accuracy",
    )
    composition_exact = _mean_metric(
        candidates,
        GLOBAL_MAP,
        seeds,
        "exact_array_accuracy",
    )
    selected_count_exact = _mean_metric(
        candidates,
        SELECTED_COUNT,
        seeds,
        "exact_array_accuracy",
    )
    composition_selected = independently_selected[GLOBAL_MAP]
    composition_qualified = not _gate_failures(composition_selected, contract)
    improvement_supported = composition_exact > selected_count_exact
    claim_supported = (
        improvement_supported
        and composition_qualified
        and independent_primary is not None
        and independent_primary["candidate_id"] == GLOBAL_MAP
    )
    report = {
        "schema_version": "ldt-status-composition-evidence-audit.v13",
        "experiment_id": preregistration["experiment_id"],
        "reviewer": {
            "role": "POST_RUN_PUBLIC_RECEIPT_AUDITOR",
            "implementation": "scripts/audit_ldt_status_composition_v13.py",
            "private_labels_available": False,
            "third_party_identity_authenticated": False,
        },
        "checks": checks,
        "measurement": {
            "seed_count": len(seeds),
            "lineage_fold_count": 6,
            "selected_count_mean_exact_status_vector_accuracy": (
                selected_count_exact
            ),
            "composition_factorized_mean_exact_status_vector_accuracy": (
                factorized_exact
            ),
            "composition_global_mean_exact_status_vector_accuracy": (
                composition_exact
            ),
            "composition_global_minus_selected_count_mean_exact_delta": (
                composition_exact - selected_count_exact
            ),
            "composition_global_minus_selected_count_per_seed_exact_deltas": (
                control_exact_deltas
            ),
            "composition_global_minus_selected_count_per_seed_worst_selected_deltas": (
                control_worst_selected_deltas
            ),
            "composition_global_minus_selected_count_per_seed_status_f1_deltas": (
                control_status_deltas
            ),
            "global_minus_factorized_per_seed_exact_deltas": (
                composition_decoder_exact_deltas
            ),
            "global_minus_factorized_per_seed_status_f1_deltas": (
                composition_decoder_status_deltas
            ),
            "global_decoder_changes_on_target_free_predictions": (
                decoder_change_count
            ),
            "frozen_selected_seed_by_candidate": {
                candidate_id: int(row["seed"])
                for candidate_id, row in sorted(independently_selected.items())
            },
            "selected_composition_gate_failures": _gate_failures(
                composition_selected,
                contract,
            ),
            "numerical_gate_passers_before_frozen_seed_selection": (
                numerical_gate_passers
            ),
            "selection_semantics": (
                "rank one seed per architecture, then apply qualification"
            ),
        },
        "resources": resources,
        "conclusions": {
            "metric_robustness": (
                "supported_on_declared_independent_v10_holdout"
            ),
            "task_result": (
                "composition_underperformed_selected_count_control"
                if not improvement_supported
                else "composition_improved_selected_count_control"
            ),
            "measurement_reliability": (
                "development_receipts_consistent_and_release_scope_blocked"
            ),
            "claim_support": "supported" if claim_supported else "not_supported",
            "operational_decision": (
                "eligible_for_one_independent_confirmation_join"
                if claim_supported
                else "preserve_confirmation_and_reject_promotion"
            ),
            "selection_note": (
                "A nonselected control seed cleared the numeric gates, but "
                "the frozen rank-then-qualify rule retained a different "
                "control seed. Post-hoc substitution is not authorized."
                if numerical_gate_passers
                else "No seed cleared every numeric development gate."
            ),
        },
        "bindings": {
            "auditor_sha256": sha256_file(Path(__file__).resolve()),
            "preregistration_sha256": sha256_file(preregistration_path),
            "review_manifest_sha256": sha256_file(review_manifest_path),
            "development_policy_sha256": sha256_file(development_policy_path),
            "development_preflight_sha256": sha256_file(
                development_preflight_path
            ),
            "base_preflight_sha256": sha256_file(base_preflight_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "merge_receipt_sha256": sha256_file(merge_path),
            "trainer_summary_sha256": sha256_file(trainer_path),
            "pair_predictions_sha256": sha256_file(pair_path),
            "array_predictions_sha256": sha256_file(array_path),
            "selection_receipt_sha256": sha256_file(selection_path),
        },
        "claim_boundary": (
            "Public development-receipt consistency only; no independent "
            "confirmation quality claim, promotion authority, or larger-model "
            "training authorization."
        ),
    }
    dump_structured(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit v13 exact status-composition evidence"
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--development-policy", type=Path, required=True)
    parser.add_argument("--development-preflight", type=Path, required=True)
    parser.add_argument("--base-preflight", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = audit(
        arguments.preregistration.resolve(),
        arguments.review_manifest.resolve(),
        arguments.development_policy.resolve(),
        arguments.development_preflight.resolve(),
        arguments.base_preflight.resolve(),
        arguments.dataset_manifest.resolve(),
        arguments.run_root.resolve(),
        arguments.out.resolve(),
    )
    print(result["conclusions"]["operational_decision"])
