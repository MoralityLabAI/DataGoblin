#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_sorter import read_jsonl, sha256_file
from datagoblin.serialization import dump_structured, load_structured

FACTORIZED = "exact-edge-h4-factorized-v15"
GLOBAL_MAP = "exact-edge-h4-global-v15"
INDEPENDENT_GLOBAL = "exact-independent-h4-global-v15"


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


def _worst_metric(candidate: dict[str, Any], metric: str) -> float:
    return min(float(fold[metric]) for fold in candidate["cv_fold_metrics"])


def _objective_without_decoder(candidate: dict[str, Any]) -> dict[str, Any]:
    objective = dict(candidate["listwise_objective"])
    objective.pop("prediction_source")
    exact = dict(objective.pop("exact_array_likelihood"))
    exact.pop("decoder")
    objective["exact_array_likelihood"] = exact
    return objective


def _objective_without_candidate_encoder(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    objective = dict(candidate["listwise_objective"])
    objective.pop("candidate_encoder")
    objective.pop("message_steps")
    objective.pop("edge_statistics", None)
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
    if _worst_metric(receipt, "exact_array_accuracy") < float(
        contract.get(
            "minimum_worst_fold_exact_array_accuracy",
            float("-inf"),
        )
    ):
        failures.append("worst_fold_exact_array_accuracy")
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
    rows = []
    valid = set(source_by_key) == expected_keys
    for candidate_id, seed in sorted(expected_keys):
        path = run_root / f"{candidate_id}-seed-{seed}" / "run_receipt.json"
        if not path.is_file():
            valid = False
            continue
        receipt = load_structured(path)
        source = source_by_key.get((candidate_id, seed), {})
        caps = receipt["caps_receipt"]
        trainer = receipt["trainer_receipt"]
        cleanup = receipt["cleanup_receipt"]
        valid &= (
            source.get("run_receipt_sha256") == sha256_file(path)
            and receipt["capped_process_exit_code"] == 0
            and receipt["wsl_stage_cleanup_exit_code"] == 0
            and bool(caps["caps_enforced"])
            and caps["exit_code"] == 0
            and trainer["status"] == "completed"
            and bool(trainer["hard_caps_verified"])
            and not trainer["heldout_pair_targets_seen"]
            and not trainer["heldout_array_targets_seen"]
            and bool(cleanup["cleanup_passed"])
            and not cleanup["lingering_owned_pids"]
            and not receipt["heldout_pair_labels_staged"]
            and not receipt["heldout_array_labels_staged"]
            and not receipt["external_checkpoint_loaded"]
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "receipt_sha256": sha256_file(path),
                "elapsed_seconds": float(trainer["elapsed_seconds"]),
                "steps_completed": int(trainer["steps_completed"]),
                "peak_ram_mb": float(caps["peak_ram_mb"]),
                "cpu_pct": float(caps["cpu_pct"]),
                "peak_io_mb_s": float(caps["peak_io_mb_s"]),
            }
        )
    return valid, {
        "shard_count": len(rows),
        "total_trainer_seconds": sum(row["elapsed_seconds"] for row in rows),
        "total_steps_completed": sum(row["steps_completed"] for row in rows),
        "maximum_peak_ram_mb": max(row["peak_ram_mb"] for row in rows),
        "maximum_cpu_pct": max(row["cpu_pct"] for row in rows),
        "maximum_peak_io_mb_s": max(row["peak_io_mb_s"] for row in rows),
        "receipt_sha256": [row["receipt_sha256"] for row in rows],
    }


def _heldout_pass(
    aggregate: dict[str, Any],
    heldout_contract: dict[str, Any],
) -> bool:
    metrics = aggregate["metrics"]
    required_flags = (
        "all_seeds_safety_qualified",
        "all_seeds_preference_qualified",
        "all_seeds_orientation_qualified",
        "all_seeds_uncertainty_qualified",
        "all_seeds_calibration_qualified",
        "all_seeds_status_qualified",
        "all_seeds_exact_array_qualified",
        "all_seeds_selected_set_qualified",
    )
    return (
        all(bool(aggregate[name]) for name in required_flags)
        and float(metrics["exact_array_accuracy"]["mean"])
        >= float(heldout_contract["minimum_exact_status_vector_accuracy"])
        and float(metrics["selected_set_exact_accuracy"]["mean"])
        >= float(heldout_contract["minimum_selected_set_exact_accuracy"])
    )


def audit(
    preregistration_path: Path,
    review_manifest_path: Path,
    development_policy_path: Path,
    development_preflight_path: Path,
    base_preflight_path: Path,
    dataset_manifest_path: Path,
    run_root: Path,
    output_path: Path,
    *,
    heldout_evaluation_path: Path | None = None,
    heldout_receipt_path: Path | None = None,
) -> dict[str, Any]:
    preregistration = load_structured(preregistration_path)
    review = load_structured(review_manifest_path)
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

    pair_identical = True
    probabilities_identical = True
    decoder_changes = 0
    edge_probability_changes = 0
    edge_status_changes = 0
    for seed in seeds:
        pair_ids = {
            identity
            for candidate_id, row_seed, identity in pair_predictions
            if candidate_id == FACTORIZED and row_seed == seed
        }
        for identity in pair_ids:
            pair_identical &= (
                _pair_behavior(pair_predictions[(FACTORIZED, seed, identity)])
                == _pair_behavior(pair_predictions[(GLOBAL_MAP, seed, identity)])
                == _pair_behavior(
                    pair_predictions[(INDEPENDENT_GLOBAL, seed, identity)]
                )
            )
        array_ids = {
            identity
            for candidate_id, row_seed, identity in array_predictions
            if candidate_id == FACTORIZED and row_seed == seed
        }
        for identity in array_ids:
            factorized = array_predictions[(FACTORIZED, seed, identity)]
            global_map = array_predictions[(GLOBAL_MAP, seed, identity)]
            probabilities_identical &= (
                factorized["status_probabilities"]
                == global_map["status_probabilities"]
            )
            decoder_changes += int(
                factorized["statuses"] != global_map["statuses"]
            )
            independent = array_predictions[
                (INDEPENDENT_GLOBAL, seed, identity)
            ]
            edge_probability_changes += int(
                independent["status_probabilities"]
                != global_map["status_probabilities"]
            )
            edge_status_changes += int(
                independent["statuses"] != global_map["statuses"]
            )
    matched = all(
        candidates[(FACTORIZED, seed)]["trainable_parameters"]
        == candidates[(GLOBAL_MAP, seed)]["trainable_parameters"]
        and _objective_without_decoder(candidates[(FACTORIZED, seed)])
        == _objective_without_decoder(candidates[(GLOBAL_MAP, seed)])
        for seed in seeds
    )
    encoder_isolated = all(
        candidates[(GLOBAL_MAP, seed)]["trainable_parameters"]
        > candidates[(INDEPENDENT_GLOBAL, seed)]["trainable_parameters"]
        and _objective_without_candidate_encoder(
            candidates[(GLOBAL_MAP, seed)]
        )
        == _objective_without_candidate_encoder(
            candidates[(INDEPENDENT_GLOBAL, seed)]
        )
        and candidates[(GLOBAL_MAP, seed)]["listwise_objective"][
            "candidate_encoder"
        ]["feature_view"]
        == "edge_difference_message"
        and candidates[(INDEPENDENT_GLOBAL, seed)]["listwise_objective"][
            "candidate_encoder"
        ]["feature_view"]
        == "independent_candidate"
        for seed in seeds
    )

    contract = preregistration["development_selection"]
    qualified_seeds: dict[str, list[int]] = {}
    independently_selected: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        receipts = [candidates[(candidate_id, seed)] for seed in seeds]
        qualified = [
            row for row in receipts if not _gate_failures(row, contract)
        ]
        qualified_seeds[candidate_id] = sorted(
            int(row["seed"]) for row in qualified
        )
        independently_selected[candidate_id] = min(
            qualified if qualified else receipts,
            key=_rank,
        )
    selection_consistent = all(
        selection["candidate_selection"][candidate_id]["selected_seed"]
        == int(independently_selected[candidate_id]["seed"])
        and selection["candidate_selection"][candidate_id][
            "observed_qualified_seeds"
        ]
        == qualified_seeds[candidate_id]
        and selection["candidate_selection"][candidate_id]["selection_lane"]
        == ("qualified" if qualified_seeds[candidate_id] else "diagnostic_only")
        for candidate_id in candidate_ids
    )
    primary_scope = set(contract["primary_candidate_ids"])
    primary_pool = [
        row
        for candidate_id, row in independently_selected.items()
        if candidate_id in primary_scope and not _gate_failures(row, contract)
    ]
    primary = min(primary_pool, key=_rank) if primary_pool else None
    primary_consistent = (
        selection["primary_candidate_id"]
        == (None if primary is None else str(primary["candidate_id"]))
        and selection["primary_seed"]
        == (None if primary is None else int(primary["seed"]))
        and selection["confirmation_join_authorized"] == (primary is not None)
    )

    packet_valid = selection["confirmation_packet"] is None
    packet_bindings: dict[str, str] = {}
    if primary is not None:
        packet = selection["confirmation_packet"]
        primary_pair_path = (
            run_root / "selected/heldout_pair_predictions.primary.jsonl"
        )
        primary_array_path = (
            run_root / "selected/heldout_array_predictions.primary.jsonl"
        )
        primary_summary_path = run_root / "selected/trainer_summary.primary.json"
        primary_summary = load_structured(primary_summary_path)
        packet_valid = (
            packet["candidate_id"] == primary["candidate_id"]
            and packet["seed"] == primary["seed"]
            and packet["candidate_count"] == 1
            and not packet["private_labels_read"]
            and packet["pair_predictions_sha256"]
            == sha256_file(primary_pair_path)
            and packet["array_predictions_sha256"]
            == sha256_file(primary_array_path)
            and packet["trainer_summary_sha256"]
            == sha256_file(primary_summary_path)
            and len(primary_summary["candidate_receipts"]) == 1
            and primary_summary["confirmation_packet_candidate_count"] == 1
        )
        packet_bindings = {
            "primary_pair_predictions_sha256": sha256_file(primary_pair_path),
            "primary_array_predictions_sha256": sha256_file(primary_array_path),
            "primary_trainer_summary_sha256": sha256_file(primary_summary_path),
        }

    resource_valid, resources = _resource_audit(
        run_root,
        expected_keys,
        trainer["source_shards"],
    )
    checks = {
        "preregistration_frozen": (
            preregistration["schema_version"] == "15.0"
            and preregistration["status"]
            == "frozen_before_dataset_generation"
        ),
        "preregistration_bound": (
            trainer["preregistration_sha256"] == sha256_file(preregistration_path)
        ),
        "dataset_bound": (
            trainer["dataset_manifest_sha256"] == sha256_file(dataset_manifest_path)
        ),
        "fresh_lineage_firewall": (
            manifest["lineage_audit"]["forbidden_prior_groups"] == 104
            and manifest["lineage_audit"]["prior_overlap"] == 0
            and manifest["lineage_audit"]["cross_split_groups"] == 0
        ),
        "frozen_array_size_support": (
            manifest["array_size_audit"]["train_distribution"]
            == {
                str(size): int(count)
                for size, count in preregistration["dataset_contract"][
                    "required_train_arrays_by_size"
                ].items()
            }
            and manifest["array_size_audit"]["heldout_distribution"]
            == {
                str(size): int(count)
                for size, count in preregistration["dataset_contract"][
                    "required_heldout_arrays_by_size"
                ].items()
            }
        ),
        "complete_candidate_seed_coverage": set(candidates) == expected_keys,
        "twelve_capped_clean_shards": (
            resource_valid
            and len(trainer["source_shards"]) == 12
            and merge["all_caps_enforced"]
            and merge["all_cleanup_passed"]
        ),
        "no_confirmation_targets_seen_in_development": (
            not trainer["heldout_pair_targets_seen"]
            and not trainer["heldout_array_targets_seen"]
            and not selection["confirmation_labels_read"]
        ),
        "gate_first_selection_independently_reproduced": selection_consistent,
        "primary_decision_independently_reproduced": primary_consistent,
        "primary_only_packet_consistent": packet_valid,
        "exact_decoder_ablation_matched": matched,
        "edge_encoder_ablation_isolated": encoder_isolated,
        "edge_encoder_changes_at_least_one_probability": (
            edge_probability_changes > 0
        ),
        "matched_pair_behavior": pair_identical,
        "matched_local_status_probabilities": probabilities_identical,
        "global_decoder_changes_at_least_one_array": decoder_changes > 0,
        "development_review_bound": (
            development_preflight["provenance"]["manifest_sha256"]
            == sha256_file(review_manifest_path)
            and base_preflight["provenance"]["manifest_sha256"]
            == sha256_file(review_manifest_path)
            and review["decision_rule"]["primary_candidate"] == GLOBAL_MAP
            and review["decision_rule"]["baseline_candidate"]
            == INDEPENDENT_GLOBAL
            and review["decision_rule"]["matched_decoder_candidate"]
            == FACTORIZED
            and review["provenance"][
                "contamination_preflight_manifest_sha256"
            ]
            == sha256_file(dataset_manifest_path)
        ),
        "scoped_development_preflight_nonblocking": (
            development_preflight["status"] in {"pass", "warn"}
            and not development_preflight["release_blocking"]
            and development_preflight["provenance"]["policy_sha256"]
            == sha256_file(development_policy_path)
        ),
        "base_preflight_release_blocking_by_design": (
            base_preflight["status"] == "fail"
            and base_preflight["release_blocking"]
        ),
    }

    heldout_present = heldout_evaluation_path is not None
    if heldout_present != (heldout_receipt_path is not None):
        raise ValueError("heldout report and receipt must be supplied together")
    heldout_passed: bool | None = None
    heldout_measurement: dict[str, Any] | None = None
    heldout_bindings: dict[str, str] = {}
    if heldout_present:
        if primary is None:
            raise ValueError("heldout evidence exists without a qualified primary")
        heldout = load_structured(heldout_evaluation_path)
        heldout_receipt = load_structured(heldout_receipt_path)
        aggregate = heldout["candidate_aggregates"][GLOBAL_MAP]
        heldout_passed = _heldout_pass(
            aggregate,
            preregistration["heldout_evaluation"],
        )
        report_hash = sha256_file(heldout_evaluation_path)
        checks["heldout_primary_only_bindings"] = (
            set(heldout["candidate_aggregates"]) == {GLOBAL_MAP}
            and heldout["development_primary_candidate_id"] == GLOBAL_MAP
            and heldout["preregistration_sha256"]
            == sha256_file(preregistration_path)
            and heldout["dataset_manifest_sha256"]
            == sha256_file(dataset_manifest_path)
            and heldout["predictions_sha256"]
            == packet_bindings["primary_pair_predictions_sha256"]
            and heldout["array_predictions_sha256"]
            == packet_bindings["primary_array_predictions_sha256"]
            and heldout["trainer_summary_sha256"]
            == packet_bindings["primary_trainer_summary_sha256"]
            and heldout["pair_labels_sha256"]
            == manifest["heldout_label_custody"]["pair_labels_sha256"]
            and heldout["array_labels_sha256"]
            == manifest["heldout_label_custody"]["array_labels_sha256"]
        )
        checks["heldout_decision_independently_reproduced"] = (
            heldout["retained_architecture_confirmed"] == heldout_passed
            and heldout["learned_pareto_set"]
            == ([GLOBAL_MAP] if heldout_passed else [])
            and heldout["heldout_reveal_count"] == 1
            and not heldout["post_reveal_tuning_permitted"]
            and not heldout["larger_model_training_authorized"]
            and heldout_receipt["heldout_evaluation_sha256"] == report_hash
            and not heldout_receipt["row_level_labels_exported"]
            and not heldout_receipt["candidate_workspace_mutated"]
        )
        heldout_measurement = {
            "exact_status_vector_accuracy": float(
                aggregate["metrics"]["exact_array_accuracy"]["mean"]
            ),
            "exact_selected_set_accuracy": float(
                aggregate["metrics"]["selected_set_exact_accuracy"]["mean"]
            ),
            "confirmed": heldout_passed,
        }
        heldout_bindings = {
            "heldout_evaluation_sha256": report_hash,
            "heldout_receipt_sha256": sha256_file(heldout_receipt_path),
        }

    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"v15 evidence audit failed: {failed}")

    global_selected = independently_selected[GLOBAL_MAP]
    development_qualified = not _gate_failures(global_selected, contract)
    if primary is None:
        operation = "preserve_confirmation_and_reject_promotion"
    elif not heldout_present:
        operation = "eligible_for_one_independent_confirmation_join"
    elif heldout_passed:
        operation = "confirmed_for_downstream_preregistration_only"
    else:
        operation = "heldout_not_confirmed_no_promotion"
    report = {
        "schema_version": "ldt-edge-exact-evidence-audit.v15",
        "experiment_id": preregistration["experiment_id"],
        "reviewer": {
            "role": "POST_RUN_PUBLIC_RECEIPT_AUDITOR",
            "implementation": "scripts/audit_ldt_edge_exact_correction_v15.py",
            "private_labels_available": False,
            "third_party_identity_authenticated": False,
        },
        "checks": checks,
        "development_measurement": {
            "qualified_seeds_by_candidate": qualified_seeds,
            "selected_seed_by_candidate": {
                candidate_id: int(row["seed"])
                for candidate_id, row in sorted(independently_selected.items())
            },
            "primary_candidate_id": (
                None if primary is None else str(primary["candidate_id"])
            ),
            "primary_seed": None if primary is None else int(primary["seed"]),
            "global_primary_gate_failures": _gate_failures(
                global_selected,
                contract,
            ),
            "global_decoder_changes_on_target_free_predictions": decoder_changes,
            "edge_encoder_probability_changes_on_target_free_predictions": (
                edge_probability_changes
            ),
            "edge_encoder_status_changes_on_target_free_predictions": (
                edge_status_changes
            ),
        },
        "heldout_measurement": heldout_measurement,
        "resources": resources,
        "conclusions": {
            "metric_robustness": (
                "supported_on_declared_independent_v10_holdout"
            ),
            "task_result": (
                "development_qualified"
                if development_qualified
                else "development_not_qualified"
            ),
            "measurement_reliability": (
                "development_receipts_consistent_and_release_scope_blocked"
                if not heldout_present
                else "single_primary_heldout_join_bound_and_reproduced"
            ),
            "development_claim_support": (
                "supported" if development_qualified else "not_supported"
            ),
            "heldout_claim_support": (
                "not_run"
                if heldout_passed is None
                else ("supported" if heldout_passed else "not_supported")
            ),
            "operational_decision": operation,
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
            **packet_bindings,
            **heldout_bindings,
        },
        "claim_boundary": (
            "Synthetic tournament-summary evidence only. Even a heldout pass "
            "authorizes only a separately frozen downstream adapter tournament; "
            "it cannot promote corpus records, deploy a sorter, or train a "
            "larger checkpoint directly."
        ),
    }
    dump_structured(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit v15 edge-exact corrective evidence"
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--development-policy", type=Path, required=True)
    parser.add_argument("--development-preflight", type=Path, required=True)
    parser.add_argument("--base-preflight", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--heldout-evaluation", type=Path)
    parser.add_argument("--heldout-receipt", type=Path)
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
        heldout_evaluation_path=(
            None
            if arguments.heldout_evaluation is None
            else arguments.heldout_evaluation.resolve()
        ),
        heldout_receipt_path=(
            None
            if arguments.heldout_receipt is None
            else arguments.heldout_receipt.resolve()
        ),
    )
    print(result["conclusions"]["operational_decision"])
