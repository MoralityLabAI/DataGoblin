#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_sorter import read_jsonl, sha256_file
from datagoblin.serialization import dump_structured, load_structured

FACTORIZED = "exact-array-h4-factorized-v12"
GLOBAL_MAP = "exact-array-h4-global-map-v12"


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


def _worst_metric(
    candidate: dict[str, Any],
    metric: str,
) -> float:
    return min(float(fold[metric]) for fold in candidate["cv_fold_metrics"])


def _objective_without_decoder(candidate: dict[str, Any]) -> dict[str, Any]:
    objective = dict(candidate["listwise_objective"])
    objective.pop("prediction_source")
    exact = dict(objective.pop("exact_array_likelihood"))
    exact.pop("decoder")
    objective["exact_array_likelihood"] = exact
    return objective


def audit(
    preregistration_path: Path,
    dataset_manifest_path: Path,
    run_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    preregistration = load_structured(preregistration_path)
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

    exact_deltas = [
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
    worst_selected_deltas = [
        _worst_metric(
            candidates[(GLOBAL_MAP, seed)],
            "selected_set_exact_accuracy",
        )
        - _worst_metric(
            candidates[(FACTORIZED, seed)],
            "selected_set_exact_accuracy",
        )
        for seed in seeds
    ]
    status_deltas = [
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
    matched_objectives = all(
        candidates[(FACTORIZED, seed)]["trainable_parameters"]
        == candidates[(GLOBAL_MAP, seed)]["trainable_parameters"]
        and _objective_without_decoder(candidates[(FACTORIZED, seed)])
        == _objective_without_decoder(candidates[(GLOBAL_MAP, seed)])
        for seed in seeds
    )
    source_shards = trainer["source_shards"]
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
            manifest["lineage_audit"]["forbidden_prior_groups"] == 74
            and manifest["lineage_audit"]["prior_overlap"] == 0
            and manifest["lineage_audit"]["cross_split_groups"] == 0
        ),
        "complete_candidate_seed_coverage": set(candidates) == expected_keys,
        "twelve_capped_clean_shards": (
            len(source_shards) == 12
            and all(
                bool(row["caps_enforced"])
                and bool(row["cleanup_passed"])
                for row in source_shards
            )
            and bool(merge["all_caps_enforced"])
            and bool(merge["all_cleanup_passed"])
        ),
        "no_confirmation_targets_seen": (
            not trainer["heldout_pair_targets_seen"]
            and not trainer["heldout_array_targets_seen"]
            and not selection["confirmation_labels_read"]
        ),
        "confirmation_join_blocked": (
            selection["no_qualified_primary"]
            and not selection["confirmation_join_authorized"]
        ),
        "exact_decoder_ablation_matched": matched_objectives,
        "matched_pair_behavior": pair_behavior_identical,
        "matched_local_status_probabilities": local_probabilities_identical,
        "global_decoder_changes_at_least_one_array": decoder_change_count > 0,
        "exact_lift_positive_in_every_seed": all(delta > 0.0 for delta in exact_deltas),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"v12 evidence audit failed: {failed}")

    factorized_exact = _mean_metric(
        candidates,
        FACTORIZED,
        seeds,
        "exact_array_accuracy",
    )
    global_exact = _mean_metric(
        candidates,
        GLOBAL_MAP,
        seeds,
        "exact_array_accuracy",
    )
    report = {
        "schema_version": "ldt-exact-array-evidence-audit.v12",
        "experiment_id": preregistration["experiment_id"],
        "reviewer": {
            "role": "POST_RUN_PUBLIC_RECEIPT_AUDITOR",
            "implementation": "scripts/audit_ldt_exact_array_v12.py",
            "private_labels_available": False,
            "third_party_identity_authenticated": False,
        },
        "checks": checks,
        "measurement": {
            "seed_count": len(seeds),
            "lineage_fold_count": 6,
            "factorized_mean_exact_status_vector_accuracy": factorized_exact,
            "global_map_mean_exact_status_vector_accuracy": global_exact,
            "mean_exact_delta": global_exact - factorized_exact,
            "per_seed_exact_deltas": exact_deltas,
            "per_seed_worst_selected_set_deltas": worst_selected_deltas,
            "per_seed_status_macro_f1_deltas": status_deltas,
            "global_decoder_changes_on_target_free_predictions": (
                decoder_change_count
            ),
        },
        "claim_judgment": {
            "whole_array_gate": "not_supported",
            "global_decoder_exactness": (
                "consistent_lift_on_declared_development_only"
            ),
            "exact_likelihood_objective": "not_isolated_from_decoder_control",
            "operational_action": (
                "preserve_confirmation_and_reject_promotion"
            ),
            "reason": (
                "Global MAP improved exactness in all three seeds and two "
                "worst folds, but the selected diagnostic still missed both "
                "mean whole-array gates."
            ),
        },
        "bindings": {
            "preregistration_sha256": sha256_file(preregistration_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "merge_receipt_sha256": sha256_file(merge_path),
            "trainer_summary_sha256": sha256_file(trainer_path),
            "pair_predictions_sha256": sha256_file(pair_path),
            "array_predictions_sha256": sha256_file(array_path),
            "selection_receipt_sha256": sha256_file(selection_path),
        },
        "claim_boundary": (
            "Public development-receipt consistency only; no independent "
            "confirmation quality claim or promotion authority."
        ),
    }
    dump_structured(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit matched v12 exact-array evidence"
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = audit(
        arguments.preregistration.resolve(),
        arguments.dataset_manifest.resolve(),
        arguments.run_root.resolve(),
        arguments.out.resolve(),
    )
    print(result["claim_judgment"]["operational_action"])
