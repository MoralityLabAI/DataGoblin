#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_sorter import read_jsonl, sha256_file
from datagoblin.serialization import dump_structured, load_structured

PAIR_CONTROL = "looped-h4-pair-control-v11"
RECURRENT = "looped-h4-recurrent-factorized-v11"
CARDINALITY_FACTORIZED = "cardinality-h4-factorized-v11"
CARDINALITY_JOINT = "cardinality-h4-joint-map-v11"


def _candidate_seed_map(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["candidate_id"]), int(row["seed"])): row
        for row in rows
    }


def _prediction_map(
    rows: list[dict[str, Any]],
    *,
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
    pair_predictions = _prediction_map(
        read_jsonl(pair_path),
        identity_field="example_id",
    )
    array_predictions = _prediction_map(
        read_jsonl(array_path),
        identity_field="array_id",
    )
    candidates = _candidate_seed_map(trainer["candidate_receipts"])
    expected_candidates = [
        str(model["model_id"])
        for model in preregistration["models"]["candidates"]
    ]
    seeds = [int(seed) for seed in preregistration["training"]["seeds"]]
    expected_keys = {
        (candidate_id, seed)
        for candidate_id in expected_candidates
        for seed in seeds
    }
    observed_keys = set(candidates)
    source_shards = trainer["source_shards"]

    matched_pair_behavior = True
    recurrent_factorized_arrays_identical = True
    cardinality_probabilities_identical = True
    joint_status_change_count = 0
    for seed in seeds:
        recurrent_pair_ids = {
            identity
            for candidate_id, row_seed, identity in pair_predictions
            if candidate_id == RECURRENT and row_seed == seed
        }
        for identity in recurrent_pair_ids:
            reference = _pair_behavior(
                pair_predictions[(RECURRENT, seed, identity)]
            )
            matched_pair_behavior &= all(
                _pair_behavior(pair_predictions[(candidate_id, seed, identity)])
                == reference
                for candidate_id in (
                    CARDINALITY_FACTORIZED,
                    CARDINALITY_JOINT,
                )
            )

        recurrent_array_ids = {
            identity
            for candidate_id, row_seed, identity in array_predictions
            if candidate_id == RECURRENT and row_seed == seed
        }
        for identity in recurrent_array_ids:
            recurrent = array_predictions[(RECURRENT, seed, identity)]
            factorized = array_predictions[
                (CARDINALITY_FACTORIZED, seed, identity)
            ]
            joint = array_predictions[(CARDINALITY_JOINT, seed, identity)]
            recurrent_factorized_arrays_identical &= (
                recurrent["statuses"] == factorized["statuses"]
                and recurrent["status_probabilities"]
                == factorized["status_probabilities"]
            )
            cardinality_probabilities_identical &= (
                factorized["status_probabilities"]
                == joint["status_probabilities"]
            )
            joint_status_change_count += int(
                factorized["statuses"] != joint["statuses"]
            )

    factorized_exact = _mean_metric(
        candidates,
        CARDINALITY_FACTORIZED,
        seeds,
        "exact_array_accuracy",
    )
    joint_exact = _mean_metric(
        candidates,
        CARDINALITY_JOINT,
        seeds,
        "exact_array_accuracy",
    )
    factorized_status = _mean_metric(
        candidates,
        CARDINALITY_FACTORIZED,
        seeds,
        "status_macro_f1",
    )
    joint_status = _mean_metric(
        candidates,
        CARDINALITY_JOINT,
        seeds,
        "status_macro_f1",
    )
    exact_deltas = [
        float(
            candidates[(CARDINALITY_JOINT, seed)]["cv_mean_metrics"][
                "exact_array_accuracy"
            ]
        )
        - float(
            candidates[(CARDINALITY_FACTORIZED, seed)]["cv_mean_metrics"][
                "exact_array_accuracy"
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
            manifest["lineage_audit"]["forbidden_prior_groups"] == 64
            and manifest["lineage_audit"]["prior_overlap"] == 0
            and manifest["lineage_audit"]["cross_split_groups"] == 0
        ),
        "complete_candidate_seed_coverage": observed_keys == expected_keys,
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
        "matched_pair_behavior": matched_pair_behavior,
        "ignored_readout_control_is_behaviorally_identical": (
            recurrent_factorized_arrays_identical
        ),
        "joint_decoder_uses_identical_status_probabilities": (
            cardinality_probabilities_identical
        ),
        "joint_decoder_changes_at_least_one_array": (
            joint_status_change_count > 0
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"v11 evidence audit failed: {failed}")

    report = {
        "schema_version": "ldt-cardinality-evidence-audit.v11",
        "experiment_id": preregistration["experiment_id"],
        "reviewer": {
            "role": "POST_RUN_PUBLIC_RECEIPT_AUDITOR",
            "implementation": "scripts/audit_ldt_cardinality_v11.py",
            "private_labels_available": False,
            "third_party_identity_authenticated": False,
        },
        "checks": checks,
        "measurement": {
            "seed_count": len(seeds),
            "lineage_fold_count": 6,
            "joint_status_change_count_on_target_free_predictions": (
                joint_status_change_count
            ),
            "factorized_mean_exact_status_vector_accuracy": factorized_exact,
            "joint_mean_exact_status_vector_accuracy": joint_exact,
            "mean_exact_delta": joint_exact - factorized_exact,
            "per_seed_exact_deltas": exact_deltas,
            "factorized_mean_status_macro_f1": factorized_status,
            "joint_mean_status_macro_f1": joint_status,
            "mean_status_macro_f1_delta": joint_status - factorized_status,
        },
        "claim_judgment": {
            "whole_array_gate": "not_supported",
            "decoder_mechanism": "directional_mixed_seed_only",
            "reason": (
                "Mean exactness rose, but one seed regressed, worst-fold "
                "exactness did not improve, and status macro-F1 declined."
            ),
            "operational_action": (
                "preserve_confirmation_and_reject_promotion"
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
        description="Audit matched v11 cardinality-decoder evidence"
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
