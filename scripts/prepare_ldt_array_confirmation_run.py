#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datagoblin.experiments.ldt_array_confirmation import (
    CONFIRMATION_FEATURE_WHITELIST,
    ConfirmationPairEncoder,
    load_confirmation_dataset,
)
from datagoblin.experiments.ldt_array_sorter import sha256_file
from datagoblin.experiments.ldt_listwise_array_v9 import (
    IndependentCandidateEncoder,
)
from datagoblin.experiments.ldt_target_independent_v7 import (
    INDEPENDENT_FEATURE_FORBIDDEN_FRAGMENTS,
)
from datagoblin.serialization import dump_structured, load_structured


def prepare(
    preregistration_path: Path,
    dataset_manifest_path: Path,
    out_path: Path,
) -> dict[str, object]:
    preregistration = load_structured(preregistration_path)
    if preregistration["status"] != "frozen_before_dataset_generation":
        raise ValueError("confirmation preregistration is not frozen")
    (
        manifest,
        train_pairs,
        heldout_pairs,
        train_arrays,
        heldout_arrays,
        train_reports,
        heldout_reports,
    ) = load_confirmation_dataset(dataset_manifest_path)
    contract = preregistration["dataset_contract"]
    observed = {
        "train_pairs": len(train_pairs),
        "heldout_pairs": len(heldout_pairs),
        "train_arrays": len(train_arrays),
        "heldout_arrays": len(heldout_arrays),
        "train_reports": len(train_reports),
        "heldout_reports": len(heldout_reports),
    }
    expected = {
        name: int(contract[f"required_{name}"])
        for name in observed
    }
    if observed != expected:
        raise ValueError("confirmation counts differ from preregistration")
    if manifest["feature_whitelist"] != list(CONFIRMATION_FEATURE_WHITELIST):
        raise ValueError("confirmation feature whitelist differs")
    feature_independence_audit: dict[str, object] | None = None
    schema_version = str(preregistration.get("schema_version"))
    if schema_version in {"7.0", "8.0", "9.0"}:
        feature_views = {
            str(model["feature_view"])
            for model in preregistration["models"]["candidates"]
        }
        if feature_views != {"independent"}:
            raise ValueError(
                "target-independent candidates include a derived feature view"
            )
        independent_names = ConfirmationPairEncoder.fit(
            train_pairs
        ).feature_names("independent")
        forbidden_hits = [
            name
            for name in independent_names
            if any(
                fragment in name
                for fragment in INDEPENDENT_FEATURE_FORBIDDEN_FRAGMENTS
            )
        ]
        if forbidden_hits:
            raise ValueError(
                "independent feature view contains forbidden proxies"
            )
        feature_independence_audit = {
            "candidate_feature_views": sorted(feature_views),
            "independent_feature_count": len(independent_names),
            "forbidden_feature_hits": forbidden_hits,
            "typed_or_lattice_views_present": False,
            "audit_split": "development_only",
        }
        if schema_version in {"8.0", "9.0"}:
            support = manifest.get("development_support_audit")
            if not support or not all(
                int(count) >= int(
                    preregistration["dataset_contract"][
                        "minimum_lineages_per_critical_class"
                    ]
                )
                for axis in support["lineage_support"].values()
                for count in axis.values()
            ):
                raise ValueError(
                    "v8 development critical-class support is incomplete"
                )
            objectives = [
                {
                    "candidate_id": str(model["model_id"]),
                    "paired_swap_batches": bool(
                        model["paired_swap_batches"]
                    ),
                    "antisymmetric_inference": bool(
                        model["antisymmetric_inference"]
                    ),
                    "array_consistency_loss_weight": float(
                        model["array_consistency_loss_weight"]
                    ),
                }
                for model in preregistration["models"]["candidates"]
            ]
            if any(
                row["array_consistency_loss_weight"] < 0.0
                or row["array_consistency_loss_weight"] > 2.0
                for row in objectives
            ):
                raise ValueError(
                    "array objective weight is out of bounds"
                )
            if schema_version == "8.0" and not any(
                row["paired_swap_batches"]
                and row["antisymmetric_inference"]
                and row["array_consistency_loss_weight"] > 0.0
                for row in objectives
            ):
                raise ValueError(
                    "v8 lacks a complete structured objective"
                )
            feature_independence_audit[
                "structured_objectives"
            ] = objectives
        if schema_version == "9.0":
            listwise = [
                {
                    "candidate_id": str(model["model_id"]),
                    **dict(model["listwise_status"]),
                }
                for model in preregistration["models"]["candidates"]
            ]
            enabled = [
                row for row in listwise if bool(row["enabled"])
            ]
            if (
                len(enabled) < 2
                or not any(
                    not bool(row["enabled"])
                    and row["family"]
                    == "deterministic_pair_reducer"
                    for row in listwise
                )
                or any(
                    row["family"]
                    != "looped_permutation_equivariant_status"
                    or not 1 <= int(row["hidden_width"]) <= 16
                    or not 1 <= int(row["loop_count"]) <= 4
                    or not 1 <= int(row["epochs"]) <= 600
                    or not 0.0
                    < float(row["learning_rate"])
                    <= 0.1
                    for row in enabled
                )
                or any(
                    float(model["array_consistency_loss_weight"])
                    != 0.0
                    for model in preregistration["models"]["candidates"]
                )
                or any(
                    not bool(model["paired_swap_batches"])
                    or not bool(model["antisymmetric_inference"])
                    for model in preregistration["models"]["candidates"]
                )
            ):
                raise ValueError(
                    "v9 listwise ablation contract is incomplete"
                )
            candidate_encoder = IndependentCandidateEncoder()
            feature_independence_audit["listwise_status"] = {
                "candidates": listwise,
                "candidate_encoder": candidate_encoder.receipt(),
                "direct_array_targets": "development_only",
                "heldout_array_targets_available": False,
            }
    hard_caps = preregistration["hard_caps"]
    if hard_caps != {
        "wrapper": "scripts/run_training_gym_capped_wsl.sh",
        "ram_mb": 2048,
        "cpu_percent": 50,
        "io_mb_per_second": 50,
        "wall_clock_seconds": 900,
        "swap_bytes": 0,
        "fallback_if_unavailable": "abort_without_training",
    }:
        raise ValueError("confirmation hard caps differ from the approved contract")
    plan: dict[str, object] = {
        "schema_version": "ldt-array-confirmation-trainer-plan.v3",
        "preregistration_schema_version": schema_version,
        "experiment_id": preregistration["experiment_id"],
        "training_task_id": preregistration["training_task_id"],
        "operator_approval": "user instruction: continue",
        "preregistration_sha256": sha256_file(preregistration_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "dataset_counts": observed,
        "preference_labels": manifest["preference_labels"],
        "rationale_labels": manifest["rationale_labels"],
        "array_status_labels": manifest["array_status_labels"],
        "candidate_models": preregistration["models"]["candidates"],
        "controls": preregistration["models"]["controls"],
        "training": preregistration["training"],
        "confidence_calibration": preregistration["confidence_calibration"],
        "hard_caps": hard_caps,
        "metric_equivalence_tolerance": manifest["metric_equivalence_tolerance"],
        "heldout_pair_label_access": False,
        "heldout_array_label_access": False,
        "external_checkpoint": None,
        "network": False,
        "sorter_authority": "MODEL_RECOMMENDATION_ONLY",
        "larger_model_training_authorized": False,
    }
    if feature_independence_audit is not None:
        plan["feature_independence_audit"] = feature_independence_audit
    dump_structured(out_path, plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a public-only LDT confirmation trainer plan"
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    prepare(arguments.preregistration, arguments.dataset_manifest, arguments.out)
    print(arguments.out)
