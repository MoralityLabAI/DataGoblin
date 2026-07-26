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
    if preregistration.get("schema_version") == "7.0":
        feature_views = {
            str(model["feature_view"])
            for model in preregistration["models"]["candidates"]
        }
        if feature_views != {"independent"}:
            raise ValueError("v7 candidates include a target-derived feature view")
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
                "v7 independent feature view contains forbidden proxies"
            )
        feature_independence_audit = {
            "candidate_feature_views": sorted(feature_views),
            "independent_feature_count": len(independent_names),
            "forbidden_feature_hits": forbidden_hits,
            "typed_or_lattice_views_present": False,
            "audit_split": "development_only",
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
