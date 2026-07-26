#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datagoblin.experiments.ldt_array_hard_surface import (
    HARD_FEATURE_WHITELIST,
    load_hard_surface_dataset,
)
from datagoblin.experiments.ldt_array_sorter import sha256_file
from datagoblin.serialization import dump_structured, load_structured


def prepare(
    preregistration_path: Path,
    dataset_manifest_path: Path,
    out_path: Path,
) -> dict[str, object]:
    preregistration = load_structured(preregistration_path)
    if preregistration["status"] != "frozen_before_dataset_generation":
        raise ValueError("hard-surface preregistration is not frozen")
    manifest, train_pairs, heldout_pairs, train_arrays, heldout_arrays = (
        load_hard_surface_dataset(dataset_manifest_path)
    )
    contract = preregistration["dataset_contract"]
    observed = {
        "train_pairs": len(train_pairs),
        "heldout_pairs": len(heldout_pairs),
        "train_arrays": len(train_arrays),
        "heldout_arrays": len(heldout_arrays),
    }
    expected = {
        "train_pairs": int(contract["required_train_pairs"]),
        "heldout_pairs": int(contract["required_heldout_pairs"]),
        "train_arrays": int(contract["required_train_arrays"]),
        "heldout_arrays": int(contract["required_heldout_arrays"]),
    }
    if observed != expected:
        raise ValueError("hard-surface counts differ from preregistration")
    if manifest["feature_whitelist"] != list(HARD_FEATURE_WHITELIST):
        raise ValueError("hard-surface feature whitelist differs from implementation")
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
        raise ValueError("hard caps differ from the approved v2 contract")
    plan: dict[str, object] = {
        "schema_version": "ldt-array-hard-surface-trainer-plan.v2",
        "experiment_id": preregistration["experiment_id"],
        "training_task_id": preregistration["training_task_id"],
        "operator_approval": "user instruction: do",
        "preregistration_sha256": sha256_file(preregistration_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "dataset_counts": observed,
        "preference_labels": manifest["preference_labels"],
        "rationale_labels": manifest["rationale_labels"],
        "array_status_labels": manifest["array_status_labels"],
        "candidate_models": preregistration["models"]["candidates"],
        "controls": preregistration["models"]["controls"],
        "training": preregistration["training"],
        "hard_caps": hard_caps,
        "metric_equivalence_tolerance": manifest["metric_equivalence_tolerance"],
        "heldout_pair_label_access": False,
        "heldout_array_label_access": False,
        "external_checkpoint": None,
        "network": False,
        "sorter_authority": "MODEL_RECOMMENDATION_ONLY",
        "larger_model_training_authorized": False,
    }
    dump_structured(out_path, plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a public-only LDT hard-surface trainer plan"
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    prepare(arguments.preregistration, arguments.dataset_manifest, arguments.out)
    print(arguments.out)
