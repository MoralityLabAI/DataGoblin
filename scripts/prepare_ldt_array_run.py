#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datagoblin.experiments.ldt_array_sorter import FEATURE_WHITELIST, load_public_dataset, sha256_file
from datagoblin.serialization import dump_structured, load_structured


def prepare(preregistration_path: Path, dataset_manifest_path: Path, out_path: Path) -> dict[str, object]:
    preregistration = load_structured(preregistration_path)
    if preregistration["status"] != "frozen_before_heldout_evaluation":
        raise ValueError("array sorter preregistration is not frozen")
    dataset_manifest, train_rows, heldout_rows = load_public_dataset(dataset_manifest_path)
    dataset_contract = preregistration["dataset_contract"]
    if dataset_contract["feature_whitelist"] != list(FEATURE_WHITELIST):
        raise ValueError("preregistration feature whitelist does not match the implementation")
    if len(train_rows) != int(dataset_contract["required_train_examples"]):
        raise ValueError("training example count differs from the preregistration")
    if len(heldout_rows) != int(dataset_contract["required_heldout_examples"]):
        raise ValueError("heldout example count differs from the preregistration")
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
        raise ValueError("hard caps differ from the approved array-gym contract")
    plan: dict[str, object] = {
        "schema_version": "ldt-array-trainer-plan.v1",
        "experiment_id": preregistration["experiment_id"],
        "training_task_id": preregistration["training_task_id"],
        "operator_approval": "user instruction: continue experimenting",
        "preregistration_sha256": sha256_file(preregistration_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "dataset_id": dataset_manifest["succession_id"],
        "dataset_counts": {"train": len(train_rows), "heldout": len(heldout_rows)},
        "candidate_models": preregistration["models"]["candidates"],
        "controls": preregistration["models"]["controls"],
        "training": preregistration["training"],
        "array_representation": preregistration["array_representation"],
        "hard_caps": hard_caps,
        "heldout_label_access": False,
        "external_checkpoint": None,
        "network": False,
        "sorter_authority": preregistration["promotion_boundary"]["sorter_authority"],
        "larger_model_training_authorized": False,
    }
    dump_structured(out_path, plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a frozen, public-only LDT array run plan")
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    prepare(arguments.preregistration, arguments.dataset_manifest, arguments.out)
    print(arguments.out)
