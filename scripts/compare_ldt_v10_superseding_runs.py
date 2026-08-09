#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_sorter import sha256_file
from datagoblin.serialization import dump_structured, load_structured


def _candidate_receipts(summary: dict[str, Any]) -> dict[tuple[str, int], Any]:
    return {
        (str(row["candidate_id"]), int(row["seed"])): row
        for row in summary["candidate_receipts"]
    }


def compare(
    original_root: Path,
    superseding_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    original_trainer_path = original_root / "merged/trainer_summary.json"
    superseding_trainer_path = (
        superseding_root / "merged/trainer_summary.json"
    )
    original_selection_path = (
        original_root / "selected/selection_receipt.json"
    )
    superseding_selection_path = (
        superseding_root / "selected/selection_receipt.json"
    )
    original_trainer = load_structured(original_trainer_path)
    superseding_trainer = load_structured(superseding_trainer_path)
    original_selection = load_structured(original_selection_path)
    superseding_selection = load_structured(superseding_selection_path)

    checks = {
        "same_experiment": (
            original_trainer["experiment_id"]
            == superseding_trainer["experiment_id"]
        ),
        "same_preregistration": (
            original_trainer["preregistration_sha256"]
            == superseding_trainer["preregistration_sha256"]
        ),
        "same_dataset": (
            original_trainer["dataset_manifest_sha256"]
            == superseding_trainer["dataset_manifest_sha256"]
        ),
        "same_pair_predictions": (
            original_trainer["predictions_sha256"]
            == superseding_trainer["predictions_sha256"]
        ),
        "same_array_predictions": (
            original_trainer["array_predictions_sha256"]
            == superseding_trainer["array_predictions_sha256"]
        ),
        "same_candidate_receipts": (
            _candidate_receipts(original_trainer)
            == _candidate_receipts(superseding_trainer)
        ),
        "same_candidate_selection": (
            original_selection["candidate_selection"]
            == superseding_selection["candidate_selection"]
        ),
        "same_selected_pair_predictions": (
            original_selection["selected_pair_predictions_sha256"]
            == superseding_selection["selected_pair_predictions_sha256"]
        ),
        "same_selected_array_predictions": (
            original_selection["selected_array_predictions_sha256"]
            == superseding_selection["selected_array_predictions_sha256"]
        ),
        "same_primary_decision": all(
            original_selection[field] == superseding_selection[field]
            for field in (
                "primary_candidate_id",
                "primary_seed",
                "no_qualified_primary",
                "confirmation_join_authorized",
                "confirmation_labels_read",
            )
        ),
        "both_target_clean": (
            not original_trainer["heldout_pair_targets_seen"]
            and not original_trainer["heldout_array_targets_seen"]
            and not superseding_trainer["heldout_pair_targets_seen"]
            and not superseding_trainer["heldout_array_targets_seen"]
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"superseding evaluator comparison failed: {failed}")
    report = {
        "schema_version": "ldt-edge-set-supersession.v10",
        "experiment_id": original_trainer["experiment_id"],
        "reason": (
            "Bind whole-array development metrics directly to the held-out-"
            "validated candidate-ID scorer without changing model behavior."
        ),
        "checks": checks,
        "status": "superseding_evaluator_equivalent",
        "bindings": {
            "preregistration_sha256": original_trainer[
                "preregistration_sha256"
            ],
            "dataset_manifest_sha256": original_trainer[
                "dataset_manifest_sha256"
            ],
            "pair_predictions_sha256": original_trainer[
                "predictions_sha256"
            ],
            "array_predictions_sha256": original_trainer[
                "array_predictions_sha256"
            ],
            "original_trainer_summary_sha256": sha256_file(
                original_trainer_path
            ),
            "superseding_trainer_summary_sha256": sha256_file(
                superseding_trainer_path
            ),
            "original_selection_sha256": sha256_file(
                original_selection_path
            ),
            "superseding_selection_sha256": sha256_file(
                superseding_selection_path
            ),
        },
        "claim_boundary": (
            "Evaluator equivalence only; it creates no additional model sample, "
            "confirmation reveal, or promotion authority."
        ),
    }
    dump_structured(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare original and superseding v10 evaluator runs"
    )
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--superseding-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = compare(
        arguments.original_root.resolve(),
        arguments.superseding_root.resolve(),
        arguments.out.resolve(),
    )
    print(result["status"])
