#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_sorter import sha256_file
from datagoblin.serialization import dump_structured, load_structured


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _qualified(
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    mean = receipt["cv_mean_metrics"]
    folds = receipt["cv_fold_metrics"]
    calibration = receipt["calibration"]["selected_metrics"]
    return (
        math.isclose(
            float(mean["safety_gate_accuracy"]),
            float(
                contract["required_mean_safety_gate_accuracy"]
            ),
            abs_tol=1e-12,
        )
        and math.isclose(
            float(mean["swap_consistency"]),
            float(contract["required_mean_swap_consistency"]),
            abs_tol=1e-12,
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
        >= float(
            contract[
                "minimum_mean_selected_set_exact_accuracy"
            ]
        )
        and min(
            float(fold["selected_set_exact_accuracy"])
            for fold in folds
        )
        >= float(
            contract[
                "minimum_worst_fold_selected_set_exact_accuracy"
            ]
        )
        and float(calibration["expected_calibration_error"])
        <= float(contract["maximum_oof_expected_calibration_error"])
    )


def _rank(receipt: Mapping[str, Any]) -> tuple[float, ...]:
    folds = receipt["cv_fold_metrics"]
    mean = receipt["cv_mean_metrics"]
    calibration = receipt["calibration"]["selected_metrics"]
    return (
        -min(
            float(fold["selected_set_exact_accuracy"])
            for fold in folds
        ),
        -min(
            float(fold["exact_array_accuracy"])
            for fold in folds
        ),
        -min(
            float(fold["status_macro_f1"])
            for fold in folds
        ),
        -min(
            float(fold["preference_macro_f1"])
            for fold in folds
        ),
        -float(mean["selected_set_exact_accuracy"]),
        -float(mean["exact_array_accuracy"]),
        -float(mean["status_macro_f1"]),
        -float(mean["preference_macro_f1"]),
        float(calibration["multiclass_negative_log_likelihood"]),
        float(receipt["trainable_parameters"]),
        float(receipt["seed"]),
    )


def _selected_rows(
    rows: Sequence[Mapping[str, Any]],
    selected_keys: set[tuple[str, int]],
    *,
    id_key: str,
) -> list[Mapping[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if (
                str(row["candidate_id"]),
                int(row["seed"]),
            )
            in selected_keys
        ),
        key=lambda row: (
            str(row["candidate_id"]),
            int(row["seed"]),
            str(row[id_key]),
        ),
    )


def select(
    *,
    preregistration_path: Path,
    trainer_summary_path: Path,
    pair_predictions_path: Path,
    array_predictions_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    preregistration = load_structured(preregistration_path)
    schema_version = str(preregistration["schema_version"])
    if (
        schema_version not in {"9.0", "10.0"}
        or preregistration["status"]
        != "frozen_before_dataset_generation"
    ):
        raise ValueError(
            "selection requires a frozen direct-array contract"
        )
    major_version = schema_version.split(".", 1)[0]
    trainer = json.loads(
        trainer_summary_path.read_text(encoding="utf-8")
    )
    if (
        trainer["status"] != "completed"
        or trainer["heldout_pair_targets_seen"]
        or trainer["heldout_array_targets_seen"]
    ):
        raise ValueError("trainer is incomplete or contaminated")
    if trainer["preregistration_sha256"] != sha256_file(
        preregistration_path
    ):
        raise ValueError("trainer used a different preregistration")
    if trainer["predictions_sha256"] != sha256_file(
        pair_predictions_path
    ):
        raise ValueError("pair predictions hash mismatch")
    if trainer["array_predictions_sha256"] != sha256_file(
        array_predictions_path
    ):
        raise ValueError("array predictions hash mismatch")

    expected_models = {
        str(row["model_id"])
        for row in preregistration["models"]["candidates"]
    }
    expected_seeds = {
        int(value) for value in preregistration["training"]["seeds"]
    }
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for receipt in trainer["candidate_receipts"]:
        if not receipt["trained"]:
            raise ValueError(
                "direct-array selection received an untrained receipt"
            )
        by_candidate[str(receipt["candidate_id"])].append(receipt)
    if set(by_candidate) != expected_models or any(
        {int(row["seed"]) for row in rows} != expected_seeds
        for rows in by_candidate.values()
    ):
        raise ValueError(
            "trainer candidate or seed set differs from contract"
        )

    contract = preregistration["development_selection"]
    selected_receipts = []
    candidate_selection = {}
    for candidate_id, receipts in sorted(by_candidate.items()):
        selected = min(receipts, key=_rank)
        selected_receipts.append(selected)
        folds = selected["cv_fold_metrics"]
        candidate_selection[candidate_id] = {
            "selected_seed": int(selected["seed"]),
            "development_qualified": _qualified(
                selected,
                contract,
            ),
            "rank_tuple": list(_rank(selected)),
            "cv_mean_metrics": selected["cv_mean_metrics"],
            "cv_worst_metrics": {
                name: min(
                    float(fold[name]) for fold in folds
                )
                for name in (
                    "preference_macro_f1",
                    "abstain_recall",
                    "status_macro_f1",
                    "exact_array_accuracy",
                    "selected_set_exact_accuracy",
                )
            },
            "oof_calibration": selected["calibration"][
                "selected_metrics"
            ],
            "listwise_objective": selected[
                "listwise_objective"
            ],
            "trainable_parameters": int(
                selected["trainable_parameters"]
            ),
        }
    qualified = [
        receipt
        for receipt in selected_receipts
        if _qualified(receipt, contract)
    ]
    primary = min(qualified, key=_rank) if qualified else None
    selected_keys = {
        (str(row["candidate_id"]), int(row["seed"]))
        for row in selected_receipts
    }
    pair_predictions = _read_jsonl(pair_predictions_path)
    array_predictions = _read_jsonl(array_predictions_path)
    selected_pair_predictions = _selected_rows(
        pair_predictions,
        selected_keys,
        id_key="example_id",
    )
    selected_array_predictions = _selected_rows(
        array_predictions,
        selected_keys,
        id_key="array_id",
    )
    if (
        len(selected_pair_predictions) * len(expected_seeds)
        != len(pair_predictions)
        or len(selected_array_predictions) * len(expected_seeds)
        != len(array_predictions)
    ):
        raise ValueError(
            "selected prediction groups are incomplete or unexpected"
        )

    out_dir.mkdir(parents=True, exist_ok=False)
    selected_pair_path = (
        out_dir / "heldout_pair_predictions.selected.jsonl"
    )
    selected_array_path = (
        out_dir / "heldout_array_predictions.selected.jsonl"
    )
    _write_jsonl(selected_pair_path, selected_pair_predictions)
    _write_jsonl(selected_array_path, selected_array_predictions)
    selection = {
        "schema_version": (
            f"ldt-listwise-array-selection.v{major_version}"
        ),
        "experiment_id": preregistration["experiment_id"],
        "selection_source": (
            "development_cv_direct_array_metrics_and_oof_pair_"
            "calibration_only"
        ),
        "selection_frozen_before_confirmation_join": True,
        "confirmation_labels_read": False,
        "candidate_selection": candidate_selection,
        "primary_candidate_id": (
            None
            if primary is None
            else str(primary["candidate_id"])
        ),
        "primary_seed": (
            None if primary is None else int(primary["seed"])
        ),
        "no_qualified_primary": primary is None,
        "confirmation_join_authorized": primary is not None,
        "ranking_rule": list(contract["ranking"]),
        "source_trainer_summary_sha256": sha256_file(
            trainer_summary_path
        ),
        "source_pair_predictions_sha256": sha256_file(
            pair_predictions_path
        ),
        "source_array_predictions_sha256": sha256_file(
            array_predictions_path
        ),
        "selected_pair_predictions_sha256": sha256_file(
            selected_pair_path
        ),
        "selected_array_predictions_sha256": sha256_file(
            selected_array_path
        ),
    }
    selected_summary = {
        **trainer,
        "candidate_receipts": selected_receipts,
        "predictions_sha256": selection[
            "selected_pair_predictions_sha256"
        ],
        "array_predictions_sha256": selection[
            "selected_array_predictions_sha256"
        ],
        "development_selection": selection,
        "source_trainer_summary_sha256": selection[
            "source_trainer_summary_sha256"
        ],
    }
    selected_summary_path = out_dir / "trainer_summary.selected.json"
    dump_structured(selected_summary_path, selected_summary)
    selection["selected_trainer_summary_sha256"] = sha256_file(
        selected_summary_path
    )
    dump_structured(out_dir / "selection_receipt.json", selection)
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze direct-array seeds and condition confirmation"
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--trainer-summary", type=Path, required=True)
    parser.add_argument(
        "--pair-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--array-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = select(
        preregistration_path=arguments.preregistration.resolve(),
        trainer_summary_path=arguments.trainer_summary.resolve(),
        pair_predictions_path=(
            arguments.pair_predictions.resolve()
        ),
        array_predictions_path=(
            arguments.array_predictions.resolve()
        ),
        out_dir=arguments.out.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
