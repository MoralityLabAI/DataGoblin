#!/usr/bin/env python3
"""Fit v2 uncertain pair sorters without access to steward targets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_hard_surface import (
    HARD_INVERT_PREFERENCE,
    HARD_PREFERENCES,
    HARD_RATIONALES,
    HardPairEncoder,
    interval_taste_policy,
    load_hard_surface_dataset,
)
from datagoblin.experiments.ldt_array_sorter import (
    classification_metrics,
    encode_examples,
    make_model,
    predict_model,
    sha256_file,
    sha256_json,
    train_array_model,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _event(path: Path, event: str, **fields: Any) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _labeled_metrics(
    model: Any,
    encoder: HardPairEncoder,
    rows: Sequence[Mapping[str, Any]],
    *,
    view: str,
) -> dict[str, Any]:
    predictions = [
        predict_model(
            model,
            encoder,
            row,
            view=view,
            preference_labels=HARD_PREFERENCES,
            rationale_labels=HARD_RATIONALES,
        )
        for row in rows
    ]
    actual_preference = [str(row["target"]["preference"]) for row in rows]
    actual_rationale = [str(row["target"]["rationale_code"]) for row in rows]
    predicted_preference = [str(row["preference"]) for row in predictions]
    predicted_rationale = [str(row["rationale_code"]) for row in predictions]
    preference = classification_metrics(
        actual_preference, predicted_preference, HARD_PREFERENCES
    )
    rationale = classification_metrics(
        actual_rationale, predicted_rationale, HARD_RATIONALES
    )
    safety_indices = [
        index for index, value in enumerate(actual_rationale) if value == "safety_gate"
    ]
    abstain_support = sum(value == "ABSTAIN" for value in actual_preference)
    return {
        "preference_accuracy": preference["accuracy"],
        "preference_macro_f1": preference["macro_f1"],
        "rationale_accuracy": rationale["accuracy"],
        "rationale_macro_f1": rationale["macro_f1"],
        "joint_exact_accuracy": sum(
            actual_preference[index] == predicted_preference[index]
            and actual_rationale[index] == predicted_rationale[index]
            for index in range(len(rows))
        )
        / len(rows),
        "safety_gate_accuracy": sum(
            actual_preference[index] == predicted_preference[index]
            for index in safety_indices
        )
        / len(safety_indices),
        "abstain_recall": (
            preference["per_class"]["ABSTAIN"]["recall"]
            if abstain_support
            else None
        ),
        "tradeoff_recall": preference["per_class"]["TRADEOFF"]["recall"],
        "swap_consistency": sum(bool(row["swap_consistent"]) for row in predictions)
        / len(rows),
    }


def _mean_metrics(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in sorted(reports[0]):
        values = [
            float(report[name])
            for report in reports
            if report[name] is not None
        ]
        result[name] = sum(values) / len(values) if values else None
    return result


def _checkpoint_writer(
    root: Path,
    events_path: Path,
    *,
    candidate_id: str,
    seed: int,
    stage: str,
) -> tuple[list[str], Any]:
    paths: list[str] = []

    def write(epoch: int, state: Mapping[str, Any]) -> None:
        path = root / candidate_id / f"seed-{seed}" / stage / f"epoch-{epoch:04d}.json"
        _write_json(
            path,
            {
                "schema_version": "ldt-array-hard-surface-checkpoint.v2",
                "candidate_id": candidate_id,
                "seed": seed,
                "stage": stage,
                "epoch": epoch,
                "state": state,
            },
        )
        paths.append(str(path))
        _event(
            events_path,
            "checkpoint",
            candidate_id=candidate_id,
            seed=seed,
            stage=stage,
            epoch=epoch,
            checkpoint_sha256=sha256_file(path),
        )

    return paths, write


def _control_prediction(
    row: Mapping[str, Any],
    *,
    control_id: str,
    majority_preference: str,
    majority_rationale: str,
    preference_frequencies: Mapping[str, float],
    tolerance: float,
) -> dict[str, Any]:
    if control_id == "majority-control-v2":
        preference = majority_preference
        rationale = majority_rationale
        probabilities = {
            label: float(preference_frequencies.get(label, 0.0))
            for label in HARD_PREFERENCES
        }
        swapped_preference = preference
    else:
        preference, rationale = interval_taste_policy(
            row["input"], tolerance=tolerance
        )
        swapped_preference, _ = interval_taste_policy(
            {"left": row["input"]["right"], "right": row["input"]["left"]},
            tolerance=tolerance,
        )
        if control_id == "forced-tradeoff-control-v2":
            if preference == "ABSTAIN":
                preference = "TRADEOFF"
                rationale = "irreducible_tradeoff"
            if swapped_preference == "ABSTAIN":
                swapped_preference = "TRADEOFF"
        elif control_id != "frozen-interval-policy-v2":
            raise ValueError(f"unknown v2 control: {control_id}")
        probabilities = {
            label: float(label == preference) for label in HARD_PREFERENCES
        }
    return {
        "preference": preference,
        "rationale_code": rationale,
        "confidence": probabilities[preference],
        "preference_probabilities": probabilities,
        "swapped_preference": swapped_preference,
        "swap_consistent": (
            swapped_preference == HARD_INVERT_PREFERENCE[preference]
        ),
        "loop_delta": 0.0,
    }


def _prediction_envelope(
    *,
    candidate_id: str,
    seed: int | None,
    row: Mapping[str, Any],
    prediction: Mapping[str, Any],
    dataset_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "ldt-array-hard-surface-prediction.v2",
        "candidate_id": candidate_id,
        "seed": seed,
        "example_id": row["example_id"],
        "array_id": row["array_id"],
        "left_index": row["left_index"],
        "right_index": row["right_index"],
        "lineage_group_sha256": row["lineage_group_sha256"],
        "input_sha256": sha256_json(row["input"]),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "preference": prediction["preference"],
        "rationale_code": prediction["rationale_code"],
        "confidence": prediction["confidence"],
        "preference_probabilities": prediction["preference_probabilities"],
        "swapped_preference": prediction["swapped_preference"],
        "swap_consistent": prediction["swap_consistent"],
        "loop_delta": prediction["loop_delta"],
        "authority": "MODEL_RECOMMENDATION_ONLY",
    }


def train(args: argparse.Namespace) -> int:
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "trainer_events.jsonl"
    events_path.touch(exist_ok=True)
    summary_path = out_dir / "trainer_summary.json"
    started = time.monotonic()
    steps_completed = 0
    checkpoints: list[str] = []
    try:
        plan = json.loads(args.trainer_plan.read_text(encoding="utf-8"))
        if plan["schema_version"] != "ldt-array-hard-surface-trainer-plan.v2":
            raise ValueError("unsupported v2 trainer plan")
        if plan["heldout_pair_label_access"] or plan["heldout_array_label_access"]:
            raise ValueError("v2 trainer plan grants forbidden heldout-label access")
        if plan["external_checkpoint"] is not None or plan["network"]:
            raise ValueError("v2 array gym must stay local and checkpoint-free")
        if os.environ.get("DATAGOBLIN_HARD_CAPS_VERIFIED") != "1":
            raise RuntimeError("hard-cap wrapper verification receipt is missing")
        _, train_rows, heldout_rows, _, _ = load_hard_surface_dataset(
            args.dataset_manifest.resolve()
        )
        manifest_sha256 = sha256_file(args.dataset_manifest.resolve())
        if manifest_sha256 != plan["dataset_manifest_sha256"]:
            raise ValueError("v2 trainer plan is bound to another dataset")
        encoder = HardPairEncoder.fit(
            train_rows,
            tolerance=float(plan["metric_equivalence_tolerance"]),
        )
        training = plan["training"]
        lineages = sorted(
            {str(row["lineage_group_sha256"]) for row in train_rows}
        )
        _event(
            events_path,
            "trainer_start",
            experiment_id=plan["experiment_id"],
            train_pairs=len(train_rows),
            heldout_pairs=len(heldout_rows),
            heldout_pair_labels_available=False,
            heldout_array_labels_available=False,
            candidate_count=len(plan["candidate_models"]),
            seeds=training["seeds"],
        )
        prediction_rows: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        train_preferences = Counter(
            str(row["target"]["preference"]) for row in train_rows
        )
        train_rationales = Counter(
            str(row["target"]["rationale_code"]) for row in train_rows
        )
        majority_preference = max(
            HARD_PREFERENCES,
            key=lambda label: (train_preferences[label], label),
        )
        majority_rationale = max(
            HARD_RATIONALES,
            key=lambda label: (train_rationales[label], label),
        )
        frequencies = {
            label: train_preferences[label] / len(train_rows)
            for label in HARD_PREFERENCES
        }
        for control in plan["controls"]:
            control_id = str(control["model_id"])
            for row in heldout_rows:
                prediction_rows.append(
                    _prediction_envelope(
                        candidate_id=control_id,
                        seed=None,
                        row=row,
                        prediction=_control_prediction(
                            row,
                            control_id=control_id,
                            majority_preference=majority_preference,
                            majority_rationale=majority_rationale,
                            preference_frequencies=frequencies,
                            tolerance=encoder.tolerance,
                        ),
                        dataset_manifest_sha256=manifest_sha256,
                    )
                )
            receipts.append(
                {
                    "candidate_id": control_id,
                    "family": control["family"],
                    "seed": None,
                    "trainable_parameters": 0,
                    "trained": False,
                    "heldout_targets_seen": False,
                }
            )
        for spec in plan["candidate_models"]:
            candidate_id = str(spec["model_id"])
            view = str(spec["feature_view"])
            input_width = len(encoder.feature_names(view))
            for seed_value in training["seeds"]:
                seed = int(seed_value)
                cv_reports: list[dict[str, Any]] = []
                seed_checkpoints: list[str] = []
                for fold_index, validation_lineage in enumerate(lineages):
                    fold_train = [
                        row
                        for row in train_rows
                        if row["lineage_group_sha256"] != validation_lineage
                    ]
                    fold_validation = [
                        row
                        for row in train_rows
                        if row["lineage_group_sha256"] == validation_lineage
                    ]
                    fold_encoder = HardPairEncoder.fit(
                        fold_train,
                        tolerance=float(plan["metric_equivalence_tolerance"]),
                    )
                    examples = encode_examples(
                        fold_train,
                        fold_encoder,
                        view=view,
                        augment_swaps=False,
                        preference_labels=HARD_PREFERENCES,
                        rationale_labels=HARD_RATIONALES,
                    )
                    model_seed = seed * 100 + fold_index
                    model = make_model(
                        spec,
                        len(fold_encoder.feature_names(view)),
                        model_seed,
                        preference_count=len(HARD_PREFERENCES),
                        rationale_count=len(HARD_RATIONALES),
                    )
                    fold_paths, callback = _checkpoint_writer(
                        out_dir / "checkpoints",
                        events_path,
                        candidate_id=candidate_id,
                        seed=seed,
                        stage=f"cv-fold-{fold_index + 1}",
                    )
                    steps_completed += train_array_model(
                        model,
                        examples,
                        epochs=int(spec["epochs"]),
                        batch_size=int(training["batch_size"]),
                        learning_rate=float(spec["learning_rate"]),
                        l2=float(training["l2"]),
                        gradient_clip=float(training["gradient_norm_clip"]),
                        rationale_loss_weight=float(
                            training["rationale_loss_weight"]
                        ),
                        seed=model_seed,
                        checkpoint_every_epochs=int(
                            training["checkpoint_interval"]["epochs"]
                        ),
                        checkpoint_every_seconds=float(
                            training["checkpoint_interval"]["seconds"]
                        ),
                        checkpoint_callback=callback,
                        preference_count=len(HARD_PREFERENCES),
                        rationale_count=len(HARD_RATIONALES),
                    )
                    seed_checkpoints.extend(fold_paths)
                    cv_reports.append(
                        _labeled_metrics(
                            model, fold_encoder, fold_validation, view=view
                        )
                    )
                    del model
                    gc.collect()
                final_examples = encode_examples(
                    train_rows,
                    encoder,
                    view=view,
                    augment_swaps=False,
                    preference_labels=HARD_PREFERENCES,
                    rationale_labels=HARD_RATIONALES,
                )
                final_seed = seed * 100 + 97
                final_model = make_model(
                    spec,
                    input_width,
                    final_seed,
                    preference_count=len(HARD_PREFERENCES),
                    rationale_count=len(HARD_RATIONALES),
                )
                final_paths, final_callback = _checkpoint_writer(
                    out_dir / "checkpoints",
                    events_path,
                    candidate_id=candidate_id,
                    seed=seed,
                    stage="final-train",
                )
                steps_completed += train_array_model(
                    final_model,
                    final_examples,
                    epochs=int(spec["epochs"]),
                    batch_size=int(training["batch_size"]),
                    learning_rate=float(spec["learning_rate"]),
                    l2=float(training["l2"]),
                    gradient_clip=float(training["gradient_norm_clip"]),
                    rationale_loss_weight=float(
                        training["rationale_loss_weight"]
                    ),
                    seed=final_seed,
                    checkpoint_every_epochs=int(
                        training["checkpoint_interval"]["epochs"]
                    ),
                    checkpoint_every_seconds=float(
                        training["checkpoint_interval"]["seconds"]
                    ),
                    checkpoint_callback=final_callback,
                    preference_count=len(HARD_PREFERENCES),
                    rationale_count=len(HARD_RATIONALES),
                )
                seed_checkpoints.extend(final_paths)
                model_payload = {
                    "schema_version": "ldt-array-hard-surface-model.v2",
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "family": spec["family"],
                    "feature_view": view,
                    "feature_names": list(encoder.feature_names(view)),
                    "encoder": encoder.receipt(),
                    "preference_labels": list(HARD_PREFERENCES),
                    "rationale_labels": list(HARD_RATIONALES),
                    "trainable_parameters": final_model.parameter_count,
                    "state": final_model.state_dict(),
                    "authority": "MODEL_RECOMMENDATION_ONLY",
                    "heldout_targets_seen": False,
                }
                model_path = (
                    out_dir / "models" / f"{candidate_id}-seed-{seed}.json"
                )
                _write_json(model_path, model_payload)
                for row in heldout_rows:
                    prediction_rows.append(
                        _prediction_envelope(
                            candidate_id=candidate_id,
                            seed=seed,
                            row=row,
                            prediction=predict_model(
                                final_model,
                                encoder,
                                row,
                                view=view,
                                preference_labels=HARD_PREFERENCES,
                                rationale_labels=HARD_RATIONALES,
                            ),
                            dataset_manifest_sha256=manifest_sha256,
                        )
                    )
                receipt = {
                    "candidate_id": candidate_id,
                    "family": spec["family"],
                    "feature_view": view,
                    "hidden_width": int(spec["hidden_width"]),
                    "loop_count": int(spec["loop_count"]),
                    "seed": seed,
                    "trainable_parameters": final_model.parameter_count,
                    "trained": True,
                    "train_pairs": len(final_examples),
                    "validation": training["validation"],
                    "cv_fold_metrics": cv_reports,
                    "cv_mean_metrics": _mean_metrics(cv_reports),
                    "model_sha256": sha256_file(model_path),
                    "checkpoint_count": len(seed_checkpoints),
                    "heldout_targets_seen": False,
                }
                receipts.append(receipt)
                checkpoints.extend(seed_checkpoints)
                _event(
                    events_path,
                    "candidate_complete",
                    candidate_id=candidate_id,
                    seed=seed,
                    cv_preference_macro_f1=receipt["cv_mean_metrics"][
                        "preference_macro_f1"
                    ],
                    cv_abstain_recall=receipt["cv_mean_metrics"][
                        "abstain_recall"
                    ],
                    model_sha256=receipt["model_sha256"],
                )
                del final_model
                gc.collect()
        predictions_path = out_dir / "heldout_pair_predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in sorted(
                prediction_rows,
                key=lambda value: (
                    str(value["candidate_id"]),
                    -1 if value["seed"] is None else int(value["seed"]),
                    str(value["example_id"]),
                ),
            ):
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        summary = {
            "schema_version": "ldt-array-hard-surface-trainer-summary.v2",
            "experiment_id": plan["experiment_id"],
            "training_task_id": plan["training_task_id"],
            "status": "completed",
            "abort_reason": None,
            "hard_caps_verified": True,
            "dataset_manifest_sha256": manifest_sha256,
            "preregistration_sha256": plan["preregistration_sha256"],
            "encoder": encoder.receipt(),
            "candidate_receipts": receipts,
            "steps_completed": steps_completed,
            "checkpoints": checkpoints,
            "predictions_sha256": sha256_file(predictions_path),
            "heldout_pair_targets_seen": False,
            "heldout_array_targets_seen": False,
            "external_checkpoint_loaded": False,
            "elapsed_seconds": time.monotonic() - started,
            "claim_boundary": (
                "Synthetic uncertain meta-selection only; all outputs are "
                "recommendation-only and no game record was trained."
            ),
        }
        _write_json(summary_path, summary)
        _event(
            events_path,
            "trainer_complete",
            steps_completed=steps_completed,
            predictions_sha256=summary["predictions_sha256"],
        )
        return 0
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        _write_json(
            summary_path,
            {
                "schema_version": "ldt-array-hard-surface-trainer-summary.v2",
                "status": "aborted",
                "abort_reason": reason,
                "steps_completed": steps_completed,
                "checkpoints": checkpoints,
                "heldout_pair_targets_seen": False,
                "heldout_array_targets_seen": False,
                "external_checkpoint_loaded": False,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        _event(
            events_path,
            "trainer_abort",
            reason=reason,
            steps_completed=steps_completed,
        )
        return 1
    finally:
        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train bounded LDT arrays on the v2 hard surface"
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--trainer-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    sys.exit(train(build_parser().parse_args()))
