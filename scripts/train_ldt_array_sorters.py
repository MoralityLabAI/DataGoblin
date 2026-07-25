#!/usr/bin/env python3
"""Fit the preregistered public-only LDT array sorter sweep.

This process must run inside the hard-cap wrapper. It never accepts a heldout-label
path and emits recommendation-only predictions for independent steward scoring.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_sorter import (
    PREFERENCES,
    RATIONALE_CODES,
    PairEncoder,
    classification_metrics,
    control_prediction,
    encode_examples,
    load_public_dataset,
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
    encoder: PairEncoder,
    rows: Sequence[Mapping[str, Any]],
    *,
    view: str,
) -> dict[str, Any]:
    predictions = [predict_model(model, encoder, row, view=view) for row in rows]
    actual_preference = [str(row["target"]["preference"]) for row in rows]
    actual_rationale = [str(row["target"]["rationale_code"]) for row in rows]
    predicted_preference = [str(row["preference"]) for row in predictions]
    predicted_rationale = [str(row["rationale_code"]) for row in predictions]
    preference = classification_metrics(actual_preference, predicted_preference, PREFERENCES)
    rationale = classification_metrics(actual_rationale, predicted_rationale, RATIONALE_CODES)
    safety_indices = [
        index for index, value in enumerate(actual_rationale) if value == "safety_gate"
    ]
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
        "safety_gate_accuracy": (
            sum(actual_preference[index] == predicted_preference[index] for index in safety_indices)
            / len(safety_indices)
            if safety_indices
            else None
        ),
        "tradeoff_recall": preference["per_class"]["TRADEOFF"]["recall"],
        "swap_consistency": sum(bool(row["swap_consistent"]) for row in predictions) / len(rows),
    }


def _mean_metrics(reports: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    names = sorted({name for report in reports for name in report})
    result: dict[str, float | None] = {}
    for name in names:
        values = [float(report[name]) for report in reports if report[name] is not None]
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
        payload = {
            "schema_version": "ldt-array-checkpoint.v1",
            "candidate_id": candidate_id,
            "seed": seed,
            "stage": stage,
            "epoch": epoch,
            "state": state,
        }
        _write_json(path, payload)
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


def _prediction_envelope(
    *,
    candidate_id: str,
    seed: int | None,
    row: Mapping[str, Any],
    prediction: Mapping[str, Any],
    dataset_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "ldt-array-prediction.v1",
        "candidate_id": candidate_id,
        "seed": seed,
        "example_id": row["example_id"],
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
    checkpoints_root = out_dir / "checkpoints"
    models_dir = out_dir / "models"
    started = time.monotonic()
    steps_completed = 0
    checkpoints: list[str] = []
    status = "aborted"
    abort_reason: str | None = None
    try:
        plan = json.loads(args.trainer_plan.read_text(encoding="utf-8"))
        if plan["schema_version"] != "ldt-array-trainer-plan.v1":
            raise ValueError("unsupported trainer plan")
        if bool(plan["heldout_label_access"]):
            raise ValueError("trainer plan grants forbidden heldout-label access")
        if plan["external_checkpoint"] is not None or bool(plan["network"]):
            raise ValueError("array gym must remain local and weight-free")
        if os.environ.get("DATAGOBLIN_HARD_CAPS_VERIFIED") != "1":
            raise RuntimeError("hard-cap wrapper verification receipt is missing")
        _, train_rows, heldout_rows = load_public_dataset(args.dataset_manifest.resolve())
        manifest_sha256 = sha256_file(args.dataset_manifest.resolve())
        if manifest_sha256 != plan["dataset_manifest_sha256"]:
            raise ValueError("trainer plan is bound to a different dataset manifest")
        encoder = PairEncoder.fit(
            train_rows,
            tolerance=float(plan["array_representation"]["metric_equivalence_tolerance"]),
        )
        training = plan["training"]
        batch_size = int(training["batch_size"])
        checkpoint_epochs = int(training["checkpoint_interval"]["epochs"])
        checkpoint_seconds = float(training["checkpoint_interval"]["seconds"])
        lineages = sorted({str(row["lineage_group_sha256"]) for row in train_rows})
        if len(lineages) < 2:
            raise ValueError("leave-one-lineage-out validation needs at least two training lineages")
        _event(
            events_path,
            "trainer_start",
            experiment_id=plan["experiment_id"],
            train_examples=len(train_rows),
            heldout_inputs=len(heldout_rows),
            heldout_labels_available=False,
            candidate_count=len(plan["candidate_models"]),
            seeds=plan["training"]["seeds"],
        )
        prediction_rows: list[dict[str, Any]] = []
        candidate_receipts: list[dict[str, Any]] = []
        train_preferences = Counter(str(row["target"]["preference"]) for row in train_rows)
        train_rationales = Counter(str(row["target"]["rationale_code"]) for row in train_rows)
        majority_preference = max(PREFERENCES, key=lambda label: (train_preferences[label], label))
        majority_rationale = max(
            RATIONALE_CODES, key=lambda label: (train_rationales[label], label)
        )
        preference_frequencies = {
            label: train_preferences[label] / len(train_rows) for label in PREFERENCES
        }
        for control in plan["controls"]:
            control_id = str(control["model_id"])
            for row in heldout_rows:
                prediction = control_prediction(
                    row,
                    control_id=control_id,
                    majority_preference=majority_preference,
                    majority_rationale=majority_rationale,
                    preference_frequencies=preference_frequencies,
                )
                prediction_rows.append(
                    _prediction_envelope(
                        candidate_id=control_id,
                        seed=None,
                        row=row,
                        prediction=prediction,
                        dataset_manifest_sha256=manifest_sha256,
                    )
                )
            candidate_receipts.append(
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
            for seed_value in plan["training"]["seeds"]:
                seed = int(seed_value)
                cv_reports = []
                seed_checkpoints: list[str] = []
                for fold_index, heldout_lineage in enumerate(lineages):
                    fold_train = [
                        row
                        for row in train_rows
                        if str(row["lineage_group_sha256"]) != heldout_lineage
                    ]
                    fold_validation = [
                        row
                        for row in train_rows
                        if str(row["lineage_group_sha256"]) == heldout_lineage
                    ]
                    fold_examples = encode_examples(
                        fold_train, encoder, view=view, augment_swaps=True
                    )
                    model_seed = seed * 100 + fold_index
                    model = make_model(spec, input_width, model_seed)
                    fold_paths, callback = _checkpoint_writer(
                        checkpoints_root,
                        events_path,
                        candidate_id=candidate_id,
                        seed=seed,
                        stage=f"cv-fold-{fold_index + 1}",
                    )
                    steps_completed += train_array_model(
                        model,
                        fold_examples,
                        epochs=int(spec["epochs"]),
                        batch_size=batch_size,
                        learning_rate=float(spec["learning_rate"]),
                        l2=float(training["l2"]),
                        gradient_clip=float(training["gradient_norm_clip"]),
                        rationale_loss_weight=float(training["rationale_loss_weight"]),
                        seed=model_seed,
                        checkpoint_every_epochs=checkpoint_epochs,
                        checkpoint_every_seconds=checkpoint_seconds,
                        checkpoint_callback=callback,
                    )
                    seed_checkpoints.extend(fold_paths)
                    cv_reports.append(
                        _labeled_metrics(model, encoder, fold_validation, view=view)
                    )
                    del model
                    gc.collect()
                final_examples = encode_examples(
                    train_rows, encoder, view=view, augment_swaps=True
                )
                final_seed = seed * 100 + 97
                final_model = make_model(spec, input_width, final_seed)
                final_paths, final_callback = _checkpoint_writer(
                    checkpoints_root,
                    events_path,
                    candidate_id=candidate_id,
                    seed=seed,
                    stage="final-train",
                )
                steps_completed += train_array_model(
                    final_model,
                    final_examples,
                    epochs=int(spec["epochs"]),
                    batch_size=batch_size,
                    learning_rate=float(spec["learning_rate"]),
                    l2=float(training["l2"]),
                    gradient_clip=float(training["gradient_norm_clip"]),
                    rationale_loss_weight=float(training["rationale_loss_weight"]),
                    seed=final_seed,
                    checkpoint_every_epochs=checkpoint_epochs,
                    checkpoint_every_seconds=checkpoint_seconds,
                    checkpoint_callback=final_callback,
                )
                seed_checkpoints.extend(final_paths)
                model_payload = {
                    "schema_version": "ldt-array-model.v1",
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "family": spec["family"],
                    "feature_view": view,
                    "feature_names": list(encoder.feature_names(view)),
                    "encoder": encoder.receipt(),
                    "trainable_parameters": final_model.parameter_count,
                    "state": final_model.state_dict(),
                    "authority": "MODEL_RECOMMENDATION_ONLY",
                    "heldout_targets_seen": False,
                    "external_checkpoint_loaded": False,
                }
                model_path = models_dir / f"{candidate_id}-seed-{seed}.json"
                _write_json(model_path, model_payload)
                for row in heldout_rows:
                    prediction_rows.append(
                        _prediction_envelope(
                            candidate_id=candidate_id,
                            seed=seed,
                            row=row,
                            prediction=predict_model(final_model, encoder, row, view=view),
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
                    "train_examples_before_swap_augmentation": len(train_rows),
                    "train_examples_after_swap_augmentation": len(final_examples),
                    "validation": "leave_one_training_lineage_out",
                    "cv_fold_metrics": cv_reports,
                    "cv_mean_metrics": _mean_metrics(cv_reports),
                    "model_sha256": sha256_file(model_path),
                    "checkpoint_count": len(seed_checkpoints),
                    "heldout_targets_seen": False,
                }
                candidate_receipts.append(receipt)
                checkpoints.extend(seed_checkpoints)
                _event(
                    events_path,
                    "candidate_complete",
                    candidate_id=candidate_id,
                    seed=seed,
                    trainable_parameters=final_model.parameter_count,
                    cv_preference_macro_f1=receipt["cv_mean_metrics"]["preference_macro_f1"],
                    model_sha256=receipt["model_sha256"],
                )
                del final_model
                gc.collect()
        predictions_path = out_dir / "heldout_predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in sorted(
                prediction_rows,
                key=lambda value: (
                    str(value["candidate_id"]),
                    -1 if value["seed"] is None else int(value["seed"]),
                    str(value["example_id"]),
                ),
            ):
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        status = "completed"
        summary = {
            "schema_version": "ldt-array-trainer-summary.v1",
            "experiment_id": plan["experiment_id"],
            "training_task_id": plan["training_task_id"],
            "status": status,
            "abort_reason": None,
            "hard_caps_verified": True,
            "dataset_manifest_sha256": manifest_sha256,
            "preregistration_sha256": plan["preregistration_sha256"],
            "encoder": encoder.receipt(),
            "candidate_receipts": candidate_receipts,
            "steps_completed": steps_completed,
            "checkpoints": checkpoints,
            "predictions_sha256": sha256_file(predictions_path),
            "heldout_targets_seen": False,
            "external_checkpoint_loaded": False,
            "elapsed_seconds": time.monotonic() - started,
            "claim_boundary": (
                "Synthetic tournament meta-selection evidence only; all outputs are "
                "recommendations and no game record or larger checkpoint was trained."
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
        abort_reason = f"{type(exc).__name__}: {exc}"
        _write_json(
            summary_path,
            {
                "schema_version": "ldt-array-trainer-summary.v1",
                "status": status,
                "abort_reason": abort_reason,
                "steps_completed": steps_completed,
                "checkpoints": checkpoints,
                "heldout_targets_seen": False,
                "external_checkpoint_loaded": False,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        _event(events_path, "trainer_abort", reason=abort_reason, steps_completed=steps_completed)
        return 1
    finally:
        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train bounded LDT array sorter candidates")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--trainer-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    random.seed(0)
    sys.exit(train(build_parser().parse_args()))
