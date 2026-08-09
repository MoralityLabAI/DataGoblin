#!/usr/bin/env python3
"""Fit v3 confirmation sorters without access to steward targets."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_confirmation import (
    ConfirmationPairEncoder,
    load_confirmation_dataset,
    partial_interval_taste_policy,
    reduce_confirmation_array,
)
from datagoblin.experiments.ldt_array_hard_surface import (
    ARRAY_STATUSES,
    HARD_INVERT_PREFERENCE,
    HARD_PREFERENCES,
    HARD_RATIONALES,
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
from datagoblin.experiments.ldt_cardinality_decoder_v11 import (
    CardinalityLoopedStatusModel,
    predict_cardinality_statuses,
)
from datagoblin.experiments.ldt_edge_set_array_v10 import (
    EdgeDifferenceCandidateEncoder,
    SetCoupledLoopedStatusModel,
    train_edge_status_model,
)
from datagoblin.experiments.ldt_exact_array_likelihood_v12 import (
    ExactArrayLikelihoodModel,
    predict_exact_array_statuses,
)
from datagoblin.experiments.ldt_listwise_array_v9 import (
    IndependentCandidateEncoder,
    LoopedSetStatusModel,
    predict_array_statuses,
    train_status_model,
)
from datagoblin.experiments.ldt_set_metric_robustness_v10 import (
    EXACT_SELECTED_SET_ACCURACY,
    EXACT_STATUS_VECTOR_ACCURACY,
    score_exact_set_metrics,
)
from datagoblin.experiments.ldt_status_composition_v13 import (
    ExactCompositionStatusModel,
    predict_composition_statuses,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _event(path: Path, event: str, **fields: Any) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _temperature_scale(
    probabilities: Mapping[str, float],
    temperature: float,
) -> dict[str, float]:
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("calibration temperature must be finite and positive")
    logits = {
        label: math.log(max(float(value), 1e-12)) / temperature
        for label, value in probabilities.items()
    }
    maximum = max(logits.values())
    exponentials = {
        label: math.exp(value - maximum) for label, value in logits.items()
    }
    total = sum(exponentials.values())
    return {label: value / total for label, value in exponentials.items()}


def _calibration_metrics(
    examples: Sequence[tuple[Mapping[str, float], str]],
    *,
    temperature: float,
    bins: int,
) -> dict[str, float]:
    scaled = [
        (_temperature_scale(probabilities, temperature), actual)
        for probabilities, actual in examples
    ]
    negative_log_likelihood = sum(
        -math.log(max(probabilities[actual], 1e-12))
        for probabilities, actual in scaled
    ) / len(scaled)
    brier = sum(
        sum(
            (
                float(probabilities[label])
                - float(label == actual)
            )
            ** 2
            for label in HARD_PREFERENCES
        )
        for probabilities, actual in scaled
    ) / len(scaled)
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for probabilities, actual in scaled:
        predicted = max(
            HARD_PREFERENCES,
            key=lambda label: probabilities[label],
        )
        confidence = float(probabilities[predicted])
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((confidence, float(predicted == actual)))
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        confidence = sum(value[0] for value in bucket) / len(bucket)
        accuracy = sum(value[1] for value in bucket) / len(bucket)
        ece += len(bucket) / len(scaled) * abs(accuracy - confidence)
    return {
        "multiclass_negative_log_likelihood": negative_log_likelihood,
        "multiclass_brier": brier,
        "expected_calibration_error": ece,
    }


def _fit_temperature(
    examples: Sequence[tuple[Mapping[str, float], str]],
    *,
    temperatures: Sequence[float],
    bins: int,
) -> dict[str, Any]:
    reports = [
        {
            "temperature": float(temperature),
            **_calibration_metrics(
                examples,
                temperature=float(temperature),
                bins=bins,
            ),
        }
        for temperature in temperatures
    ]
    selected = min(
        reports,
        key=lambda report: (
            report["multiclass_negative_log_likelihood"],
            abs(math.log(report["temperature"])),
        ),
    )
    return {
        "method": "frozen_temperature_grid",
        "fit_split": "out_of_fold_development_predictions_only",
        "selected_temperature": selected["temperature"],
        "selected_metrics": {
            key: value
            for key, value in selected.items()
            if key != "temperature"
        },
        "grid_reports": reports,
    }


def _scaled_prediction(
    prediction: Mapping[str, Any],
    *,
    temperature: float,
) -> dict[str, Any]:
    probabilities = _temperature_scale(
        prediction["preference_probabilities"],
        temperature,
    )
    preference = max(
        HARD_PREFERENCES,
        key=lambda label: probabilities[label],
    )
    return {
        **prediction,
        "preference": preference,
        "confidence": probabilities[preference],
        "preference_probabilities": probabilities,
    }


def _labeled_metrics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    actual_preference = [str(row["target"]["preference"]) for row in rows]
    actual_rationale = [str(row["target"]["rationale_code"]) for row in rows]
    predicted_preference = [str(row["preference"]) for row in predictions]
    predicted_rationale = [str(row["rationale_code"]) for row in predictions]
    preference = classification_metrics(
        actual_preference,
        predicted_preference,
        HARD_PREFERENCES,
    )
    rationale = classification_metrics(
        actual_rationale,
        predicted_rationale,
        HARD_RATIONALES,
    )
    safety_indices = [
        index
        for index, value in enumerate(actual_rationale)
        if value == "safety_gate"
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
        "swap_consistency": sum(
            bool(row["swap_consistent"]) for row in predictions
        )
        / len(rows),
    }


def _array_labeled_metrics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, Any]],
    *,
    direct_statuses: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, float]:
    pair_predictions: dict[
        str,
        dict[tuple[int, int], str],
    ] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        pair_predictions.setdefault(str(row["array_id"]), {})[
            (int(row["left_index"]), int(row["right_index"]))
        ] = str(prediction["preference"])
    actual_statuses = []
    predicted_statuses = []
    exact_arrays = 0
    selected_sets = 0
    for array_row in arrays:
        array_id = str(array_row["array_id"])
        if direct_statuses is None:
            candidates = [
                candidate["features"]
                for candidate in array_row["candidates"]
            ]
            predicted = reduce_confirmation_array(
                candidates,
                pair_predictions[array_id],
            )["statuses"]
        else:
            predicted = list(direct_statuses[array_id])
        actual = [
            str(value) for value in array_row["target"]["statuses"]
        ]
        actual_statuses.extend(actual)
        predicted_statuses.extend(predicted)
        candidate_ids = [
            str(candidate["tournament_report_sha256"])
            for candidate in array_row["candidates"]
        ]
        robust_scores = score_exact_set_metrics(
            tuple(zip(candidate_ids, actual, strict=True)),
            tuple(zip(candidate_ids, predicted, strict=True)),
        )
        exact_arrays += robust_scores[EXACT_STATUS_VECTOR_ACCURACY]
        selected_sets += robust_scores[EXACT_SELECTED_SET_ACCURACY]
    status = classification_metrics(
        actual_statuses,
        predicted_statuses,
        ARRAY_STATUSES,
    )
    return {
        "status_macro_f1": float(status["macro_f1"]),
        "exact_array_accuracy": exact_arrays / len(arrays),
        "selected_set_exact_accuracy": selected_sets / len(arrays),
    }


def _paired_training_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    lookup = {
        (
            str(row["array_id"]),
            int(row["left_index"]),
            int(row["right_index"]),
        ): row
        for row in rows
    }
    if len(lookup) != len(rows):
        raise ValueError("paired training rows contain duplicate orientations")
    result = []
    for row in sorted(
        rows,
        key=lambda value: (
            str(value["array_id"]),
            int(value["left_index"]),
            int(value["right_index"]),
        ),
    ):
        left = int(row["left_index"])
        right = int(row["right_index"])
        if left >= right:
            continue
        reverse = lookup[(str(row["array_id"]), right, left)]
        if (
            str(reverse["target"]["preference"])
            != HARD_INVERT_PREFERENCE[
                str(row["target"]["preference"])
            ]
            or str(reverse["target"]["rationale_code"])
            != str(row["target"]["rationale_code"])
        ):
            raise ValueError("paired orientations disagree on target binding")
        result.extend((row, reverse))
    if len(result) != len(rows):
        raise ValueError("paired training rows are incomplete")
    return result


def _array_consistency_weights(
    rows: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, Any]],
    *,
    extra_weight: float,
) -> tuple[list[float], dict[str, float | int]]:
    if extra_weight < 0.0 or not math.isfinite(extra_weight):
        raise ValueError("array consistency weight must be non-negative")
    statuses = {
        str(row["array_id"]): [
            str(status) for status in row["target"]["statuses"]
        ]
        for row in arrays
    }
    weights = []
    critical = 0
    for row in rows:
        array_statuses = statuses[str(row["array_id"])]
        left_status = array_statuses[int(row["left_index"])]
        right_status = array_statuses[int(row["right_index"])]
        preference = str(row["target"]["preference"])
        rationale = str(row["target"]["rationale_code"])
        loser_status = (
            right_status
            if preference == "LEFT"
            else left_status
            if preference == "RIGHT"
            else None
        )
        structural = (
            (rationale == "safety_gate" and loser_status == "UNSAFE")
            or (
                rationale == "pareto_dominance"
                and loser_status == "DOMINATED"
            )
            or (
                preference == "ABSTAIN"
                and "UNRESOLVED" in {left_status, right_status}
            )
            or (
                preference == "TRADEOFF"
                and left_status == right_status == "SELECTED"
            )
        )
        critical += structural
        weights.append(1.0 + extra_weight * float(structural))
    return weights, {
        "example_count": len(rows),
        "structural_edge_count": critical,
        "base_weight": 1.0,
        "extra_structural_weight": extra_weight,
        "mean_weight": sum(weights) / len(weights),
        "maximum_weight": max(weights),
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
        path = (
            root
            / candidate_id
            / f"seed-{seed}"
            / stage
            / f"epoch-{epoch:04d}.json"
        )
        _write_json(
            path,
            {
                "schema_version": "ldt-array-confirmation-checkpoint.v3",
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
    if control_id == "majority-control-v3":
        preference = majority_preference
        rationale = majority_rationale
        probabilities = {
            label: float(preference_frequencies.get(label, 0.0))
            for label in HARD_PREFERENCES
        }
        swapped_preference = preference
    else:
        preference, rationale = partial_interval_taste_policy(
            row["input"],
            tolerance=tolerance,
        )
        swapped_preference, _ = partial_interval_taste_policy(
            {
                "left": row["input"]["right"],
                "right": row["input"]["left"],
            },
            tolerance=tolerance,
        )
        if control_id == "forced-tradeoff-control-v3":
            if preference == "ABSTAIN":
                preference = "TRADEOFF"
                rationale = "irreducible_tradeoff"
            if swapped_preference == "ABSTAIN":
                swapped_preference = "TRADEOFF"
        elif control_id != "frozen-partial-interval-policy-v3":
            raise ValueError(f"unknown confirmation control: {control_id}")
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
    calibration_temperature: float,
) -> dict[str, Any]:
    return {
        "schema_version": "ldt-array-confirmation-prediction.v3",
        "candidate_id": candidate_id,
        "seed": seed,
        "example_id": row["example_id"],
        "array_id": row["array_id"],
        "left_index": row["left_index"],
        "right_index": row["right_index"],
        "left_tournament_report_sha256": row[
            "left_tournament_report_sha256"
        ],
        "right_tournament_report_sha256": row[
            "right_tournament_report_sha256"
        ],
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
        "calibration_temperature": calibration_temperature,
        "authority": "MODEL_RECOMMENDATION_ONLY",
    }


def _array_prediction_envelope(
    *,
    candidate_id: str,
    seed: int,
    row: Mapping[str, Any],
    prediction: Mapping[str, Any],
    dataset_manifest_sha256: str,
    prediction_source: str,
) -> dict[str, Any]:
    return {
        "schema_version": "ldt-listwise-array-prediction.v9",
        "candidate_id": candidate_id,
        "seed": seed,
        "array_id": row["array_id"],
        "lineage_group_sha256": row["lineage_group_sha256"],
        "candidate_count": row["candidate_count"],
        "candidate_input_sha256": sha256_json(
            row["candidates"]
        ),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "prediction_source": prediction_source,
        "statuses": list(prediction["statuses"]),
        "status_probabilities": list(
            prediction["status_probabilities"]
        ),
        "authority": "MODEL_RECOMMENDATION_ONLY",
    }


def _reduced_status_predictions(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    pair_predictions: dict[
        str,
        dict[tuple[int, int], str],
    ] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        pair_predictions.setdefault(str(row["array_id"]), {})[
            (int(row["left_index"]), int(row["right_index"]))
        ] = str(prediction["preference"])
    result = {}
    for array_row in arrays:
        array_id = str(array_row["array_id"])
        candidates = [
            candidate["features"]
            for candidate in array_row["candidates"]
        ]
        statuses = reduce_confirmation_array(
            candidates,
            pair_predictions[array_id],
        )["statuses"]
        result[array_id] = {
            "statuses": statuses,
            "status_probabilities": [
                {
                    label: float(label == status)
                    for label in ARRAY_STATUSES
                }
                for status in statuses
            ],
        }
    return result


def _candidate_encoder(listwise_spec: Mapping[str, Any]) -> Any:
    family = str(listwise_spec.get("family"))
    encoder_name = str(listwise_spec.get("candidate_encoder", ""))
    if encoder_name == "edge_difference_message" or (
        not encoder_name
        and family == "edge_difference_permutation_equivariant_status"
    ):
        return EdgeDifferenceCandidateEncoder()
    if encoder_name in {"", "independent_candidate", "none"} and family in {
        "deterministic_pair_reducer",
        "exact_composition_normalized_array_status",
        "exact_globally_normalized_array_status",
        "looped_cardinality_permutation_equivariant_status",
        "looped_permutation_equivariant_status",
    }:
        return IndependentCandidateEncoder()
    raise ValueError(f"unsupported listwise family: {family}")


def _make_status_model(
    listwise_spec: Mapping[str, Any],
    *,
    input_width: int,
    seed: int,
) -> Any:
    family = str(listwise_spec["family"])
    common = {
        "input_width": input_width,
        "hidden_width": int(listwise_spec["hidden_width"]),
        "loop_count": int(listwise_spec["loop_count"]),
        "seed": seed,
    }
    if family == "looped_permutation_equivariant_status":
        return LoopedSetStatusModel(**common)
    if family == "looped_cardinality_permutation_equivariant_status":
        cardinality = dict(listwise_spec["selected_cardinality"])
        return CardinalityLoopedStatusModel(
            **common,
            max_candidates=int(cardinality["max_candidates"]),
            cardinality_loss_weight=float(cardinality["loss_weight"]),
            decoder=str(cardinality["decoder"]),
        )
    if family == "exact_globally_normalized_array_status":
        exact_array = dict(listwise_spec["exact_array_likelihood"])
        return ExactArrayLikelihoodModel(
            **common,
            max_candidates=int(exact_array["max_candidates"]),
            decoder=str(exact_array["decoder"]),
        )
    if family == "exact_composition_normalized_array_status":
        composition = dict(listwise_spec["composition_likelihood"])
        return ExactCompositionStatusModel(
            **common,
            max_candidates=int(composition["max_candidates"]),
            decoder=str(composition["decoder"]),
        )
    if family == "edge_difference_permutation_equivariant_status":
        set_loss = dict(listwise_spec["set_coupled_loss"])
        return SetCoupledLoopedStatusModel(
            **common,
            set_loss_weight=float(set_loss["weight"]),
            set_loss_smoothing=float(set_loss["smoothing"]),
        )
    raise ValueError(f"unsupported enabled listwise family: {family}")


def _train_direct_status_model(
    model: Any,
    encoder: Any,
    arrays: Sequence[Mapping[str, Any]],
    listwise_spec: Mapping[str, Any],
    training: Mapping[str, Any],
    *,
    seed: int,
    checkpoint_callback: Any,
) -> int:
    common = {
        "epochs": int(listwise_spec["epochs"]),
        "learning_rate": float(listwise_spec["learning_rate"]),
        "l2": float(training["l2"]),
        "gradient_clip": float(training["gradient_norm_clip"]),
        "seed": seed,
        "checkpoint_every_epochs": int(
            training["listwise_checkpoint_interval_epochs"]
        ),
        "checkpoint_callback": checkpoint_callback,
    }
    if isinstance(model, ExactArrayLikelihoodModel):
        return train_status_model(model, encoder, arrays, **common)
    if isinstance(model, ExactCompositionStatusModel):
        return train_status_model(model, encoder, arrays, **common)
    if isinstance(model, SetCoupledLoopedStatusModel):
        return train_edge_status_model(model, encoder, arrays, **common)
    return train_status_model(model, encoder, arrays, **common)


def _predict_direct_statuses(
    model: Any,
    encoder: Any,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(model, CardinalityLoopedStatusModel):
        return predict_cardinality_statuses(model, encoder, row)
    if isinstance(model, ExactArrayLikelihoodModel):
        return predict_exact_array_statuses(model, encoder, row)
    if isinstance(model, ExactCompositionStatusModel):
        return predict_composition_statuses(model, encoder, row)
    return predict_array_statuses(model, encoder, row)


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
        if plan["schema_version"] != "ldt-array-confirmation-trainer-plan.v3":
            raise ValueError("unsupported confirmation trainer plan")
        if plan["heldout_pair_label_access"] or plan["heldout_array_label_access"]:
            raise ValueError("confirmation plan grants heldout-label access")
        if plan["external_checkpoint"] is not None or plan["network"]:
            raise ValueError("confirmation must stay local and checkpoint-free")
        if os.environ.get("DATAGOBLIN_HARD_CAPS_VERIFIED") != "1":
            raise RuntimeError("hard-cap wrapper verification receipt is missing")
        is_v9 = plan.get("preregistration_schema_version") in {
            "9.0",
            "10.0",
            "11.0",
            "12.0",
            "13.0",
            "14.0",
            "15.0",
        }
        (
            _,
            train_rows,
            heldout_rows,
            train_arrays,
            heldout_arrays,
            _,
            _,
        ) = load_confirmation_dataset(args.dataset_manifest.resolve())
        manifest_sha256 = sha256_file(args.dataset_manifest.resolve())
        if manifest_sha256 != plan["dataset_manifest_sha256"]:
            raise ValueError("confirmation plan is bound to another dataset")
        encoder = ConfirmationPairEncoder.fit(
            train_rows,
            tolerance=float(plan["metric_equivalence_tolerance"]),
        )
        training = plan["training"]
        candidate_models = list(plan["candidate_models"])
        seed_values = [
            int(value) for value in training["seeds"]
        ]
        shard_requested = (
            args.candidate_id is not None or args.seed is not None
        )
        if shard_requested:
            if args.candidate_id is None:
                raise ValueError(
                    "a seed shard requires a candidate filter"
                )
            candidate_models = [
                model
                for model in candidate_models
                if str(model["model_id"]) == args.candidate_id
            ]
            if args.seed is not None:
                seed_values = [
                    seed
                    for seed in seed_values
                    if seed == args.seed
                ]
            if (
                len(candidate_models) != 1
                or not seed_values
            ):
                raise ValueError(
                    "trainer shard is outside the frozen sweep"
                )
        calibration = plan["confidence_calibration"]
        temperatures = [
            float(value) for value in calibration["temperatures"]
        ]
        bins = int(calibration["expected_calibration_error_bins"])
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
            candidate_count=len(candidate_models),
            seeds=seed_values,
            shard_requested=shard_requested,
            temperature_grid=temperatures,
        )
        prediction_rows: list[dict[str, Any]] = []
        array_prediction_rows: list[dict[str, Any]] = []
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
        for control in (
            [] if shard_requested else plan["controls"]
        ):
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
                        calibration_temperature=1.0,
                    )
                )
            receipts.append(
                {
                    "candidate_id": control_id,
                    "family": control["family"],
                    "seed": None,
                    "trainable_parameters": 0,
                    "trained": False,
                    "calibration_temperature": 1.0,
                    "heldout_targets_seen": False,
                }
            )
        for spec in candidate_models:
            candidate_id = str(spec["model_id"])
            view = str(spec["feature_view"])
            paired_swap_batches = bool(
                spec.get("paired_swap_batches", False)
            )
            antisymmetric_inference = bool(
                spec.get("antisymmetric_inference", False)
            )
            array_consistency_loss_weight = float(
                spec.get("array_consistency_loss_weight", 0.0)
            )
            listwise_spec = dict(
                spec.get(
                    "listwise_status",
                    {
                        "enabled": False,
                        "family": "deterministic_pair_reducer",
                    },
                )
            )
            listwise_enabled = bool(
                listwise_spec.get("enabled", False)
            )
            candidate_encoder = _candidate_encoder(listwise_spec)
            input_width = len(encoder.feature_names(view))
            for seed_value in seed_values:
                seed = int(seed_value)
                cv_reports: list[dict[str, Any]] = []
                calibration_examples: list[
                    tuple[Mapping[str, float], str]
                ] = []
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
                    fold_train_arrays = [
                        row
                        for row in train_arrays
                        if row["lineage_group_sha256"]
                        != validation_lineage
                    ]
                    fold_validation_arrays = [
                        row
                        for row in train_arrays
                        if row["lineage_group_sha256"]
                        == validation_lineage
                    ]
                    fold_encoder = ConfirmationPairEncoder.fit(
                        fold_train,
                        tolerance=float(
                            plan["metric_equivalence_tolerance"]
                        ),
                    )
                    ordered_fold_train = (
                        _paired_training_rows(fold_train)
                        if paired_swap_batches
                        else fold_train
                    )
                    examples = encode_examples(
                        ordered_fold_train,
                        fold_encoder,
                        view=view,
                        augment_swaps=False,
                        preference_labels=HARD_PREFERENCES,
                        rationale_labels=HARD_RATIONALES,
                    )
                    sample_weights, _ = _array_consistency_weights(
                        ordered_fold_train,
                        fold_train_arrays,
                        extra_weight=array_consistency_loss_weight,
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
                        gradient_clip=float(
                            training["gradient_norm_clip"]
                        ),
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
                        sample_weights=sample_weights,
                        paired_group_size=(
                            2 if paired_swap_batches else 1
                        ),
                    )
                    seed_checkpoints.extend(fold_paths)
                    fold_predictions = [
                        predict_model(
                            model,
                            fold_encoder,
                            row,
                            view=view,
                            preference_labels=HARD_PREFERENCES,
                            rationale_labels=HARD_RATIONALES,
                            antisymmetric=antisymmetric_inference,
                        )
                        for row in fold_validation
                    ]
                    fold_direct_statuses = None
                    status_model = None
                    if listwise_enabled:
                        listwise_seed = (
                            seed * 1000 + fold_index
                        )
                        status_model = _make_status_model(
                            listwise_spec,
                            input_width=len(
                                candidate_encoder.feature_names()
                            ),
                            seed=listwise_seed,
                        )
                        (
                            status_paths,
                            status_callback,
                        ) = _checkpoint_writer(
                            out_dir / "checkpoints",
                            events_path,
                            candidate_id=candidate_id,
                            seed=seed,
                            stage=(
                                f"cv-fold-{fold_index + 1}"
                                "-listwise"
                            ),
                        )
                        steps_completed += _train_direct_status_model(
                            status_model,
                            candidate_encoder,
                            fold_train_arrays,
                            listwise_spec,
                            training,
                            seed=listwise_seed,
                            checkpoint_callback=status_callback,
                        )
                        seed_checkpoints.extend(status_paths)
                        fold_direct_statuses = {
                            str(array_row["array_id"]): (
                                _predict_direct_statuses(
                                    status_model,
                                    candidate_encoder,
                                    array_row,
                                )["statuses"]
                            )
                            for array_row in (
                                fold_validation_arrays
                            )
                        }
                    cv_reports.append(
                        {
                            **_labeled_metrics(
                                fold_validation,
                                fold_predictions,
                            ),
                            **_array_labeled_metrics(
                                fold_validation,
                                fold_predictions,
                                fold_validation_arrays,
                                direct_statuses=(
                                    fold_direct_statuses
                                ),
                            ),
                        }
                    )
                    calibration_examples.extend(
                        (
                            prediction["preference_probabilities"],
                            str(row["target"]["preference"]),
                        )
                        for row, prediction in zip(
                            fold_validation,
                            fold_predictions,
                            strict=True,
                        )
                    )
                    del model
                    if status_model is not None:
                        del status_model
                    gc.collect()
                calibration_receipt = _fit_temperature(
                    calibration_examples,
                    temperatures=temperatures,
                    bins=bins,
                )
                selected_temperature = float(
                    calibration_receipt["selected_temperature"]
                )
                ordered_train_rows = (
                    _paired_training_rows(train_rows)
                    if paired_swap_batches
                    else train_rows
                )
                final_examples = encode_examples(
                    ordered_train_rows,
                    encoder,
                    view=view,
                    augment_swaps=False,
                    preference_labels=HARD_PREFERENCES,
                    rationale_labels=HARD_RATIONALES,
                )
                final_sample_weights, array_weight_receipt = (
                    _array_consistency_weights(
                        ordered_train_rows,
                        train_arrays,
                        extra_weight=array_consistency_loss_weight,
                    )
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
                    sample_weights=final_sample_weights,
                    paired_group_size=(
                        2 if paired_swap_batches else 1
                    ),
                )
                seed_checkpoints.extend(final_paths)
                final_status_model = None
                listwise_parameter_count = 0
                if listwise_enabled:
                    listwise_final_seed = seed * 1000 + 997
                    final_status_model = _make_status_model(
                        listwise_spec,
                        input_width=len(
                            candidate_encoder.feature_names()
                        ),
                        seed=listwise_final_seed,
                    )
                    (
                        final_status_paths,
                        final_status_callback,
                    ) = _checkpoint_writer(
                        out_dir / "checkpoints",
                        events_path,
                        candidate_id=candidate_id,
                        seed=seed,
                        stage="final-train-listwise",
                    )
                    steps_completed += _train_direct_status_model(
                        final_status_model,
                        candidate_encoder,
                        train_arrays,
                        listwise_spec,
                        training,
                        seed=listwise_final_seed,
                        checkpoint_callback=(
                            final_status_callback
                        ),
                    )
                    seed_checkpoints.extend(final_status_paths)
                    listwise_parameter_count = (
                        final_status_model.parameter_count
                    )
                total_parameter_count = (
                    final_model.parameter_count
                    + listwise_parameter_count
                )
                model_payload = {
                    "schema_version": "ldt-array-confirmation-model.v3",
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "family": spec["family"],
                    "feature_view": view,
                    "feature_names": list(encoder.feature_names(view)),
                    "encoder": encoder.receipt(),
                    "preference_labels": list(HARD_PREFERENCES),
                    "rationale_labels": list(HARD_RATIONALES),
                    "trainable_parameters": total_parameter_count,
                    "pair_trainable_parameters": (
                        final_model.parameter_count
                    ),
                    "listwise_trainable_parameters": (
                        listwise_parameter_count
                    ),
                    "calibration": calibration_receipt,
                    "structured_objectives": {
                        "paired_swap_batches": paired_swap_batches,
                        "antisymmetric_inference": (
                            antisymmetric_inference
                        ),
                        "array_consistency_loss_weight": (
                            array_consistency_loss_weight
                        ),
                        "array_weight_receipt": array_weight_receipt,
                    },
                    "listwise_objective": {
                        **listwise_spec,
                        "candidate_encoder": (
                            candidate_encoder.receipt()
                        ),
                        "development_array_targets_seen": (
                            listwise_enabled
                        ),
                        "heldout_array_targets_seen": False,
                    },
                    "state": final_model.state_dict(),
                    "listwise_state": (
                        None
                        if final_status_model is None
                        else final_status_model.state_dict()
                    ),
                    "authority": "MODEL_RECOMMENDATION_ONLY",
                    "heldout_targets_seen": False,
                }
                model_path = (
                    out_dir / "models" / f"{candidate_id}-seed-{seed}.json"
                )
                _write_json(model_path, model_payload)
                heldout_pair_predictions = []
                for row in heldout_rows:
                    prediction = predict_model(
                        final_model,
                        encoder,
                        row,
                        view=view,
                        preference_labels=HARD_PREFERENCES,
                        rationale_labels=HARD_RATIONALES,
                        antisymmetric=antisymmetric_inference,
                    )
                    scaled_prediction = _scaled_prediction(
                        prediction,
                        temperature=selected_temperature,
                    )
                    heldout_pair_predictions.append(
                        scaled_prediction
                    )
                    prediction_rows.append(
                        _prediction_envelope(
                            candidate_id=candidate_id,
                            seed=seed,
                            row=row,
                            prediction=scaled_prediction,
                            dataset_manifest_sha256=manifest_sha256,
                            calibration_temperature=selected_temperature,
                        )
                    )
                if final_status_model is None:
                    heldout_status_predictions = (
                        _reduced_status_predictions(
                            heldout_rows,
                            heldout_pair_predictions,
                            heldout_arrays,
                        )
                    )
                    prediction_source = (
                        "deterministic_pair_reducer"
                    )
                else:
                    heldout_status_predictions = {
                        str(array_row["array_id"]): (
                            _predict_direct_statuses(
                                final_status_model,
                                candidate_encoder,
                                array_row,
                            )
                        )
                        for array_row in heldout_arrays
                    }
                    prediction_source = str(
                        listwise_spec.get(
                            "prediction_source",
                            "direct_listwise_status_head",
                        )
                    )
                for array_row in heldout_arrays:
                    if is_v9:
                        array_prediction_rows.append(
                            _array_prediction_envelope(
                                candidate_id=candidate_id,
                                seed=seed,
                                row=array_row,
                                prediction=(
                                    heldout_status_predictions[
                                        str(
                                            array_row["array_id"]
                                        )
                                    ]
                                ),
                                dataset_manifest_sha256=(
                                    manifest_sha256
                                ),
                                prediction_source=(
                                    prediction_source
                                ),
                            )
                        )
                receipt = {
                    "candidate_id": candidate_id,
                    "family": spec["family"],
                    "feature_view": view,
                    "hidden_width": int(spec["hidden_width"]),
                    "loop_count": int(spec["loop_count"]),
                    "seed": seed,
                    "trainable_parameters": total_parameter_count,
                    "pair_trainable_parameters": (
                        final_model.parameter_count
                    ),
                    "listwise_trainable_parameters": (
                        listwise_parameter_count
                    ),
                    "trained": True,
                    "train_pairs": len(final_examples),
                    "validation": training["validation"],
                    "cv_fold_metrics": cv_reports,
                    "cv_mean_metrics": _mean_metrics(cv_reports),
                    "calibration": calibration_receipt,
                    "model_sha256": sha256_file(model_path),
                    "checkpoint_count": len(seed_checkpoints),
                    "heldout_targets_seen": False,
                    "paired_swap_batches": paired_swap_batches,
                    "antisymmetric_inference": (
                        antisymmetric_inference
                    ),
                    "array_consistency_loss_weight": (
                        array_consistency_loss_weight
                    ),
                    "array_weight_receipt": array_weight_receipt,
                    "listwise_objective": {
                        **listwise_spec,
                        "candidate_encoder": (
                            candidate_encoder.receipt()
                        ),
                        "development_array_targets_seen": (
                            listwise_enabled
                        ),
                        "heldout_array_targets_seen": False,
                    },
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
                    calibration_temperature=selected_temperature,
                    model_sha256=receipt["model_sha256"],
                )
                del final_model
                if final_status_model is not None:
                    del final_status_model
                gc.collect()
        predictions_path = out_dir / "heldout_pair_predictions.jsonl"
        with predictions_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
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
        array_predictions_path = (
            out_dir / "heldout_array_predictions.jsonl"
        )
        if is_v9:
            with array_predictions_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                for row in sorted(
                    array_prediction_rows,
                    key=lambda value: (
                        str(value["candidate_id"]),
                        int(value["seed"]),
                        str(value["array_id"]),
                    ),
                ):
                    handle.write(
                        json.dumps(
                            row,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        summary = {
            "schema_version": "ldt-array-confirmation-trainer-summary.v3",
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
            "shard_filter": (
                {
                    "candidate_id": args.candidate_id,
                    "seed": args.seed,
                }
                if shard_requested
                else None
            ),
            "elapsed_seconds": time.monotonic() - started,
            "claim_boundary": (
                "Fresh content-free tournament-summary confirmation only; "
                "outputs are recommendation-only and no game record was trained."
            ),
        }
        if is_v9:
            summary["array_predictions_sha256"] = sha256_file(
                array_predictions_path
            )
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
                "schema_version": "ldt-array-confirmation-trainer-summary.v3",
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
        description="Train bounded LDT arrays on the v3 confirmation surface"
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--trainer-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--seed", type=int)
    return parser


if __name__ == "__main__":
    sys.exit(train(build_parser().parse_args()))
