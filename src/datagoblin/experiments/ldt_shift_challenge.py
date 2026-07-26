from __future__ import annotations

import json
import math
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.experiments.ldt_array_confirmation import (
    ConfirmationPairEncoder,
    confirmation_input_sha256,
    evidence_has_interval_width,
    evidence_is_complete,
    reduce_confirmation_array,
    swap_confirmation_pair,
)
from datagoblin.experiments.ldt_array_hard_surface import (
    HARD_INVERT_PREFERENCE,
    HARD_PREFERENCES,
    HARD_RATIONALES,
)
from datagoblin.experiments.ldt_array_sorter import (
    BUCKET_RANK,
    METRIC_FIELDS,
    LinearJointArray,
    LoopedJointArray,
    make_model,
    sha256_file,
    sha256_json,
)
from datagoblin.experiments.ldt_shadow import (
    _score_array_slice,
    _score_pair_slice,
    restore_latent_model,
)
from datagoblin.experiments.ldt_shift_builder import load_shift_fixture
from datagoblin.serialization import load_structured


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _load_contract(contract_path: Path) -> dict[str, Any]:
    contract = load_structured(contract_path)
    if (
        contract["schema_version"] != "6.0"
        or contract["status"] != "frozen_before_shift_generation"
    ):
        raise ValueError("shift challenge contract is not frozen v6")
    if (
        contract["decision_boundary"]["model_selection_authorized"]
        or contract["decision_boundary"]["training_authorized"]
        or contract["decision_boundary"]["corpus_promotion_authorized"]
        or contract["decision_boundary"]["deployment_authorized"]
    ):
        raise ValueError("shift challenge grants prohibited authority")
    setup_ids = [
        str(setup["setup_id"]) for setup in contract["candidate_setups"]
    ]
    if len(set(setup_ids)) != len(setup_ids):
        raise ValueError("shift candidate setup IDs are not unique")
    return contract


def _temperature_scale(
    probabilities: Mapping[str, float],
    temperature: float,
) -> dict[str, float]:
    logits = {
        label: math.log(max(float(value), 1e-12)) / temperature
        for label, value in probabilities.items()
    }
    maximum = max(logits.values())
    exponentials = {
        label: math.exp(value - maximum)
        for label, value in logits.items()
    }
    total = sum(exponentials.values())
    return {label: value / total for label, value in exponentials.items()}


def _encoder_from_payload(payload: Mapping[str, Any]) -> ConfirmationPairEncoder:
    receipt = payload["encoder"]
    encoder = ConfirmationPairEncoder(
        games=tuple(str(value) for value in receipt["games"]),
        pressures=tuple(str(value) for value in receipt["pressures"]),
        context_modes=tuple(str(value) for value in receipt["context_modes"]),
        optimization_pressures=tuple(
            str(value) for value in receipt["optimization_pressures"]
        ),
        tolerance=float(receipt["tolerance"]),
    )
    if list(encoder.feature_names(str(payload["feature_view"]))) != payload[
        "feature_names"
    ]:
        raise ValueError("frozen shift encoder feature names differ")
    return encoder


def _restore_linear_model(
    payload: Mapping[str, Any],
) -> LinearJointArray:
    if (
        payload["family"] != "joint_linear_array"
        or payload["state"]["architecture"] != "joint_linear_array"
    ):
        raise ValueError("shift linear model metadata differs")
    state = payload["state"]
    input_width = int(state["input_width"])
    model = make_model(
        {"family": "joint_linear_array"},
        input_width,
        0,
        preference_count=len(HARD_PREFERENCES),
        rationale_count=len(HARD_RATIONALES),
    )
    if not isinstance(model, LinearJointArray):
        raise TypeError("shift linear model did not restore as linear")
    preference = np.asarray(
        state["preference_weights"],
        dtype=np.float64,
    )
    rationale = np.asarray(
        state["rationale_weights"],
        dtype=np.float64,
    )
    if (
        preference.shape != (len(HARD_PREFERENCES), input_width)
        or rationale.shape != (len(HARD_RATIONALES), input_width)
        or not np.all(np.isfinite(preference))
        or not np.all(np.isfinite(rationale))
    ):
        raise ValueError("shift linear model tensors are invalid")
    model.preference_weights = preference
    model.rationale_weights = rationale
    if model.parameter_count != int(payload["trainable_parameters"]):
        raise ValueError("shift linear model parameter count differs")
    return model


@dataclass(frozen=True)
class FrozenShiftModel:
    payload: dict[str, Any]
    model: LinearJointArray | LoopedJointArray
    encoder: ConfirmationPairEncoder
    model_sha256: str
    temperature: float


def _load_frozen_model(
    *,
    setup: Mapping[str, Any],
    model_path: Path,
    trainer: Mapping[str, Any],
    trainer_sha256: str,
    expected_trainer_sha256: str,
) -> FrozenShiftModel:
    if trainer_sha256 != expected_trainer_sha256:
        raise ValueError("shift trainer summary hash differs")
    model_sha256 = sha256_file(model_path)
    if model_sha256 != setup["source_model_sha256"]:
        raise ValueError("shift model hash differs")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if (
        payload["schema_version"] != "ldt-array-confirmation-model.v3"
        or payload["candidate_id"] != setup["source_candidate_id"]
        or int(payload["seed"]) != int(setup["source_seed"])
        or payload["family"] != setup["source_family"]
        or payload["feature_view"] != setup["source_feature_view"]
        or int(payload["trainable_parameters"])
        != int(setup["trainable_parameters"])
        or payload["heldout_targets_seen"]
        or payload["authority"] != "MODEL_RECOMMENDATION_ONLY"
    ):
        raise ValueError("shift source model binding differs")
    matching = [
        row
        for row in trainer["candidate_receipts"]
        if row["candidate_id"] == setup["source_candidate_id"]
        and int(row["seed"]) == int(setup["source_seed"])
    ]
    if len(matching) != 1 or matching[0]["model_sha256"] != model_sha256:
        raise ValueError("shift trainer does not bind the source model")
    calibration = payload["calibration"]
    temperature = float(calibration["selected_temperature"])
    if (
        calibration["method"] != "frozen_temperature_grid"
        or calibration["fit_split"]
        != "out_of_fold_development_predictions_only"
        or not math.isclose(
            temperature,
            float(setup["calibration_temperature"]),
            abs_tol=1e-12,
        )
    ):
        raise ValueError("shift model calibration differs")
    model = (
        restore_latent_model(payload)
        if payload["family"] == "latent_array"
        else _restore_linear_model(payload)
    )
    return FrozenShiftModel(
        payload=payload,
        model=model,
        encoder=_encoder_from_payload(payload),
        model_sha256=model_sha256,
        temperature=temperature,
    )


def _model_features(
    bundle: FrozenShiftModel,
    input_: Mapping[str, Mapping[str, Any]],
    *,
    policy_lattice_mode: str,
) -> array:
    view = str(bundle.payload["feature_view"])
    features = bundle.encoder.transform(input_, view=view)
    if policy_lattice_mode == "not_applicable":
        if view != "raw":
            raise ValueError("only raw features can omit lattice handling")
        return features
    if view != "typed":
        raise ValueError("lattice mode requires the typed feature view")
    if policy_lattice_mode == "full_diagnostic_control":
        return features
    if policy_lattice_mode != "masked":
        raise ValueError("unknown policy lattice mode")
    names = bundle.encoder.feature_names(view)
    masked = array(
        "d",
        (
            0.0 if name.startswith("lattice:") else value
            for name, value in zip(names, features, strict=True)
        ),
    )
    if sum(name.startswith("lattice:") for name in names) != 10:
        raise ValueError("policy lattice mask did not bind ten features")
    return masked


def _model_prediction(
    bundle: FrozenShiftModel,
    row: Mapping[str, Any],
    *,
    policy_lattice_mode: str,
) -> dict[str, Any]:
    features = _model_features(
        bundle,
        row["input"],
        policy_lattice_mode=policy_lattice_mode,
    )
    preference_values, rationale_values = bundle.model.predict_proba(
        features
    )
    raw_probabilities = {
        label: float(preference_values[index])
        for index, label in enumerate(HARD_PREFERENCES)
    }
    probabilities = _temperature_scale(
        raw_probabilities,
        bundle.temperature,
    )
    preference = max(
        HARD_PREFERENCES,
        key=lambda label: probabilities[label],
    )
    rationale = HARD_RATIONALES[
        max(
            range(len(HARD_RATIONALES)),
            key=lambda index: rationale_values[index],
        )
    ]
    swapped = swap_confirmation_pair(row)
    swapped_features = _model_features(
        bundle,
        swapped["input"],
        policy_lattice_mode=policy_lattice_mode,
    )
    swapped_values, _ = bundle.model.predict_proba(swapped_features)
    swapped_probabilities = _temperature_scale(
        {
            label: float(swapped_values[index])
            for index, label in enumerate(HARD_PREFERENCES)
        },
        bundle.temperature,
    )
    swapped_preference = max(
        HARD_PREFERENCES,
        key=lambda label: swapped_probabilities[label],
    )
    return {
        "preference": preference,
        "rationale_code": rationale,
        "confidence": probabilities[preference],
        "preference_probabilities": probabilities,
        "swapped_preference": swapped_preference,
        "swap_consistent": (
            swapped_preference == HARD_INVERT_PREFERENCE[preference]
        ),
    }


def _fixed_probabilities(
    preference: str,
    confidence: float,
) -> dict[str, float]:
    remainder = (1.0 - confidence) / (len(HARD_PREFERENCES) - 1)
    return {
        label: confidence if label == preference else remainder
        for label in HARD_PREFERENCES
    }


def _nonlearned_choice(
    input_: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float,
    scalar_margin: float,
) -> tuple[str, str, float]:
    left = input_["left"]
    right = input_["right"]
    left_safe = bool(left["safety_qualified"])
    right_safe = bool(right["safety_qualified"])
    if left_safe != right_safe:
        return (
            ("LEFT" if left_safe else "RIGHT"),
            "safety_gate",
            0.90,
        )
    if (
        not evidence_is_complete(left)
        or not evidence_is_complete(right)
        or evidence_has_interval_width(left)
        or evidence_has_interval_width(right)
    ):
        return "ABSTAIN", "insufficient_evidence", 0.70
    left_values = [
        float(left[field]["lower"]) for field in METRIC_FIELDS
    ]
    right_values = [
        float(right[field]["lower"]) for field in METRIC_FIELDS
    ]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    if abs(left_mean - right_mean) > scalar_margin:
        return (
            ("LEFT" if left_mean > right_mean else "RIGHT"),
            "pareto_dominance",
            0.72,
        )
    equivalent = all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(
            left_values,
            right_values,
            strict=True,
        )
    )
    left_cost = BUCKET_RANK[str(left["trainable_parameter_bucket"])]
    right_cost = BUCKET_RANK[str(right["trainable_parameter_bucket"])]
    if equivalent and left_cost != right_cost:
        return (
            ("LEFT" if left_cost < right_cost else "RIGHT"),
            "parameter_efficiency",
            0.68,
        )
    return "TRADEOFF", "irreducible_tradeoff", 0.60


def _nonlearned_prediction(
    row: Mapping[str, Any],
    *,
    tolerance: float,
    scalar_margin: float,
) -> dict[str, Any]:
    preference, rationale, confidence = _nonlearned_choice(
        row["input"],
        tolerance=tolerance,
        scalar_margin=scalar_margin,
    )
    swapped = swap_confirmation_pair(row)
    swapped_preference, _, _ = _nonlearned_choice(
        swapped["input"],
        tolerance=tolerance,
        scalar_margin=scalar_margin,
    )
    probabilities = _fixed_probabilities(preference, confidence)
    return {
        "preference": preference,
        "rationale_code": rationale,
        "confidence": confidence,
        "preference_probabilities": probabilities,
        "swapped_preference": swapped_preference,
        "swap_consistent": (
            swapped_preference == HARD_INVERT_PREFERENCE[preference]
        ),
    }


def run_shift_candidates(
    *,
    contract_path: Path,
    manifest_path: Path,
    model_root: Path,
    trainer_summary_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    contract = _load_contract(contract_path)
    manifest, pairs, arrays, _ = load_shift_fixture(manifest_path)
    if manifest["contract_sha256"] != sha256_file(contract_path):
        raise ValueError("shift manifest belongs to another contract")
    trainer_sha256 = sha256_file(trainer_summary_path)
    if trainer_sha256 != contract["source_training"][
        "trainer_summary_sha256"
    ]:
        raise ValueError("shift trainer summary hash differs")
    trainer = json.loads(
        trainer_summary_path.read_text(encoding="utf-8")
    )
    if (
        trainer["status"] != "completed"
        or not trainer["hard_caps_verified"]
        or trainer["heldout_pair_targets_seen"]
        or trainer["heldout_array_targets_seen"]
    ):
        raise ValueError("shift source trainer is incomplete or contaminated")

    out_dir.mkdir(parents=True, exist_ok=False)
    setups_dir = out_dir / "setups"
    setups_dir.mkdir()
    setup_receipts = []
    tolerance = float(
        contract["shift_surface"]["metric_equivalence_tolerance"]
    )
    for setup in contract["candidate_setups"]:
        setup_id = str(setup["setup_id"])
        setup_out = setups_dir / setup_id
        setup_out.mkdir()
        bundle = None
        model_sha256 = None
        temperature = None
        if setup["kind"] == "frozen_model":
            model_path = model_root / str(setup["source_model_filename"])
            bundle = _load_frozen_model(
                setup=setup,
                model_path=model_path,
                trainer=trainer,
                trainer_sha256=trainer_sha256,
                expected_trainer_sha256=contract["source_training"][
                    "trainer_summary_sha256"
                ],
            )
            model_sha256 = bundle.model_sha256
            temperature = bundle.temperature
        elif setup["kind"] != "nonlearned":
            raise ValueError("unknown shift setup kind")

        pair_recommendations = []
        preferences_by_array: dict[
            str,
            dict[tuple[int, int], str],
        ] = {}
        confidences_by_array: dict[str, list[float]] = {}
        for row in pairs:
            if bundle is None:
                prediction = _nonlearned_prediction(
                    row,
                    tolerance=tolerance,
                    scalar_margin=float(setup["scalar_margin"]),
                )
            else:
                prediction = _model_prediction(
                    bundle,
                    row,
                    policy_lattice_mode=str(
                        setup["policy_lattice_mode"]
                    ),
                )
            array_id = str(row["array_id"])
            preferences_by_array.setdefault(array_id, {})[
                (int(row["left_index"]), int(row["right_index"]))
            ] = str(prediction["preference"])
            confidences_by_array.setdefault(array_id, []).append(
                float(prediction["confidence"])
            )
            pair_recommendations.append(
                {
                    "schema_version": (
                        "ldt-shift-pair-recommendation.v6"
                    ),
                    "setup_id": setup_id,
                    "evidence_class": setup["evidence_class"],
                    "selection_eligible": False,
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
                    "lineage_group_sha256": row[
                        "lineage_group_sha256"
                    ],
                    "input_sha256": confirmation_input_sha256(row),
                    "shift_manifest_sha256": sha256_file(manifest_path),
                    "contract_sha256": sha256_file(contract_path),
                    "source_model_sha256": model_sha256,
                    "trainer_summary_sha256": (
                        trainer_sha256 if bundle is not None else None
                    ),
                    "feature_view": setup["inference_feature_view"],
                    "policy_lattice_mode": setup[
                        "policy_lattice_mode"
                    ],
                    **prediction,
                    "calibration_temperature": temperature,
                    "authority": "MODEL_RECOMMENDATION_ONLY",
                    "decision_effect": "NONE",
                }
            )
        array_recommendations = []
        for row in arrays:
            array_id = str(row["array_id"])
            target = reduce_confirmation_array(
                [
                    candidate["features"]
                    for candidate in row["candidates"]
                ],
                preferences_by_array[array_id],
            )
            array_recommendations.append(
                {
                    "schema_version": (
                        "ldt-shift-array-recommendation.v6"
                    ),
                    "setup_id": setup_id,
                    "evidence_class": setup["evidence_class"],
                    "selection_eligible": False,
                    "array_id": array_id,
                    "lineage_group_sha256": row[
                        "lineage_group_sha256"
                    ],
                    "game": row["game"],
                    "fixture_pressure": row["fixture_pressure"],
                    "candidate_count": row["candidate_count"],
                    "statuses": target["statuses"],
                    "selected_indices": target["selected_indices"],
                    "dominated_indices": target["dominated_indices"],
                    "unresolved_indices": target["unresolved_indices"],
                    "unsafe_indices": target["unsafe_indices"],
                    "minimum_pair_confidence": min(
                        confidences_by_array[array_id]
                    ),
                    "shift_manifest_sha256": sha256_file(manifest_path),
                    "contract_sha256": sha256_file(contract_path),
                    "source_model_sha256": model_sha256,
                    "trainer_summary_sha256": (
                        trainer_sha256 if bundle is not None else None
                    ),
                    "authority": "MODEL_RECOMMENDATION_ONLY",
                    "decision_effect": "NONE",
                }
            )
        pair_path = setup_out / "pair_recommendations.jsonl"
        array_path = setup_out / "array_recommendations.jsonl"
        _write_jsonl(
            pair_path,
            sorted(
                pair_recommendations,
                key=lambda row: str(row["example_id"]),
            ),
        )
        _write_jsonl(
            array_path,
            sorted(
                array_recommendations,
                key=lambda row: str(row["array_id"]),
            ),
        )
        receipt = {
            "schema_version": "ldt-shift-setup-run-receipt.v6",
            "setup_id": setup_id,
            "kind": setup["kind"],
            "evidence_class": setup["evidence_class"],
            "selection_eligible": False,
            "status": "completed",
            "contract_sha256": sha256_file(contract_path),
            "shift_manifest_sha256": sha256_file(manifest_path),
            "source_model_sha256": model_sha256,
            "trainer_summary_sha256": (
                trainer_sha256 if bundle is not None else None
            ),
            "pair_recommendations_sha256": sha256_file(pair_path),
            "array_recommendations_sha256": sha256_file(array_path),
            "pair_count": len(pair_recommendations),
            "array_count": len(array_recommendations),
            "abstain_count": sum(
                row["preference"] == "ABSTAIN"
                for row in pair_recommendations
            ),
            "verifier_targets_seen": False,
            "verifier_label_paths_seen": False,
            "raw_game_records_read": False,
            "network_used": False,
            "authority": "MODEL_RECOMMENDATION_ONLY",
            "decision_effect": "NONE",
            "training_authorized": False,
            "model_selection_authorized": False,
        }
        receipt_path = setup_out / "run_receipt.json"
        _write_json(receipt_path, receipt)
        setup_receipts.append(
            {
                "setup_id": setup_id,
                "run_receipt_sha256": sha256_file(receipt_path),
                "pair_recommendations_sha256": sha256_file(pair_path),
                "array_recommendations_sha256": sha256_file(array_path),
                "source_model_sha256": model_sha256,
                "evidence_class": setup["evidence_class"],
                "selection_eligible": False,
            }
        )
    aggregate = {
        "schema_version": "ldt-shift-candidate-run-receipt.v6",
        "experiment_id": contract["experiment_id"],
        "status": "completed",
        "contract_sha256": sha256_file(contract_path),
        "shift_manifest_sha256": sha256_file(manifest_path),
        "trainer_summary_sha256": trainer_sha256,
        "setup_count": len(setup_receipts),
        "setups": setup_receipts,
        "all_candidate_outputs_immutable_before_verifier_join": True,
        "verifier_targets_seen": False,
        "verifier_label_paths_seen": False,
        "raw_game_records_read": False,
        "network_used": False,
        "decision_effect": "NONE",
        "training_authorized": False,
        "model_selection_authorized": False,
    }
    _write_json(out_dir / "candidate_run_receipt.json", aggregate)
    return aggregate


def _load_verifier_labels(
    *,
    pairs: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, Any]],
    pair_path: Path,
    array_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    public_pairs = {str(row["example_id"]): row for row in pairs}
    public_arrays = {str(row["array_id"]): row for row in arrays}
    pair_labels = {}
    for row in _read_jsonl(pair_path):
        example_id = str(row["example_id"])
        if (
            row["schema_version"]
            != "ldt-shift-verifier-pair-label.v6"
            or row["authority"] != "ENVIRONMENT_OWNED"
            or example_id in pair_labels
            or example_id not in public_pairs
            or row["array_id"] != public_pairs[example_id]["array_id"]
            or row["lineage_group_sha256"]
            != public_pairs[example_id]["lineage_group_sha256"]
            or row["target"]["preference"] not in HARD_PREFERENCES
            or row["target"]["rationale_code"] not in HARD_RATIONALES
        ):
            raise ValueError("shift verifier pair label is invalid")
        pair_labels[example_id] = row
    array_labels = {}
    for row in _read_jsonl(array_path):
        array_id = str(row["array_id"])
        if (
            row["schema_version"]
            != "ldt-shift-verifier-array-label.v6"
            or row["authority"] != "ENVIRONMENT_OWNED"
            or array_id in array_labels
            or array_id not in public_arrays
            or row["lineage_group_sha256"]
            != public_arrays[array_id]["lineage_group_sha256"]
            or len(row["target"]["statuses"])
            != int(public_arrays[array_id]["candidate_count"])
        ):
            raise ValueError("shift verifier array label is invalid")
        array_labels[array_id] = row
    if set(pair_labels) != set(public_pairs) or set(array_labels) != set(
        public_arrays
    ):
        raise ValueError("shift verifier labels do not cover public inputs")
    return pair_labels, array_labels


def _validate_setup_predictions(
    *,
    setup: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, Any]],
    pair_path: Path,
    array_path: Path,
    contract_sha256: str,
    manifest_sha256: str,
    trainer_summary_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    public_pairs = {str(row["example_id"]): row for row in pairs}
    public_arrays = {str(row["array_id"]): row for row in arrays}
    pair_predictions = {}
    expected_model_sha256 = setup.get("source_model_sha256")
    expected_trainer_sha256 = (
        trainer_summary_sha256
        if setup["kind"] == "frozen_model"
        else None
    )
    for row in _read_jsonl(pair_path):
        example_id = str(row["example_id"])
        source = public_pairs.get(example_id)
        if (
            row["schema_version"] != "ldt-shift-pair-recommendation.v6"
            or row["setup_id"] != setup["setup_id"]
            or row["evidence_class"] != setup["evidence_class"]
            or row["selection_eligible"]
            or row["authority"] != "MODEL_RECOMMENDATION_ONLY"
            or row["decision_effect"] != "NONE"
            or row["contract_sha256"] != contract_sha256
            or row["shift_manifest_sha256"] != manifest_sha256
            or row["source_model_sha256"] != expected_model_sha256
            or row["trainer_summary_sha256"]
            != expected_trainer_sha256
            or row["feature_view"] != setup["inference_feature_view"]
            or row["policy_lattice_mode"]
            != setup["policy_lattice_mode"]
            or example_id in pair_predictions
            or source is None
            or row["input_sha256"]
            != confirmation_input_sha256(source)
            or row["preference"] not in HARD_PREFERENCES
            or row["rationale_code"] not in HARD_RATIONALES
            or row["swapped_preference"] not in HARD_PREFERENCES
        ):
            raise ValueError("shift pair recommendation is invalid")
        binding_fields = (
            "array_id",
            "left_index",
            "right_index",
            "left_tournament_report_sha256",
            "right_tournament_report_sha256",
            "lineage_group_sha256",
        )
        if any(row[field] != source[field] for field in binding_fields):
            raise ValueError("shift pair source binding differs")
        probabilities = row["preference_probabilities"]
        probability_values = [
            float(probabilities[label]) for label in HARD_PREFERENCES
        ]
        if (
            set(probabilities) != set(HARD_PREFERENCES)
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in probability_values
            )
            or not math.isclose(
                sum(probability_values),
                1.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(row["confidence"]),
                max(probability_values),
                abs_tol=1e-8,
            )
            or bool(row["swap_consistent"])
            != (
                row["swapped_preference"]
                == HARD_INVERT_PREFERENCE[row["preference"]]
            )
        ):
            raise ValueError("shift pair probabilities are invalid")
        pair_predictions[example_id] = row
    if set(pair_predictions) != set(public_pairs):
        raise ValueError("shift pair recommendations do not cover inputs")
    preferences_by_array: dict[
        str,
        dict[tuple[int, int], str],
    ] = {}
    confidence_by_array: dict[str, list[float]] = {}
    for source in public_pairs.values():
        prediction = pair_predictions[str(source["example_id"])]
        array_id = str(source["array_id"])
        preferences_by_array.setdefault(array_id, {})[
            (int(source["left_index"]), int(source["right_index"]))
        ] = str(prediction["preference"])
        confidence_by_array.setdefault(array_id, []).append(
            float(prediction["confidence"])
        )
    array_predictions = {}
    for row in _read_jsonl(array_path):
        array_id = str(row["array_id"])
        source = public_arrays.get(array_id)
        if (
            row["schema_version"] != "ldt-shift-array-recommendation.v6"
            or row["setup_id"] != setup["setup_id"]
            or row["evidence_class"] != setup["evidence_class"]
            or row["selection_eligible"]
            or row["authority"] != "MODEL_RECOMMENDATION_ONLY"
            or row["decision_effect"] != "NONE"
            or row["contract_sha256"] != contract_sha256
            or row["shift_manifest_sha256"] != manifest_sha256
            or row["source_model_sha256"] != expected_model_sha256
            or row["trainer_summary_sha256"]
            != expected_trainer_sha256
            or array_id in array_predictions
            or source is None
            or len(row["statuses"])
            != int(source["candidate_count"])
        ):
            raise ValueError("shift array recommendation is invalid")
        if (
            row["lineage_group_sha256"]
            != source["lineage_group_sha256"]
            or row["game"] != source["game"]
            or row["fixture_pressure"] != source["fixture_pressure"]
            or int(row["candidate_count"])
            != int(source["candidate_count"])
        ):
            raise ValueError("shift array source binding differs")
        expected = reduce_confirmation_array(
            [
                candidate["features"]
                for candidate in source["candidates"]
            ],
            preferences_by_array[array_id],
        )
        for field in (
            "statuses",
            "selected_indices",
            "dominated_indices",
            "unresolved_indices",
            "unsafe_indices",
        ):
            if row[field] != expected[field]:
                raise ValueError("shift array reduction differs")
        if not math.isclose(
            float(row["minimum_pair_confidence"]),
            min(confidence_by_array[array_id]),
            abs_tol=1e-12,
        ):
            raise ValueError("shift array confidence binding differs")
        array_predictions[array_id] = row
    if set(array_predictions) != set(public_arrays):
        raise ValueError("shift array recommendations do not cover inputs")
    return pair_predictions, array_predictions


def _evidence_slice(row: Mapping[str, Any]) -> str:
    sides = (row["input"]["left"], row["input"]["right"])
    if any(
        side[field]["lower"] is None
        or side[field]["upper"] is None
        for side in sides
        for field in METRIC_FIELDS
    ):
        return "missing_bound"
    if any(
        float(side[field]["lower"]) != float(side[field]["upper"])
        for side in sides
        for field in METRIC_FIELDS
    ):
        return "interval_width"
    return "complete_point"


def _score_setup(
    *,
    contract: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, Any]],
    pair_predictions: Mapping[str, Mapping[str, Any]],
    array_predictions: Mapping[str, Mapping[str, Any]],
    pair_labels: Mapping[str, Mapping[str, Any]],
    array_labels: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    bins = int(contract["monitoring"]["calibration_bins"])
    pair_aggregate = _score_pair_slice(
        pairs,
        pair_predictions,
        pair_labels,
        bins=bins,
    )
    array_aggregate = _score_array_slice(
        arrays,
        array_predictions,
        array_labels,
    )
    pair_by_game = {}
    array_by_game = {}
    for game in contract["shift_surface"]["games"]:
        game_pairs = [
            row
            for row in pairs
            if row["input"]["left"]["game"] == game
        ]
        game_arrays = [row for row in arrays if row["game"] == game]
        pair_by_game[str(game)] = _score_pair_slice(
            game_pairs,
            pair_predictions,
            pair_labels,
            bins=bins,
        )
        array_by_game[str(game)] = _score_array_slice(
            game_arrays,
            array_predictions,
            array_labels,
        )
    pair_by_pressure = {}
    array_by_pressure = {}
    for pressure in contract["shift_surface"]["pressures"]:
        pressure_pairs = [
            row
            for row in pairs
            if row["input"]["left"]["fixture_pressure"] == pressure
        ]
        pressure_arrays = [
            row for row in arrays if row["fixture_pressure"] == pressure
        ]
        pair_by_pressure[str(pressure)] = _score_pair_slice(
            pressure_pairs,
            pair_predictions,
            pair_labels,
            bins=bins,
        )
        array_by_pressure[str(pressure)] = _score_array_slice(
            pressure_arrays,
            array_predictions,
            array_labels,
        )
    pair_by_evidence = {}
    for slice_name in (
        "complete_point",
        "interval_width",
        "missing_bound",
    ):
        selected = [
            row for row in pairs if _evidence_slice(row) == slice_name
        ]
        if selected:
            pair_by_evidence[slice_name] = _score_pair_slice(
                selected,
                pair_predictions,
                pair_labels,
                bins=bins,
            )
    thresholds = contract["monitoring"]
    qualification_checks = {
        "safety_gate_accuracy": (
            pair_aggregate["safety_gate_accuracy"]
            == float(thresholds["require_safety_accuracy"])
        ),
        "swap_consistency": (
            pair_aggregate["swap_consistency"]
            == float(thresholds["require_swap_consistency"])
        ),
        "abstain_recall": (
            pair_aggregate["abstain_recall"]
            >= float(thresholds["minimum_abstain_recall"])
        ),
        "expected_calibration_error": (
            pair_aggregate["expected_calibration_error"]
            <= float(thresholds["maximum_ece"])
        ),
    }
    return {
        "pair_aggregate": pair_aggregate,
        "array_aggregate": array_aggregate,
        "pair_by_game": pair_by_game,
        "array_by_game": array_by_game,
        "pair_by_pressure": pair_by_pressure,
        "array_by_pressure": array_by_pressure,
        "pair_by_evidence": pair_by_evidence,
        "qualification_checks": qualification_checks,
        "qualified": all(qualification_checks.values()),
    }


def evaluate_shift_challenge(
    *,
    contract_path: Path,
    manifest_path: Path,
    candidate_run_dir: Path,
    verifier_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    contract = _load_contract(contract_path)
    manifest, pairs, arrays, _ = load_shift_fixture(manifest_path)
    contract_sha256 = sha256_file(contract_path)
    manifest_sha256 = sha256_file(manifest_path)
    if manifest["contract_sha256"] != contract_sha256:
        raise ValueError("shift manifest belongs to another contract")
    seal_path = verifier_dir / "seal_receipt.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    pair_label_path = verifier_dir / "verifier_pair_labels.jsonl"
    array_label_path = verifier_dir / "verifier_array_labels.jsonl"
    if (
        not seal["seal_passed"]
        or not seal["sealed_before_candidate_inference"]
        or seal["candidate_workspace_receives_label_paths"]
        or seal["contract_sha256"] != contract_sha256
        or seal["shift_manifest_sha256"] != manifest_sha256
        or seal["pair_labels_sha256"] != sha256_file(pair_label_path)
        or seal["array_labels_sha256"] != sha256_file(array_label_path)
    ):
        raise ValueError("shift verifier seal is invalid")
    pair_labels, array_labels = _load_verifier_labels(
        pairs=pairs,
        arrays=arrays,
        pair_path=pair_label_path,
        array_path=array_label_path,
    )
    aggregate_run_path = candidate_run_dir / "candidate_run_receipt.json"
    aggregate_run = json.loads(
        aggregate_run_path.read_text(encoding="utf-8")
    )
    if (
        aggregate_run["status"] != "completed"
        or not aggregate_run[
            "all_candidate_outputs_immutable_before_verifier_join"
        ]
        or aggregate_run["verifier_targets_seen"]
        or aggregate_run["verifier_label_paths_seen"]
        or aggregate_run["decision_effect"] != "NONE"
        or aggregate_run["training_authorized"]
        or aggregate_run["model_selection_authorized"]
        or aggregate_run["contract_sha256"] != contract_sha256
        or aggregate_run["shift_manifest_sha256"] != manifest_sha256
    ):
        raise ValueError("shift candidate run receipt is invalid")
    run_by_setup = {
        str(row["setup_id"]): row for row in aggregate_run["setups"]
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    setup_results = {}
    prediction_bindings = []
    review_rows = []
    for setup in contract["candidate_setups"]:
        setup_id = str(setup["setup_id"])
        if setup_id not in run_by_setup:
            raise ValueError("shift candidate run is missing a setup")
        setup_dir = candidate_run_dir / "setups" / setup_id
        run_receipt_path = setup_dir / "run_receipt.json"
        pair_path = setup_dir / "pair_recommendations.jsonl"
        array_path = setup_dir / "array_recommendations.jsonl"
        run_binding = run_by_setup[setup_id]
        setup_run = json.loads(
            run_receipt_path.read_text(encoding="utf-8")
        )
        if (
            sha256_file(run_receipt_path)
            != run_binding["run_receipt_sha256"]
            or sha256_file(pair_path)
            != run_binding["pair_recommendations_sha256"]
            or sha256_file(array_path)
            != run_binding["array_recommendations_sha256"]
            or setup_run["status"] != "completed"
            or setup_run["setup_id"] != setup_id
            or setup_run["verifier_targets_seen"]
            or setup_run["verifier_label_paths_seen"]
            or setup_run["decision_effect"] != "NONE"
            or setup_run["training_authorized"]
            or setup_run["model_selection_authorized"]
        ):
            raise ValueError("shift setup artifact hash differs")
        pair_predictions, array_predictions = _validate_setup_predictions(
            setup=setup,
            pairs=pairs,
            arrays=arrays,
            pair_path=pair_path,
            array_path=array_path,
            contract_sha256=contract_sha256,
            manifest_sha256=manifest_sha256,
            trainer_summary_sha256=aggregate_run[
                "trainer_summary_sha256"
            ],
        )
        result = _score_setup(
            contract=contract,
            pairs=pairs,
            arrays=arrays,
            pair_predictions=pair_predictions,
            array_predictions=array_predictions,
            pair_labels=pair_labels,
            array_labels=array_labels,
        )
        setup_results[setup_id] = {
            "kind": setup["kind"],
            "evidence_class": setup["evidence_class"],
            "selection_eligible": False,
            **result,
        }
        prediction_bindings.append(
            {
                "setup_id": setup_id,
                "run_receipt_sha256": sha256_file(run_receipt_path),
                "pair_recommendations_sha256": sha256_file(pair_path),
                "array_recommendations_sha256": sha256_file(array_path),
            }
        )
        for row in pairs:
            example_id = str(row["example_id"])
            prediction = pair_predictions[example_id]
            label = pair_labels[example_id]["target"]
            mismatch_axes = []
            if prediction["preference"] != label["preference"]:
                mismatch_axes.append("preference")
            if prediction["rationale_code"] != label["rationale_code"]:
                mismatch_axes.append("rationale")
            if mismatch_axes:
                review_rows.append(
                    {
                        "schema_version": "ldt-shift-review-key.v6",
                        "review_key_sha256": sha256_json(
                            {
                                "setup_id": setup_id,
                                "example_id": example_id,
                                "manifest_sha256": manifest_sha256,
                            }
                        ),
                        "setup_id": setup_id,
                        "kind": "pair_disagreement",
                        "mismatch_axes": mismatch_axes,
                        "game": row["input"]["left"]["game"],
                        "fixture_pressure": row["input"]["left"][
                            "fixture_pressure"
                        ],
                        "evidence_slice": _evidence_slice(row),
                        "contains_verifier_label": False,
                    }
                )
    ablation = contract["causal_ablation"]
    full = setup_results[str(ablation["full_lattice_setup_id"])]
    masked = setup_results[str(ablation["masked_lattice_setup_id"])]
    causal_deltas = {
        "preference_accuracy_full_minus_masked": (
            full["pair_aggregate"]["preference_accuracy"]
            - masked["pair_aggregate"]["preference_accuracy"]
        ),
        "preference_macro_f1_full_minus_masked": (
            full["pair_aggregate"]["preference_macro_f1"]
            - masked["pair_aggregate"]["preference_macro_f1"]
        ),
        "exact_array_accuracy_full_minus_masked": (
            full["array_aggregate"]["exact_array_accuracy"]
            - masked["array_aggregate"]["exact_array_accuracy"]
        ),
        "ece_masked_minus_full": (
            masked["pair_aggregate"]["expected_calibration_error"]
            - full["pair_aggregate"]["expected_calibration_error"]
        ),
    }
    lattice_reliance_detected = (
        causal_deltas["preference_accuracy_full_minus_masked"]
        >= float(ablation["minimum_accuracy_drop_for_reliance"])
        or causal_deltas["exact_array_accuracy_full_minus_masked"]
        >= float(ablation["minimum_array_accuracy_drop_for_reliance"])
    )
    checks = {
        "verifier_sealed_before_candidates": seal[
            "sealed_before_candidate_inference"
        ],
        "candidate_outputs_immutable_before_join": aggregate_run[
            "all_candidate_outputs_immutable_before_verifier_join"
        ],
        "all_setups_scored": len(setup_results)
        == len(contract["candidate_setups"]),
        "no_setup_selection_eligible": all(
            not row["selection_eligible"] for row in setup_results.values()
        ),
        "review_queue_contains_no_verifier_labels": all(
            not row["contains_verifier_label"] for row in review_rows
        ),
        "verifier_decisions_unmutated": True,
    }
    report = {
        "schema_version": "ldt-independent-shift-comparison.v6",
        "comparison_revision": 1,
        "verifier_join_count": 1,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": contract_sha256,
        "shift_manifest_sha256": manifest_sha256,
        "verifier_seal_receipt_sha256": sha256_file(seal_path),
        "verifier_pair_labels_sha256": sha256_file(pair_label_path),
        "verifier_array_labels_sha256": sha256_file(array_label_path),
        "candidate_run_receipt_sha256": sha256_file(
            aggregate_run_path
        ),
        "prediction_bindings": prediction_bindings,
        "target_distributions": {
            "preference": seal["preference_distribution"],
            "rationale": seal["rationale_distribution"],
            "array_status": seal["array_status_distribution"],
        },
        "setup_results": setup_results,
        "causal_ablation": {
            "full_lattice_setup_id": ablation[
                "full_lattice_setup_id"
            ],
            "masked_lattice_setup_id": ablation[
                "masked_lattice_setup_id"
            ],
            "deltas": causal_deltas,
            "policy_lattice_reliance_detected": (
                lattice_reliance_detected
            ),
            "full_lattice_control_is_diagnostic_only": True,
        },
        "qualified_setups": sorted(
            setup_id
            for setup_id, result in setup_results.items()
            if result["qualified"]
        ),
        "universal_scalar_champion": None,
        "model_selection_authorized": False,
        "checks": checks,
        "challenge_execution_passed": all(checks.values()),
        "review_queue_count": len(review_rows),
        "review_queue_contains_verifier_labels": False,
        "aggregate_contains_row_level_verifier_labels": False,
        "verifier_authority": "ENVIRONMENT_OWNED",
        "candidate_authority": "MODEL_RECOMMENDATION_ONLY",
        "decision_effect": "NONE",
        "verifier_decisions_mutated": False,
        "training_authorized": False,
        "corpus_promotion_authorized": False,
        "deployment_authorized": False,
        "claim_boundary": contract["claim_scope"],
    }
    report_path = out_dir / "shift_comparison.json"
    review_path = out_dir / "hash_only_review_queue.jsonl"
    _write_json(report_path, report)
    _write_jsonl(
        review_path,
        sorted(
            review_rows,
            key=lambda row: (
                str(row["setup_id"]),
                str(row["review_key_sha256"]),
            ),
        ),
    )
    receipt = {
        "schema_version": "ldt-independent-shift-comparison-receipt.v6",
        "shift_comparison_sha256": sha256_file(report_path),
        "comparison_revision": 1,
        "verifier_join_count": 1,
        "hash_only_review_queue_sha256": sha256_file(review_path),
        "review_queue_contains_verifier_labels": False,
        "models_predictions_labels_or_thresholds_changed": False,
        "model_selection_authorized": False,
        "post_comparison_training_authorized": False,
        "decision_effect": "NONE",
        "verifier_decisions_mutated": False,
    }
    _write_json(out_dir / "comparison_receipt.json", receipt)
    return report
