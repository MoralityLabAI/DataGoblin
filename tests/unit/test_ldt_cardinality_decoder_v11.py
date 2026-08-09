from __future__ import annotations

from pathlib import Path

import numpy as np

from datagoblin.experiments.ldt_array_sorter import METRIC_FIELDS
from datagoblin.experiments.ldt_cardinality_decoder_v11 import (
    CardinalityLoopedStatusModel,
    joint_selected_count_map,
)
from datagoblin.experiments.ldt_listwise_array_v9 import (
    IndependentCandidateEncoder,
    train_status_model,
)
from datagoblin.serialization import load_structured

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    ROOT
    / "configs"
    / "preregistrations"
    / "ldt_cardinality_decoder_v11.yaml"
)


def _candidate(offset: float) -> dict[str, object]:
    result: dict[str, object] = {
        "trainable_parameter_bucket": "medium",
        "safety_qualified": True,
    }
    for index, field in enumerate(METRIC_FIELDS):
        value = offset + index * 0.04
        result[field] = {"lower": value, "upper": value + 0.01}
    return result


def _arrays() -> list[dict[str, object]]:
    return [
        {
            "array_id": "array-one",
            "candidates": [
                {"features": _candidate(0.20)},
                {"features": _candidate(0.40)},
                {"features": _candidate(0.60)},
                {"features": _candidate(0.75)},
            ],
            "target": {
                "statuses": [
                    "DOMINATED",
                    "UNRESOLVED",
                    "SELECTED",
                    "UNSAFE",
                ]
            },
        },
        {
            "array_id": "array-two",
            "candidates": [
                {"features": _candidate(0.25)},
                {"features": _candidate(0.45)},
                {"features": _candidate(0.65)},
                {"features": _candidate(0.80)},
            ],
            "target": {
                "statuses": [
                    "DOMINATED",
                    "UNRESOLVED",
                    "SELECTED",
                    "UNSAFE",
                ]
            },
        },
    ]


def test_joint_map_uses_count_and_is_permutation_equivariant() -> None:
    probabilities = np.asarray(
        [
            [0.40, 0.50, 0.05, 0.05],
            [0.35, 0.34, 0.16, 0.15],
            [0.10, 0.60, 0.20, 0.10],
        ],
        dtype=np.float64,
    )
    count_probabilities = np.asarray(
        [0.01, 0.95, 0.03, 0.01, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    decoded = joint_selected_count_map(
        probabilities,
        count_probabilities,
    )
    assert decoded.tolist() == [1, 0, 1]

    order = np.asarray([2, 0, 1])
    permuted = joint_selected_count_map(
        probabilities[order],
        count_probabilities,
    )
    assert permuted.tolist() == decoded[order].tolist()


def test_matched_cardinality_models_train_identically() -> None:
    encoder = IndependentCandidateEncoder()
    arrays = _arrays()
    common = {
        "input_width": len(encoder.feature_names()),
        "hidden_width": 4,
        "loop_count": 3,
        "seed": 41,
        "max_candidates": 6,
        "cardinality_loss_weight": 0.5,
    }
    factorized = CardinalityLoopedStatusModel(
        **common,
        decoder="factorized",
    )
    joint = CardinalityLoopedStatusModel(
        **common,
        decoder="joint_selected_count_map",
    )
    before = factorized.loss(
        encoder.transform_array(
            [candidate["features"] for candidate in arrays[0]["candidates"]]
        ),
        [1, 2, 0, 3],
    )
    for model in (factorized, joint):
        steps = train_status_model(
            model,
            encoder,
            arrays,
            epochs=50,
            learning_rate=0.025,
            l2=0.0005,
            gradient_clip=5.0,
            seed=41,
            checkpoint_every_epochs=25,
        )
        assert steps == 100
    after = factorized.loss(
        encoder.transform_array(
            [candidate["features"] for candidate in arrays[0]["candidates"]]
        ),
        [1, 2, 0, 3],
    )
    assert after < before
    assert factorized.parameter_count == joint.parameter_count == 206
    assert 245 + factorized.parameter_count == 451
    assert np.array_equal(factorized.input_weights, joint.input_weights)
    assert np.array_equal(
        factorized.cardinality_weights,
        joint.cardinality_weights,
    )


def test_v11_preregistration_freezes_matched_decoder_ablation() -> None:
    config = load_structured(PREREGISTRATION)
    assert config["schema_version"] == "11.0"
    assert config["status"] == "frozen_before_dataset_generation"
    assert len(config["surface"]["forbidden_prior_lineage_sha256"]) == 64
    cardinality = [
        model["listwise_status"]
        for model in config["models"]["candidates"]
        if model["listwise_status"]["family"]
        == "looped_cardinality_permutation_equivariant_status"
    ]
    assert len(cardinality) == 2
    left = dict(cardinality[0])
    right = dict(cardinality[1])
    left.pop("prediction_source")
    right.pop("prediction_source")
    left_cardinality = dict(left.pop("selected_cardinality"))
    right_cardinality = dict(right.pop("selected_cardinality"))
    assert left == right
    assert left_cardinality.pop("decoder") == "factorized"
    assert right_cardinality.pop("decoder") == "joint_selected_count_map"
    assert left_cardinality == right_cardinality
