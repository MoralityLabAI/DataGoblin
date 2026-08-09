from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from datagoblin.experiments.ldt_array_sorter import METRIC_FIELDS
from datagoblin.experiments.ldt_listwise_array_v9 import (
    IndependentCandidateEncoder,
    train_status_model,
)
from datagoblin.experiments.ldt_status_composition_v13 import (
    ExactCompositionStatusModel,
    _assignment_table,
    _composition_space,
)
from datagoblin.serialization import load_structured

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    ROOT
    / "configs"
    / "preregistrations"
    / "ldt_status_composition_v13.yaml"
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


def test_composition_space_and_assignment_table_are_complete() -> None:
    vectors, index = _composition_space(6)
    assignments, composition_ids = _assignment_table(6, 6)
    assert len(vectors) == len(index) == 210
    assert assignments.shape == (4096, 6)
    assert composition_ids.shape == (4096,)
    assert len(set(composition_ids.tolist())) == 84


def test_composition_statistics_match_manual_full_enumeration() -> None:
    model = ExactCompositionStatusModel(
        input_width=3,
        hidden_width=2,
        loop_count=1,
        seed=23,
        max_candidates=6,
    )
    vectors, composition_index = _composition_space(6)
    model.composition_potential[:] = [
        0.07 * vector[0]
        - 0.03 * vector[1]
        + 0.05 * vector[2]
        - 0.02 * vector[3]
        for vector in vectors
    ]
    probabilities = np.asarray(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.25, 0.35, 0.25, 0.15],
            [0.15, 0.30, 0.40, 0.15],
        ],
        dtype=np.float64,
    )
    marginals, composition_marginals, log_partition = (
        model._structured_statistics(probabilities)
    )
    assignments = list(itertools.product(range(4), repeat=3))
    scores = []
    composition_ids = []
    for assignment in assignments:
        composition = tuple(
            assignment.count(status) for status in range(4)
        )
        composition_id = composition_index[composition]
        composition_ids.append(composition_id)
        scores.append(
            sum(
                math.log(probabilities[position, status])
                for position, status in enumerate(assignment)
            )
            + model.composition_potential[composition_id]
        )
    score_array = np.asarray(scores)
    maximum = float(np.max(score_array))
    weights = np.exp(score_array - maximum)
    weights /= np.sum(weights)
    expected_marginals = np.zeros_like(probabilities)
    expected_compositions = np.zeros(210, dtype=np.float64)
    for assignment, composition_id, weight in zip(
        assignments,
        composition_ids,
        weights,
        strict=True,
    ):
        for position, status in enumerate(assignment):
            expected_marginals[position, status] += weight
        expected_compositions[composition_id] += weight
    expected_log_partition = maximum + math.log(
        float(np.sum(np.exp(score_array - maximum)))
    )
    assert np.allclose(marginals, expected_marginals, atol=1e-12)
    assert np.allclose(
        composition_marginals,
        expected_compositions,
        atol=1e-12,
    )
    assert math.isclose(log_partition, expected_log_partition, abs_tol=1e-12)


def test_matched_composition_models_train_identically() -> None:
    encoder = IndependentCandidateEncoder()
    arrays = _arrays()
    common = {
        "input_width": len(encoder.feature_names()),
        "hidden_width": 4,
        "loop_count": 3,
        "seed": 53,
        "max_candidates": 6,
    }
    factorized = ExactCompositionStatusModel(
        **common,
        decoder="factorized",
    )
    global_map = ExactCompositionStatusModel(
        **common,
        decoder="global_composition_map",
    )
    first_features = encoder.transform_array(
        [candidate["features"] for candidate in arrays[0]["candidates"]]
    )
    before = factorized.loss(first_features, [1, 2, 0, 3])
    for model in (factorized, global_map):
        assert (
            train_status_model(
                model,
                encoder,
                arrays,
                epochs=30,
                learning_rate=0.025,
                l2=0.0005,
                gradient_clip=5.0,
                seed=53,
                checkpoint_every_epochs=15,
            )
            == 60
        )
    assert factorized.loss(first_features, [1, 2, 0, 3]) < before
    assert factorized.parameter_count == global_map.parameter_count == 374
    assert 245 + factorized.parameter_count == 619
    assert np.array_equal(factorized.input_weights, global_map.input_weights)
    assert np.array_equal(
        factorized.composition_potential,
        global_map.composition_potential,
    )


def test_v13_preregistration_freezes_matched_composition_ablation() -> None:
    config = load_structured(PREREGISTRATION)
    assert config["schema_version"] == "13.0"
    assert config["status"] == "frozen_before_dataset_generation"
    assert len(config["surface"]["forbidden_prior_lineage_sha256"]) == 84
    composition = [
        model["listwise_status"]
        for model in config["models"]["candidates"]
        if model["listwise_status"]["family"]
        == "exact_composition_normalized_array_status"
    ]
    assert len(composition) == 2
    left = dict(composition[0])
    right = dict(composition[1])
    left.pop("prediction_source")
    right.pop("prediction_source")
    left_objective = dict(left.pop("composition_likelihood"))
    right_objective = dict(right.pop("composition_likelihood"))
    assert left == right
    assert left_objective.pop("decoder") == "factorized"
    assert right_objective.pop("decoder") == "global_composition_map"
    assert left_objective == right_objective
