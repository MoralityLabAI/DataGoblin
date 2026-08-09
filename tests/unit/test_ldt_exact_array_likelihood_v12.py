from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from datagoblin.experiments.ldt_array_sorter import METRIC_FIELDS
from datagoblin.experiments.ldt_exact_array_likelihood_v12 import (
    ExactArrayLikelihoodModel,
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
    / "ldt_exact_array_likelihood_v12.yaml"
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


def test_exact_structured_statistics_match_full_enumeration() -> None:
    model = ExactArrayLikelihoodModel(
        input_width=3,
        hidden_width=2,
        loop_count=1,
        seed=17,
        max_candidates=6,
    )
    model.selected_count_potential[:] = [0.2, -0.1, 0.4, 0.0, 0.0, 0.0, 0.0]
    probabilities = np.asarray(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.25, 0.35, 0.25, 0.15],
            [0.15, 0.30, 0.40, 0.15],
        ],
        dtype=np.float64,
    )
    marginals, count_marginals, log_partition = (
        model._structured_statistics(probabilities)
    )

    assignments = list(itertools.product(range(4), repeat=3))
    scores = np.asarray(
        [
            sum(
                math.log(probabilities[index, status])
                for index, status in enumerate(assignment)
            )
            + model.selected_count_potential[assignment.count(0)]
            for assignment in assignments
        ]
    )
    maximum = float(np.max(scores))
    weights = np.exp(scores - maximum)
    weights /= np.sum(weights)
    expected_marginals = np.zeros_like(probabilities)
    expected_counts = np.zeros(7, dtype=np.float64)
    for assignment, weight in zip(assignments, weights, strict=True):
        for index, status in enumerate(assignment):
            expected_marginals[index, status] += weight
        expected_counts[assignment.count(0)] += weight
    expected_log_partition = maximum + math.log(
        float(np.sum(np.exp(scores - maximum)))
    )
    assert np.allclose(marginals, expected_marginals, atol=1e-12)
    assert np.allclose(count_marginals, expected_counts, atol=1e-12)
    assert math.isclose(log_partition, expected_log_partition, abs_tol=1e-12)


def test_matched_exact_array_models_train_identically() -> None:
    encoder = IndependentCandidateEncoder()
    arrays = _arrays()
    common = {
        "input_width": len(encoder.feature_names()),
        "hidden_width": 4,
        "loop_count": 3,
        "seed": 47,
        "max_candidates": 6,
    }
    factorized = ExactArrayLikelihoodModel(
        **common,
        decoder="factorized",
    )
    global_map = ExactArrayLikelihoodModel(
        **common,
        decoder="global_selected_count_map",
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
                epochs=50,
                learning_rate=0.025,
                l2=0.0005,
                gradient_clip=5.0,
                seed=47,
                checkpoint_every_epochs=25,
            )
            == 100
        )
    assert factorized.loss(first_features, [1, 2, 0, 3]) < before
    assert factorized.parameter_count == global_map.parameter_count == 171
    assert 245 + factorized.parameter_count == 416
    assert np.array_equal(factorized.input_weights, global_map.input_weights)
    assert np.array_equal(
        factorized.selected_count_potential,
        global_map.selected_count_potential,
    )


def test_v12_preregistration_freezes_matched_exact_decoder_ablation() -> None:
    config = load_structured(PREREGISTRATION)
    assert config["schema_version"] == "12.0"
    assert config["status"] == "frozen_before_dataset_generation"
    assert len(config["surface"]["forbidden_prior_lineage_sha256"]) == 74
    exact = [
        model["listwise_status"]
        for model in config["models"]["candidates"]
        if model["listwise_status"]["family"]
        == "exact_globally_normalized_array_status"
    ]
    assert len(exact) == 2
    left = dict(exact[0])
    right = dict(exact[1])
    left.pop("prediction_source")
    right.pop("prediction_source")
    left_objective = dict(left.pop("exact_array_likelihood"))
    right_objective = dict(right.pop("exact_array_likelihood"))
    assert left == right
    assert left_objective.pop("decoder") == "factorized"
    assert right_objective.pop("decoder") == "global_selected_count_map"
    assert left_objective == right_objective
