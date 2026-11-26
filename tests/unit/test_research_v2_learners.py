"""Public learner fixtures, analytic-gradient checks and boundary attacks."""

import copy
import itertools

import numpy as np
import pytest

from datagoblin.research_v2 import learners


@pytest.mark.parametrize("kind", learners.KINDS)
def test_fit_is_reproducible_and_has_real_parameter_updates(kind):
    config = {"steps": 40, "train_examples": 48}
    trained = learners.fit(kind, 17, config)
    assert trained == learners.fit(kind, 17, config)
    receipt = trained["receipt"]
    assert receipt["optimizer_updates"] == 40
    assert receipt["init_weights_sha256"] != receipt["weights_sha256"]
    assert receipt["loss_after"] < receipt["loss_before"]
    assert receipt["parameters"] == (129 if kind == "trm_style" else 75)
    learners.validate_artifact(trained)
    zero = learners.fit(kind, 17, {**config, "steps": 0})
    assert zero["receipt"]["optimizer_updates"] == 0
    assert zero["receipt"]["cell_calls"] == 0
    assert zero["receipt"]["init_weights_sha256"] == zero["receipt"]["weights_sha256"]
    assert zero["receipt"]["loss_before"] == zero["receipt"]["loss_after"]
    assert zero["receipt"]["training_roots_sha256"] == receipt["training_roots_sha256"]


def test_trm_backpropagates_through_both_shared_phases():
    config = learners.normalize_config("trm_style", {"steps": 1, "width": 4, "recursions": 3})
    weights = learners._weights("trm_style", 7, config)
    data = learners._data("trm_style", 13, "train", 16)
    _, gradients = learners._trm_gradients(weights, data, config)
    # Central differences cover every parameter, including the shared phase bias.
    epsilon = 1e-5
    for name, parameter in weights.items():
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = learners._loss(learners._trm_forward(weights, data, 3)[0], data["targets"])
            parameter[index] = original - epsilon
            minus = learners._loss(learners._trm_forward(weights, data, 3)[0], data["targets"])
            parameter[index] = original
            assert gradients[name][index] == pytest.approx((plus - minus) / (2 * epsilon), abs=2e-8)


def test_ldt_gradient_matches_central_differences():
    config = learners.normalize_config("ldt_filter", {})
    weights = learners._weights("ldt_filter", 7, config)
    data = learners._data("ldt_filter", 13, "train", 16)
    _, gradients = learners._ldt_gradients(weights, data)
    epsilon = 1e-5
    for name, parameter in weights.items():
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = learners._loss(learners._probabilities("ldt_filter", weights, data, config), data["targets"])
            parameter[index] = original - epsilon
            minus = learners._loss(learners._probabilities("ldt_filter", weights, data, config), data["targets"])
            parameter[index] = original
            assert gradients[name][index] == pytest.approx((plus - minus) / (2 * epsilon), abs=2e-8)


@pytest.mark.parametrize("kind", learners.KINDS)
def test_exact_roots_cannot_cross_split_even_with_same_seed(kind):
    datasets = {split: learners._data(kind, 11, split, 128) for split in learners.SPLITS}
    for first, second in itertools.combinations(datasets.values(), 2):
        assert not set(first["roots"]) & set(second["roots"])
    for data in datasets.values():
        assert len(data["roots"]) == len(set(data["roots"]))
        assert len(data["targets"]) == 128


def test_graph_revisions_change_truth_and_targets_agree_with_integer_walk():
    data = learners._data("trm_style", 7, "calibration", 32)
    for i, graph in enumerate(data["edges"]):
        edges = graph.argmax(axis=1).tolist()
        start = int(data["start"][i].argmax())
        node = start
        for _ in range(int(data["hops"][i])):
            node = edges[node]
        assert data["targets"][i] == node
    assert np.all(data["targets"][::2] != data["targets"][1::2])
    assert np.all(data["prior"][::2] == 0)
    np.testing.assert_array_equal(data["prior"][1::2].argmax(axis=1), data["targets"][::2])


def test_positive_persona_tags_cannot_override_explicit_critical_veto():
    data = learners._data("ldt_persona", 3, "ood", 32)
    # An extreme accept-only head represents the learned-filter failure mode.
    probabilities = np.tile([0.99, 0.005, 0.005], (32, 1))
    guarded = learners._metric_values("ldt_persona", probabilities, data, {"critical_veto": True})
    unguarded = learners._metric_values("ldt_persona", probabilities, data, {"critical_veto": False})
    assert guarded["raw_critical_false_accepts"] == 32
    assert guarded["critical_false_accepts"] == 0
    assert guarded["correct"] == 32
    assert unguarded["critical_false_accepts"] == 32
    assert unguarded["correct"] == 0


def test_exact_tag_policy_ignores_style_and_keeps_uncertain_rows_on_hold():
    for positive in ([0] * 16, [1] * 16):
        assert learners._tag_target([0] * 8 + positive) == 0
        assert learners._tag_target([0] * 4 + [1, 0, 0, 0] + positive) == 1
        assert learners._tag_target([1, 0, 0, 0] + [0] * 4 + positive) == 2


@pytest.mark.parametrize("config", [
    {"steps": True}, {"steps": -1}, {"steps": 129}, {"train_examples": 257},
    {"train_examples": 17}, {"width": 33}, {"width": False}, {"recursions": 9},
    {"recursions": 0}, {"learning_rate": float("nan")}, {"learning_rate": float("inf")},
    {"learning_rate": True}, {"learning_rate": 0}, {"device": "cuda"}, {"critical_veto": True},
])
def test_unsafe_or_unknown_trm_config_is_rejected(config):
    with pytest.raises(ValueError):
        learners.fit("trm_style", 7, config)


@pytest.mark.parametrize("config", [{"critical_veto": 1}, {"width": 16}, {"recursions": 4}, {"steps": 1.0}])
def test_unsafe_or_unknown_ldt_config_is_rejected(config):
    with pytest.raises(ValueError):
        learners.fit("ldt_filter", 7, config)


@pytest.mark.parametrize("seed", [True, -1, 2**63, "7"])
def test_seeds_are_bounded_integers(seed):
    with pytest.raises(ValueError):
        learners.fit("ldt_filter", seed, {})


def test_artifact_tampering_and_malformed_shapes_fail_before_evaluation():
    artifact = learners.fit("ldt_filter", 11, {"steps": 2})
    tampered = copy.deepcopy(artifact)
    tampered["weights"]["up"][0][0] += 1
    with pytest.raises(ValueError, match="digest"):
        learners.evaluate(tampered, 7, "calibration")
    malformed = copy.deepcopy(artifact)
    malformed["weights"]["up"] = [[0.0]]
    malformed["receipt"]["weights_sha256"] = learners._digest(malformed["weights"])
    malformed["artifact_sha256"] = learners._digest({
        key: value for key, value in malformed.items() if key != "artifact_sha256"
    })
    with pytest.raises(ValueError, match="shape"):
        learners.evaluate(malformed, 7, "calibration")
    nan = copy.deepcopy(artifact)
    nan["weights"]["bias"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        learners.evaluate(nan, 7, "calibration")


def test_evaluate_uses_requested_split_and_counts_costs():
    artifact = learners.fit("trm_style", 17, {"steps": 40, "recursions": 4})
    result = learners.evaluate(artifact, 23, "calibration", examples=32)
    assert result["example_count"] == 32
    assert result["root_count"] == 16
    assert result["cell_calls"] == 32 * 2 * learners.NODES * 4
    assert result == learners.evaluate(artifact, 23, "calibration", examples=32)
    assert result["dataset_roots_sha256"] != learners.evaluate(artifact, 23, "evaluation")["dataset_roots_sha256"]
    with pytest.raises(ValueError):
        learners.evaluate(artifact, 23, "hidden")
    with pytest.raises(ValueError):
        learners.evaluate(artifact, 23, "calibration", examples=257)
