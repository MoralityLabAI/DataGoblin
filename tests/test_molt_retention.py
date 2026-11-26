import itertools
from dataclasses import replace

import numpy as np
import pytest

from datagoblin.training_gym.molt_retention import (
    baseline_scores,
    episode,
    exact_expected_success,
    issue,
    make_world,
    public_features,
    retain,
    verify,
)
from datagoblin.training_gym.molt_retention_train import TiedScorer, dataset, features, fit, run


def test_public_scores_and_receipt_retraction():
    world = make_world("calibration_short", 2, 0)
    altered = replace(world, query=tuple(i for i in range(10) if i not in world.query)[:2],
                      edited_source=(world.edited_source + 1) % 10)
    assert public_features(world) == public_features(altered)
    for p in ("fifo", "random", "scope", "generator_rank"):
        assert baseline_scores(world, p) == baseline_scores(altered, p)
    model = TiedScorer(7)
    np.testing.assert_array_equal(model.forward(features(world), 3)[0], model.forward(features(altered), 3)[0])
    changed = replace(world, edited_source=world.query[0])
    clean = tuple(issue(changed, i, "clean") for i in changed.query)
    answer = sum(r.value for r in clean) % 17
    assert verify(changed, "clean", clean, answer)
    assert not verify(changed, "revised", clean, answer)
    scores = tuple(float(i in changed.query) for i in range(10))
    result = episode(changed, scores, 2, 1, "revised")
    assert result["verified"] and result["verifier_calls"] == 1
    assert not episode(changed, scores, 2, 0, "revised")["verified"]


def test_generator_rank_exhaustive_expected_optimality():
    for i in range(3):
        world = make_world("calibration_short", 8, i)
        for k in (1, 2, 4):
            bank = retain(baseline_scores(world, "generator_rank"), k)
            for b, revised in itertools.product((0, 1), (False, True)):
                best = max(exact_expected_success(world, s, b, revised)
                           for s in itertools.combinations(range(10), k))
                assert exact_expected_success(world, bank, b, revised) == best


def test_bptt_against_finite_differences():
    model = TiedScorer(1, width=4)
    x = features(make_world("train", 2, 0))
    y = np.asarray([1.0] * 2 + [0.0] * 8)
    for loops in (1, 3):
        _, grads = model.loss_grad(x, y, loops)
        for key, param in model.params.items():
            index = (0,) * param.ndim
            old = param[index]
            param[index] = old + 1e-6
            high, _ = model.loss_grad(x, y, loops)
            param[index] = old - 1e-6
            low, _ = model.loss_grad(x, y, loops)
            param[index] = old
            assert grads[key][index] == pytest.approx((high - low) / 2e-6, abs=1e-7)
        if loops == 1:
            assert not np.any(grads["h"])
        else:
            assert np.linalg.norm(grads["h"]) > 0


def test_fit_matched_controls_and_labels():
    worlds = [make_world("train", 3, i) for i in range(16)]
    with pytest.raises(ValueError):
        dataset([make_world("calibration_short", 3, 0)])
    x, y = dataset(worlds)
    x2, y2 = dataset(worlds, 72)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y.reshape(-1, 10).sum(1), y2.reshape(-1, 10).sum(1))
    assert not np.array_equal(y, y2)
    _, a = fit(worlds, "tied3", 31, 4)
    _, b = fit(worlds, "single_reinvest", 31, 4)
    assert a["initial_weights_digest"] == b["initial_weights_digest"]
    assert a["forward_cell_example_visits"] == b["forward_cell_example_visits"]
    assert a["processed_examples"] * 3 == b["processed_examples"]
    assert a["weights_digest"] != b["weights_digest"]


def test_small_panel_complete_denominator():
    report = run(8, 2, 2, (31, 37))
    assert len(report["fits"]) == 8
    assert len(report["rows"]) == 4 * 13 * 2 * 4 * 2
    assert not report["promotion_authorized"]
    assert all(len(r["stored"]) <= r["capacity"] and r["verifier_calls"] <= r["verifier_budget"]
               for r in report["rows"])
    assert all(r["world_roots"] == 2 for r in report["summary"])
