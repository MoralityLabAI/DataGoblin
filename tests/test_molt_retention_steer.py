from dataclasses import replace

import numpy as np
import pytest

from datagoblin.training_gym.molt_retention import make_world
from datagoblin.training_gym.molt_retention_steer import (
    choose_components,
    factorize,
    frame,
    interventions,
    jacobian,
    matched_vector,
    readout,
)
from datagoblin.training_gym.molt_retention_train import TiedScorer, features


def test_reconstruction_and_exact_jacobian():
    model = TiedScorer(31)
    w = make_world("train", 7, 0)
    x = features(w)
    factors = factorize(model)
    f = frame(model, x, w.root_id, factors)
    np.testing.assert_allclose(f.components.sum(1), f.source, atol=1e-12)
    j = jacobian(model, f.fixed, f.corrupted)
    for d in (0, 5, 15):
        plus, minus = f.corrupted.copy(), f.corrupted.copy()
        plus[0, d] += 1e-6
        minus[0, d] -= 1e-6
        numerical = (readout(model, f.fixed, plus) - readout(model, f.fixed, minus)) / 2e-6
        assert numerical[0] == pytest.approx(j[0, d], abs=1e-8)
        np.testing.assert_allclose(numerical[1:], 0, atol=1e-10)


def test_strong_sham_matches_signed_full_readout_and_radius():
    rng = np.random.default_rng(7)
    for _ in range(64):
        d, j, z = rng.standard_normal((3, 16))
        v, status = matched_vector(d, j, z, rng, strong=True)
        assert status == "matched"
        assert np.linalg.norm(v) == pytest.approx(np.linalg.norm(d), abs=1e-10)
        assert j @ v == pytest.approx(j @ d, abs=1e-10)
        assert np.linalg.norm(z + v) == pytest.approx(np.linalg.norm(z + d), abs=1e-10)
        assert np.linalg.norm(v - d) > 1e-5


def test_degenerate_match_is_visible():
    rng = np.random.default_rng(7)
    d = np.asarray([1.0, 0.0, 0.0])
    _, status = matched_vector(d, d, np.asarray([0.0, 1.0, 0.0]), rng, strong=True)
    assert status == "no_distinct_match"
    _, status = matched_vector(d * 0, d, d, rng, strong=True)
    assert status == "zero_dose"


def test_frozen_selection_and_no_query_access():
    model = TiedScorer(7)
    factors = factorize(model)
    train = [make_world("train", 3, i) for i in range(12)]
    ids, _ = choose_components(model, train, factors)
    assert len(ids) == 4 and len(set(ids)) == 4
    world = make_world("calibration_long", 3, 0)
    with pytest.raises(ValueError):
        choose_components(model, [world], factors)
    changed = replace(world, query=tuple(i for i in range(18) if i not in world.query)[:2])
    before = model.identity()
    a, audit = interventions(model, features(world), world.root_id, factors, ids)
    b, _ = interventions(model, features(changed), changed.root_id, factors, ids)
    for condition in a:
        for arm in a[condition]:
            np.testing.assert_array_equal(a[condition][arm], b[condition][arm])
    assert audit["full_restore_max_abs"] < 1e-10
    assert model.identity() == before
    for condition in audit["conditions"].values():
        assert condition["strong"]["all_matched"]
        assert condition["strong"]["max_signed_j_error"] < 1e-9
