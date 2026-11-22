"""TRAIN/CAL-only semantic Mathlib ranker wiring, with test-only labels."""

from __future__ import annotations

import numpy as np
import pytest

from datagoblin.training_gym.rlt_gym.hard02_mathlib import (
    _witness,
    calibrate,
    fit_policy,
    generate,
    order,
    rank_scores,
)
from datagoblin.training_gym.rlt_gym.hard02_semantic_features import DIMENSION


def _synthetic_rows(split: str, *, seed: int):
    # These labels are unit-test fixtures only, not experimental evidence.
    roots = [r for r in generate(per_split=8, seed=seed, matched_menus=True) if r.split == split]
    result = []
    for index, root in enumerate(roots):
        _, _, witness = _witness(
            split, index, numeric_shift=17 + seed % 41, depth_shift=1 + seed % 3
        )
        checks = [{
            "candidate_id": cid,
            "receipt": {"passed": tactic.strip() == witness},
        } for cid, tactic in enumerate(root.menu)]
        result.append({
            "split": split, "root": root, "goal_pp": f"⊢ {root.statement}",
            "menu": root.menu, "checks": checks,
        })
    return result


def test_native_training_wrapper_uses_full_script_semantic_features():
    seed = 2026112211
    train = _synthetic_rows("train", seed=seed)
    cal = _synthetic_rows("calibration", seed=seed)
    weights = fit_policy(train, (), semantic=True)
    assert weights.shape == (DIMENSION,)
    assert np.isfinite(weights).all()
    for root in train:
        scores = rank_scores(root["goal_pp"], root["menu"], weights, (), semantic=True)
        assert scores.shape == (5,)
        assert np.isfinite(scores).all()
        assert len(order(root["root"], "learned", root["goal_pp"], weights, (), semantic=True)) == 5
    threshold, frontier = calibrate(cal, weights, (), semantic=True)
    assert threshold in {row["threshold"] for row in frontier}
    assert all(row["certified"] <= 8 for row in frontier)


def test_semantic_model_training_rejects_heldout_and_calibration_roots():
    seed = 2026112211
    with pytest.raises(ValueError, match="TRAIN"):
        fit_policy(_synthetic_rows("ood_test", seed=seed), (), semantic=True)
    train = _synthetic_rows("train", seed=seed)
    weights = fit_policy(train, (), semantic=True)
    with pytest.raises(ValueError, match="CAL"):
        calibrate(train, weights, (), semantic=True)
