"""Candidate-content feature invariance and near-miss distinctness contracts."""

from __future__ import annotations

import numpy as np
import pytest

from datagoblin.training_gym.rlt_gym.hard02_mathlib import generate
from datagoblin.training_gym.rlt_gym.hard02_semantic_features import (
    DIMENSION,
    FEATURE_VERSION,
    features,
)


def test_content_features_distinguish_all_five_matched_candidates():
    roots = generate(per_split=8, seed=2026112211, matched_menus=True)
    assert len(roots) == 32
    assert FEATURE_VERSION == "rlt_hard02_lexical_hash_v1"
    for root in roots:
        x = features(f"⊢ {root.statement}", root.menu)
        assert x.shape == (5, DIMENSION)
        assert np.isfinite(x).all()
        assert len({tuple(np.round(row, 12)) for row in x}) == 5


def test_features_are_candidate_order_independent_and_ignore_padding():
    root = generate(per_split=8, seed=2026112211, matched_menus=True)[0]
    goal = f"⊢ {root.statement}"
    original = features(goal, root.menu)
    permuted = features(goal, tuple(reversed(root.menu)))
    stripped = features(goal, tuple(item.rstrip() for item in root.menu))
    assert np.array_equal(original, permuted[::-1])
    assert np.array_equal(original, stripped)


def test_features_reject_empty_candidates_and_unbounded_menu():
    with pytest.raises(ValueError, match="nonempty"):
        features("", ["rfl"])
    with pytest.raises(ValueError, match="empty"):
        features("⊢ True", ["  "])
    with pytest.raises(ValueError, match="bounded"):
        features("⊢ True", ["rfl"] * 33)
