import numpy as np
import pytest


def test_candidate_loss_excludes_unknowns_and_ignores_pool_padding():
    pytest.importorskip("autograd")
    from datagoblin.training_gym.rlt_lean_train_rank import initialize, objective, scores

    params = initialize(seed=7, dimension=3, width=5)
    x = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=float)
    known = np.array([[1, 0, -1]])
    mask = np.ones((1, 3), dtype=bool)
    extra = np.concatenate([x, np.full((1, 1, 3), 999.0)], axis=1)
    padded = np.array([[True, True, True, False]])

    for mode in ("direct", "rlt", "matched_damping"):
        logits = list(scores(params, x, mask, mode, 6))
        expected = sum(np.logaddexp(0, row[0, 1] - row[0, 0]) for row in logits) / 6
        assert objective(params, x, mask, known, mode, 6) == pytest.approx(expected)
        assert objective(params, extra, padded, np.array([[1, 0, -1, -1]]), mode, 6) == pytest.approx(expected)
