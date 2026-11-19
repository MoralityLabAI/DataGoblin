from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


def test_reference_clock_and_feedback_pulse_use_separate_controls() -> None:
    pytest.importorskip("autograd")
    from datagoblin.training_gym import rlt_pilot as pilot

    weights = pilot.initialize(17)
    ids = np.asarray([[1, 2, 3, 4]])
    delta = np.random.default_rng(7).normal(size=(1, 4, pilot.WIDTH)).astype(np.float32) * 0.01
    base = pilot.logits_and_states(weights, ids, "rlt", 4, all_visits=True)
    explicit = pilot.logits_and_states(weights, ids, "rlt", 4, all_visits=True, clock_depths=(4, 4))
    late = pilot.logits_and_states(weights, ids, "rlt", 4, all_visits=True, feedback_pulse=(3, delta))
    for clean, reference, final_only in zip(base, explicit, late, strict=True):
        np.testing.assert_array_equal(clean, reference)
        np.testing.assert_array_equal(clean, final_only)
    early = pilot.logits_and_states(weights, ids, "rlt", 4, all_visits=True, feedback_pulse=(0, delta))
    np.testing.assert_array_equal(base[0], early[0])
    assert not np.allclose(base[-1], early[-1])


def test_feedback_controls_and_causal_prefix() -> None:
    pytest.importorskip("autograd")
    from datagoblin.training_gym import rlt_pilot as pilot

    assert [pilot.feedback_weight("rlt", i, 4) for i in range(4)] == [1 / 4, 1 / 3, 1 / 2, 1]
    assert math.isclose(
        sum(pilot.feedback_weight("matched_damping", i, 4) for i in range(4)),
        sum(pilot.feedback_weight("rlt", i, 4) for i in range(4)),
    )
    weights = pilot.initialize(5)
    prefix = np.array([[1, 2, 3, 4]])
    future_changed = np.array([[1, 2, 9, 8]])
    for mode in pilot.MODES:
        old = pilot.logits_and_states(weights, prefix, mode, 4, all_visits=True)
        new = pilot.logits_and_states(weights, future_changed, mode, 4, all_visits=True)
        for left, right in zip(old, new, strict=True):
            assert np.allclose(left[:, :2], right[:, :2], atol=1e-6)


def test_shared_block_gradient_through_rlt_schedule() -> None:
    pytest.importorskip("autograd")
    from autograd import grad

    from datagoblin.training_gym import rlt_pilot as pilot

    initial = pilot.initialize(5)
    ids = np.array([[1, 2, 3, 4]])
    target = np.array([2, 3, 4, 5])

    def objective(qkv):
        weights = initial | {"qkv": qkv}
        logits = pilot.logits_and_states(weights, ids, "rlt", 4, all_visits=True)
        return sum(pilot.target_nll(visit[0], target).mean() for visit in logits) / len(logits)

    matrix = initial["qkv"].copy()
    derivative = grad(objective)(matrix)
    location = np.unravel_index(np.argmax(np.abs(derivative)), derivative.shape)
    epsilon = 1e-3
    matrix[location] += epsilon
    plus = objective(matrix)
    matrix[location] -= 2 * epsilon
    minus = objective(matrix)
    assert abs(derivative[location]) > 1e-3
    assert derivative[location] == pytest.approx((plus - minus) / (2 * epsilon), abs=1e-3, rel=0.03)


def test_paused_optimizer_resumes_same_trajectory(tmp_path: Path) -> None:
    pytest.importorskip("autograd")
    from datagoblin.training_gym import rlt_pilot as pilot
    from datagoblin.training_gym.rlt_revision import build

    build(tmp_path / "rules", seed=12, train=2, per_split=1)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    text = [[pilot.BOS, *([i] * 96)] for i in (65, 66, 67)]
    (frozen / "text_train.json").write_text(pilot.canonical_json(text), encoding="utf-8")
    continuous = tmp_path / "complete"
    resumed = tmp_path / "resumed"
    continuous.mkdir()
    resumed.mkdir()

    def continue_run():
        return

    checks = iter((True, False))

    def interrupt_at_second_step():
        if not next(checks):
            raise OSError("test host RAM pause")

    pilot.train(tmp_path, continuous, "rlt", 8, 3, continue_run)
    with pytest.raises(OSError, match="test host RAM pause"):
        pilot.train(tmp_path, resumed, "rlt", 8, 3, interrupt_at_second_step)
    assert pilot.jsonl(resumed / "train.jsonl")[-1]["update"] == 1
    pilot.train(tmp_path, resumed, "rlt", 8, 3, continue_run)
    with np.load(continuous / "weights.npz") as left, np.load(resumed / "weights.npz") as right:
        assert all(np.array_equal(left[name], right[name]) for name in left)
