import numpy as np
import pytest


def test_carry_condition_exposes_fast_feedback_placement() -> None:
    pytest.importorskip("autograd")
    from datagoblin.training_gym import rlt_hrm_carry as pilot

    w = {
        "low": np.array([[-40, -40, 0], [-40, 40, 0], [-40, 40, 0], [40, -40, 0]], dtype=np.float32),
        "high": np.array([[-40, 40, 0], [40, -40, 0], [40, -40, 0], [-40, 40, 0]], dtype=np.float32),
    }
    inputs = pilot.encode(np.array([65535]), np.array([1]), width=16)
    for arm in ("direct", "slow_rlt", "slow_damping"):
        assert pilot.words(pilot.refine(w, inputs, arm, 8)[1]).item() == 65536
    for arm in ("fast_rlt", "fast_damping"):
        assert pilot.words(pilot.refine(w, inputs, arm, 8)[1]).item() != 65536


def test_two_slow_visits_make_the_matched_schedules_identical() -> None:
    pytest.importorskip("autograd")
    from datagoblin.training_gym import rlt_hrm_carry as pilot

    rng = np.random.default_rng(17)
    w = {name: rng.normal(size=(4, 3)) for name in ("low", "high")}
    inputs = pilot.encode(np.array([1, 65535]), np.array([65535, 1]), width=16)
    rlt = pilot.refine(w, inputs, "slow_rlt", 2)
    damping = pilot.refine(w, inputs, "slow_damping", 2)
    for x, y in zip(rlt, damping, strict=True):
        np.testing.assert_array_equal(x, y)
