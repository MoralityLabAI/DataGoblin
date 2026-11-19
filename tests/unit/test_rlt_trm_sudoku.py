import math

import pytest

from datagoblin.training_gym.rlt_trm_sudoku import feedback


def test_trm_schedules_change_only_the_selected_state() -> None:
    direct_y, direct_z = feedback("direct")
    y_rlt, z_direct = feedback("answer_rlt")
    y_damped, z_direct_again = feedback("answer_damping")
    y_direct, z_rlt = feedback("reasoning_rlt")
    y_direct_again, z_damped = feedback("reasoning_damping")
    assert direct_y == y_direct == y_direct_again == (1.0,) * 3
    assert direct_z == z_direct == z_direct_again == (1.0,) * 6
    assert y_rlt == (1 / 3, 1 / 2, 1)
    assert y_damped == pytest.approx((5 / 12, 5 / 12, 1))
    assert z_rlt != z_damped and len(z_rlt) == len(z_damped) == 6
    assert math.isclose(sum(z_rlt), sum(z_damped))
    with pytest.raises(ValueError, match="unknown TRM"):
        feedback("unregistered")
