import math
import pickle
from io import BytesIO

import numpy as np
import pytest

from datagoblin.training_gym.rlt_hrm_sudoku import slow_coefficients, sudoku_valid
from datagoblin.training_gym.rlt_hrm_sudoku_cpu import bf16
from datagoblin.training_gym.rlt_hrm_sudoku_weights import LayoutReader


def test_sudoku_gate_checks_givens_and_boxes() -> None:
    solved = "".join(str((3 * r + r // 3 + c) % 9 + 1) for r in range(9) for c in range(9))
    latin_square = "".join(str((r + c) % 9 + 1) for r in range(9) for c in range(9))
    assert sudoku_valid(solved[0] + "0" * 80, solved)
    assert not sudoku_valid("2" + "0" * 80, solved)
    assert not sudoku_valid("0" * 81, latin_square)
    assert not sudoku_valid("0" * 81, solved[:-1] + "0")


def test_slow_schedule_respects_segment_and_global_counts() -> None:
    assert slow_coefficients("segment_rlt", 16) == (0.5, 1.0) * 16
    assert slow_coefficients("global_rlt", 1) == slow_coefficients("global_damping", 1)
    rlt = slow_coefficients("global_rlt", 16)
    damped = slow_coefficients("global_damping", 16)
    assert rlt[0] == 1 / 32 and rlt[-1] == damped[-1] == 1
    assert math.isclose(sum(rlt), sum(damped))
    assert len(rlt) == len(damped) == 32


def test_tensor_layout_reader_rejects_unlisted_globals() -> None:
    with pytest.raises(pickle.UnpicklingError, match="unsupported checkpoint global"):
        LayoutReader(BytesIO(pickle.dumps(complex))).load()


def test_bfloat16_rounding_uses_nearest_even() -> None:
    midpoints = np.array([1 + 1 / 256, 1 + 3 / 256, -1 - 1 / 256], dtype=np.float32)
    np.testing.assert_array_equal(bf16(midpoints), np.array([1, 1 + 1 / 64, -1], dtype=np.float32))
