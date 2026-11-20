"""Validated feedback schedules shared by the optional pilots and CPU gym."""

from __future__ import annotations

import math
from numbers import Integral

MODES = ("direct", "rlt", "matched_damping")


def feedback_weight(mode: str, visit: int, depth: int) -> float:
    """Return the coefficient for a zero-based visit within a fixed segment.

    Damping matches RLT's coefficient sum and full terminal replacement.
    This matches coefficients, not the displacement a changing endpoint causes.
    """
    if mode not in MODES:
        raise ValueError(f"unknown feedback {mode!r}")
    if isinstance(depth, bool) or not isinstance(depth, Integral) or depth < 1:
        raise ValueError("planned depth must be a positive integer")
    if isinstance(visit, bool) or not isinstance(visit, Integral) or not 0 <= visit < depth:
        raise ValueError("visit must be a zero-based integer below planned depth")
    if mode == "direct" or visit == depth - 1:
        return 1.0
    if mode == "rlt":
        return 1 / (depth - visit)
    return math.fsum(1 / j for j in range(2, depth + 1)) / (depth - 1)
