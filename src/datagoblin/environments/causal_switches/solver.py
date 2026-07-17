from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MechanicsState:
    intervention_seen: bool
    elapsed: int
    delay: int


def exact_label(state: MechanicsState) -> int:
    """Environment-owned label: the delayed effect is visible iff its clock matured."""
    if state.delay < 1:
        raise ValueError("delay must be positive")
    if state.elapsed < 0:
        raise ValueError("elapsed must be nonnegative")
    return int(state.intervention_seen and state.elapsed >= state.delay)
