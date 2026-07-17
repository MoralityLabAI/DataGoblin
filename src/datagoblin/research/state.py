from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datagoblin.domain.experiments import LifecycleStatus
from datagoblin.serialization import to_jsonable

_ORDER = {
    LifecycleStatus.DRAFT: 0,
    LifecycleStatus.CRITIQUED: 1,
    LifecycleStatus.VALIDATED: 2,
    LifecycleStatus.TICKETED: 3,
    LifecycleStatus.BUILT: 4,
    LifecycleStatus.EXECUTED: 5,
    LifecycleStatus.EVALUATED: 6,
    LifecycleStatus.PROMOTED: 7,
    LifecycleStatus.REJECTED: 7,
}


@dataclass(frozen=True)
class ResearchState:
    state_id: str
    lifecycle_status: LifecycleStatus
    candidate_hypothesis_ids: tuple[str, ...]
    frozen_evaluation_hash: str | None
    public_payload: Mapping[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


def transition_status(
    current: LifecycleStatus,
    target: LifecycleStatus,
    *,
    authority: str,
) -> LifecycleStatus:
    if current in {LifecycleStatus.PROMOTED, LifecycleStatus.REJECTED} and target != current:
        raise ValueError("terminal lifecycle status is immutable")
    if _ORDER[target] < _ORDER[current]:
        raise ValueError("lifecycle transitions must be monotone")
    if authority == "model" and target in {
        LifecycleStatus.EXECUTED,
        LifecycleStatus.EVALUATED,
        LifecycleStatus.PROMOTED,
    }:
        raise ValueError("model authority cannot mark execution, evaluation, or promotion")
    if current == LifecycleStatus.EVALUATED and target not in {
        LifecycleStatus.EVALUATED,
        LifecycleStatus.PROMOTED,
        LifecycleStatus.REJECTED,
    }:
        raise ValueError("evaluated state cannot be mutated")
    return target
