from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from datagoblin.domain.experiments import LifecycleStatus
from datagoblin.serialization import freeze_jsonable, to_jsonable

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

_NEXT = {
    LifecycleStatus.DRAFT: LifecycleStatus.CRITIQUED,
    LifecycleStatus.CRITIQUED: LifecycleStatus.VALIDATED,
    LifecycleStatus.VALIDATED: LifecycleStatus.TICKETED,
    LifecycleStatus.TICKETED: LifecycleStatus.BUILT,
    LifecycleStatus.BUILT: LifecycleStatus.EXECUTED,
    LifecycleStatus.EXECUTED: LifecycleStatus.EVALUATED,
}


class TransitionAuthority(str, Enum):
    MODEL = "model"
    ORCHESTRATOR = "orchestrator"
    BUILDER = "builder"
    HARNESS = "harness"
    ENVIRONMENT = "environment"
    HUMAN = "human"


_AUTHORITY_TARGETS = {
    TransitionAuthority.MODEL: {LifecycleStatus.CRITIQUED},
    TransitionAuthority.ORCHESTRATOR: {
        LifecycleStatus.CRITIQUED,
        LifecycleStatus.VALIDATED,
        LifecycleStatus.TICKETED,
        LifecycleStatus.PROMOTED,
        LifecycleStatus.REJECTED,
    },
    TransitionAuthority.BUILDER: {LifecycleStatus.BUILT, LifecycleStatus.REJECTED},
    TransitionAuthority.HARNESS: {
        LifecycleStatus.EXECUTED,
        LifecycleStatus.EVALUATED,
        LifecycleStatus.REJECTED,
    },
    TransitionAuthority.ENVIRONMENT: {
        LifecycleStatus.EXECUTED,
        LifecycleStatus.EVALUATED,
        LifecycleStatus.REJECTED,
    },
    TransitionAuthority.HUMAN: {LifecycleStatus.PROMOTED, LifecycleStatus.REJECTED},
}


@dataclass(frozen=True)
class ResearchState:
    state_id: str
    lifecycle_status: LifecycleStatus
    candidate_hypothesis_ids: tuple[str, ...]
    frozen_evaluation_hash: str | None
    public_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_hypothesis_ids", tuple(self.candidate_hypothesis_ids))
        object.__setattr__(self, "public_payload", freeze_jsonable(self.public_payload))

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


def transition_status(
    current: LifecycleStatus,
    target: LifecycleStatus,
    *,
    authority: TransitionAuthority | str,
) -> LifecycleStatus:
    try:
        typed_authority = TransitionAuthority(authority)
    except ValueError as exc:
        raise ValueError(f"unknown transition authority: {authority}") from exc
    if current in {LifecycleStatus.PROMOTED, LifecycleStatus.REJECTED} and target != current:
        raise ValueError("terminal lifecycle status is immutable")
    if _ORDER[target] < _ORDER[current]:
        raise ValueError("lifecycle transitions must be monotone")
    if target == current:
        return target
    allowed_target = LifecycleStatus.REJECTED if target == LifecycleStatus.REJECTED else _NEXT.get(current)
    if current == LifecycleStatus.EVALUATED and target == LifecycleStatus.PROMOTED:
        allowed_target = LifecycleStatus.PROMOTED
    if target != allowed_target:
        raise ValueError(f"lifecycle transition must be adjacent: {current.value} -> {target.value}")
    if target not in _AUTHORITY_TARGETS[typed_authority]:
        raise ValueError(f"{typed_authority.value} authority cannot mark {target.value.lower()}")
    return target
