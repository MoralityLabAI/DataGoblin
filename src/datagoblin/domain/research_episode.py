from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datagoblin.serialization import to_jsonable

from .builders import BuilderTicket
from .experiments import ExperimentRecord
from .hypotheses import ResearchHypothesis
from .learner import LearnerSnapshot
from .recipes import DataRecipe


@dataclass(frozen=True)
class ResearchEpisode:
    episode_id: str
    schema_version: str
    learner_snapshot: LearnerSnapshot
    scout_findings: Mapping[str, Any]
    hypothesis_set: tuple[ResearchHypothesis, ...]
    proposed_recipe: DataRecipe
    rat_critique: Mapping[str, Any]
    repaired_recipe: DataRecipe
    constitutional_decision: Mapping[str, Any]
    builder_tickets: tuple[BuilderTicket, ...]
    experiment_outcome: ExperimentRecord | None
    prediction_error: Mapping[str, float]
    interpretation: str
    next_action: str

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> ResearchEpisode:
        outcome = data.get("experiment_outcome")
        return cls(
            episode_id=str(data["episode_id"]),
            schema_version=str(data["schema_version"]),
            learner_snapshot=LearnerSnapshot.from_jsonable(data["learner_snapshot"]),
            scout_findings=dict(data["scout_findings"]),
            hypothesis_set=tuple(ResearchHypothesis.from_jsonable(item) for item in data["hypothesis_set"]),
            proposed_recipe=DataRecipe.from_jsonable(data["proposed_recipe"]),
            rat_critique=dict(data["rat_critique"]),
            repaired_recipe=DataRecipe.from_jsonable(data["repaired_recipe"]),
            constitutional_decision=dict(data["constitutional_decision"]),
            builder_tickets=tuple(BuilderTicket.from_jsonable(item) for item in data["builder_tickets"]),
            experiment_outcome=ExperimentRecord.from_jsonable(outcome) if outcome else None,
            prediction_error={str(k): float(v) for k, v in data.get("prediction_error", {}).items()},
            interpretation=str(data["interpretation"]),
            next_action=str(data["next_action"]),
        )
