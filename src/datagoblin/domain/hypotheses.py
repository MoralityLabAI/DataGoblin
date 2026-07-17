from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from datagoblin.serialization import to_jsonable

from .evidence import EvidenceType


class HypothesisStatus(str, Enum):
    PROPOSED = "PROPOSED"
    ACTIVE = "ACTIVE"
    FALSIFIED = "FALSIFIED"
    SUPPORTED = "SUPPORTED"
    ABSTAINED = "ABSTAINED"


@dataclass(frozen=True)
class ResearchHypothesis:
    hypothesis_id: str
    statement: str
    mechanism: str
    supporting_observations: tuple[str, ...]
    contradicting_observations: tuple[str, ...]
    confidence: float
    competing_hypothesis_ids: tuple[str, ...]
    predicted_signatures: tuple[str, ...]
    falsification_conditions: tuple[str, ...]
    evidence_type: EvidenceType
    status: HypothesisStatus = HypothesisStatus.PROPOSED

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> ResearchHypothesis:
        return cls(
            hypothesis_id=str(data["hypothesis_id"]),
            statement=str(data["statement"]),
            mechanism=str(data["mechanism"]),
            supporting_observations=tuple(str(item) for item in data.get("supporting_observations", ())),
            contradicting_observations=tuple(str(item) for item in data.get("contradicting_observations", ())),
            confidence=float(data["confidence"]),
            competing_hypothesis_ids=tuple(str(item) for item in data.get("competing_hypothesis_ids", ())),
            predicted_signatures=tuple(str(item) for item in data["predicted_signatures"]),
            falsification_conditions=tuple(str(item) for item in data["falsification_conditions"]),
            evidence_type=EvidenceType(data["evidence_type"]),
            status=HypothesisStatus(data.get("status", HypothesisStatus.PROPOSED.value)),
        )
