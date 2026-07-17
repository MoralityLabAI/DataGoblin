from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from datagoblin.serialization import to_jsonable


class EvidenceType(str, Enum):
    ENVIRONMENT_VERIFIED = "ENVIRONMENT_VERIFIED"
    INDEPENDENT_HARNESS = "INDEPENDENT_HARNESS"
    HUMAN_REVIEWED = "HUMAN_REVIEWED"
    MODEL_INFERRED = "MODEL_INFERRED"
    EXPERIENCE_INFERRED = "EXPERIENCE_INFERRED"
    SIMULATED = "SIMULATED"
    UNKNOWN = "UNKNOWN"

    @property
    def may_hard_promote(self) -> bool:
        return self in {self.ENVIRONMENT_VERIFIED, self.INDEPENDENT_HARNESS}


@dataclass(frozen=True)
class EvidenceClaim:
    evidence_type: EvidenceType
    source_component: str
    artifact_refs: tuple[str, ...] = ()
    asserted_by: str = ""
    verified_by: str | None = None
    note: str = ""

    @property
    def hard_authoritative(self) -> bool:
        return self.evidence_type.may_hard_promote and bool(self.verified_by)

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> EvidenceClaim:
        return cls(
            evidence_type=EvidenceType(data["evidence_type"]),
            source_component=str(data["source_component"]),
            artifact_refs=tuple(str(item) for item in data.get("artifact_refs", ())),
            asserted_by=str(data.get("asserted_by", "")),
            verified_by=str(data["verified_by"]) if data.get("verified_by") else None,
            note=str(data.get("note", "")),
        )
