from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datagoblin.serialization import to_jsonable

from .evidence import EvidenceType


@dataclass(frozen=True)
class CapabilityObservation:
    capability_id: str
    evaluation_family: str
    point_estimate: float
    uncertainty_interval: tuple[float, float] | None
    standard_error: float | None
    sample_count: int
    evidence_type: EvidenceType
    artifact_refs: tuple[str, ...] = ()

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> CapabilityObservation:
        interval = data.get("uncertainty_interval")
        return cls(
            capability_id=str(data["capability_id"]),
            evaluation_family=str(data["evaluation_family"]),
            point_estimate=float(data["point_estimate"]),
            uncertainty_interval=(float(interval[0]), float(interval[1])) if interval else None,
            standard_error=float(data["standard_error"]) if data.get("standard_error") is not None else None,
            sample_count=int(data["sample_count"]),
            evidence_type=EvidenceType(data["evidence_type"]),
            artifact_refs=tuple(str(item) for item in data.get("artifact_refs", ())),
        )


@dataclass(frozen=True)
class FailureCluster:
    cluster_id: str
    observable_pattern: str
    affected_capabilities: tuple[str, ...]
    example_refs: tuple[str, ...]
    competing_explanations: tuple[str, ...]
    severity: float
    confidence: float
    evidence_type: EvidenceType

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> FailureCluster:
        return cls(
            cluster_id=str(data["cluster_id"]),
            observable_pattern=str(data["observable_pattern"]),
            affected_capabilities=tuple(str(item) for item in data["affected_capabilities"]),
            example_refs=tuple(str(item) for item in data.get("example_refs", ())),
            competing_explanations=tuple(str(item) for item in data["competing_explanations"]),
            severity=float(data["severity"]),
            confidence=float(data["confidence"]),
            evidence_type=EvidenceType(data["evidence_type"]),
        )
