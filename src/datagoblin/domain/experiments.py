from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from datagoblin.serialization import to_jsonable

from .evidence import EvidenceType


class LifecycleStatus(str, Enum):
    DRAFT = "DRAFT"
    CRITIQUED = "CRITIQUED"
    VALIDATED = "VALIDATED"
    TICKETED = "TICKETED"
    BUILT = "BUILT"
    EXECUTED = "EXECUTED"
    EVALUATED = "EVALUATED"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    schema_version: str
    snapshot_hash: str
    hypothesis_hashes: tuple[str, ...]
    recipe_hash: str
    constitution_hash: str
    builder_ticket_hashes: tuple[str, ...]
    build_artifact_hashes: tuple[str, ...]
    generator_hash: str
    solver_hash: str
    verifier_hash: str
    learner_hash: str
    train_split_manifest: Mapping[str, Any]
    evaluation_split_manifest: Mapping[str, Any]
    predicted_outcomes: tuple[Mapping[str, Any], ...]
    actual_metrics: Mapping[str, float]
    run_status: LifecycleStatus
    violations: tuple[str, ...]
    resource_use: Mapping[str, float]
    seeds: Mapping[str, tuple[int, ...]]
    errors: tuple[str, ...]
    interpretation: str
    next_experiment_suggestions: tuple[str, ...]
    evidence_type: EvidenceType
    claim_boundary: str
    timestamps: Mapping[str, str]

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> ExperimentRecord:
        return cls(
            experiment_id=str(data["experiment_id"]),
            schema_version=str(data["schema_version"]),
            snapshot_hash=str(data["snapshot_hash"]),
            hypothesis_hashes=tuple(str(item) for item in data["hypothesis_hashes"]),
            recipe_hash=str(data["recipe_hash"]),
            constitution_hash=str(data["constitution_hash"]),
            builder_ticket_hashes=tuple(str(item) for item in data["builder_ticket_hashes"]),
            build_artifact_hashes=tuple(str(item) for item in data["build_artifact_hashes"]),
            generator_hash=str(data["generator_hash"]),
            solver_hash=str(data["solver_hash"]),
            verifier_hash=str(data["verifier_hash"]),
            learner_hash=str(data["learner_hash"]),
            train_split_manifest=dict(data["train_split_manifest"]),
            evaluation_split_manifest=dict(data["evaluation_split_manifest"]),
            predicted_outcomes=tuple(dict(item) for item in data["predicted_outcomes"]),
            actual_metrics={str(k): float(v) for k, v in data["actual_metrics"].items()},
            run_status=LifecycleStatus(data["run_status"]),
            violations=tuple(str(item) for item in data.get("violations", ())),
            resource_use={str(k): float(v) for k, v in data["resource_use"].items()},
            seeds={str(k): tuple(int(x) for x in v) for k, v in data["seeds"].items()},
            errors=tuple(str(item) for item in data.get("errors", ())),
            interpretation=str(data["interpretation"]),
            next_experiment_suggestions=tuple(str(item) for item in data["next_experiment_suggestions"]),
            evidence_type=EvidenceType(data["evidence_type"]),
            claim_boundary=str(data["claim_boundary"]),
            timestamps={str(k): str(v) for k, v in data["timestamps"].items()},
        )
