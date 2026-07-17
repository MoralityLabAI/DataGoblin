from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datagoblin.serialization import to_jsonable

from .capabilities import CapabilityObservation, FailureCluster
from .evidence import EvidenceClaim


@dataclass(frozen=True)
class LearnerSnapshot:
    snapshot_id: str
    schema_version: str
    learner_id: str
    architecture_family: str
    parameter_count: int
    checkpoint_ref: str
    training_compute_used: Mapping[str, float]
    capability_observations: tuple[CapabilityObservation, ...]
    failure_clusters: tuple[FailureCluster, ...]
    calibration_observations: Mapping[str, float]
    representation_diagnostics: Mapping[str, Any]
    current_data_inventory: Mapping[str, int]
    current_objective_mixture: Mapping[str, float]
    recent_experiment_references: tuple[str, ...]
    available_training_budget: Mapping[str, float]
    provenance: EvidenceClaim
    created_at: str

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> LearnerSnapshot:
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            schema_version=str(data["schema_version"]),
            learner_id=str(data["learner_id"]),
            architecture_family=str(data["architecture_family"]),
            parameter_count=int(data["parameter_count"]),
            checkpoint_ref=str(data["checkpoint_ref"]),
            training_compute_used={str(k): float(v) for k, v in data["training_compute_used"].items()},
            capability_observations=tuple(
                CapabilityObservation.from_jsonable(item) for item in data["capability_observations"]
            ),
            failure_clusters=tuple(FailureCluster.from_jsonable(item) for item in data["failure_clusters"]),
            calibration_observations={str(k): float(v) for k, v in data["calibration_observations"].items()},
            representation_diagnostics=dict(data["representation_diagnostics"]),
            current_data_inventory={str(k): int(v) for k, v in data["current_data_inventory"].items()},
            current_objective_mixture={str(k): float(v) for k, v in data["current_objective_mixture"].items()},
            recent_experiment_references=tuple(str(item) for item in data.get("recent_experiment_references", ())),
            available_training_budget={str(k): float(v) for k, v in data["available_training_budget"].items()},
            provenance=EvidenceClaim.from_jsonable(data["provenance"]),
            created_at=str(data["created_at"]),
        )
