from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datagoblin.serialization import to_jsonable

from .evidence import EvidenceClaim, EvidenceType


@dataclass(frozen=True)
class TrajectorySource:
    source: str
    share: float
    label_provenance: EvidenceType

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> TrajectorySource:
        return cls(str(data["source"]), float(data["share"]), EvidenceType(data["label_provenance"]))


@dataclass(frozen=True)
class PredictedOutcome:
    metric: str
    lower: float
    upper: float
    confidence: float
    evidence_type: EvidenceType
    evidence_refs: tuple[str, ...] = ()

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> PredictedOutcome:
        return cls(
            metric=str(data["metric"]),
            lower=float(data["lower"]),
            upper=float(data["upper"]),
            confidence=float(data["confidence"]),
            evidence_type=EvidenceType(data["evidence_type"]),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", ())),
        )


@dataclass(frozen=True)
class DataRecipe:
    recipe_id: str
    schema_version: str
    title: str
    hypothesis_ids: tuple[str, ...]
    target_capabilities: tuple[str, ...]
    environment_specification: Mapping[str, Any]
    generator_specification: Mapping[str, Any]
    trajectory_source_mixture: tuple[TrajectorySource, ...]
    supervision_objectives: tuple[Mapping[str, Any], ...]
    curriculum_controller: Mapping[str, Any]
    split_contract: Mapping[str, Any]
    counterfactual_and_symmetry_transforms: tuple[str, ...]
    reference_solver_requirements: Mapping[str, Any]
    independent_verifier_requirements: Mapping[str, Any]
    builder_permissions: Mapping[str, Any]
    resource_budget: Mapping[str, Any]
    ablations: tuple[Mapping[str, Any], ...]
    controls: tuple[Mapping[str, Any], ...]
    success_criteria: tuple[str, ...]
    falsification_criteria: tuple[str, ...]
    predicted_outcomes: tuple[PredictedOutcome, ...]
    uncertainty: Mapping[str, Any]
    provenance: EvidenceClaim
    constitution_version: str
    claim_boundary: str
    metadata: Mapping[str, Any]

    @property
    def trajectory_share_total(self) -> float:
        return sum(item.share for item in self.trajectory_source_mixture)

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> DataRecipe:
        return cls(
            recipe_id=str(data["recipe_id"]),
            schema_version=str(data["schema_version"]),
            title=str(data["title"]),
            hypothesis_ids=tuple(str(item) for item in data["hypothesis_ids"]),
            target_capabilities=tuple(str(item) for item in data["target_capabilities"]),
            environment_specification=dict(data["environment_specification"]),
            generator_specification=dict(data["generator_specification"]),
            trajectory_source_mixture=tuple(
                TrajectorySource.from_jsonable(item) for item in data["trajectory_source_mixture"]
            ),
            supervision_objectives=tuple(dict(item) for item in data["supervision_objectives"]),
            curriculum_controller=dict(data["curriculum_controller"]),
            split_contract=dict(data["split_contract"]),
            counterfactual_and_symmetry_transforms=tuple(
                str(item) for item in data["counterfactual_and_symmetry_transforms"]
            ),
            reference_solver_requirements=dict(data["reference_solver_requirements"]),
            independent_verifier_requirements=dict(data["independent_verifier_requirements"]),
            builder_permissions=dict(data["builder_permissions"]),
            resource_budget=dict(data["resource_budget"]),
            ablations=tuple(dict(item) for item in data.get("ablations", ())),
            controls=tuple(dict(item) for item in data.get("controls", ())),
            success_criteria=tuple(str(item) for item in data["success_criteria"]),
            falsification_criteria=tuple(str(item) for item in data["falsification_criteria"]),
            predicted_outcomes=tuple(PredictedOutcome.from_jsonable(item) for item in data["predicted_outcomes"]),
            uncertainty=dict(data["uncertainty"]),
            provenance=EvidenceClaim.from_jsonable(data["provenance"]),
            constitution_version=str(data["constitution_version"]),
            claim_boundary=str(data["claim_boundary"]),
            metadata=dict(data.get("metadata", {})),
        )
