from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datagoblin.serialization import to_jsonable


@dataclass(frozen=True)
class ResourceBudget:
    cpu_seconds: int
    memory_mb: int
    storage_mb: int
    wall_clock_seconds: int
    process_count: int

    @property
    def bounded(self) -> bool:
        return all(value > 0 for value in self.to_jsonable().values())

    def to_jsonable(self) -> dict[str, int]:
        return {
            "cpu_seconds": self.cpu_seconds,
            "memory_mb": self.memory_mb,
            "storage_mb": self.storage_mb,
            "wall_clock_seconds": self.wall_clock_seconds,
            "process_count": self.process_count,
        }

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> ResourceBudget:
        return cls(*(int(data[name]) for name in cls.__dataclass_fields__))


_HIDDEN_MARKERS = (
    "hidden_evaluation_seed",
    "private_evaluation",
    "expected_answer",
    "hidden_result",
)


@dataclass(frozen=True)
class BuilderTicket:
    ticket_id: str
    experiment_id: str
    component_type: str
    purpose: str
    required_inputs: tuple[str, ...]
    declared_outputs: tuple[str, ...]
    interface_contract: Mapping[str, Any]
    invariants: tuple[str, ...]
    property_tests: tuple[str, ...]
    dependencies: tuple[str, ...]
    allowed_filesystem_roots: tuple[str, ...]
    forbidden_filesystem_roots: tuple[str, ...]
    network_permission: bool
    subprocess_permission: bool
    resource_budget: ResourceBudget
    acceptance_criteria: tuple[str, ...]
    expected_artifact_manifest: tuple[str, ...]
    evaluator_separation_requirements: tuple[str, ...]

    def __post_init__(self) -> None:
        exposed = " ".join((*self.required_inputs, *self.declared_outputs)).lower()
        if any(marker in exposed for marker in _HIDDEN_MARKERS):
            raise ValueError("builder ticket exposes hidden evaluation material")

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> BuilderTicket:
        return cls(
            ticket_id=str(data["ticket_id"]),
            experiment_id=str(data["experiment_id"]),
            component_type=str(data["component_type"]),
            purpose=str(data["purpose"]),
            required_inputs=tuple(str(item) for item in data["required_inputs"]),
            declared_outputs=tuple(str(item) for item in data["declared_outputs"]),
            interface_contract=dict(data["interface_contract"]),
            invariants=tuple(str(item) for item in data["invariants"]),
            property_tests=tuple(str(item) for item in data["property_tests"]),
            dependencies=tuple(str(item) for item in data.get("dependencies", ())),
            allowed_filesystem_roots=tuple(str(item) for item in data["allowed_filesystem_roots"]),
            forbidden_filesystem_roots=tuple(str(item) for item in data["forbidden_filesystem_roots"]),
            network_permission=bool(data["network_permission"]),
            subprocess_permission=bool(data["subprocess_permission"]),
            resource_budget=ResourceBudget.from_jsonable(data["resource_budget"]),
            acceptance_criteria=tuple(str(item) for item in data["acceptance_criteria"]),
            expected_artifact_manifest=tuple(str(item) for item in data["expected_artifact_manifest"]),
            evaluator_separation_requirements=tuple(str(item) for item in data["evaluator_separation_requirements"]),
        )
