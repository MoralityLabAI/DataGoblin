from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datagoblin.serialization import to_jsonable


@dataclass(frozen=True)
class CausalSwitchesSpec:
    schema_version: str = "1.0"
    environment_id: str = "causal_switches_v1"
    object_count: int = 3
    maximum_delay: int = 4
    feature_names: tuple[str, ...] = (
        "color_red",
        "color_green",
        "color_blue",
        "intervention_seen",
        "elapsed_scaled",
        "delay_scaled",
        "bias",
    )
    claim_boundary: str = (
        "Toy finite mechanics for pipeline smoke testing; not evidence about real agents or model families."
    )

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> CausalSwitchesSpec:
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            environment_id=str(data.get("environment_id", "causal_switches_v1")),
            object_count=int(data.get("object_count", 3)),
            maximum_delay=int(data.get("maximum_delay", 4)),
            feature_names=tuple(str(item) for item in data.get("feature_names", cls.feature_names)),
            claim_boundary=str(data.get("claim_boundary", cls.claim_boundary)),
        )
