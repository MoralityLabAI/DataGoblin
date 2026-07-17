from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from datagoblin.domain.recipes import DataRecipe
from datagoblin.serialization import to_jsonable


@dataclass(frozen=True)
class RatCritique:
    constitutional_violations: tuple[str, ...]
    leakage_paths: tuple[str, ...]
    shortcut_opportunities: tuple[str, ...]
    confounds: tuple[str, ...]
    unverifiable_labels: tuple[str, ...]
    metric_gaming_opportunities: tuple[str, ...]
    unnecessary_permissions: tuple[str, ...]
    missing_negative_result_interpretation: tuple[str, ...]
    minimal_repairs: tuple[str, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


class Rat:
    def critique_and_repair(self, recipe: DataRecipe) -> tuple[RatCritique, DataRecipe]:
        permissions = dict(recipe.builder_permissions)
        unnecessary = ()
        repairs = []
        violations = []
        if permissions.get("network"):
            unnecessary = ("Builder requests network access despite a fully local task.",)
            violations.append("unnecessary_network")
            repairs.append("Deny builder network access.")
            permissions["network"] = False
        metadata = dict(recipe.metadata)
        metadata["negative_result_interpretation"] = (
            "A null result weakens the shortcut-only explanation; "
            "a history-feature interaction supports the capacity explanation."
        )
        critique = RatCritique(
            constitutional_violations=tuple(violations),
            leakage_paths=(
                "Evaluation generator must remain harness-owned; only its public hash and count may be disclosed.",
            ),
            shortcut_opportunities=(
                "Delay and intervention frequency could become replacement shortcuts if not stratified.",
            ),
            confounds=("Role permutation and delay width need separate ablations.",),
            unverifiable_labels=(),
            metric_gaming_opportunities=("Accuracy alone can hide calibration and subgroup failures.",),
            unnecessary_permissions=unnecessary,
            missing_negative_result_interpretation=("Original recipe did not specify how to read a null result.",),
            minimal_repairs=(*repairs, "Add explicit null-result interpretation."),
        )
        return critique, replace(recipe, builder_permissions=permissions, metadata=metadata)
