from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from datagoblin.serialization import to_jsonable


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class RuleViolation:
    rule_id: str
    severity: Severity
    message: str
    evidence_references: tuple[str, ...]
    minimal_repair: str

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


RULE_MESSAGES = {
    "evaluation_leakage": "Evaluation-only fields or seeds are exposed to the proposer or builder.",
    "seed_namespace_overlap": "Training and evaluation seed namespaces or generator families overlap.",
    "missing_independent_verifier": "No independent verifier is declared.",
    "builder_evaluator_identity": "The proposer/builder and verifier are the same component.",
    "unbounded_budget": "The resource budget is missing, nonpositive, or above policy bounds.",
    "unnecessary_network": "Network access is requested without an explicit approved justification.",
    "unnecessary_subprocess": "Builder subprocess access is requested without an explicit justification.",
    "missing_falsifier": "The recipe has no falsification criterion.",
    "confounded_ablation": "An ablation changes multiple unrelated variables.",
    "trajectory_mixture_invalid": "Trajectory-source shares do not sum to one.",
    "uncertified_task": "Executable tasks lack solver/invariant certification or an uncertainty label.",
    "unsupported_expected_gain": "An expected gain is exact or lacks evidence and uncertainty.",
    "metric_mutated_after_freeze": "The success metric changed after outcome inspection.",
    "curriculum_diversity_collapse": "The curriculum duplicates or collapses its declared diversity.",
    "cherry_picked_seeds": "The evaluation seed list was selected after or by outcome.",
    "hidden_result_exposed": "A hidden result or metric is exposed in builder-facing material.",
    "builder_privilege_violation": "The builder can modify the evaluator or ledger.",
    "self_certified_environment_evidence": "A proposer self-authored environment-verified claim.",
    "label_provenance_missing": "A label source has unknown or undeclared provenance.",
    "held_out_measurement_missing": "Capability gain is not measured on a held-out split.",
    "control_or_ablation_missing": "No control or ablation is declared.",
    "architecture_alternatives_missing": "An architectural conclusion omits data, optimization, and compute controls.",
    "artifact_provenance_missing": "Required provenance or artifact hash declarations are absent.",
    "failed_seed_exclusion": "Failed seeds may be silently excluded.",
    "unsafe_mechanics": "The proposal requests mechanics outside the harmless v0 scope.",
}


def violation(
    rule_id: str,
    repair: str,
    *,
    severity: Severity = Severity.ERROR,
    evidence: tuple[str, ...] = (),
) -> RuleViolation:
    return RuleViolation(rule_id, severity, RULE_MESSAGES[rule_id], evidence, repair)
