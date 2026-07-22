from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from datagoblin.domain.builders import BuilderTicket
from datagoblin.domain.evidence import EvidenceType
from datagoblin.domain.hypotheses import ResearchHypothesis
from datagoblin.domain.recipes import DataRecipe
from datagoblin.hashing import sha256_json
from datagoblin.serialization import to_jsonable

from .charter import Charter
from .rules import RuleViolation, Severity, violation


class DecisionKind(str, Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    AUDIT = "AUDIT"
    ABSTAIN = "ABSTAIN"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ConstitutionalDecision:
    decision: DecisionKind
    constitution_version: str
    constitution_hash: str
    recipe_hash: str
    hypothesis_hashes: tuple[str, ...]
    ticket_hashes: tuple[str, ...]
    subject_hash: str
    triggered_rules: tuple[RuleViolation, ...]
    violation_severity: Severity | None
    evidence_references: tuple[str, ...]
    minimal_repairs: tuple[str, ...]
    execution_permitted: bool
    decision_receipt_hash: str

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


class ConstitutionalValidator:
    def __init__(self, charter: Charter | None = None) -> None:
        self.charter = charter or Charter.load()
        defaults = self.charter.data["defaults"]
        self.tolerance = float(defaults["trajectory_share_tolerance"])
        self.max_budget = {str(k): float(v) for k, v in defaults["maximum_budget"].items()}

    def validate(
        self,
        recipe: DataRecipe,
        hypotheses: Iterable[ResearchHypothesis] = (),
        tickets: Iterable[BuilderTicket] = (),
    ) -> ConstitutionalDecision:
        hypothesis_records = tuple(hypotheses)
        ticket_records = tuple(tickets)
        issues = list(self._recipe_issues(recipe, hypothesis_records))
        issues.extend(self._ticket_issues(ticket_records))
        severity = self._maximum_severity(issues)
        if any(item.severity == Severity.CRITICAL for item in issues):
            kind = DecisionKind.REJECT
        elif issues:
            kind = DecisionKind.REVISE
        else:
            kind = DecisionKind.ACCEPT
        recipe_hash = sha256_json(recipe)
        hypothesis_hashes = tuple(sha256_json(item) for item in hypothesis_records)
        ticket_hashes = tuple(sha256_json(item) for item in ticket_records)
        subject = {
            "recipe_hash": recipe_hash,
            "hypothesis_hashes": hypothesis_hashes,
            "ticket_hashes": ticket_hashes,
        }
        subject_hash = sha256_json(subject)
        evidence_references = tuple(dict.fromkeys(ref for item in issues for ref in item.evidence_references))
        minimal_repairs = tuple(dict.fromkeys(item.minimal_repair for item in issues))
        payload = {
            "decision": kind.value,
            "constitution_version": self.charter.version,
            "constitution_hash": self.charter.hash,
            "subject_hash": subject_hash,
            **subject,
            "triggered_rules": [item.to_jsonable() for item in issues],
            "evidence_references": evidence_references,
            "minimal_repairs": minimal_repairs,
            "execution_permitted": kind == DecisionKind.ACCEPT,
        }
        return ConstitutionalDecision(
            decision=kind,
            constitution_version=self.charter.version,
            constitution_hash=self.charter.hash,
            recipe_hash=recipe_hash,
            hypothesis_hashes=hypothesis_hashes,
            ticket_hashes=ticket_hashes,
            subject_hash=subject_hash,
            triggered_rules=tuple(issues),
            violation_severity=severity,
            evidence_references=evidence_references,
            minimal_repairs=minimal_repairs,
            execution_permitted=kind == DecisionKind.ACCEPT,
            decision_receipt_hash=sha256_json(payload),
        )

    def _recipe_issues(self, recipe: DataRecipe, hypotheses: tuple[ResearchHypothesis, ...]) -> Iterable[RuleViolation]:
        expected_schema_version = str(self.charter.data["schema_version"])
        if recipe.schema_version != expected_schema_version:
            yield violation(
                "unsupported_recipe_schema_version",
                f"Use recipe schema version {expected_schema_version}.",
                severity=Severity.CRITICAL,
            )
        if recipe.constitution_version != self.charter.version:
            yield violation(
                "constitution_version_mismatch",
                f"Revalidate the recipe against constitution version {self.charter.version}.",
                severity=Severity.CRITICAL,
            )
        split = recipe.split_contract
        train_ns = str(split.get("train_seed_namespace", ""))
        eval_ns = str(split.get("evaluation_seed_namespace", ""))
        exposed = {str(item).lower() for item in split.get("builder_exposed_fields", [])}
        if split.get("evaluation_seeds_exposed") or any(
            "hidden" in item or "evaluation_seed" in item or "expected_answer" in item for item in exposed
        ):
            yield violation(
                "evaluation_leakage",
                "Remove evaluation seeds and hidden fields from builder inputs.",
                severity=Severity.CRITICAL,
            )
        train_family = split.get("train_generator_family")
        eval_family = split.get("evaluation_generator_family")
        if not train_ns or not eval_ns or train_ns == eval_ns or (train_family and train_family == eval_family):
            yield violation(
                "seed_namespace_overlap",
                "Use disjoint namespaces and generator families for train and evaluation.",
            )

        verifier = recipe.independent_verifier_requirements
        if not verifier.get("required", False) or not verifier.get("component"):
            yield violation(
                "missing_independent_verifier",
                "Declare a required independent verifier component.",
                severity=Severity.CRITICAL,
            )
        proposer = str(recipe.provenance.asserted_by or recipe.provenance.source_component)
        verifier_component = str(verifier.get("component", ""))
        builder_component = str(recipe.builder_permissions.get("component", "template_builder"))
        if verifier_component and verifier_component in {proposer, builder_component}:
            yield violation(
                "builder_evaluator_identity",
                "Assign a verifier owned by a distinct component.",
                severity=Severity.CRITICAL,
            )

        budget = recipe.resource_budget
        if not self._bounded_budget(budget):
            yield violation(
                "unbounded_budget",
                "Declare finite positive resource limits within constitution caps.",
                severity=Severity.CRITICAL,
            )
        if recipe.builder_permissions.get("network", False) and not recipe.builder_permissions.get(
            "network_justification"
        ):
            yield violation(
                "unnecessary_network",
                "Deny network access or add a reviewed necessity justification.",
            )
        if recipe.builder_permissions.get("subprocess", False) and not recipe.builder_permissions.get(
            "subprocess_justification"
        ):
            yield violation(
                "unnecessary_subprocess",
                "Deny subprocess access or add a reviewed necessity justification.",
            )
        if recipe.builder_permissions.get("may_modify_ledger") or recipe.builder_permissions.get(
            "may_modify_evaluator"
        ):
            yield violation(
                "builder_privilege_violation",
                "Remove ledger and evaluator write permissions.",
                severity=Severity.CRITICAL,
            )

        hypothesis_map = {item.hypothesis_id: item for item in hypotheses}
        if not recipe.falsification_criteria or any(
            item in hypothesis_map and not hypothesis_map[item].falsification_conditions
            for item in recipe.hypothesis_ids
        ):
            yield violation(
                "missing_falsifier",
                "Add an observable condition that would falsify each hypothesis.",
            )
        if not recipe.controls and not recipe.ablations:
            yield violation(
                "control_or_ablation_missing",
                "Add at least one matched control or single-variable ablation.",
            )
        for item in recipe.ablations:
            changed = item.get("changed_variables", [])
            if item.get("confounded") or (len(changed) > 1 and not item.get("joint_change_justification")):
                yield violation("confounded_ablation", "Split the ablation into one-variable contrasts.")
                break
        if not math.isclose(recipe.trajectory_share_total, 1.0, abs_tol=self.tolerance):
            yield violation("trajectory_mixture_invalid", "Normalize trajectory shares to sum to one.")

        solver = recipe.reference_solver_requirements
        task_kind = str(recipe.environment_specification.get("task_kind", "executable"))
        invariant_certificate = bool(recipe.environment_specification.get("invariant_certificate"))
        if (
            task_kind not in {"impossibility", "uncertainty"}
            and not solver.get("required")
            and not invariant_certificate
        ):
            yield violation(
                "uncertified_task",
                "Require an exact solver, invariant certificate, or uncertainty label.",
                severity=Severity.CRITICAL,
            )
        for outcome in recipe.predicted_outcomes:
            unsupported = outcome.evidence_type == EvidenceType.UNKNOWN or (
                outcome.lower == outcome.upper and not outcome.evidence_refs
            )
            if unsupported:
                yield violation(
                    "unsupported_expected_gain",
                    "Use a bounded interval with confidence and evidence or mark UNKNOWN.",
                )
                break
        if split.get("metric_changed_after_run"):
            yield violation(
                "metric_mutated_after_freeze",
                "Restore the preregistered metric and audit the run.",
                severity=Severity.CRITICAL,
            )
        if (
            recipe.curriculum_controller.get("diversity_collapsed")
            or recipe.curriculum_controller.get("unique_fraction", 1.0) < 0.5
        ):
            yield violation("curriculum_diversity_collapse", "Restore role, delay, and composition diversity.")
        if split.get("cherry_picked_seeds") or split.get("exclude_failed_seeds"):
            code = "cherry_picked_seeds" if split.get("cherry_picked_seeds") else "failed_seed_exclusion"
            yield violation(
                code,
                "Use the frozen seed namespace and report every attempted seed.",
                severity=Severity.CRITICAL,
            )
        if split.get("hidden_result_exposed") or recipe.builder_permissions.get("hidden_metric_access"):
            yield violation(
                "hidden_result_exposed",
                "Remove hidden metrics and results from builder-facing fields.",
                severity=Severity.CRITICAL,
            )

        if recipe.provenance.evidence_type == EvidenceType.ENVIRONMENT_VERIFIED and (
            not recipe.provenance.verified_by or recipe.provenance.verified_by == proposer
        ):
            yield violation(
                "self_certified_environment_evidence",
                "Downgrade to MODEL_INFERRED or attach an independent environment receipt.",
                severity=Severity.CRITICAL,
            )
        if any(item.label_provenance == EvidenceType.UNKNOWN for item in recipe.trajectory_source_mixture):
            yield violation("label_provenance_missing", "Declare the generating authority for every label.")
        if not split.get("held_out", False):
            yield violation(
                "held_out_measurement_missing",
                "Freeze a held-out evaluation family before execution.",
            )
        if recipe.metadata.get("architectural_conclusion") and not recipe.metadata.get("matched_alternatives"):
            yield violation(
                "architecture_alternatives_missing",
                "Add matched data, optimization, and compute alternatives.",
            )
        if not recipe.provenance.artifact_refs or not recipe.claim_boundary:
            yield violation(
                "artifact_provenance_missing",
                "Attach source artifact references and an explicit claim boundary.",
            )
        if recipe.environment_specification.get("unsafe_mechanics"):
            yield violation(
                "unsafe_mechanics",
                "Restrict mechanics to harmless synthetic tasks.",
                severity=Severity.CRITICAL,
            )

    def _ticket_issues(self, tickets: tuple[BuilderTicket, ...]) -> Iterable[RuleViolation]:
        for ticket in tickets:
            if ticket.network_permission:
                yield violation("unnecessary_network", f"Deny network on ticket {ticket.ticket_id}.")
            if not ticket.resource_budget.bounded:
                yield violation(
                    "unbounded_budget",
                    f"Bound all resources on ticket {ticket.ticket_id}.",
                    severity=Severity.CRITICAL,
                )
            roots = " ".join(ticket.allowed_filesystem_roots).lower()
            if "ledger" in roots or "evaluator" in roots or "private" in roots:
                yield violation(
                    "builder_privilege_violation",
                    f"Remove privileged roots from ticket {ticket.ticket_id}.",
                    severity=Severity.CRITICAL,
                )

    def _bounded_budget(self, budget: Mapping[str, Any]) -> bool:
        for name, maximum in self.max_budget.items():
            try:
                value = float(budget[name])
            except (KeyError, TypeError, ValueError):
                return False
            if not math.isfinite(value) or value <= 0 or value > maximum:
                return False
        return True

    @staticmethod
    def _maximum_severity(issues: list[RuleViolation]) -> Severity | None:
        if not issues:
            return None
        order = {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2, Severity.CRITICAL: 3}
        return max((item.severity for item in issues), key=order.__getitem__)
