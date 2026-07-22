from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datagoblin.constitution.validator import ConstitutionalDecision, ConstitutionalValidator
from datagoblin.domain.builders import BuilderTicket, ResourceBudget
from datagoblin.domain.evidence import EvidenceClaim, EvidenceType
from datagoblin.domain.experiments import LifecycleStatus
from datagoblin.domain.hypotheses import ResearchHypothesis
from datagoblin.domain.learner import LearnerSnapshot
from datagoblin.domain.recipes import DataRecipe
from datagoblin.hashing import sha256_json
from datagoblin.ids import content_id

from .backend import GoblinBackend
from .membrane import MembraneDecision, ProposalAction, ResearchMembrane, ResearchProposal
from .rat import Rat, RatCritique
from .scout import ScoutFindings
from .state import ResearchState, TransitionAuthority


@dataclass(frozen=True)
class OrchestrationResult:
    experiment_id: str
    scout_findings: ScoutFindings
    hypotheses: tuple[ResearchHypothesis, ...]
    original_recipe: DataRecipe
    rat_critique: RatCritique
    repaired_recipe: DataRecipe
    constitutional_decision: ConstitutionalDecision
    tickets: tuple[BuilderTicket, ...]
    research_state: ResearchState
    membrane_decisions: tuple[MembraneDecision, ...]


class ResearchOrchestrator:
    def __init__(self, backend: GoblinBackend, validator: ConstitutionalValidator | None = None) -> None:
        self.backend = backend
        self.validator = validator or ConstitutionalValidator()
        self.rat = Rat()
        self.membrane = ResearchMembrane()

    def propose(self, snapshot: LearnerSnapshot) -> OrchestrationResult:
        state = ResearchState(
            state_id=content_id("state", snapshot.to_jsonable()),
            lifecycle_status=LifecycleStatus.DRAFT,
            candidate_hypothesis_ids=(),
            frozen_evaluation_hash=None,
            public_payload={"learner_snapshot": snapshot.to_jsonable()},
        )
        proposal = self.backend.propose(state)
        critique, repaired = self.rat.critique_and_repair(proposal.recipe)
        decisions: list[MembraneDecision] = []
        state = self._advance(
            state,
            ResearchProposal(
                ProposalAction.NARROW_HYPOTHESES,
                {
                    "hypothesis_ids": [item.hypothesis_id for item in proposal.hypotheses],
                    "rat_critique_hash": sha256_json(critique),
                    "repaired_recipe_hash": sha256_json(repaired),
                },
                EvidenceClaim(EvidenceType.MODEL_INFERRED, "rat", asserted_by="orchestrator"),
                LifecycleStatus.CRITIQUED,
            ),
            TransitionAuthority.ORCHESTRATOR,
            decisions,
        )
        experiment_id = content_id("experiment", {"snapshot": snapshot.snapshot_id, "recipe": repaired.to_jsonable()})
        tickets = emit_builder_tickets(experiment_id, repaired)
        decision = self.validator.validate(repaired, proposal.hypotheses, tickets)
        if decision.execution_permitted:
            state = self._advance(
                state,
                ResearchProposal(
                    ProposalAction.REQUEST_EXPERIMENT,
                    {
                        "constitutional_decision_receipt": decision.decision_receipt_hash,
                        "frozen_evaluation_hash": sha256_json(repaired.split_contract),
                    },
                    EvidenceClaim(EvidenceType.SIMULATED, "constitutional_validator", asserted_by="orchestrator"),
                    LifecycleStatus.VALIDATED,
                ),
                TransitionAuthority.ORCHESTRATOR,
                decisions,
            )
            state = self._advance(
                state,
                ResearchProposal(
                    ProposalAction.REQUEST_EXPERIMENT,
                    {"builder_ticket_ids": [ticket.ticket_id for ticket in tickets]},
                    EvidenceClaim(EvidenceType.SIMULATED, "ticket_writer", asserted_by="orchestrator"),
                    LifecycleStatus.TICKETED,
                ),
                TransitionAuthority.ORCHESTRATOR,
                decisions,
            )
        return OrchestrationResult(
            experiment_id,
            proposal.scout_findings,
            proposal.hypotheses,
            proposal.recipe,
            critique,
            repaired,
            decision,
            tickets,
            state,
            tuple(decisions),
        )

    def _advance(
        self,
        state: ResearchState,
        proposal: ResearchProposal,
        authority: TransitionAuthority,
        decisions: list[MembraneDecision],
    ) -> ResearchState:
        decision = self.membrane.apply(state, proposal, authority=authority)
        decisions.append(decision)
        if not decision.accepted:
            raise RuntimeError(decision.reason)
        return decision.after


def emit_builder_tickets(experiment_id: str, recipe: DataRecipe) -> tuple[BuilderTicket, ...]:
    budget = ResourceBudget.from_jsonable(recipe.resource_budget)
    shared: dict[str, Any] = {
        "experiment_id": experiment_id,
        "allowed_filesystem_roots": ("assigned_build_root",),
        "forbidden_filesystem_roots": ("ledger", "evaluator", "private_evaluation"),
        "network_permission": False,
        "subprocess_permission": False,
        "resource_budget": budget,
        "dependencies": ("numpy",),
        "evaluator_separation_requirements": (
            "Builder receives no hidden evaluation seeds, answers, metrics, or evaluator source.",
        ),
    }
    recipe_hash = sha256_json(recipe.to_jsonable())
    generator = BuilderTicket(
        ticket_id=content_id("ticket", {"experiment": experiment_id, "component": "generator"}),
        component_type="environment_generator",
        purpose="Materialize the trusted causal_switches_v1 training generator and public split manifest.",
        required_inputs=(f"recipe_sha256:{recipe_hash}", "public_split_contract"),
        declared_outputs=(
            "train_dataset.npz",
            "train_split_manifest.json",
            "provenance_manifest.json",
        ),
        interface_contract={"environment": "causal_switches_v1", "schema_version": "1.0"},
        invariants=("deterministic_for_seed", "labels_from_exact_solver", "no_eval_namespace"),
        property_tests=("counterfactual_pair_consistency", "role_permutation_coverage"),
        acceptance_criteria=("All property tests pass.", "Every output hash is recorded."),
        expected_artifact_manifest=(
            "train_dataset.npz",
            "train_split_manifest.json",
            "provenance_manifest.json",
        ),
        **shared,
    )
    solver = BuilderTicket(
        ticket_id=content_id("ticket", {"experiment": experiment_id, "component": "solver"}),
        component_type="reference_solver",
        purpose="Declare the trusted exact label solver interface used by the environment owner.",
        required_inputs=(f"recipe_sha256:{recipe_hash}", "environment_public_spec"),
        declared_outputs=("solver_contract.json", "solver_property_tests.json"),
        interface_contract={"solver": "causal_switches_exact_solver", "authority": "environment"},
        invariants=("solver_does_not_call_model", "solver_is_deterministic"),
        property_tests=("transition_replay_matches_label",),
        acceptance_criteria=("Solver tests pass on builder-visible generated cases.",),
        expected_artifact_manifest=("solver_contract.json", "solver_property_tests.json"),
        **shared,
    )
    return generator, solver
