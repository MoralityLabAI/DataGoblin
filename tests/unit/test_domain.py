from __future__ import annotations

import json
from pathlib import Path

import pytest

from datagoblin.constitution.validator import ConstitutionalValidator, DecisionKind
from datagoblin.domain.builders import BuilderTicket
from datagoblin.domain.evidence import EvidenceClaim, EvidenceType
from datagoblin.domain.experiments import LifecycleStatus
from datagoblin.domain.hypotheses import ResearchHypothesis
from datagoblin.domain.learner import LearnerSnapshot
from datagoblin.domain.recipes import DataRecipe
from datagoblin.hashing import sha256_json
from datagoblin.research.membrane import (
    EvidenceVerification,
    ProposalAction,
    ResearchMembrane,
    ResearchProposal,
)
from datagoblin.research.state import ResearchState, transition_status
from datagoblin.serialization import dump_structured, load_structured


def test_core_records_round_trip(snapshot, orchestration) -> None:
    assert LearnerSnapshot.from_jsonable(snapshot.to_jsonable()) == snapshot
    for hypothesis in orchestration.hypotheses:
        assert ResearchHypothesis.from_jsonable(hypothesis.to_jsonable()) == hypothesis
    assert DataRecipe.from_jsonable(orchestration.repaired_recipe.to_jsonable()) == orchestration.repaired_recipe
    for ticket in orchestration.tickets:
        assert BuilderTicket.from_jsonable(ticket.to_jsonable()) == ticket


def test_canonical_hash_is_stable() -> None:
    assert sha256_json({"b": 2, "a": [1, 3]}) == sha256_json({"a": [1, 3], "b": 2})


def test_yaml_and_json_loaders_agree(tmp_path: Path, good_recipe: DataRecipe) -> None:
    json_path = tmp_path / "recipe.json"
    yaml_path = tmp_path / "recipe.yaml"
    dump_structured(json_path, good_recipe.to_jsonable())
    dump_structured(yaml_path, good_recipe.to_jsonable())
    assert load_structured(json_path) == load_structured(yaml_path)


def test_json_loader_accepts_windows_utf8_bom_receipt(tmp_path: Path) -> None:
    path = tmp_path / "powershell-receipt.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"status": "completed"}).encode("utf-8"))
    assert load_structured(path) == {"status": "completed"}


def test_unknown_enum_fails(snapshot: LearnerSnapshot) -> None:
    value = snapshot.to_jsonable()
    value["provenance"]["evidence_type"] = "MADE_UP"
    with pytest.raises(ValueError):
        LearnerSnapshot.from_jsonable(value)


def test_trajectory_shares_and_required_controls(good_recipe: DataRecipe) -> None:
    value = good_recipe.to_jsonable()
    value["trajectory_source_mixture"][0]["share"] = 0.2
    decision = ConstitutionalValidator().validate(DataRecipe.from_jsonable(value))
    assert "trajectory_mixture_invalid" in {item.rule_id for item in decision.triggered_rules}
    assert good_recipe.falsification_criteria
    assert good_recipe.controls or good_recipe.ablations
    assert ConstitutionalValidator().validate(good_recipe).decision == DecisionKind.ACCEPT


def test_hidden_fields_cannot_enter_builder_ticket(orchestration) -> None:
    value = orchestration.tickets[0].to_jsonable()
    value["required_inputs"].append("hidden_evaluation_seed")
    with pytest.raises(ValueError, match="hidden evaluation"):
        BuilderTicket.from_jsonable(value)


def test_model_evidence_is_soft_and_cannot_promote(snapshot: LearnerSnapshot) -> None:
    state = ResearchState("state", LifecycleStatus.DRAFT, (), None, {})
    proposal = ResearchProposal(
        ProposalAction.PROPOSE_HYPOTHESIS,
        {"id": "hyp"},
        EvidenceClaim(EvidenceType.MODEL_INFERRED, "model", asserted_by="model"),
        LifecycleStatus.PROMOTED,
    )
    decision = ResearchMembrane().apply(state, proposal)
    assert not decision.accepted
    assert decision.soft_stored
    assert decision.after == state


def test_public_research_state_is_deeply_immutable() -> None:
    state = ResearchState(
        "state",
        LifecycleStatus.DRAFT,
        (),
        None,
        {"accepted_proposals": [], "nested": {"values": [1]}},
    )
    proposal = ResearchProposal(
        ProposalAction.PROPOSE_HYPOTHESIS,
        {"id": "hyp"},
        EvidenceClaim(EvidenceType.MODEL_INFERRED, "model"),
    )
    decision = ResearchMembrane().apply(state, proposal)
    assert decision.accepted
    assert state.public_payload["accepted_proposals"] == ()
    assert len(decision.after.public_payload["accepted_proposals"]) == 1
    assert state.public_payload["nested"]["values"] == (1,)
    with pytest.raises(TypeError):
        state.public_payload["new"] = "mutation"  # type: ignore[index]


def test_promotion_requires_verification_bound_to_state_and_proposal() -> None:
    state = ResearchState("state", LifecycleStatus.EVALUATED, (), "eval-hash", {})
    proposal = ResearchProposal(
        ProposalAction.REQUEST_EXPERIMENT,
        {},
        EvidenceClaim(EvidenceType.MODEL_INFERRED, "model"),
        LifecycleStatus.PROMOTED,
    )
    unverified = ResearchMembrane().apply(state, proposal, authority="orchestrator")
    assert not unverified.accepted
    verification = EvidenceVerification.issue(
        state,
        proposal,
        evidence_type=EvidenceType.INDEPENDENT_HARNESS,
        verifier_id="independent-verifier",
        artifact_refs=("sha256:receipt",),
    )
    untrusted_verifier = ResearchMembrane().apply(
        state,
        proposal,
        verification=verification,
        authority="orchestrator",
    )
    assert not untrusted_verifier.accepted
    assert "not trusted" in untrusted_verifier.reason
    membrane = ResearchMembrane(
        {"independent-verifier": frozenset({EvidenceType.INDEPENDENT_HARNESS})}
    )
    promoted = membrane.apply(
        state,
        proposal,
        verification=verification,
        authority="orchestrator",
    )
    assert promoted.accepted
    assert promoted.after.lifecycle_status == LifecycleStatus.PROMOTED
    assert promoted.verification_receipt_hash == verification.receipt_hash


def test_evaluation_requires_independent_verification() -> None:
    state = ResearchState("state", LifecycleStatus.EXECUTED, (), "eval-hash", {})
    proposal = ResearchProposal(
        ProposalAction.REQUEST_EXPERIMENT,
        {},
        EvidenceClaim(EvidenceType.SIMULATED, "harness"),
        LifecycleStatus.EVALUATED,
    )
    decision = ResearchMembrane().apply(state, proposal, authority="harness")
    assert not decision.accepted
    assert "require a bound independent verification" in decision.reason


def test_verification_receipt_cannot_be_replayed_for_another_proposal() -> None:
    state = ResearchState("state", LifecycleStatus.EVALUATED, (), "eval-hash", {})
    issued_for = ResearchProposal(
        ProposalAction.REQUEST_EXPERIMENT,
        {},
        EvidenceClaim(EvidenceType.MODEL_INFERRED, "model"),
        LifecycleStatus.PROMOTED,
    )
    verification = EvidenceVerification.issue(
        state,
        issued_for,
        evidence_type=EvidenceType.ENVIRONMENT_VERIFIED,
        verifier_id="environment",
        artifact_refs=("sha256:environment-receipt",),
    )
    different = ResearchProposal(
        ProposalAction.RECOMMEND_ABSTENTION,
        {},
        EvidenceClaim(EvidenceType.MODEL_INFERRED, "model"),
        LifecycleStatus.PROMOTED,
    )
    membrane = ResearchMembrane({"environment": frozenset({EvidenceType.ENVIRONMENT_VERIFIED})})
    decision = membrane.apply(
        state,
        different,
        verification=verification,
        authority="orchestrator",
    )
    assert not decision.accepted
    assert "not bound" in decision.reason


def test_status_transitions_are_monotone() -> None:
    assert (
        transition_status(LifecycleStatus.DRAFT, LifecycleStatus.CRITIQUED, authority="model")
        == LifecycleStatus.CRITIQUED
    )
    with pytest.raises(ValueError, match="monotone"):
        transition_status(LifecycleStatus.BUILT, LifecycleStatus.DRAFT, authority="harness")
    with pytest.raises(ValueError, match="adjacent"):
        transition_status(LifecycleStatus.DRAFT, LifecycleStatus.EVALUATED, authority="model")
    with pytest.raises(ValueError, match="unknown transition authority"):
        transition_status(LifecycleStatus.DRAFT, LifecycleStatus.CRITIQUED, authority="claimed-harness")


def test_json_is_strict_about_nonfinite() -> None:
    with pytest.raises(ValueError):
        json.loads('{"x": NaN}', parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
