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
from datagoblin.research.membrane import ProposalAction, ResearchMembrane, ResearchProposal
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


def test_status_transitions_are_monotone() -> None:
    assert (
        transition_status(LifecycleStatus.DRAFT, LifecycleStatus.CRITIQUED, authority="model")
        == LifecycleStatus.CRITIQUED
    )
    with pytest.raises(ValueError, match="monotone"):
        transition_status(LifecycleStatus.BUILT, LifecycleStatus.DRAFT, authority="harness")
    with pytest.raises(ValueError, match="cannot mark"):
        transition_status(LifecycleStatus.DRAFT, LifecycleStatus.EVALUATED, authority="model")


def test_json_is_strict_about_nonfinite() -> None:
    with pytest.raises(ValueError):
        json.loads('{"x": NaN}', parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
