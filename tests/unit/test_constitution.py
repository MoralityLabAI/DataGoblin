from __future__ import annotations

import pytest

from datagoblin.constitution.validator import ConstitutionalValidator, DecisionKind
from datagoblin.datasets.corruptions import MUTATION_NAMES, apply_mutation
from datagoblin.domain.recipes import DataRecipe


@pytest.mark.parametrize("operator", MUTATION_NAMES)
def test_mutation_declares_detected_violation_and_original_repairs(operator, good_recipe) -> None:
    corrupted, receipt = apply_mutation(good_recipe, operator, seed=17)
    decision = ConstitutionalValidator().validate(corrupted)
    detected = {item.rule_id for item in decision.triggered_rules}
    assert set(receipt.expected_violation_mask) <= detected
    assert decision.decision != DecisionKind.ACCEPT
    assert ConstitutionalValidator().validate(good_recipe).decision == DecisionKind.ACCEPT
    assert receipt.before_hash != receipt.after_hash


def test_evaluation_leakage_example_is_rejected() -> None:
    from pathlib import Path

    from datagoblin.serialization import load_structured

    root = Path(__file__).resolve().parents[2]
    recipe = DataRecipe.from_jsonable(load_structured(root / "examples/recipe.invalid_eval_leakage.yaml"))
    decision = ConstitutionalValidator().validate(recipe)
    assert decision.decision == DecisionKind.REJECT
    assert "evaluation_leakage" in {item.rule_id for item in decision.triggered_rules}


def test_self_certified_environment_evidence_is_rejected(good_recipe) -> None:
    corrupted, _ = apply_mutation(good_recipe, "self_certify_environment_evidence", 2)
    rules = {item.rule_id for item in ConstitutionalValidator().validate(corrupted).triggered_rules}
    assert "self_certified_environment_evidence" in rules


def test_separation_seed_budget_and_network_rules(good_recipe) -> None:
    expected = {
        "merge_builder_and_evaluator": "builder_evaluator_identity",
        "create_train_eval_generator_overlap": "seed_namespace_overlap",
        "unbound_compute": "unbounded_budget",
        "request_unnecessary_network": "unnecessary_network",
    }
    for operator, rule in expected.items():
        corrupted, _ = apply_mutation(good_recipe, operator, 3)
        rules = {item.rule_id for item in ConstitutionalValidator().validate(corrupted).triggered_rules}
        assert rule in rules
