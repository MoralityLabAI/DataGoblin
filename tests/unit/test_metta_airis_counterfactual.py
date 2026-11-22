"""AIRIS-style paired world interventions: verifier, split and evidence-custody contracts."""
from __future__ import annotations

from copy import deepcopy

import pytest

from MeTTa import airis_counterfactual as airis
from MeTTa.looped_sft import root_id, split_for
from MeTTa.pattern_miner import _internal_partition, structural_features
from MeTTa.sft_oracle import evaluate_subset, first_state, make_task, validate_task


def _train_root(family="budget", confirmation=True):
    for seed in range(512):
        if split_for(family, seed) != "train_candidate":
            continue
        if (_internal_partition(root_id(family, seed)) == "confirmation") == confirmation:
            return family, seed
    raise AssertionError("expected train root")


def test_public_proposer_mutates_one_fact_and_regenerates_real_metta_facts():
    family, seed = _train_root("budget")
    task = make_task(family, seed)
    selected = first_state(task, seed)
    options = airis.propose_interventions(task, selected)
    assert 1 <= len(options) <= 5
    assert len(set(x.mechanism for x in options)) == len(options)
    for intervention in options:
        modified = airis.apply_intervention(task, intervention)
        validate_task(modified)
        assert modified["program"] != task["program"]
        assert len(airis._one_fact_difference(task, modified)) == 1
        assert selected == first_state(task, seed)
        assert len(task["items"]) == len(modified["items"])
        for field in ("goals", "conflicts", "max_items", "family", "source_kind"):
            assert task[field] == modified[field]


def test_budget_intervention_changes_environment_violation():
    family, seed = _train_root("budget")
    task = make_task(family, seed)
    selected = first_state(task, seed)
    budget = next(x for x in airis.propose_interventions(task, selected) if x.mechanism == "budget")
    modified = airis.apply_intervention(task, budget)
    before = evaluate_subset(task, selected)
    after = evaluate_subset(modified, selected)
    assert ("token_budget" in before["violations"]) != ("token_budget" in after["violations"])
    assert airis.direct_target(task, selected, "budget") != airis.direct_target(modified, selected, "budget")
    assert task["program"] != modified["program"]


def test_extra_mutation_and_stale_interventions_fail_closed():
    family, seed = _train_root("lineage")
    task = make_task(family, seed)
    selected = first_state(task, seed)
    intervention = airis.propose_interventions(task, selected)[0]
    corrupted = deepcopy(task)
    corrupted["max_items"] += 1
    corrupted["items"][0]["quality"] += 1
    with pytest.raises(ValueError, match="exactly one public fact"):
        airis._one_fact_difference(task, corrupted)
    forged = airis.Intervention(intervention.mechanism, intervention.field,
                                intervention.item_id, "not current", intervention.after)
    with pytest.raises(ValueError, match="stale"):
        airis.apply_intervention(task, forged)
    with pytest.raises(ValueError, match="unregistered"):
        airis.apply_intervention(task, airis.Intervention("other", "x", None, 1, 2))


def test_dependency_proposals_respect_acyclic_prerequisite_constraints():
    for seed in range(40):
        task = make_task("dependency", seed)
        initial = first_state(task, seed)
        for intervention in airis.propose_interventions(task, initial):
            if intervention.mechanism == "dependency":
                modified = airis.apply_intervention(task, intervention)
                validate_task(modified)
                assert len(
                    set(intervention.before) ^ set(intervention.after)
                ) == 1


def test_policy_miner_uses_discovery_labels_and_masks_internal_confirmation(monkeypatch):
    original_miner = airis.mine_motifs
    observed = {}

    def audit_mask(rows, **kwargs):
        observed["masked"] = sum(
            row.failures == () for row in rows if _internal_partition(row.root) == "confirmation"
        )
        observed["confirmed"] = sum(_internal_partition(row.root) == "confirmation" for row in rows)
        assert all(row.failures == () for row in rows if _internal_partition(row.root) == "confirmation")
        return original_miner(rows, **kwargs)

    monkeypatch.setattr(airis, "mine_motifs", audit_mask)
    motifs, counts = airis.discover_policy(roots_per_family=32, min_support=2, top_k=8)
    assert observed["masked"] == observed["confirmed"] == counts["confirmation"]
    assert motifs and len(motifs) <= 8
    assert counts["discovery"] + counts["confirmation"] > 0


def test_paired_root_never_receives_training_or_admission_authority():
    family, seed = _train_root("coverage")
    task = make_task(family, seed)
    selected = first_state(task, seed)
    options = airis.propose_interventions(task, selected)
    row = airis.evaluate_pair(family, seed, [("selected_ge_3",)], control_seed=77)
    assert row is not None and row["root"] == root_id(family, seed)
    assert row["split"] == "train_candidate"
    assert row["internal_partition"] == "confirmation"
    assert row["ticket_custody"] == "shadow_only_not_admitted"
    assert row["training_ready"] is False
    assert row["new_sft_presentations"] == 0
    assert row["zero_change_sham_passed"] is True
    assert row["guided"]["one_fact_delta"] == 1
    assert row["uniform"]["one_fact_delta"] == 1
    assert row["budget_first"]["one_fact_delta"] == 1
    assert row["guided"]["native_metta_executed"] is False
    assert len(row["source_world_sha256"]) == 64
    assert len(row["guided"]["world_sha256"]) == 64
    assert len(options) >= 1


def test_reserved_ood_and_discovery_roots_are_refused():
    family, seed = _train_root("budget", confirmation=False)
    with pytest.raises(ValueError, match="exploration root"):
        airis.evaluate_pair(family, seed, [], control_seed=1)
    with pytest.raises(ValueError, match="reserved"):
        airis.evaluate_pair("conflict", 0, [], control_seed=1)
    reserved = next(s for s in range(512) if split_for("budget", s) == "test_reserved")
    with pytest.raises(ValueError, match="reserved"):
        airis.evaluate_pair("budget", reserved, [], control_seed=1)


def test_pairing_is_deterministic_and_same_root_for_both_arms():
    report_a, receipts_a = airis.run_experiment(roots_per_family=48, case_limit=12,
                                                 min_support=2, top_k=8, seed=19)
    report_b, receipts_b = airis.run_experiment(roots_per_family=48, case_limit=12,
                                                 min_support=2, top_k=8, seed=19)
    assert report_a == report_b and receipts_a == receipts_b
    assert report_a["actual_paired_cases"] == 12
    assert report_a["discovery_root_count"] + report_a["internal_confirmation_root_count"] > 0
    assert report_a["internal_confirmation_labels_used_for_policy"] is False
    assert report_a["reserved_dev_test_ood_roots_accessed"] == 0
    assert report_a["new_sft_presentations"] == 0
    assert report_a["training_ready"] is False
    assert report_a["native_airis_executed"] is False
    assert report_a["all_zero_change_shams_passed"] is True
    assert report_a["paired_policy"]["budget_first"]["paired_roots"] == 12
    assert report_a["paired_outcomes"]["guided_only_target_flip"] + (
        report_a["paired_outcomes"]["uniform_only_target_flip"]
        + report_a["paired_outcomes"]["ties"]
    ) == 12
    assert len({r["root"] for r in receipts_a}) == 12


def test_uniform_control_is_seeded_and_uses_unchanged_public_candidates():
    task = make_task("lineage", 4)
    initial = first_state(task, 4)
    options = airis.propose_interventions(task, initial)
    first = airis.choose_uniform(options, root="source-root-1", seed=17)
    again = airis.choose_uniform(list(reversed(options)), root="source-root-1", seed=17)
    # Control samples a canonical ordered mechanism list.
    assert first.mechanism in airis.MECHANISMS
    assert again.mechanism in airis.MECHANISMS
    assert first == airis.choose_uniform(options, root="source-root-1", seed=17)
    features = structural_features(task, initial)
    guided = airis.choose_guided(options, features, [("selected_cost_ge_75pct",)])
    assert guided in options


def test_case_bounds_are_strict():
    with pytest.raises(ValueError, match="frozen synthetic corpus"):
        airis.discover_policy(roots_per_family=513)
    with pytest.raises(ValueError, match="case limit"):
        airis._confirmation_cases(32, 0)
    with pytest.raises(ValueError, match="no intervention"):
        airis.choose_guided([], (), [])
    with pytest.raises(ValueError, match="no control"):
        airis.choose_uniform([], root="r", seed=1)
