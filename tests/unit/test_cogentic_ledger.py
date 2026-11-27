from __future__ import annotations

import copy

import pytest

from datagoblin.training_gym.rlt_gym import cogentic_ledger as c


def _rehash(trace):
    previous = c.digest({"task_digest": trace["task_digest"], "policy": trace["policy"]})
    for step in trace["steps"]:
        step["previous_digest"] = previous
        step["step_digest"] = c.digest({k: v for k, v in step.items() if k != "step_digest"})
        previous = step["step_digest"]
    trace["final_digest"] = previous
    trace["trace_digest"] = c.digest({k: v for k, v in trace.items() if k != "trace_digest"})


def test_generation_is_deterministic_and_roots_never_cross_splits():
    tasks = c.generate_tasks(12)
    assert tasks == c.generate_tasks(12)
    assert len({task["root_id"] for task in tasks}) == 48
    assert len({task["task_id"] for task in tasks}) == 48
    assert {task["family"] for task in tasks} == set(c.FAMILIES)
    assert all(task["provenance"]["depth"] >= 6 for task in tasks if task["split"] == "ood_test")
    assert all(task["provenance"]["depth"] <= 5 for task in tasks if task["split"] != "ood_test")


def test_every_family_contains_negative_duplicate_and_complete_teacher_proof():
    for task in c.generate_tasks(8):
        trace = c.collect_trace(task)
        assert trace["solved"]
        assert c.replay_trace(task, trace)["passed"]
        codes = [step["verification"]["code"] for step in trace["steps"]]
        assert codes.count("missing_premise") == 1
        assert codes.count("duplicate_lemma") == 1
        assert codes.count("proved_lemma") >= 3


def test_duplicate_lemma_costs_one_call_and_adds_no_fact():
    task = c.generate_tasks(1)[0]
    known = c.initial_state(task)
    action = c.reference_action(task, known)
    first = c.check_step(task, known, action)
    second = c.check_step(task, first["next_known"], action)
    assert second["accepted"] is True
    assert second["new_fact"] is None
    assert second["code"] == "duplicate_lemma"
    assert second["verifier_calls"] == 1
    assert second["next_known"] == first["next_known"]


@pytest.mark.parametrize("action", [-1, 1_000_000, True, False, "1", None])
def test_invalid_actions_have_cost_and_leave_state_unchanged(action):
    task = c.generate_tasks(1)[0]
    known = c.initial_state(task)
    receipt = c.check_step(task, known, action)
    assert receipt["accepted"] is False
    assert receipt["code"] == "invalid_action"
    assert receipt["next_known"] == known
    assert receipt["verifier_calls"] == 1


def test_public_projection_excludes_metadata_and_copies_state():
    task = c.generate_tasks(1)[0]
    task["hidden_answer"] = "canary"
    known = c.initial_state(task)
    observation = c.public_observation(task, known)
    assert set(observation) == {"world", "known", "history"}
    assert "canary" not in c.canonical(observation)
    assert "root_id" not in c.canonical(observation)
    observation["world"]["facts"].clear()
    observation["known"].clear()
    assert task["world"]["facts"] == known


def test_rehashed_forged_success_is_rejected_by_independent_replay():
    task = c.generate_tasks(1)[0]
    trace = c.collect_trace(task)
    forged = copy.deepcopy(trace)
    forged["steps"][0]["verification"]["accepted"] = True
    _rehash(forged)
    with pytest.raises(ValueError, match="checker or cost"):
        c.replay_trace(task, forged)


def test_rehashed_cost_erasure_is_rejected():
    task = c.generate_tasks(1)[0]
    trace = c.collect_trace(task)
    trace["steps"][0]["cost"]["verifier_calls"] = 0
    _rehash(trace)
    with pytest.raises(ValueError, match="checker or cost"):
        c.replay_trace(task, trace)


def test_rehashed_injected_fact_is_rejected():
    task = c.generate_tasks(1)[0]
    trace = c.collect_trace(task)
    trace["steps"][0]["observation"]["known"].append(task["world"]["goal"])
    _rehash(trace)
    with pytest.raises(ValueError, match="public state"):
        c.replay_trace(task, trace)


def test_rehashed_provenance_substitution_is_rejected():
    task = c.generate_tasks(1)[0]
    trace = c.collect_trace(task)
    trace["provenance"]["cogentic_paper_trace"] = True
    _rehash(trace)
    with pytest.raises(ValueError, match="provenance"):
        c.replay_trace(task, trace)


def test_export_custody_and_negative_loss_boundaries():
    for task in c.generate_tasks(4):
        trace = c.collect_trace(task)
        streams = c.export_streams(task, trace)
        if task["split"] != "train":
            assert not any(streams.values())
            continue
        good_steps = {step["step_digest"] for step in trace["steps"] if step["verification"]["new_fact"] is not None}
        assert {r["step_digest"] for r in streams["sft"]} == good_steps
        assert any(not r["target"]["accepted"] for r in streams["outcomes"])
        assert len(streams["repairs"]) == len(streams["sft"])


def test_labels_are_separate_and_features_ignore_split_metadata():
    task = c.generate_tasks(1)[0]
    known = c.initial_state(task)
    row = c.ranking_row(task, known)
    assert len(row["candidates"]) == len(task["world"]["rules"])
    assert all(len(x["features"]) == len(c.FEATURE_NAMES) for x in row["candidates"])
    assert "labels" not in row["public"]
    changed = copy.deepcopy(task)
    changed["split"] = "ordinary_test"
    changed["root_id"] = "CANARY"
    assert c.candidate_features(task, known, 0) == c.candidate_features(changed, known, 0)


def test_lean_source_uses_only_replayed_terms_and_no_admissions():
    task = c.generate_tasks(1)[0]
    source = c.lean_source(task, c.collect_trace(task), "cogTest")
    assert "#print axioms cogTest" in source
    assert "sorry" not in source and "axiom " not in source
    with pytest.raises(ValueError):
        c.lean_source(task, c.collect_trace(task), "bad; axiom injected : False")
