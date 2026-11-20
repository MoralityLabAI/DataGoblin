"""Feedback semantics, verifier custody and replay checks for the CPU control."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from datagoblin.serialization import canonical_json, dump_structured
from datagoblin.training_gym.rlt_feedback import MODES, feedback_weight
from datagoblin.training_gym.rlt_gym import tasks
from datagoblin.training_gym.rlt_gym.dynamics import NativeRanker


@pytest.fixture
def cohort():
    return tasks.generate_tasks(seed=17, per_family=1)


def _train(cohort):
    return [task for task in cohort if task["split"] == "train"]


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def test_constant_endpoint_reaches_evenly_spaced_states():
    state = 0.0
    path = []
    for visit in range(3):
        state += feedback_weight("rlt", visit, 3) * (6.0 - state)
        path.append(state)
    assert path == [2.0, 4.0, 6.0]
    for depth in (1, 2, 4, 8, 31):
        rectified = [feedback_weight("rlt", v, depth) for v in range(depth)]
        damped = [feedback_weight("matched_damping", v, depth) for v in range(depth)]
        assert sum(rectified) == pytest.approx(sum(damped))
        assert rectified[-1] == damped[-1] == 1.0
        assert all(0 < eta <= 1 for eta in rectified + damped)


@pytest.mark.parametrize(("mode", "visit", "depth"), [
    ("typo", 0, 1), ("typo", 3, 4), ("direct", -1, 4), ("rlt", 4, 4),
    ("direct", 0, 0), ("rlt", True, 4), ("rlt", 0, True), ("rlt", 1.5, 4),
])
def test_feedback_rejects_invalid_contract_even_at_terminal_visit(mode, visit, depth):
    with pytest.raises(ValueError):
        feedback_weight(mode, visit, depth)


def test_each_arm_has_identical_one_visit_execution(cohort):
    model = NativeRanker(13)
    task = _train(cohort)[0]
    rows = [next(model.visits(task, mode, 1)) for mode in MODES]
    assert all(row["eta"] == 1 for row in rows)
    assert all(row["state_summaries"] == rows[0]["state_summaries"] for row in rows)
    assert all(row["candidate_scores"] == rows[0]["candidate_scores"] for row in rows)


def test_readout_site_and_fixed_planned_clock_are_observable(cohort):
    model = NativeRanker(13)
    task = _train(cohort)[0]
    carried = list(model.visits(task, "rlt", 4))
    endpoints = list(model.visits(task, "rlt", 4, readout_site="endpoint"))
    first = carried[0]["state_summaries"]
    assert first["carried_answer"]["rms"] == pytest.approx(first["endpoint"]["rms"] / 4)
    assert carried[0]["candidate_scores"] != endpoints[0]["candidate_scores"]
    assert carried[-1]["candidate_scores"] == endpoints[-1]["candidate_scores"]
    assert [row["eta"] for row in carried] == [0.25, 1 / 3, 0.5, 1.0]
    assert all(row["planned_depth"] == 4 and row["step_size"] == 0.25 for row in carried)
    prefix = model.visits(task, "rlt", 4)
    assert [next(prefix), next(prefix)] == carried[:2]
    assert carried[0]["state_summaries"]["before"]["l2"] == 0
    assert "values" not in canonical_json(carried)


def test_matched_fit_uses_same_checked_examples_and_frozen_reservoir(cohort):
    models = []
    for mode in MODES:
        model = NativeRanker(23)
        reservoir = {key: value.copy() for key, value in model.weights.items() if key != "readout"}
        receipt = model.fit(_train(cohort), training_mode=mode, planned_depth=4)
        assert receipt["fitted"] is True
        assert receipt["model_kind"] == "frozen_random_reservoir"
        assert receipt["architecture_promotion_eligible"] is False
        assert receipt["training_readout_examples"] == 4 * receipt["training_candidates"]
        assert receipt["training_verifier_calls"] == receipt["training_candidates"]
        for key, expected in reservoir.items():
            np.testing.assert_array_equal(model.weights[key], expected)
        models.append(model)
    for key in ("initialization_digest", "training_fingerprint", "training_receipts_digest",
                "training_readout_examples", "training_block_calls"):
        assert len({model.metadata[key] for model in models}) == 1
    assert len({model.model_digest for model in models}) == len(MODES)
    with pytest.raises(ValueError, match="immutable"):
        models[0].fit(_train(cohort))


def test_evidence_split_is_rejected_before_any_label_access(cohort, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("verifier was touched before split preflight")

    monkeypatch.setattr(tasks, "verify", forbidden)
    mixed = [_train(cohort)[0], next(task for task in cohort if task["split"] == "ordinary_test")]
    model = NativeRanker(4)
    initial = model.model_digest
    with pytest.raises(ValueError, match="train split"):
        model.fit(mixed)
    assert model.model_digest == initial


def test_forged_verifier_receipt_cannot_train_a_readout(cohort, monkeypatch):
    original = tasks.verify

    def forged(task, candidate):
        return {**original(task, candidate), "candidate_digest": "0" * 64}

    monkeypatch.setattr(tasks, "verify", forged)
    model = NativeRanker(4)
    initial = model.model_digest
    with pytest.raises(ValueError, match="does not bind"):
        model.fit(_train(cohort))
    assert model.model_digest == initial


def test_relabeling_training_lineage_or_input_cannot_enter_evaluation(cohort):
    model = NativeRanker(4)
    model.fit(_train(cohort))
    task = copy.deepcopy(_train(cohort)[0])
    task["split"] = "ordinary_test"
    task["task_digest"] = _digest({k: v for k, v in task.items() if k != "task_digest"})
    with pytest.raises(ValueError, match="training lineage"):
        list(model.visits(task, "direct", 4))
    task["task_id"] = "new-evaluation-id"
    task["lineage_id"] = "new-evaluation-lineage"
    task["task_digest"] = _digest({k: v for k, v in task.items() if k != "task_digest"})
    with pytest.raises(ValueError, match="training input"):
        list(model.visits(task, "direct", 4))


def test_checkpoint_replays_visits_and_rejects_tampering(cohort, tmp_path):
    model = NativeRanker(4)
    model.fit(_train(cohort), training_mode="rlt")
    task = next(task for task in cohort if task["split"] == "ordinary_test")
    path = tmp_path / "model.json"
    model.save(path)
    restored = NativeRanker.load(path)
    assert restored.model_digest == model.model_digest
    assert list(restored.visits(task, "rlt", 8)) == list(model.visits(task, "rlt", 8))
    tampered = model.snapshot()
    tampered["weights"]["readout"][0] += 1
    dump_structured(path, tampered)
    with pytest.raises(ValueError, match="digest"):
        NativeRanker.load(path)
    polluted = model.snapshot()
    polluted["metadata"]["training_manifest"][0]["split"] = "ordinary_test"
    polluted["model_digest"] = _digest({k: v for k, v in polluted.items() if k != "model_digest"})
    dump_structured(path, polluted)
    with pytest.raises(ValueError, match="evidence split"):
        NativeRanker.load(path)


def test_named_pulse_has_replayable_matched_dose_and_future_effect(cohort):
    model = NativeRanker(7)
    task = _train(cohort)[0]
    clean = list(model.visits(task, "rlt", 4))
    spec = {"kind": "random_pulse", "boundary": "carried_answer", "visit": 1,
            "relative_dose": 0.1, "seed": 31}
    perturbed = list(model.visits(task, "rlt", 4, intervention=spec))
    assert perturbed == list(model.visits(task, "rlt", 4, intervention=spec))
    assert clean[0]["state_summaries"] == perturbed[0]["state_summaries"]
    receipt = perturbed[1]["intervention"]
    assert receipt["realized_relative_dose"] == pytest.approx(0.1)
    assert receipt["bpd_mechanism_verified"] is False
    assert receipt["first_affected_readout_visit"] == 1
    assert clean[2]["state_summaries"]["endpoint"] != perturbed[2]["state_summaries"]["endpoint"]
    sham = copy.deepcopy(spec)
    sham.pop("relative_dose")
    sham.update({"absolute_dose": receipt["realized_absolute_dose"], "seed": 32})
    comparison = list(model.visits(task, "rlt", 4, intervention=sham))
    assert comparison[1]["intervention"]["realized_absolute_dose"] == pytest.approx(
        receipt["realized_absolute_dose"])
    with pytest.raises(ValueError, match="boundary"):
        list(model.visits(task, "rlt", 4, intervention={**spec, "boundary": "unknown"}))
    terminal = {**spec, "visit": 3}
    unobserved = list(model.visits(task, "rlt", 4, readout_site="endpoint", intervention=terminal))
    assert unobserved[-1]["intervention"]["first_affected_readout_visit"] is None


def test_ranker_keeps_metrics_finite_on_each_split(cohort):
    model = NativeRanker(2)
    model.fit(_train(cohort))
    for task in cohort:
        row = next(model.visits(task, "matched_damping", 4))
        assert row["candidate"] in task["candidates"]
        assert row["block_calls"] == 1
        assert row["evidence_epoch"] == task["evidence_epoch"]
        assert np.isfinite(row["candidate_scores"]).all()
