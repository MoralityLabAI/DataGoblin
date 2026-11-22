"""CPU student and bounded recurrent rollout contracts for AIRIS-02."""
from __future__ import annotations

import json

import numpy as np
import pytest

from MeTTa.airis02_fresh import ARMS, _frozen_motifs, build_bundle, build_row
from MeTTa.airis02_student import (
    ACTION_CLASSES,
    INPUT_DIM,
    LinearStudent,
    RecurrentStudent,
    action_index,
    decode_action,
    encode_public,
    evaluate,
    legal_action_mask,
    run_students,
)
from MeTTa.sft_oracle import first_state, make_task


def test_student_features_are_fixed_shape_and_public_only():
    task = make_task("lineage", 512)
    state = first_state(task, 512)
    features = encode_public(task, state)
    assert features.shape == (INPUT_DIM,)
    assert np.all(np.isfinite(features))
    assert len(features) < 256
    mask = legal_action_mask(task, state)
    assert mask.shape == (ACTION_CLASSES,)
    assert mask[20]
    assert int(mask.sum()) == len(task["items"]) + 1
    with pytest.raises(ValueError, match="duplicate selected"):
        encode_public(task, [state[0], state[0]])
    with pytest.raises(ValueError, match="absent"):
        encode_public(task, ["unknown"])


def test_repair_action_classes_roundtrip():
    for index in range(ACTION_CLASSES):
        assert action_index(decode_action(index)) == index
    with pytest.raises(ValueError, match="invalid action logit"):
        decode_action(40)
    with pytest.raises(ValueError, match="invalid canonical"):
        action_index({"op": "add", "id": "r010"})
    with pytest.raises(ValueError, match="invalid curation"):
        action_index({"op": "rm", "id": "r3"})


def test_student_models_are_deterministic_and_loss_reduces():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(16, INPUT_DIM))
    y = np.full(16, 10, dtype=np.int64)
    first = LinearStudent(12)
    second = LinearStudent(12)
    a = first.fit(x, y, steps=12, lr=0.15)
    b = second.fit(x, y, steps=12, lr=0.15)
    assert a == b and a[-1] < a[0]
    assert first.digest() == second.digest()
    assert first.params > 0
    third = RecurrentStudent(3, hidden=8)
    losses = third.fit(x, y, steps=14, lr=0.15, train_loops=4)
    assert losses[-1] < losses[0]
    assert len(third.digest()) == 64
    for loops in (2, 4, 8):
        assert third.logits(x, loops=loops).shape == (16, ACTION_CLASSES)
    with pytest.raises(ValueError, match="unroll"):
        third.logits(x, loops=9)


def test_small_bundle_fits_research_models_after_independent_audit(tmp_path):
    build_bundle(tmp_path, train_roots_per_family=3, dev_roots_per_family=2)
    report = run_students(tmp_path, steps=5, lr=0.1, seed=5, hidden=8)
    assert report["model_trained"] is True
    assert report["student_training_policy"] == "structural_membership_action_mask_plus_hash_fixed_point_stop"
    assert report["synthetic_research_fit_eligible"] is True
    assert report["source_data_admitted"] is False
    assert report["training_ready"] is False
    assert report["holdout_steward_opened"] is False
    assert report["development_evaluation_world_policy"] == "shared_uniform_intervention"
    assert len(report["development_evaluation_worlds_sha256"]) == 64
    assert report["train_cases_per_arm"] == 15
    assert report["development_cases_per_arm"] == 10
    assert report["optimizer_steps_per_arm"] == 5
    assert set(report["trained_models"]) == set(ARMS)
    for value in report["trained_models"].values():
        assert value["linear"]["parameters"] > 0
        assert value["recurrent"]["parameters"] > 0
        assert len(value["linear"]["checkpoint_sha256"]) == 64
        assert len(value["recurrent"]["checkpoint_sha256"]) == 64
        assert set(value["recurrent"]["development"]) == {"2", "4", "8"}
        assert value["linear"]["within_arm_diagnostic"]["cases"] == 10
        assert value["recurrent"]["within_arm_diagnostic"]["cases"] == 10
        for arm in [value["linear"]["development"], *value["recurrent"]["development"].values()]:
            assert arm["cases"] == 10
            assert 0 <= arm["solved_and_stopped"] <= 10
            assert 0 <= arm["premature_stops"] <= 10
            assert arm["mean_remaining_canonical_edits"] >= 0
        assert len(json.dumps(value)) > 50


def test_student_refuses_changed_teacher_bundle(tmp_path):
    build_bundle(tmp_path, train_roots_per_family=2, dev_roots_per_family=1)
    file = tmp_path / "train_candidate.jsonl"
    file.write_text(file.read_text() + '{"forged":"entry"}\n')
    with pytest.raises(ValueError, match="digest mismatch"):
        run_students(tmp_path, steps=1, hidden=8)


def test_rollout_uses_one_source_root_for_each_action_arm():
    motifs = _frozen_motifs()
    rows = [build_row("dependency", 512, arm, motifs) for arm in ARMS]
    model = LinearStudent(7)
    for row in rows:
        report = evaluate(model, [row], loops=1)
        assert report["cases"] == 1
        assert report["exact_first_edit"] in (0, 1)
        assert report["solved_and_stopped"] in (0, 1)
