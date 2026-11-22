from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from datagoblin.training_gym.sgrt_holoarc_gym import make_task
from datagoblin.training_gym.sgrt_learned import (
    CANDIDATE_DIM,
    EVAL_ARMS,
    PUBLIC_EVAL_SEED,
    PUBLIC_TRAIN_SEED,
    RANK,
    TASK_DIM,
    TRAINED_ARMS,
    LowRankHead,
    TrainingItem,
    _loss_gradients,
    _training_items,
    candidate_features,
    context_transport,
    evaluate,
    fit,
    ranked_candidates,
    task_features,
)


def test_observation_contract_ignores_target_and_generator_family() -> None:
    task = make_task(seed=PUBLIC_TRAIN_SEED, root=2)
    edited = replace(task, target=("id", "id"), family="mixed")
    assert task.public() != edited.public()  # public metadata includes family
    a = task_features(task.public())
    b = task_features(edited.public())
    assert a.shape == b.shape == (TASK_DIM,)
    assert np.array_equal(a, b)
    candidate = ("xor1", "add1")
    assert candidate_features(candidate, task.anchor).shape == (CANDIDATE_DIM,)
    for arm in TRAINED_ARMS:
        assert np.array_equal(
            context_transport(task.public(), arm=arm),
            context_transport(edited.public(), arm=arm),
        )


def test_geometry_controls_are_orthogonal_and_path_order_sensitive() -> None:
    public = make_task(seed=PUBLIC_TRAIN_SEED, root=6).public()
    transforms = {
        arm: context_transport(public, arm=arm)
        for arm in TRAINED_ARMS
    }
    identity = np.eye(RANK)
    assert np.linalg.norm(transforms["flat_lowrank"] - identity) < 1e-10
    assert np.linalg.norm(transforms["holoarc_gear_lowrank"] - identity) > 1e-4
    assert np.linalg.norm(
        transforms["holoarc_gear_lowrank"] - transforms["holoarc_step_shuffle_lowrank"]
    ) > 1e-5
    for transform in transforms.values():
        assert transform.shape == (RANK, RANK)
        assert np.linalg.norm(transform.T @ transform - identity) < 1e-10
        assert np.linalg.det(transform) == pytest.approx(1.0, abs=1e-10)


def test_softmax_training_derivative_matches_finite_difference() -> None:
    item = _training_items(roots=16, arm="holoarc_gear_lowrank")[0]
    head = LowRankHead.initialize(seed=42)
    _loss, grads = _loss_gradients(head, item)
    params = (head.task_matrix, head.candidate_matrix, head.gate_vector)
    eps = 1e-5
    for p, gradient in zip(params, grads, strict=True):
        index = (0, 0) if p.ndim == 2 else (0,)
        original = float(p[index])
        p[index] = original + eps
        high, _ = _loss_gradients(head, item)
        p[index] = original - eps
        low, _ = _loss_gradients(head, item)
        p[index] = original
        assert gradient[index] == pytest.approx((high - low) / (2 * eps), abs=1e-6)


def test_training_split_excludes_mixed_family() -> None:
    for arm in ("semantic_lowrank", "holoarc_gear_lowrank"):
        items = _training_items(roots=20, arm=arm)
        assert len(items) == 15
        for item in items:
            assert isinstance(item, TrainingItem)
            assert item.candidates.shape == (18, CANDIDATE_DIM)
            assert item.positives.any()
    for index in range(20):
        if index % 4 == 3:
            assert make_task(seed=PUBLIC_TRAIN_SEED, root=index).family == "mixed"


def test_learned_heads_share_capacity_and_are_deterministic() -> None:
    trained = {}
    for arm in TRAINED_ARMS:
        head, log = fit(arm=arm, seed=7, train_roots=20, updates=20)
        assert log["trainable_parameters"] == TASK_DIM * RANK + CANDIDATE_DIM * RANK + TASK_DIM
        assert np.isfinite(log["last32_loss_mean"])
        assert np.all(np.isfinite(head.task_matrix))
        assert np.all(np.isfinite(head.candidate_matrix))
        assert np.all(np.isfinite(head.gate_vector))
        trained[arm] = head
    repeat, _ = fit(arm="holoarc_gear_lowrank", seed=7, train_roots=20, updates=20)
    assert trained["holoarc_gear_lowrank"].digest() == repeat.digest()
    public = make_task(seed=PUBLIC_EVAL_SEED, root=0).public()
    candidate_set = ranked_candidates(trained["holoarc_gear_lowrank"], public, arm="holoarc_gear_lowrank", budget=8)
    assert len(candidate_set) == len(set(candidate_set)) == 8
    rows, summary = evaluate(trained, eval_roots=8, budget=8)
    assert len(rows) == 8
    assert len(summary["family_ood_mixed"]) == len(EVAL_ARMS)
    assert summary["family_ood_mixed"]["semantic_lowrank"]["roots"] == 2
    for arm, counts in summary["all"].items():
        assert counts["roots"] == 8
        assert counts["verifier_calls"] == (8 if arm == "rigid_top1_proxy" else 64)
