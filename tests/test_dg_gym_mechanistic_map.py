"""Unit tests for DG-MEM-FB-01 persistent RAM-bounded mechanistic map."""

from __future__ import annotations

import math

import numpy as np
import pytest

from datagoblin.training_gym.mechanistic_map import (
    CausalContribution,
    ComponentStability,
    InterventionOutcome,
    JacobianGradAlignment,
    MarginalGainEstimate,
    MechanisticMapStore,
)
from datagoblin.training_gym.steering_feedback import Cost


def test_budget_validation_and_zero_mode() -> None:
    for bad in (-1.0, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            MechanisticMapStore(bad)

    zero = MechanisticMapStore(0.0)
    assert not zero.enabled
    assert zero.budget_bytes == 0
    ok = zero.put_bpd_component(
        component_id="l0:c0",
        layer="l0",
        rank_idx=0,
        u=np.ones(8, dtype=np.float32),
        v=np.ones(8, dtype=np.float32),
        singular_value=1.0,
        weight_sha256="abc",
        reconstruction_cosine=0.9,
    )
    assert not ok
    assert zero.allocated_bytes == 0
    assert zero.budget_rejections == 1


def test_all_eight_sections_and_strict_ram_cap() -> None:
    store = MechanisticMapStore(mms_ram_budget_mb=0.5)
    assert store.enabled

    assert store.put_bpd_component(
        component_id="l1_out:c0",
        layer="l1_out",
        rank_idx=0,
        u=np.ones(64, dtype=np.float32),
        v=np.ones(64, dtype=np.float32),
        singular_value=2.5,
        weight_sha256="sha0",
        reconstruction_cosine=0.98,
    )
    assert store.put_causal_contribution(
        CausalContribution("l1_out", 2, "l1_out:c0", 0.18, 1.4, 0.75, 64)
    )
    assert store.put_activation_stats(
        layer="l1_out",
        visit=2,
        mean=np.zeros(64, dtype=np.float32),
        var=np.ones(64, dtype=np.float32),
        norm_mean=3.2,
        steering_direction=np.ones(64, dtype=np.float32) / 8.0,
        sample_count=64,
    )
    assert store.put_jacobian_grad_alignment(
        JacobianGradAlignment("l1_out", 2, "l1_out:c0", 1.7, 0.82, 0.71, 0.55)
    )
    rng = np.random.default_rng(7)
    assert store.put_cross_pass_transport(
        src_visit=1,
        dst_visit=2,
        layer="l1_out",
        matrix=rng.standard_normal((64, 64)).astype(np.float32),
        max_rank=8,
        r2_fidelity=0.88,
    )
    assert store.record_intervention_outcome(
        InterventionOutcome(10, "sha0", "l1_out:c0", "bpd_repair", 0.62, 0.79, 0.17, 0.12, 1024, True)
    )
    assert store.put_component_stability(
        ComponentStability("l1_out:c0", 10, 0.995, 0.992, 0.98, 0.97, 0.01)
    )
    assert store.put_marginal_gain(
        MarginalGainEstimate("l1_out:c0", "l1_out", 2, "l1_out:c0", 0.15, 512, 10_000_000, 4096, 0.9)
    )

    rcpt = store.receipt()
    assert rcpt["allocated_bytes"] <= rcpt["budget_bytes"]
    assert all(rcpt["section_counts"][s] == 1 for s in MechanisticMapStore.SECTIONS)
    assert store.get_bpd_component("l1_out:c0") is not None
    assert store.get_causal_contribution("l1_out", 2, "l1_out:c0") is not None
    assert store.get_activation_stats("l1_out", 2) is not None
    assert store.get_jacobian_grad_alignment("l1_out", 2, "l1_out:c0") is not None
    assert store.get_cross_pass_transport("l1_out", 1, 2) is not None
    assert len(store.recent_outcomes("l1_out:c0")) == 1
    assert store.get_component_stability("l1_out:c0") is not None
    assert not store.should_reaudit(drift_threshold=0.05)


def test_priority_eviction_under_tight_budget() -> None:
    # ~4 KiB budget: holds only a few 256-dim float32 components (~2176 bytes each).
    store = MechanisticMapStore(mms_ram_budget_mb=4096 / (1024 * 1024))
    vec = np.ones(200, dtype=np.float32)  # u + v = 1600 bytes + 128 = 1728 bytes
    assert store.put_bpd_component(
        component_id="low_a",
        layer="l0",
        rank_idx=0,
        u=vec,
        v=vec,
        singular_value=0.1,
        weight_sha256="s",
        reconstruction_cosine=0.5,
        priority=0.1,
        step=0,
    )
    assert store.put_bpd_component(
        component_id="high_b",
        layer="l0",
        rank_idx=1,
        u=vec,
        v=vec,
        singular_value=5.0,
        weight_sha256="s",
        reconstruction_cosine=0.9,
        priority=10.0,
        step=1,
    )
    # Adding a third medium-priority item must evict low_a and keep high_b.
    assert store.put_bpd_component(
        component_id="med_c",
        layer="l0",
        rank_idx=2,
        u=vec,
        v=vec,
        singular_value=2.0,
        weight_sha256="s",
        reconstruction_cosine=0.8,
        priority=5.0,
        step=2,
    )
    assert store.evictions == 1
    assert store.allocated_bytes <= store.budget_bytes
    assert store.get_bpd_component("low_a") is None
    assert store.get_bpd_component("high_b") is not None
    assert store.get_bpd_component("med_c") is not None


def test_select_top_targets_and_steering_feedback_bridge() -> None:
    store = MechanisticMapStore(mms_ram_budget_mb=1.0)
    store.put_causal_contribution(CausalContribution("l1", 1, "c1", 0.25, 1.8, 0.85, 64))
    store.put_causal_contribution(CausalContribution("l1", 1, "c2", 0.05, 0.3, 0.20, 64))
    store.put_marginal_gain(MarginalGainEstimate("c1", "l1", 1, "c1", 0.20, 400, 5_000_000, 1024, 0.9))
    store.put_marginal_gain(MarginalGainEstimate("c2", "l1", 1, "c2", 0.04, 400, 5_000_000, 1024, 0.5))

    top = store.select_top_targets(k=1)
    assert len(top) == 1 and top[0].target_id == "c1"

    decision = store.select_feedback_window(
        checkpoint="ckpt_sha",
        feedback_cohort="dev_audit",
        step=10,
        window=20,
        remaining=Cost(100_000_000, 100_000_000, 0, 10_000, 10, 1_000_000, 0),
        competent=True,
    )
    assert decision.action is not None
    assert decision.action.key == "c1"
    assert decision.stop_step == 30
