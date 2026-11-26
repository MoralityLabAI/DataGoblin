"""Unit tests for MMS-05 utility-directed mechanistic feedback scheduler."""

from __future__ import annotations

import numpy as np
import pytest

from datagoblin.training_gym.mms_utility import FeedbackSignal, UtilityDirectedScheduler


def test_partition_isolation_rejects_final_and_sealed() -> None:
    for bad_part in ("dev_final", "confirmation", "sealed", "test"):
        with pytest.raises(ValueError, match="forbidden feedback partition"):
            FeedbackSignal(
                step=1,
                site="trm0.ff",
                ram_mib=128.0,
                partition=bad_part,
                signal_kind="lean_proof_progress",
                delta_v=0.05,
                delta_cpu_seconds=1.0,
                delta_ram_mib=128.0,
            )


def test_utility_scheduler_beats_hit_proxy_when_high_hit_site_is_low_value() -> None:
    """High-hit site with zero/negative downstream value must lose to moderate-hit high-value site."""
    sched = UtilityDirectedScheduler(
        sites=("noisy_high_hit", "useful_site"),
        ram_tiers_mib=(64.0, 256.0),
        lambda_ram=0.005,
        exploration_bonus=0.01,
    )
    rng = np.random.default_rng(42)
    for step in range(12):
        # noisy_high_hit gets tons of cache hits/admissions, high gradient variance, zero proof progress
        sched.record_proxy_counts("noisy_high_hit", 256.0, hits=500, admitted=400, comp_grad=rng.normal(0, 5.0, 8))
        sched.observe_feedback(
            FeedbackSignal(
                step=step,
                site="noisy_high_hit",
                ram_mib=256.0,
                partition="dev_train",
                signal_kind="lean_proof_progress",
                delta_v=0.001,
                delta_cpu_seconds=2.0,
                delta_ram_mib=256.0,
            )
        )
        # useful_site has fewer hits, consistent gradient direction, strong downstream progress at lower RAM
        sched.record_proxy_counts(
            "useful_site", 64.0, hits=40, admitted=30, comp_grad=np.ones(8) + rng.normal(0, 0.05, 8)
        )
        sched.observe_feedback(
            FeedbackSignal(
                step=step,
                site="useful_site",
                ram_mib=64.0,
                partition="dev_train",
                signal_kind="lbf_bellman_regret",
                delta_v=0.12,
                delta_cpu_seconds=1.0,
                delta_ram_mib=64.0,
                state_signature=np.array([1.0, 0.0, 0.5, 0.0], dtype=np.float32),
                component_ids=("l1_down:c2", "l1_down:c3"),
            )
        )

    hit_choice = sched.select(policy="hit_proxy")
    util_choice = sched.select(
        policy="utility", query_signature=np.array([0.9, 0.1, 0.4, 0.0], dtype=np.float32)
    )
    assert hit_choice.site == "noisy_high_hit"
    assert util_choice.site == "useful_site"
    assert util_choice.ram_mib == 64.0
    assert "l1_down:c2" in util_choice.retrieved_components


def test_compressed_coordinate_history_respects_byte_budget() -> None:
    sched = UtilityDirectedScheduler(
        sites=("s0",),
        ram_tiers_mib=(16.0,),
        history_budget_bytes=2048,
    )
    for step in range(50):
        sched.record_coordinate_delta(step, "s0", np.ones(64, dtype=np.float32) * step)
    rcpt = sched.receipt()
    assert rcpt["stored_coord_deltas"] <= 4  # 64*4 = 256 B each, half budget = 1024 B -> max 4 entries
    assert rcpt["extra_forward_passes"] == 0
    assert rcpt["extra_backward_passes"] == 0
