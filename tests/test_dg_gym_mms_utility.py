"""Regression tests for repaired MMS-05/06 UtilityDirectedScheduler."""

from __future__ import annotations

import math

import numpy as np
import pytest

from datagoblin.training_gym.mms_utility import (
    ORDINARY_SITE,
    FeedbackContext,
    FeedbackSignal,
    UtilityDirectedScheduler,
)


def _ctx(
    site: str = "block",
    *,
    task_id: str = "lbf20d",
    metric_name: str = "mean_regret",
    metric_direction: int = -1,
    metric_scale: float = 1.0,
    basis_digest: str = "basis_v1",
) -> FeedbackContext:
    return FeedbackContext(
        task_id=task_id,
        metric_name=metric_name,
        metric_direction=metric_direction,
        metric_scale=metric_scale,
        split_digest="split_dev_train_sha",
        feedback_horizon=40,
        checkpoint_sha256="ckpt_sha_0",
        optimizer_sha256="opt_sha_0",
        site=site,
        basis_digest=basis_digest,
    )


def test_validation_rejects_invalid_configs_and_partitions() -> None:
    with pytest.raises(ValueError, match="duplicate site"):
        UtilityDirectedScheduler(("a", "a"), (4.0,))
    with pytest.raises(ValueError, match="ORDINARY_SITE"):
        UtilityDirectedScheduler((ORDINARY_SITE,), (4.0,))
    with pytest.raises(ValueError, match="duplicate ram_tiers"):
        UtilityDirectedScheduler(("a",), (4.0, 4.0))
    with pytest.raises(ValueError, match="finite"):
        UtilityDirectedScheduler(("a",), (math.nan,))
    with pytest.raises(ValueError, match="cov_dim"):
        UtilityDirectedScheduler(("a",), (4.0,), cov_dim=0)
    with pytest.raises(ValueError, match="ema_alpha"):
        UtilityDirectedScheduler(("a",), (4.0,), ema_alpha=0.0)
    with pytest.raises(ValueError, match="exploration_bonus"):
        UtilityDirectedScheduler(("a",), (4.0,), exploration_bonus=-0.1)

    for bad_part in ("dev_final", "confirmation", "sealed", "test"):
        with pytest.raises(ValueError, match="forbidden feedback partition"):
            FeedbackSignal(
                step=1,
                trajectory_id="t0",
                site="block",
                ram_mib=4.0,
                partition=bad_part,
                signal_kind="lbf_bellman_regret",
                delta_v=-0.05,
                delta_cpu_seconds=1.0,
                delta_ram_mib=4.0,
                context=_ctx("block"),
            )


def test_owned_compact_coords_no_view_retention_and_mutation_isolation() -> None:
    sched = UtilityDirectedScheduler(("s0",), (4.0,), history_budget_bytes=4096)
    huge_base = np.arange(100_000, dtype=np.float32)
    small_view = huge_base[10:18]  # 8-element view retaining huge_base
    assert small_view.base is huge_base

    assert sched.record_coordinate_delta(1, "s0", small_view)
    stored = sched._coord_history[0][2]
    assert stored.base is None  # owned compact allocation, does NOT retain huge_base
    assert stored.nbytes == 32

    # Mutating caller buffer must not mutate stored record
    small_view[:] = -999.0
    assert np.allclose(stored, np.arange(10, 18, dtype=np.float32))


def test_zero_byte_history_prevents_empty_records_and_cov_bytes_scale_with_dim() -> None:
    sched0 = UtilityDirectedScheduler(("s0",), (4.0,), history_budget_bytes=0, cov_dim=16)
    assert not sched0.record_coordinate_delta(0, "s0", np.array([], dtype=np.float32))
    assert not sched0.record_coordinate_delta(1, "s0", np.ones(4, dtype=np.float32))
    assert len(sched0._coord_history) == 0
    assert sched0.coord_rejections == 2

    # Covariance owned bytes derive from actual cov_dim (16 floats + 16x16 floats = 272 * 4 = 1088 B per action)
    ctx = _ctx("s0")
    sched0.record_proxy_counts(ctx, "s0", 4.0, hits=1, admitted=1)
    # Two actions in table: (__ordinary__, 0.0) and ("s0", 4.0) -> 2 * 1088 = 2176 owned bytes
    assert sched0.owned_array_bytes == 2 * (16 + 16 * 16) * 4
    rcpt = sched0.receipt(measured_process_rss_bytes=None)
    assert rcpt["memory"]["owned_array_bytes"] == 2176
    assert rcpt["memory"]["estimated_charge_bytes"] > rcpt["memory"]["owned_array_bytes"]
    assert rcpt["memory"]["measured_process_rss_bytes"] is None


def test_trajectory_isolation_independent_negative_feedback_does_not_leak() -> None:
    """Negative feedback on site_a trajectory must stay completely independent of positive site_b."""
    sched = UtilityDirectedScheduler(("site_a", "site_b"), (4.0,))
    ctx_a = _ctx("site_a", metric_name="acc", metric_direction=1)
    ctx_b = _ctx("site_b", metric_name="acc", metric_direction=1)

    sched.record_action("traj_b", step=1, site="site_b", ram_mib=4.0)
    sched.observe_feedback(
        FeedbackSignal(
            step=1,
            trajectory_id="traj_b",
            site="site_b",
            ram_mib=4.0,
            partition="dev_train",
            signal_kind="lbf_bellman_regret",
            delta_v=0.20,
            delta_cpu_seconds=1.0,
            delta_ram_mib=4.0,
            context=ctx_b,
        )
    )
    tbl = sched._ctx_table(ctx_b.metric_context_key)
    b_before = (tbl[("site_b", 4.0)].ema_delta_v, tbl[("site_b", 4.0)].count)

    # Now record strongly negative feedback on an independent trajectory for site_a
    sched.record_action("traj_a", step=2, site="site_a", ram_mib=4.0)
    sched.observe_feedback(
        FeedbackSignal(
            step=2,
            trajectory_id="traj_a",
            site="site_a",
            ram_mib=4.0,
            partition="dev_train",
            signal_kind="lbf_bellman_regret",
            delta_v=-5.0,
            delta_cpu_seconds=1.0,
            delta_ram_mib=4.0,
            context=ctx_a,
        )
    )
    b_after = (tbl[("site_b", 4.0)].ema_delta_v, tbl[("site_b", 4.0)].count)
    assert b_after == b_before
    assert tbl[("site_a", 4.0)].ema_delta_v < 0


def test_ordinary_abstention_budget_filtering_and_cross_basis_guard() -> None:
    sched = UtilityDirectedScheduler(
        ("block",),
        (4.0, 64.0),
        research_mode=False,
        exploration_budget=0,
    )
    ctx = _ctx("block", basis_digest="basis_A")

    # Without positive evidence and outside research_mode, must choose ORDINARY_SITE
    choice0 = sched.select(context=ctx, policy="utility")
    assert choice0.site == ORDINARY_SITE
    assert choice0.reason == "no_favorable_gain_evidence"

    # Give positive gain to 64 MiB tier and negative to 4 MiB, then cap RAM headroom at 16 MiB
    sched.record_action("t1", 1, "block", 64.0)
    sched.observe_feedback(
        FeedbackSignal(
            step=1,
            trajectory_id="t1",
            site="block",
            ram_mib=64.0,
            partition="dev_train",
            signal_kind="lbf_bellman_regret",
            delta_v=-0.10,  # direction=-1 -> normalized gain = +0.10
            delta_cpu_seconds=1.0,
            delta_ram_mib=64.0,
            context=ctx,
            state_signature=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            component_ids=("c0",),
        )
    )
    # 64 MiB exceeds 16 MiB live headroom -> filtered out -> falls back to ordinary!
    choice_capped = sched.select(context=ctx, policy="utility", live_ram_headroom_bytes=16 * 1024 * 1024)
    assert choice_capped.site == ORDINARY_SITE

    # Retrieval under same basis succeeds; cross-basis rejected unless validated_transport_ok=True
    same = sched.retrieve_repairs(np.array([1.0, 0.0, 0.0], dtype=np.float32), query_context=ctx)
    assert len(same) == 1
    diff_ctx = _ctx("block", basis_digest="basis_B")
    cross_blocked = sched.retrieve_repairs(np.array([1.0, 0.0, 0.0], dtype=np.float32), query_context=diff_ctx)
    assert len(cross_blocked) == 0
    assert sched.cross_basis_rejections == 1
    cross_ok = sched.retrieve_repairs(
        np.array([1.0, 0.0, 0.0], dtype=np.float32), query_context=diff_ctx, validated_transport_ok=True
    )
    assert len(cross_ok) == 1


def test_separate_metric_contexts_do_not_mix_units() -> None:
    sched = UtilityDirectedScheduler(("block",), (4.0,))
    ctx_regret = _ctx("block", task_id="lbf20d", metric_name="regret", metric_direction=-1, metric_scale=1.0)
    ctx_proof = _ctx("block", task_id="lean", metric_name="close_at_8", metric_direction=1, metric_scale=100.0)

    sched.record_action("tr", 1, "block", 4.0)
    sched.observe_feedback(
        FeedbackSignal(
            step=1,
            trajectory_id="tr",
            site="block",
            ram_mib=4.0,
            partition="dev_train",
            signal_kind="lbf_bellman_regret",
            delta_v=-0.5,
            delta_cpu_seconds=1.0,
            delta_ram_mib=4.0,
            context=ctx_regret,
        )
    )
    # Proof context must remain uncontaminated (falls back to ordinary)
    assert sched.select(context=ctx_proof, policy="utility").site == ORDINARY_SITE
    assert sched.select(context=ctx_regret, policy="utility").site == "block"
