"""MMS-06 Sections 3 & 5: GradientBank profiler & repaired LBF-20D break-even campaign.

1. Profiles LoopedTransformers@3dad5bd9 GradientBank vs vectorized FastGradientBank
   with per-component timing (key construction, lookup, insertion, copying, refresh,
   hook dispatch, scheduler).
2. Runs repaired LBF-20D comparisons under BOTH fixed total CPU budget (B_CPU) and
   fixed exposure (updates=400), with:
   - Strictly disjoint train (96), feedback_dev_train (dev6[:16]), final_dev (dev6[16:32]).
   - Shared initial checkpoint, optimizer, RNG stream, and identical CPU stopping rule.
   - Complete non-overlapping 8-bucket CostLedger (CPU + wall separately).
   - Method-by-refresh 2x2 (svd_frozen, svd_refreshed, sparse_frozen, sparse_refreshed),
     plus ordinary, dense, retrospective_end_diagnostic, utility_scheduler, shuffled_feedback.
   - Paired per-root & per-seed Delta_V_pi(B_CPU, B_RAM) = V(theta_pi) - V(theta_ordinary).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from datagoblin.training_gym.factor_maps import (
    CanonicalFactorMap,
    canonical_svd_factors,
    canonicalize_factors,
    select_and_project_numpy,
)
from datagoblin.training_gym.fast_gradient_bank import CostLedger, FastGradientBank
from datagoblin.training_gym.mms_utility import (
    ORDINARY_SITE,
    FeedbackContext,
    FeedbackSignal,
    UtilityDirectedScheduler,
)

# Import sparse factor fitter from sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_bpd_map import fit_sparse_factors  # type: ignore[import-not-found]


def state_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().numpy().astype("<f4").tobytes())
    return h.hexdigest()


def opt_sha256(opt: torch.optim.Optimizer) -> str:
    h = hashlib.sha256()
    sd = opt.state_dict()
    for k in sorted(sd.get("state", {})):
        entry = sd["state"][k]
        for fk in sorted(entry):
            v = entry[fk]
            if isinstance(v, torch.Tensor):
                h.update(v.detach().cpu().numpy().astype("<f4").tobytes())
            else:
                h.update(str(v).encode("utf-8"))
    return h.hexdigest()


def split_digest(problems: list[Any]) -> str:
    h = hashlib.sha256()
    for p in problems:
        h.update(json.dumps({"masks": list(p.masks), "weights": list(p.weights)}, sort_keys=True).encode())
    return h.hexdigest()


def profile_gradient_banks(lt_repo: Path) -> dict[str, Any]:
    """Profile 3dad5bd9 GradientBank vs vectorized FastGradientBank on exact LBF-20D shape."""
    sys.path.insert(0, str(lt_repo))
    from research_gym.interp.mms import GradientBank as OldBank  # type: ignore[import-not-found]

    torch.set_num_threads(2)
    rng = np.random.default_rng(2026)
    shape = (64,)
    mib = 4.0
    n_steps = 240
    batch = 128

    keys_seq = [rng.integers(0, 12288, size=batch, dtype=np.int64) for _ in range(n_steps)]
    vals_seq = [rng.standard_normal((batch, *shape)).astype(np.float32) for _ in range(n_steps)]

    # 1. Profile old 3dad5bd9 GradientBank
    old_bank = OldBank(mib, shape)
    t_key_c = t_read_c = t_write_c = t_copy_c = 0.0
    t_old_w0, t_old_c0 = time.perf_counter(), time.process_time()
    for step in range(n_steps):
        tc = time.process_time()
        tk = torch.from_numpy(keys_seq[step]) * 3 + 1
        t_key_c += time.process_time() - tc

        tc = time.process_time()
        tv = torch.from_numpy(vals_seq[step])
        t_copy_c += time.process_time() - tc

        tc = time.process_time()
        _, _ = old_bank.read(tk, step, 32768)
        t_read_c += time.process_time() - tc

        tc = time.process_time()
        old_bank.write(tk, tv, step)
        t_write_c += time.process_time() - tc
    old_wall = time.perf_counter() - t_old_w0
    old_cpu = time.process_time() - t_old_c0

    # 2. Profile new vectorized FastGradientBank
    new_bank = FastGradientBank(mib, shape)
    nf_key_c = nf_read_c = nf_write_c = 0.0
    t_new_w0, t_new_c0 = time.perf_counter(), time.process_time()
    for step in range(n_steps):
        tc = time.process_time()
        nk = keys_seq[step] * 3 + 1
        nf_key_c += time.process_time() - tc

        tc = time.process_time()
        _, _ = new_bank.read(nk, step, 32768)
        nf_read_c += time.process_time() - tc

        tc = time.process_time()
        new_bank.write(nk, vals_seq[step], step)
        nf_write_c += time.process_time() - tc
    new_wall = time.perf_counter() - t_new_w0
    new_cpu = time.process_time() - t_new_c0

    # 3. Profile canonical SVD refresh vs sparse MSE+L1 fit on (64, 64) weight
    w_sample = torch.randn(64, 64)
    tc = time.process_time()
    for _ in range(10):
        _ = canonical_svd_factors(w_sample.numpy(), rank=16)
    svd_10_cpu = time.process_time() - tc

    _, _, sp_stats = fit_sparse_factors(w_sample, components=16, steps=60, seed=101)

    return {
        "shape": list(shape),
        "mib": mib,
        "steps": n_steps,
        "batch": batch,
        "old_3dad5bd9": {
            "total_wall_s": round(old_wall, 6),
            "total_cpu_s": round(old_cpu, 6),
            "key_cpu_s": round(t_key_c, 6),
            "copy_cpu_s": round(t_copy_c, 6),
            "read_cpu_s": round(t_read_c, 6),
            "write_cpu_s": round(t_write_c, 6),
            "hits": old_bank.hits,
            "queries": old_bank.queries,
            "collisions": old_bank.collisions,
            "allocated_bytes": old_bank.allocated_bytes,
        },
        "fast_vectorized": {
            "total_wall_s": round(new_wall, 6),
            "total_cpu_s": round(new_cpu, 6),
            "key_cpu_s": round(nf_key_c, 6),
            "read_cpu_s": round(nf_read_c, 6),
            "write_cpu_s": round(nf_write_c, 6),
            "hits": new_bank.hits,
            "queries": new_bank.queries,
            "collisions": new_bank.collisions,
            "allocated_bytes": new_bank.allocated_bytes,
        },
        "speedup_wall_x": round(old_wall / max(new_wall, 1e-6), 2),
        "speedup_cpu_x": round(old_cpu / max(new_cpu, 1e-6), 2),
        "svd_refresh_cpu_ms_per_call": round(100.0 * svd_10_cpu, 4),
        "sparse_fit_60steps_cpu_ms": round(1000.0 * sp_stats["timing"]["optimization_cpu_s"], 4),
    }


class FastMechanisticHook:
    """Vectorized MMS hook using FastGradientBank, canonical factor maps, and CostLedger."""

    def __init__(
        self,
        module: torch.nn.Linear,
        *,
        mib: float,
        mode: str,
        method: str,  # 'svd' | 'sparse' | 'frozen_external' | 'dense'
        refresh_every: int,  # 0 => frozen
        rank: int = 16,
        topk: int = 4,
        ratio: float = 0.2,
        max_age: int = 32768,
        sparse_steps: int = 60,
        frozen_fmap: CanonicalFactorMap | None = None,
        ledger: CostLedger,
    ) -> None:
        self.module = module
        self.mib = float(mib)
        self.mode = mode
        self.method = method
        self.refresh_every = int(refresh_every)
        self.rank = int(rank)
        self.topk = int(topk)
        self.ratio = float(ratio)
        self.max_age = int(max_age)
        self.sparse_steps = int(sparse_steps)
        self.ledger = ledger

        self.bank: FastGradientBank | None = None
        self.fmap: CanonicalFactorMap | None = frozen_fmap
        self.map_step = -1 if frozen_fmap is None else 0
        self.map_refreshes = 0
        self.admitted = 0
        self.updates = 0
        self.last_batch_hits = 0
        self.last_batch_admitted = 0
        self.last_comp_grad: np.ndarray | None = None

    def _ensure_map(self, step: int) -> None:
        if self.mode in ("observe", "dense"):
            return
        if self.fmap is not None and (self.refresh_every <= 0 or (step - self.map_step) < self.refresh_every):
            return
        bucket = "map" if self.fmap is None else "refresh"
        with self.ledger.scope(bucket):
            w_np = self.module.weight.detach().cpu().float().numpy()
            if self.method == "svd":
                self.fmap = canonical_svd_factors(w_np, rank=self.rank)
            elif self.method == "sparse":
                v_t, u_t, _ = fit_sparse_factors(
                    self.module.weight.detach().cpu(),
                    components=self.rank,
                    steps=self.sparse_steps,
                    seed=101 + step,
                )
                self.fmap = canonicalize_factors(v_t.numpy(), u_t.numpy())
            self.map_step = step
            self.map_refreshes += 1

    def _on_grad(self, keys_np: np.ndarray, step: int, x_t: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        with self.ledger.scope("lookup"):
            g_np = grad.detach().cpu().float().numpy()
            if not np.isfinite(g_np).all():
                return grad
            if self.bank is None:
                self.bank = FastGradientBank(self.mib, g_np.shape[1:])
            old, hit = self.bank.read(keys_np, step, self.max_age)
            flat = g_np.reshape(g_np.shape[0], -1)
            norms = np.linalg.norm(flat, axis=1, keepdims=True)
            normed = (flat / np.maximum(norms, 1e-12)).reshape(g_np.shape)
            self.bank.write(keys_np, normed, step)
            self.updates += g_np.shape[0]
            self.last_batch_hits = int(hit.sum())

        if self.mode == "observe" or not bool(hit.any()):
            self.last_batch_admitted = 0
            return grad

        with self.ledger.scope("feedback"):
            h_hit = old[hit]
            c_hit = g_np[hit]
            if self.mode == "dense":
                h_flat = h_hit.reshape(len(h_hit), -1)
                c_flat = c_hit.reshape(len(c_hit), -1)
                c_norm = np.linalg.norm(c_flat, axis=1)
                h_norm = np.linalg.norm(h_flat, axis=1)
                dot = np.sum(h_flat * c_flat, axis=1)
                elig = (h_norm > 1e-12) & (dot > 0) & (c_norm > 1e-12)
                scale = (self.ratio * c_norm * elig.astype(np.float32)) / np.maximum(h_norm, 1e-12)
                delta = (h_flat * scale[:, None]).reshape(c_hit.shape)
            else:
                assert self.fmap is not None
                x_hit = x_t.detach().cpu().float().numpy()[hit]
                _, delta, elig = select_and_project_numpy(
                    x_hit, h_hit, c_hit, self.fmap, topk=self.topk, mode=self.mode, ratio=self.ratio
                )
                # Compact component gradient projection for scheduler covariance tracking
                self.last_comp_grad = (c_hit.mean(axis=0) @ self.fmap.u.T)[:8].astype(np.float32)

            out_np = g_np.copy()
            out_np[hit] += delta
            n_adm = int(elig.sum())
            self.admitted += n_adm
            self.last_batch_admitted = n_adm
            return torch.from_numpy(out_np).to(dtype=grad.dtype)

    @contextmanager
    def capture(self, ids_np: np.ndarray, step: int, visit: int, visits: int):
        keys_np = ids_np.astype(np.int64) * visits + visit
        self._ensure_map(step)
        calls = 0

        def fwd_hook(_mod: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            nonlocal calls
            ordinal = calls
            calls += 1
            if ordinal != visit:
                return
            x_in = inputs[0].detach()
            output.register_hook(lambda g: self._on_grad(keys_np, step, x_in, g))

        handle = self.module.register_forward_hook(fwd_hook)
        try:
            yield
        finally:
            handle.remove()

    @property
    def bank_bytes(self) -> int:
        return 0 if self.bank is None else self.bank.allocated_bytes

    def receipt(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "method": self.method,
            "mib": self.mib,
            "refresh_every": self.refresh_every,
            "map_refreshes": self.map_refreshes,
            "basis_digest": None if self.fmap is None else self.fmap.basis_digest,
            "admitted": self.admitted,
            "updates": self.updates,
            "bank": None if self.bank is None else self.bank.receipt(),
        }


def run_lbf20d_arm(
    gym: Any,
    *,
    arm_name: str,
    seed: int,
    init_state: dict[str, torch.Tensor],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    m_train: torch.Tensor,
    fb_items: list[Any],
    final_items: list[Any],
    fb_split_digest: str,
    mode: str,  # 'fixed_cpu' | 'fixed_exposure'
    cpu_budget_s: float,
    max_updates: int,
    bank_mib: float,
    retrospective_fmap: CanonicalFactorMap | None = None,
) -> dict[str, Any]:
    """Run one registered LBF-20D arm with exact non-overlapping CostLedger and shared init."""
    ledger = CostLedger()
    proc = psutil.Process(os.getpid())
    model = gym.Net("recur3")
    model.load_state_dict({k: v.clone() for k, v in init_state.items()})
    start_ckpt_sha = state_sha256(model)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    gen = torch.Generator().manual_seed(seed + 7919)
    stream_h = hashlib.sha256()

    # Configure arm hooks / scheduler
    hook_main: FastMechanisticHook | None = None
    hooks_by_tier: dict[float, FastMechanisticHook] = {}
    scheduler: UtilityDirectedScheduler | None = None
    scheduler_log: list[dict[str, Any]] = []
    shuf_rng = random.Random(seed + 99001)

    if arm_name == "ordinary":
        pass
    elif arm_name == "dense":
        hook_main = FastMechanisticHook(
            model.blocks[0], mib=bank_mib, mode="dense", method="dense", refresh_every=0, ledger=ledger
        )
    elif arm_name == "svd_frozen":
        hook_main = FastMechanisticHook(
            model.blocks[0], mib=bank_mib, mode="bpd", method="svd", refresh_every=0, ledger=ledger
        )
    elif arm_name == "svd_refreshed":
        hook_main = FastMechanisticHook(
            model.blocks[0], mib=bank_mib, mode="bpd", method="svd", refresh_every=64, ledger=ledger
        )
    elif arm_name == "sparse_frozen":
        hook_main = FastMechanisticHook(
            model.blocks[0], mib=bank_mib, mode="bpd", method="sparse", refresh_every=0, sparse_steps=60, ledger=ledger
        )
    elif arm_name == "sparse_refreshed":
        hook_main = FastMechanisticHook(
            model.blocks[0], mib=bank_mib, mode="bpd", method="sparse", refresh_every=64, sparse_steps=60, ledger=ledger
        )
    elif arm_name == "retrospective_end_diagnostic":
        assert retrospective_fmap is not None
        hook_main = FastMechanisticHook(
            model.blocks[0],
            mib=bank_mib,
            mode="bpd",
            method="frozen_external",
            refresh_every=0,
            frozen_fmap=retrospective_fmap,
            ledger=ledger,
        )
    elif arm_name in ("utility_scheduler", "shuffled_feedback"):
        scheduler = UtilityDirectedScheduler(
            sites=("block",),
            ram_tiers_mib=(0.5, bank_mib),
            lambda_ram=0.005,
            exploration_bonus=0.02,
            history_budget_bytes=128 * 1024,
            cov_dim=8,
            research_mode=True,
            exploration_budget=3,
        )
        for tier in (0.5, bank_mib):
            hooks_by_tier[tier] = FastMechanisticHook(
                model.blocks[0], mib=tier, mode="bpd", method="svd", refresh_every=64, ledger=ledger
            )
    else:
        raise ValueError(f"unknown arm: {arm_name}")

    # Track initial feedback-partition regret for scheduler arms (charged to feedback/verifier!)
    last_fb_regret: float | None = None
    active_sched_site = ORDINARY_SITE
    active_sched_tier = 0.0
    window_start_cpu = 0.0
    window_steps = 40
    total_ram_cap_bytes = int((bank_mib + 1.0) * (1 << 20))  # bank + scheduler cap

    if scheduler is not None:
        with ledger.scope("verifier"):
            rows_fb = gym.evaluate(model, fb_items)
            last_fb_regret = float(statistics.mean(r["regret"] for r in rows_fb))

    updates = 0
    step_cap = max_updates if mode == "fixed_exposure" else 100_000
    batch = gym.BATCH
    visits = 3
    peak_rss_bytes = int(proc.memory_info().rss)

    for step in range(step_cap):
        # Identical CPU stopping rule across all arms in fixed_cpu mode
        if mode == "fixed_cpu" and ledger.total_cpu_seconds >= cpu_budget_s:
            break

        # Scheduler decision & feedback observation every window_steps
        if scheduler is not None and (step % window_steps == 0):
            if step > 0 and last_fb_regret is not None:
                with ledger.scope("verifier"):
                    rows_fb = gym.evaluate(model, fb_items)
                    cur_fb_regret = float(statistics.mean(r["regret"] for r in rows_fb))
                delta_regret = cur_fb_regret - last_fb_regret
                if arm_name == "shuffled_feedback":
                    # Permuted/sign-randomized feedback control
                    delta_regret = shuf_rng.choice([-1.0, 1.0]) * abs(delta_regret)
                window_cpu = max(1e-6, ledger.total_cpu_seconds - window_start_cpu)
                with ledger.scope("scheduler"):
                    basis_dig = (
                        hooks_by_tier[active_sched_tier].fmap.basis_digest
                        if active_sched_site != ORDINARY_SITE
                        and hooks_by_tier[active_sched_tier].fmap is not None
                        else "ordinary_none"
                    )
                    ctx = FeedbackContext(
                        task_id="lbf20d",
                        metric_name="mean_regret",
                        metric_direction=-1,
                        metric_scale=1.0,
                        split_digest=fb_split_digest,
                        feedback_horizon=window_steps,
                        checkpoint_sha256=state_sha256(model),
                        optimizer_sha256=opt_sha256(opt),
                        site=active_sched_site,
                        basis_digest=basis_dig,
                    )
                    scheduler.observe_feedback(
                        FeedbackSignal(
                            step=step,
                            trajectory_id=f"seed{seed}_w{step // window_steps}",
                            site=active_sched_site,
                            ram_mib=active_sched_tier,
                            partition="dev_train",
                            signal_kind="lbf_bellman_regret",
                            delta_v=delta_regret,
                            delta_cpu_seconds=window_cpu,
                            delta_ram_mib=active_sched_tier,
                            context=ctx,
                        )
                    )
                last_fb_regret = cur_fb_regret

            # Choose next window action subject to remaining CPU and live RAM cap
            with ledger.scope("scheduler"):
                rem_cpu = max(0.0, cpu_budget_s - ledger.total_cpu_seconds) if mode == "fixed_cpu" else None
                live_bank = sum(h.bank_bytes for h in hooks_by_tier.values())
                avail_headroom = max(0, total_ram_cap_bytes - scheduler.estimated_charge_bytes - live_bank)
                sel_ctx = FeedbackContext(
                    task_id="lbf20d",
                    metric_name="mean_regret",
                    metric_direction=-1,
                    metric_scale=1.0,
                    split_digest=fb_split_digest,
                    feedback_horizon=window_steps,
                    checkpoint_sha256=state_sha256(model),
                    optimizer_sha256=opt_sha256(opt),
                    site="block",
                    basis_digest="block_svd",
                )
                choice = scheduler.select(
                    context=sel_ctx,
                    policy="utility",
                    remaining_cpu_seconds=rem_cpu,
                    live_ram_headroom_bytes=avail_headroom if live_bank == 0 else total_ram_cap_bytes,
                )
                active_sched_site = choice.site
                active_sched_tier = choice.ram_mib
                scheduler.record_action(
                    f"seed{seed}_w{(step // window_steps) + 1}", step, active_sched_site, active_sched_tier
                )
                window_start_cpu = ledger.total_cpu_seconds
                scheduler_log.append(
                    {
                        "step": step,
                        "site": choice.site,
                        "ram_mib": choice.ram_mib,
                        "reason": choice.reason,
                        "used_exploration": choice.used_exploration,
                        "empirical_gain": round(choice.empirical_gain, 6),
                        "utility": round(choice.utility, 6),
                    }
                )
                if mode == "fixed_cpu" and ledger.total_cpu_seconds >= cpu_budget_s:
                    break

        active_hook = (
            hook_main
            if scheduler is None
            else (None if active_sched_site == ORDINARY_SITE else hooks_by_tier[active_sched_tier])
        )

        with ledger.scope("train"):
            ids = torch.randint(len(x_train), (batch,), generator=gen)
            ids_np = ids.numpy()
            stream_h.update(ids_np.tobytes())
            opt.zero_grad(set_to_none=True)
            ctx_mgr = (
                active_hook.capture(ids_np, step, visits // 2, visits)
                if active_hook is not None
                else nullcontext()
            )
            with ctx_mgr:
                pred = model(x_train[ids])
                valid = m_train[ids]
                val = ((pred - y_train[ids]).square() * valid).sum() / valid.sum()
                val.backward()
            opt.step()

        if scheduler is not None and active_hook is not None:
            with ledger.scope("scheduler"):
                px_ctx = FeedbackContext(
                    task_id="lbf20d",
                    metric_name="mean_regret",
                    metric_direction=-1,
                    metric_scale=1.0,
                    split_digest=fb_split_digest,
                    feedback_horizon=window_steps,
                    checkpoint_sha256=start_ckpt_sha,
                    optimizer_sha256="live",
                    site="block",
                    basis_digest="block_svd",
                )
                scheduler.record_proxy_counts(
                    px_ctx,
                    "block",
                    active_sched_tier,
                    hits=active_hook.last_batch_hits,
                    admitted=active_hook.last_batch_admitted,
                    comp_grad=active_hook.last_comp_grad,
                )

        updates += 1
        if (updates & 31) == 0:
            rss_now = int(proc.memory_info().rss)
            if rss_now > peak_rss_bytes:
                peak_rss_bytes = rss_now

    # Final unsteered evaluation on strictly disjoint final_dev roots (outside training CPU cap)
    model.eval()
    eval_w0, eval_c0 = time.perf_counter(), time.process_time()
    final_rows = gym.evaluate(model, final_items)
    eval_wall = time.perf_counter() - eval_w0
    eval_cpu = time.process_time() - eval_c0
    mean_regret = float(statistics.mean(r["regret"] for r in final_rows))
    mean_exp_q = float(statistics.mean(r["expected_queries"] for r in final_rows))

    rss_end = int(proc.memory_info().rss)
    peak_rss_bytes = max(peak_rss_bytes, rss_end)
    live_bank_bytes = (
        hook_main.bank_bytes
        if hook_main is not None
        else sum(h.bank_bytes for h in hooks_by_tier.values())
    )
    sched_est_bytes = 0 if scheduler is None else scheduler.estimated_charge_bytes
    sched_owned_bytes = 0 if scheduler is None else scheduler.owned_array_bytes

    return {
        "arm": arm_name,
        "mode": mode,
        "seed": seed,
        "start_checkpoint_sha256": start_ckpt_sha,
        "end_checkpoint_sha256": state_sha256(model),
        "stream_sha256": stream_h.hexdigest(),
        "updates": updates,
        "executed_visits": updates * batch * visits,
        "cpu_budget_s": cpu_budget_s if mode == "fixed_cpu" else None,
        "cpu_overshoot_or_unused_s": round(ledger.total_cpu_seconds - cpu_budget_s, 6)
        if mode == "fixed_cpu"
        else None,
        "ledger": ledger.receipt(),
        "final_dev": {
            "mean_regret": round(mean_regret, 6),
            "value_neg_regret": round(-mean_regret, 6),
            "mean_expected_queries": round(mean_exp_q, 6),
            "per_root_regret": [round(float(r["regret"]), 6) for r in final_rows],
            "eval_wall_s": round(eval_wall, 6),
            "eval_cpu_s": round(eval_cpu, 6),
        },
        "memory": {
            "bank_owned_bytes": live_bank_bytes,
            "scheduler_owned_bytes": sched_owned_bytes,
            "scheduler_estimated_charge_bytes": sched_est_bytes,
            "total_estimated_live_bytes": live_bank_bytes + sched_est_bytes,
            "total_cap_bytes": total_ram_cap_bytes,
            "measured_process_rss_peak_bytes": peak_rss_bytes,
        },
        "hook_receipt": None if hook_main is None else hook_main.receipt(),
        "scheduler_receipt": None
        if scheduler is None
        else scheduler.receipt(measured_process_rss_bytes=peak_rss_bytes),
        "scheduler_log": scheduler_log,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lt-repo", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 107, 113])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cpu-budget", type=float, default=4.0)
    p.add_argument("--fixed-updates", type=int, default=400)
    p.add_argument("--bank-mib", type=float, default=4.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    lbf_dir = args.lt_repo / "research" / "lite_braille_fresco"
    sys.path.insert(0, str(lbf_dir))
    import lbf20d_learning as gym  # type: ignore[import-not-found]

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    # 1. Profile old 3dad5bd9 GradientBank vs vectorized FastGradientBank
    bank_profile = profile_gradient_banks(args.lt_repo)
    print(
        f"BANK PROFILE: old={bank_profile['old_3dad5bd9']['total_cpu_s']:.3f}s CPU "
        f"vs fast={bank_profile['fast_vectorized']['total_cpu_s']:.3f}s CPU "
        f"({bank_profile['speedup_cpu_x']}x CPU speedup, {bank_profile['speedup_wall_x']}x wall speedup)",
        flush=True,
    )

    # 2. Build strictly disjoint cohorts: train (96), feedback_dev_train (dev6[:16]), final_dev (dev6[16:32])
    cohorts = gym.tasks()
    train_probs = cohorts["train"][:96]
    fb_probs = cohorts["dev6"][:16]
    final_probs = cohorts["dev6"][16:32]

    rng_data = random.Random(2026112510)
    train_data = [gym.dataset(pr, 128, rng_data) for pr in train_probs]
    x_train, y_train, m_train = (torch.cat([d[i] for d in train_data]) for i in (1, 2, 3))
    fb_items = [(pr, gym.dataset(pr), gym.analytic(pr)) for pr in fb_probs]
    final_items = [(pr, gym.dataset(pr), gym.analytic(pr)) for pr in final_probs]

    digests = {
        "train_96": split_digest(train_probs),
        "feedback_dev_train_16": split_digest(fb_probs),
        "final_dev_16": split_digest(final_probs),
    }

    arms_order = (
        "ordinary",
        "dense",
        "svd_frozen",
        "svd_refreshed",
        "sparse_frozen",
        "sparse_refreshed",
        "retrospective_end_diagnostic",
        "utility_scheduler",
        "shuffled_feedback",
    )

    all_runs: dict[str, list[dict[str, Any]]] = {"fixed_cpu": [], "fixed_exposure": []}

    for seed in args.seeds:
        torch.manual_seed(seed)
        base_model = gym.Net("recur3")
        init_state = copy.deepcopy(base_model.state_dict())

        # First train a separate end-of-training ordinary model for the retrospective diagnostic map
        retro_model = gym.Net("recur3")
        retro_model.load_state_dict({k: v.clone() for k, v in init_state.items()})
        retro_opt = torch.optim.AdamW(retro_model.parameters(), lr=0.002, weight_decay=0.0001)
        retro_gen = torch.Generator().manual_seed(seed + 7919)
        for _ in range(args.fixed_updates):
            ids = torch.randint(len(x_train), (gym.BATCH,), generator=retro_gen)
            retro_opt.zero_grad(set_to_none=True)
            pred = retro_model(x_train[ids])
            valid = m_train[ids]
            loss = ((pred - y_train[ids]).square() * valid).sum() / valid.sum()
            loss.backward()
            retro_opt.step()
        v_ret, u_ret, _ = fit_sparse_factors(retro_model.blocks[0].weight, components=16, steps=60, seed=seed)
        retro_fmap = canonicalize_factors(v_ret.numpy(), u_ret.numpy())

        for mode in ("fixed_cpu", "fixed_exposure"):
            print(f"\n=== Seed {seed} | {mode} ===", flush=True)
            for arm_name in arms_order:
                row = run_lbf20d_arm(
                    gym,
                    arm_name=arm_name,
                    seed=seed,
                    init_state=init_state,
                    x_train=x_train,
                    y_train=y_train,
                    m_train=m_train,
                    fb_items=fb_items,
                    final_items=final_items,
                    fb_split_digest=digests["feedback_dev_train_16"],
                    mode=mode,
                    cpu_budget_s=args.cpu_budget,
                    max_updates=args.fixed_updates,
                    bank_mib=args.bank_mib,
                    retrospective_fmap=retro_fmap if arm_name == "retrospective_end_diagnostic" else None,
                )
                all_runs[mode].append(row)
                ld = row["ledger"]
                print(
                    f"  {arm_name:28s} upd={row['updates']:4d} regret={row['final_dev']['mean_regret']:.4f} "
                    f"C_tot={ld['total_cpu_seconds']:5.2f}s (tr={ld['cpu_seconds']['train']:4.2f} "
                    f"lk={ld['cpu_seconds']['lookup']:4.2f} fb={ld['cpu_seconds']['feedback']:4.2f} "
                    f"mp={ld['cpu_seconds']['map']+ld['cpu_seconds']['refresh']:4.2f} "
                    f"vr={ld['cpu_seconds']['verifier']:4.2f}) wall={ld['total_wall_seconds']:5.2f}s",
                    flush=True,
                )

    # Aggregate Delta_V_pi = V(pi) - V(ordinary) = -(regret(pi) - regret(ordinary))
    summaries: dict[str, Any] = {}
    for mode in ("fixed_cpu", "fixed_exposure"):
        by_arm: dict[str, list[dict[str, Any]]] = {}
        for r in all_runs[mode]:
            by_arm.setdefault(r["arm"], []).append(r)
        ord_by_seed = {r["seed"]: r for r in by_arm["ordinary"]}
        mode_sum: dict[str, Any] = {}
        for arm_name in arms_order:
            rows = by_arm[arm_name]
            per_seed_dv = [
                round(r["final_dev"]["value_neg_regret"] - ord_by_seed[r["seed"]]["final_dev"]["value_neg_regret"], 6)
                for r in rows
            ]
            # Per-root mean delta_V averaged across seeds (unit of root independence = 16 disjoint roots)
            per_root_dv = []
            for r_i in range(len(final_items)):
                vals = [
                    -(
                        r["final_dev"]["per_root_regret"][r_i]
                        - ord_by_seed[r["seed"]]["final_dev"]["per_root_regret"][r_i]
                    )
                    for r in rows
                ]
                per_root_dv.append(statistics.mean(vals))
            pr_arr = np.asarray(per_root_dv, dtype=np.float64)
            ord_cpu_mean = statistics.mean(ord_by_seed[r["seed"]]["ledger"]["total_cpu_seconds"] for r in rows)
            arm_cpu_mean = statistics.mean(r["ledger"]["total_cpu_seconds"] for r in rows)
            ord_step_cpu = statistics.mean(
                ord_by_seed[r["seed"]]["ledger"]["total_cpu_seconds"] / max(1, ord_by_seed[r["seed"]]["updates"])
                for r in rows
            )
            arm_step_cpu = statistics.mean(r["ledger"]["total_cpu_seconds"] / max(1, r["updates"]) for r in rows)
            mode_sum[arm_name] = {
                "mean_updates": round(statistics.mean(r["updates"] for r in rows), 1),
                "mean_regret": round(statistics.mean(r["final_dev"]["mean_regret"] for r in rows), 6),
                "mean_delta_V_vs_ordinary": round(statistics.mean(per_seed_dv), 6),
                "per_seed_delta_V": per_seed_dv,
                "positive_all_seeds": all(dv > 0 for dv in per_seed_dv),
                "per_root_delta_V_mean": round(float(pr_arr.mean()), 6),
                "per_root_delta_V_se": round(float(pr_arr.std(ddof=1) / np.sqrt(len(pr_arr))), 6),
                "mean_total_cpu_s": round(arm_cpu_mean, 4),
                "mean_total_wall_s": round(statistics.mean(r["ledger"]["total_wall_seconds"] for r in rows), 4),
                "step_cpu_overhead_ratio_vs_ordinary": round(arm_step_cpu / max(ord_step_cpu, 1e-9), 3),
                "total_cpu_ratio_vs_ordinary": round(arm_cpu_mean / max(ord_cpu_mean, 1e-9), 3),
                "mean_ledger_cpu": {
                    k: round(statistics.mean(r["ledger"]["cpu_seconds"][k] for r in rows), 4)
                    for k in rows[0]["ledger"]["cpu_seconds"]
                },
                "max_bank_owned_bytes": max(r["memory"]["bank_owned_bytes"] for r in rows),
                "max_scheduler_est_bytes": max(r["memory"]["scheduler_estimated_charge_bytes"] for r in rows),
                "max_process_rss_bytes": max(r["memory"]["measured_process_rss_peak_bytes"] for r in rows),
            }
        summaries[mode] = mode_sum

    receipt = {
        "status": "PUBLIC_DEVELOPMENT_ONLY",
        "experiment": "MMS-06-LBF20D-BREAK-EVEN",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": args.threads,
        "seeds": args.seeds,
        "cpu_budget_s": args.cpu_budget,
        "fixed_updates": args.fixed_updates,
        "bank_mib": args.bank_mib,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split_digests": digests,
        "bank_profile": bank_profile,
        "summaries": summaries,
        "runs": all_runs,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raw_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    compact = {k: v for k, v in receipt.items() if k != "runs"}
    (args.out / "summary.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
