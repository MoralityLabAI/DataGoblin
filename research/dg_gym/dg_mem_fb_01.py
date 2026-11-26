"""DG-MEM-FB-01: Resource-aware mechanistic feedback on heterogeneous localized causal defects.

Public development only; local hardware, no paid compute.
Demonstrates a small tied recurrent reasoning model on a verified multi-regime
modular-state task with localized BPD-component causal defects, repaired under
matched visit-credit budgets using a persistent RAM-bounded mechanistic map
controlled by ``mms_ram_budget_mb``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F

from datagoblin.training_gym.mechanistic_map import (
    CausalContribution,
    ComponentStability,
    InterventionOutcome,
    JacobianGradAlignment,
    MarginalGainEstimate,
    MechanisticMapStore,
)
from datagoblin.training_gym.resource_plan import paired_outcomes
from datagoblin.training_gym.steering_feedback import Cost

MODULUS = 7
NUM_REGIMES = 4
VISITS = 4
PERM = (3, 5, 0, 6, 2, 1, 4)
TARGET_LAYERS = ("l0_up", "l0_down", "l1_up", "l1_down")
BPD_RANK = 16


def sample_verified_batch(
    rng: np.random.Generator,
    n: int,
    visits: int = VISITS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample verified multi-regime modular reasoning instances with exact step traces.

    Returns:
        regimes: [n] int64 in {0..3}
        init_state: [n] int64 in {0..6}
        ops: [n, visits, 2] int64 (a in 1..6, b in 0..6)
        states: [n, visits] int64 exact intermediate & terminal states
    """
    regimes = rng.integers(0, NUM_REGIMES, size=n, dtype=np.int64)
    init_state = rng.integers(0, MODULUS, size=n, dtype=np.int64)
    a = rng.integers(1, MODULUS, size=(n, visits), dtype=np.int64)
    b = rng.integers(0, MODULUS, size=(n, visits), dtype=np.int64)
    ops = np.stack([a, b], axis=-1)
    states = np.zeros((n, visits), dtype=np.int64)
    perm_arr = np.array(PERM, dtype=np.int64)

    s = init_state.copy()
    for t in range(visits):
        at = a[:, t]
        bt = b[:, t]
        s0 = (s + at + bt) % MODULUS
        s1 = np.where((s % 2) == 0, (s + at) % MODULUS, (s - bt) % MODULUS)
        s2 = (s + np.where(s < 3, at, bt)) % MODULUS
        s3 = (perm_arr[(s + at) % MODULUS] + bt) % MODULUS
        s = np.where(
            regimes == 0,
            s0,
            np.where(regimes == 1, s1, np.where(regimes == 2, s2, s3)),
        )
        states[:, t] = s

    return (
        torch.from_numpy(regimes),
        torch.from_numpy(init_state),
        torch.from_numpy(ops),
        torch.from_numpy(states),
    )


class RecurrentReasoner(nn.Module):
    """Small tied FiLM-gated recurrent reasoning model with explicit linear sites for BPD."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_emb = nn.Embedding(MODULUS, hidden_size)
        self.regime_emb = nn.Embedding(NUM_REGIMES, hidden_size)
        self.a_emb = nn.Embedding(MODULUS, hidden_size // 2)
        self.b_emb = nn.Embedding(MODULUS, hidden_size // 2)
        self.visit_emb = nn.Embedding(VISITS, hidden_size)

        self.gate0 = nn.Linear(hidden_size * 2, hidden_size)
        self.gate1 = nn.Linear(hidden_size * 2, hidden_size)

        self.norm0 = nn.LayerNorm(hidden_size)
        self.l0_up = nn.Linear(hidden_size, hidden_size)
        self.l0_down = nn.Linear(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.l1_up = nn.Linear(hidden_size, hidden_size)
        self.l1_down = nn.Linear(hidden_size, hidden_size)

        self.out_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, MODULUS)

    def forward_with_traces(
        self,
        regimes: torch.Tensor,
        init_state: torch.Tensor,
        ops: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, list[torch.Tensor]]]:
        """Run V tied recurrent visits, returning logits [B, V, M], states, and layer acts."""
        r_emb = self.regime_emb(regimes)
        h = self.state_emb(init_state) + r_emb
        step_logits: list[torch.Tensor] = []
        visit_states: list[torch.Tensor] = []
        layer_acts: dict[str, list[torch.Tensor]] = {name: [] for name in TARGET_LAYERS}
        n_visits = ops.shape[1]
        for v in range(n_visits):
            op_vec = torch.cat([self.a_emb(ops[:, v, 0]), self.b_emb(ops[:, v, 1])], dim=-1)
            cond = torch.cat([op_vec, r_emb], dim=-1)
            v_idx = torch.full((h.shape[0],), v, dtype=torch.long, device=h.device)

            h_in = self.norm0(h + op_vec + self.visit_emb(v_idx))
            u0 = self.l0_up(h_in) * torch.tanh(self.gate0(cond))
            layer_acts["l0_up"].append(u0)
            d0 = self.l0_down(F.gelu(u0))
            layer_acts["l0_down"].append(d0)
            h = h + 0.5 * d0

            h_mid = self.norm1(h)
            u1 = self.l1_up(h_mid) * torch.tanh(self.gate1(cond))
            layer_acts["l1_up"].append(u1)
            d1 = self.l1_down(F.gelu(u1))
            layer_acts["l1_down"].append(d1)
            h = h + 0.5 * d1

            visit_states.append(h)
            step_logits.append(self.head(self.out_norm(h)))
        return torch.stack(step_logits, dim=1), visit_states, layer_acts


def checkpoint_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, param in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().numpy().astype("<f4").tobytes())
    return h.hexdigest()


def weight_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().astype("<f4").tobytes()).hexdigest()


def bpd_svd_decompose(weight: torch.Tensor, rank: int = BPD_RANK) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute rank-R BPD/SVD component decomposition W ≈ sum_c s_c u_c v_c^T."""
    w = weight.detach().cpu().float().numpy()
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    r = min(rank, len(s))
    u_r, s_r, vt_r = u[:, :r], s[:r], vt[:r, :]
    recon = (u_r * s_r[None, :]) @ vt_r
    denom = float(np.linalg.norm(w) * np.linalg.norm(recon))
    cos = float(np.sum(w * recon) / max(denom, 1e-12))
    return u_r.astype(np.float32), s_r.astype(np.float32), vt_r.astype(np.float32), cos


@torch.no_grad()
def evaluate_model(
    model: RecurrentReasoner,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    regimes, init_state, ops, states = (t.to(device) for t in batch)
    logits, _, _ = model.forward_with_traces(regimes, init_state, ops)
    term_logits = logits[:, -1, :]
    term_target = states[:, -1]
    preds = term_logits.argmax(dim=-1)
    correct = (preds == term_target).cpu().numpy().astype(np.int64)
    loss = float(F.cross_entropy(term_logits, term_target).item())
    per_regime: dict[str, float] = {}
    reg_np = regimes.cpu().numpy()
    for r in range(NUM_REGIMES):
        mask = reg_np == r
        per_regime[f"regime_{r}"] = float(correct[mask].mean()) if mask.any() else 0.0
    return {
        "accuracy": float(correct.mean()),
        "loss": loss,
        "per_regime": per_regime,
        "binary_outcomes": correct.tolist(),
    }


def train_competent_base(
    seed: int,
    device: torch.device,
    steps: int = 2200,
    batch_size: int = 128,
) -> tuple[RecurrentReasoner, dict[str, Any]]:
    """Train a competent base checkpoint on the verified multi-regime task."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    model = RecurrentReasoner(hidden_size=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-4)
    t0 = time.perf_counter()
    for _step in range(steps):
        regimes, init_state, ops, states = (t.to(device) for t in sample_verified_batch(rng, batch_size))
        logits, _, _ = model.forward_with_traces(regimes, init_state, ops)
        # Intermediate + terminal supervision gives crisp per-visit representations.
        loss = F.cross_entropy(logits.reshape(-1, MODULUS), states.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    wall_s = time.perf_counter() - t0
    return model, {"pretrain_steps": steps, "pretrain_wall_s": round(wall_s, 4)}


def inject_localized_bpd_defects(
    model: RecurrentReasoner,
    defect_layer: str = "l1_down",
    defect_components: tuple[int, ...] = (0, 1, 2, 3),
) -> dict[str, Any]:
    """Inject heterogeneous localized causal defects into a sparse subset of BPD components.

    Corrupts only ``defect_components`` in ``defect_layer`` with heterogeneous attenuation/flip
    magnitudes while leaving all other layers and remaining BPD components intact.
    """
    layer_mod: nn.Linear = getattr(model, defect_layer)
    w = layer_mod.weight.detach().cpu().float().numpy()
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    # Heterogeneous localized fault magnitudes (preserves healthy circuits while creating clear localized deficit).
    scales = (-0.8, -0.5, -0.2, 0.1)
    w_corrupt = w.copy()
    defect_ids: list[str] = []
    for idx, comp_idx in enumerate(defect_components):
        scale = scales[idx % len(scales)]
        delta = (scale - 1.0) * s[comp_idx] * np.outer(u[:, comp_idx], vt[comp_idx, :])
        w_corrupt += delta
        defect_ids.append(f"{defect_layer}:c{comp_idx}")
    with torch.no_grad():
        layer_mod.weight.copy_(torch.from_numpy(w_corrupt).to(layer_mod.weight.device, dtype=layer_mod.weight.dtype))
    return {
        "defect_layer": defect_layer,
        "defect_components": list(defect_components),
        "defect_ids": defect_ids,
        "defect_scales": list(scales[: len(defect_components)]),
    }


def populate_mechanistic_map(
    model: RecurrentReasoner,
    store: MechanisticMapStore,
    audit_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    *,
    step: int,
    base_bpd: dict[str, tuple[np.ndarray, np.ndarray]],
    prev_bpd: dict[str, tuple[np.ndarray, np.ndarray]],
    prev_grads: dict[str, np.ndarray],
) -> tuple[int, dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    """Populate all 8 sections of ``MechanisticMapStore`` on the feedback-audit cohort.

    Returns:
        audit_credits: exact visit credits consumed (1/fwd + 2/bwd per example-visit).
        cur_bpd: current (u, v) dict for stability tracking.
        cur_grads: current layer weight gradients for alignment tracking.
    """
    regimes, init_state, ops, states = (t.to(device) for t in audit_batch)
    n_audit = int(regimes.shape[0])
    audit_credits = 0

    # 1. Forward + backward pass on audit cohort for activation stats, gradients & Jacobian probes.
    model.zero_grad(set_to_none=True)
    logits, _visit_states, layer_acts = model.forward_with_traces(regimes, init_state, ops)
    term_logits = logits[:, -1, :]
    term_target = states[:, -1]
    loss = F.cross_entropy(term_logits, term_target)
    loss.backward()
    # Charge 1 fwd + 2 bwd per example-visit.
    audit_credits += 3 * n_audit * VISITS

    base_preds = term_logits.detach().argmax(dim=-1)
    base_correct = (base_preds == term_target).float()
    base_acc = float(base_correct.mean().item())
    # True class logit minus max non-target logit
    target_logits = term_logits.detach().gather(1, term_target[:, None]).squeeze(1)
    masked = term_logits.detach().clone()
    masked.scatter_(1, term_target[:, None], -1e9)
    base_margin = float((target_logits - masked.max(dim=1).values).mean().item())

    cur_bpd: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cur_grads: dict[str, np.ndarray] = {}

    # 2. Activation statistics + steering directions + cross-pass transport maps.
    with torch.no_grad():
        wrong_mask = base_preds != term_target
        right_mask = ~wrong_mask
        for layer_name in TARGET_LAYERS:
            acts_list = [a.detach().cpu().float().numpy() for a in layer_acts[layer_name]]
            for v, act in enumerate(acts_list):
                mean = act.mean(axis=0)
                var = act.var(axis=0)
                norm_mean = float(np.linalg.norm(act, axis=1).mean())
                if bool(wrong_mask.any()) and bool(right_mask.any()):
                    steer = act[right_mask.cpu().numpy()].mean(axis=0) - act[wrong_mask.cpu().numpy()].mean(axis=0)
                else:
                    steer = mean
                sn = float(np.linalg.norm(steer))
                steer = steer / max(sn, 1e-8)
                store.put_activation_stats(
                    layer=layer_name,
                    visit=v,
                    mean=mean,
                    var=var,
                    norm_mean=norm_mean,
                    steering_direction=steer,
                    sample_count=n_audit,
                    priority=1.0,
                    step=step,
                )
            # Cross-pass transport T_{v -> v+1} via ridge regression.
            for v in range(VISITS - 1):
                x_src = acts_list[v]
                y_dst = acts_list[v + 1]
                d = x_src.shape[1]
                reg = 1e-2 * np.eye(d, dtype=np.float32)
                t_mat = np.linalg.solve(x_src.T @ x_src + reg, x_src.T @ y_dst).T  # [d, d]
                pred_dst = x_src @ t_mat.T
                ss_res = float(np.sum((y_dst - pred_dst) ** 2))
                ss_tot = float(np.sum((y_dst - y_dst.mean(axis=0, keepdims=True)) ** 2)) + 1e-8
                r2 = max(0.0, 1.0 - ss_res / ss_tot)
                store.put_cross_pass_transport(
                    src_visit=v,
                    dst_visit=v + 1,
                    layer=layer_name,
                    matrix=t_mat,
                    max_rank=16,
                    r2_fidelity=r2,
                    priority=0.6 + 0.4 * r2,
                    step=step,
                )

    # 3. BPD decompositions, component stability, Jacobian/gradient alignment, causal ablation & marginal gain.
    # Use a compact 32-example subset for per-component causal ablation to keep audit overhead realistic.
    sub_n = min(32, n_audit)
    sub_reg, sub_init, sub_ops, sub_states = regimes[:sub_n], init_state[:sub_n], ops[:sub_n], states[:sub_n]

    for layer_name in TARGET_LAYERS:
        mod: nn.Linear = getattr(model, layer_name)
        w_sha = weight_sha256(mod.weight)
        u_r, s_r, vt_r, recon_cos = bpd_svd_decompose(mod.weight, rank=BPD_RANK)
        g_mat = (
            mod.weight.grad.detach().cpu().float().numpy()
            if mod.weight.grad is not None
            else np.zeros_like(mod.weight.detach().cpu().float().numpy())
        )
        cur_grads[layer_name] = g_mat
        prev_g = prev_grads.get(layer_name)

        # Layer-level gradient norm proxy for Jacobian sensitivity across visits
        w_orig = mod.weight.detach().clone()
        for c in range(u_r.shape[1]):
            cid = f"{layer_name}:c{c}"
            u_c = u_r[:, c]
            v_c = vt_r[c, :]
            sigma_c = float(s_r[c])
            cur_bpd[cid] = (u_c, v_c)

            # Projected gradient magnitude on rank-1 component (u_c, v_c): |u_c^T G v_c|
            g_proj = float(u_c @ g_mat @ v_c)
            g_comp_norm = float(np.linalg.norm(g_mat @ v_c) + np.linalg.norm(u_c @ g_mat))
            if prev_g is not None:
                prev_proj = float(u_c @ prev_g @ v_c)
                grad_cos_prev = float(np.sign(g_proj * prev_proj)) if abs(g_proj * prev_proj) > 1e-12 else 0.0
            else:
                grad_cos_prev = 1.0
            grad_cos_base = float(abs(g_proj) / (np.linalg.norm(g_mat) + 1e-8))

            # Component stability vs previous step & base
            u_prev, v_prev = prev_bpd.get(cid, (u_c, v_c))
            u_base, v_base = base_bpd.get(cid, (u_c, v_c))
            cos_u_prev = float(abs(np.dot(u_c, u_prev)))
            cos_v_prev = float(abs(np.dot(v_c, v_prev)))
            cos_u_base = float(abs(np.dot(u_c, u_base)))
            cos_v_base = float(abs(np.dot(v_c, v_base)))
            drift = max(0.0, 1.0 - 0.5 * (cos_u_prev + cos_v_prev))

            # Causal contribution via rank-1 sign-probe / ablation on audit subset (charged to audit_credits!)
            # Only probe top-8 singular components per layer by gradient projection to keep audit cost bounded.
            delta_acc = 0.0
            delta_margin = 0.0
            spec = 0.0
            if c < 8:
                with torch.no_grad():
                    # Test restoring/perturbing component c along negative gradient direction
                    probe_step = -np.sign(g_proj) * 0.5 * sigma_c
                    delta_w = torch.from_numpy((probe_step * np.outer(u_c, v_c)).astype(np.float32)).to(device)
                    mod.weight.copy_(w_orig + delta_w)
                    p_logits, _, _ = model.forward_with_traces(sub_reg, sub_init, sub_ops)
                    mod.weight.copy_(w_orig)
                    audit_credits += sub_n * VISITS  # 1 forward credit per example-visit

                    pt = p_logits[:, -1, :]
                    p_acc = float((pt.argmax(dim=-1) == sub_states[:, -1]).float().mean().item())
                    t_log = pt.gather(1, sub_states[:, -1, None]).squeeze(1)
                    m_log = pt.clone()
                    m_log.scatter_(1, sub_states[:, -1, None], -1e9)
                    p_margin = float((t_log - m_log.max(dim=1).values).mean().item())
                    delta_acc = p_acc - base_acc
                    delta_margin = p_margin - base_margin
                    spec = float(np.clip(delta_margin / (abs(base_margin) + 1.0), -1.0, 1.0))

            # Priority: components with large causal sensitivity + large gradient projection get highest priority.
            prio = 1.0 + 5.0 * max(0.0, delta_margin) + 2.0 * abs(g_proj)
            store.put_bpd_component(
                component_id=cid,
                layer=layer_name,
                rank_idx=c,
                u=u_c,
                v=v_c,
                singular_value=sigma_c,
                weight_sha256=w_sha,
                reconstruction_cosine=recon_cos,
                priority=prio,
                step=step,
            )
            store.put_causal_contribution(
                CausalContribution(
                    layer=layer_name,
                    visit=VISITS - 1,
                    component_id=cid,
                    accuracy_delta=delta_acc,
                    logit_diff_delta=delta_margin,
                    specificity=spec,
                    sample_count=sub_n if c < 8 else 0,
                ),
                priority=prio,
                step=step,
            )
            store.put_jacobian_grad_alignment(
                JacobianGradAlignment(
                    layer=layer_name,
                    visit=VISITS - 1,
                    component_id=cid,
                    jacobian_norm=g_comp_norm,
                    grad_cosine_prev=grad_cos_prev,
                    grad_cosine_base=grad_cos_base,
                    directional_sensitivity=abs(g_proj),
                ),
                priority=prio,
                step=step,
            )
            store.put_component_stability(
                ComponentStability(
                    component_id=cid,
                    step=step,
                    cos_u_prev=cos_u_prev,
                    cos_v_prev=cos_v_prev,
                    cos_u_base=cos_u_base,
                    cos_v_base=cos_v_base,
                    drift_rate=drift,
                ),
                priority=1.2,
            )
            # Combine empirical prior from previous intervention outcomes if available.
            past = store.recent_outcomes(cid)
            emp_bonus = float(np.mean([o.delta_acc for o in past[-3:]])) if past else 0.0
            est_gain = max(0.0, 0.6 * max(0.0, delta_margin) + 0.4 * abs(g_proj) + 0.5 * max(0.0, emp_bonus))
            store.put_marginal_gain(
                MarginalGainEstimate(
                    target_id=cid,
                    layer=layer_name,
                    visit=VISITS - 1,
                    component_id=cid,
                    estimated_gain=est_gain,
                    estimated_cost_visits=3 * 64 * VISITS,
                    estimated_wall_ns=2_000_000,
                    estimated_ram_bytes=512,
                    confidence=0.85 if c < 8 else 0.35,
                ),
                step=step,
            )

    return audit_credits, cur_bpd, cur_grads


def apply_bpd_subspace_step(
    model: RecurrentReasoner,
    opt: torch.optim.Optimizer,
    targets: list[tuple[str, int, np.ndarray, np.ndarray]],
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    *,
    subspace_boost: float = 4.0,
    residual_scale: float = 0.4,
) -> int:
    """Apply an AdamW step whose gradient is concentrated on selected BPD (u_c, v_c) subspaces.

    Uses the exact same AdamW optimizer class as ``task_reinvest``, but projects and boosts
    the gradient along the map-selected defective BPD components while damping updates on
    healthy layers/components to prevent destructive drift.
    """
    regimes, init_state, ops, states = (t.to(device) for t in batch)
    opt.zero_grad(set_to_none=True)
    logits, _, _ = model.forward_with_traces(regimes, init_state, ops)
    loss = F.cross_entropy(logits.reshape(-1, MODULUS), states.reshape(-1))
    loss.backward()

    by_layer: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for layer_name, _c_idx, u_c, v_c in targets:
        by_layer.setdefault(layer_name, []).append(
            (torch.from_numpy(u_c).to(device), torch.from_numpy(v_c).to(device))
        )

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad
            layer_key = name.split(".")[0]
            if name.endswith(".weight") and layer_key in by_layer:
                proj_g = torch.zeros_like(g)
                for u_t, v_t in by_layer[layer_key]:
                    ut_g = u_t @ g
                    g_v = g @ v_t
                    scalar = torch.dot(u_t, g_v)
                    proj_g += torch.outer(u_t, ut_g) + torch.outer(g_v, v_t) - scalar * torch.outer(u_t, v_t)
                param.grad.copy_(subspace_boost * proj_g + residual_scale * g)
            else:
                param.grad.mul_(residual_scale)

    opt.step()
    return 3 * int(regimes.shape[0]) * VISITS


def run_arm(
    *,
    arm: str,
    mms_ram_budget_mb: float,
    seed: int,
    defect_state: dict[str, torch.Tensor],
    defect_info: dict[str, Any],
    audit_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    eval_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    credit_cap: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run one repair arm from the exact same defective checkpoint under a matched credit cap."""
    model = RecurrentReasoner(hidden_size=64).to(device)
    model.load_state_dict({k: v.to(device).clone() for k, v in defect_state.items()})
    start_sha = checkpoint_sha256(model)

    rng = np.random.default_rng(50000 + seed)
    store = MechanisticMapStore(mms_ram_budget_mb=mms_ram_budget_mb)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    batch_size = 64
    step_credits = 3 * batch_size * VISITS
    credits = 0
    audit_credits_total = 0
    train_credits_total = 0
    updates = 0
    audits_run = 0
    audits_skipped = 0
    window_size = 25

    base_bpd: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    prev_bpd: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    prev_grads: dict[str, np.ndarray] = {}
    selected_targets: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    selected_ids_history: list[list[str]] = []

    pre_eval = evaluate_model(model, audit_batch, device)
    last_audit_acc = pre_eval["accuracy"]
    steps_since_audit = window_size

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    wall0, cpu0 = time.perf_counter(), time.process_time()

    window_idx = 0
    while credits + step_credits <= credit_cap:
        if (
            arm in ("mms_guided_tiny", "mms_guided_med", "mms_guided_full", "shuffled_map")
            and updates % window_size == 0
        ):
            need_audit = (updates == 0) or store.should_reaudit(
                drift_threshold=0.06, min_steps_since_audit=window_size, steps_since_audit=steps_since_audit
            )
            # Ensure we have enough remaining credits to audit + still do useful updates
            est_audit_cost = 3 * int(audit_batch[0].shape[0]) * VISITS + 32 * 32 * VISITS
            if need_audit and credits + est_audit_cost + step_credits <= credit_cap:
                a_cred, cur_bpd, cur_grads = populate_mechanistic_map(
                    model,
                    store,
                    audit_batch,
                    device,
                    step=updates,
                    base_bpd=base_bpd,
                    prev_bpd=prev_bpd,
                    prev_grads=prev_grads,
                )
                if not base_bpd:
                    base_bpd = cur_bpd
                prev_bpd = cur_bpd
                prev_grads = cur_grads
                credits += a_cred
                audit_credits_total += a_cred
                audits_run += 1
                steps_since_audit = 0

                # Select top-4 targets from map (or shuffled permutation for shuffled_map)
                if arm == "shuffled_map":
                    all_cids = list(cur_bpd.keys())
                    shuf_rng = np.random.default_rng(77000 + seed * 100 + window_idx)
                    chosen_cids = list(shuf_rng.choice(all_cids, size=4, replace=False))
                else:
                    top_est = store.select_top_targets(k=4, min_gain_per_visit=0.0)
                    chosen_cids = [e.component_id for e in top_est]
                    # Verify bridge to steering_feedback.select_window
                    _dec = store.select_feedback_window(
                        checkpoint=checkpoint_sha256(model),
                        feedback_cohort="audit_64",
                        step=updates,
                        window=window_size,
                        remaining=Cost(10**12, 10**12, 0, max(1, credit_cap - credits), 100, 10**9, 0),
                        competent=True,
                    )
                selected_targets = []
                for cid in chosen_cids:
                    comp = store.get_bpd_component(cid)
                    if comp is not None:
                        selected_targets.append((comp.layer, comp.rank_idx, comp.u, comp.v))
                    elif cid in cur_bpd:
                        # Cache miss due to tight mms_ram_budget_mb eviction: fallback to current factor
                        layer_n, c_str = cid.split(":c")
                        selected_targets.append((layer_n, int(c_str), cur_bpd[cid][0], cur_bpd[cid][1]))
                selected_ids_history.append(chosen_cids)
                window_idx += 1
            else:
                audits_skipped += 1

        if credits + step_credits > credit_cap:
            break

        batch = sample_verified_batch(rng, batch_size)
        if arm == "task_reinvest":
            regimes, init_state, ops, states = (t.to(device) for t in batch)
            logits, _, _ = model.forward_with_traces(regimes, init_state, ops)
            loss = F.cross_entropy(logits.reshape(-1, MODULUS), states.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            c_step = step_credits
        elif arm == "static_cyclic":
            layer_name = TARGET_LAYERS[(updates // 10) % len(TARGET_LAYERS)]
            mod: nn.Linear = getattr(model, layer_name)
            u_r, _, vt_r, _ = bpd_svd_decompose(mod.weight, rank=4)
            cyc_targets = [(layer_name, c, u_r[:, c], vt_r[c, :]) for c in range(4)]
            c_step = apply_bpd_subspace_step(model, opt, cyc_targets, batch, device)
        else:
            c_step = apply_bpd_subspace_step(model, opt, selected_targets, batch, device)

        credits += c_step
        train_credits_total += c_step
        updates += 1
        steps_since_audit += 1

        # Record intervention outcome at end of each window into persistent map
        if store.enabled and updates % window_size == 0 and selected_targets:
            cur_a = evaluate_model(model, audit_batch, device)["accuracy"]
            delta_a = cur_a - last_audit_acc
            for layer_n, c_idx, _, _ in selected_targets[:2]:
                store.record_intervention_outcome(
                    InterventionOutcome(
                        step=updates,
                        checkpoint_sha256=checkpoint_sha256(model),
                        target_id=f"{layer_n}:c{c_idx}",
                        action_kind="bpd_subspace_repair",
                        pre_acc=last_audit_acc,
                        post_acc=cur_a,
                        delta_acc=delta_a,
                        wall_seconds=time.perf_counter() - wall0,
                        cost_visits=window_size * step_credits,
                        accepted=delta_a >= 0.0,
                    )
                )
            last_audit_acc = cur_a

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall0
    cpu_s = time.process_time() - cpu0
    vram_peak_mb = round(torch.cuda.max_memory_allocated(device) / 1e6, 3) if device.type == "cuda" else 0.0

    final_eval = evaluate_model(model, eval_batch, device)
    # Localization precision: fraction of selected IDs that hit the true injected defect_ids
    true_defects = set(defect_info["defect_ids"])
    if selected_ids_history:
        first_sel = set(selected_ids_history[0])
        loc_recall = len(first_sel & true_defects) / max(1, len(true_defects))
    else:
        loc_recall = 0.0

    return {
        "arm": arm,
        "seed": seed,
        "mms_ram_budget_mb": mms_ram_budget_mb,
        "start_checkpoint_sha256": start_sha,
        "end_checkpoint_sha256": checkpoint_sha256(model),
        "updates": updates,
        "credits": credits,
        "audit_credits": audit_credits_total,
        "train_credits": train_credits_total,
        "unused_credits": credit_cap - credits,
        "audits_run": audits_run,
        "audits_skipped": audits_skipped,
        "defect_localization_recall": round(loc_recall, 4),
        "selected_windows": selected_ids_history,
        "eval_accuracy": round(final_eval["accuracy"], 6),
        "eval_loss": round(final_eval["loss"], 6),
        "per_regime_accuracy": {k: round(v, 4) for k, v in final_eval["per_regime"].items()},
        "binary_outcomes": final_eval["binary_outcomes"],
        "wall_seconds": round(wall_s, 4),
        "cpu_seconds": round(cpu_s, 4),
        "visits_per_wall_second": round(credits / max(wall_s, 1e-6), 1),
        "vram_peak_mb": vram_peak_mb,
        "map_receipt": store.receipt(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--pretrain-steps", type=int, default=900)
    parser.add_argument("--credit-cap", type=int, default=160_000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)

    args.out.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process(os.getpid())

    # Strictly disjoint feedback-audit cohort (64) and held-out evaluation cohort (512).
    audit_batch = sample_verified_batch(np.random.default_rng(91001), 64)
    eval_batch = sample_verified_batch(np.random.default_rng(92001), 512)

    arm_specs = [
        ("task_reinvest", 0.0),
        ("static_cyclic", 0.0),
        ("shuffled_map", 4.0),
        ("mms_guided_tiny", 0.02),
        ("mms_guided_med", 0.25),
        ("mms_guided_full", 4.0),
    ]

    all_runs: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []

    for seed in args.seeds:
        base_model, pre_info = train_competent_base(seed, device, steps=args.pretrain_steps)
        base_eval = evaluate_model(base_model, eval_batch, device)
        base_sha = checkpoint_sha256(base_model)

        defect_info = inject_localized_bpd_defects(base_model, defect_layer="l1_down", defect_components=(0, 1, 2, 3))
        defect_eval = evaluate_model(base_model, eval_batch, device)
        defect_sha = checkpoint_sha256(base_model)
        defect_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}

        seed_summaries.append(
            {
                "seed": seed,
                "parameters": sum(p.numel() for p in base_model.parameters()),
                "base_checkpoint_sha256": base_sha,
                "base_eval_accuracy": round(base_eval["accuracy"], 6),
                "defect_checkpoint_sha256": defect_sha,
                "defect_eval_accuracy": round(defect_eval["accuracy"], 6),
                "defect_info": defect_info,
                **pre_info,
            }
        )
        print(
            f"[seed={seed}] base_acc={base_eval['accuracy']:.4f} -> defect_acc={defect_eval['accuracy']:.4f} "
            f"(pretrain={pre_info['pretrain_wall_s']}s)",
            flush=True,
        )

        for arm_name, ram_mb in arm_specs:
            res = run_arm(
                arm=arm_name,
                mms_ram_budget_mb=ram_mb,
                seed=seed,
                defect_state=defect_state,
                defect_info=defect_info,
                audit_batch=audit_batch,
                eval_batch=eval_batch,
                credit_cap=args.credit_cap,
                device=device,
            )
            all_runs.append(res)
            print(
                f"  {arm_name:16s} ram={ram_mb:4.2f}MB acc={res['eval_accuracy']:.4f} "
                f"upd={res['updates']:3d} loc_rec={res['defect_localization_recall']:.2f} "
                f"evict={res['map_receipt']['evictions']:3d} wall={res['wall_seconds']:.2f}s",
                flush=True,
            )

    # Aggregate by arm & compute paired outcomes vs task_reinvest
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in all_runs:
        by_arm.setdefault(r["arm"], []).append(r)

    task_by_seed = {r["seed"]: r for r in by_arm["task_reinvest"]}
    summary_by_arm: dict[str, Any] = {}
    for arm_name, rows in by_arm.items():
        paired_gains = [r["eval_accuracy"] - task_by_seed[r["seed"]]["eval_accuracy"] for r in rows]
        # Aggregate binary outcomes across seeds (descriptive)
        a_bin = [b for r in rows for b in task_by_seed[r["seed"]]["binary_outcomes"]]
        b_bin = [b for r in rows for b in r["binary_outcomes"]]
        po = paired_outcomes(a_bin, b_bin)
        summary_by_arm[arm_name] = {
            "mms_ram_budget_mb": rows[0]["mms_ram_budget_mb"],
            "mean_eval_accuracy": round(float(np.mean([r["eval_accuracy"] for r in rows])), 6),
            "mean_gain_vs_task_reinvest": round(float(np.mean(paired_gains)), 6),
            "per_seed_gain_vs_task_reinvest": [round(g, 6) for g in paired_gains],
            "mean_defect_localization_recall": round(
                float(np.mean([r["defect_localization_recall"] for r in rows])), 4
            ),
            "mean_updates": round(float(np.mean([r["updates"] for r in rows])), 1),
            "mean_credits": round(float(np.mean([r["credits"] for r in rows])), 1),
            "mean_audit_credits": round(float(np.mean([r["audit_credits"] for r in rows])), 1),
            "mean_wall_seconds": round(float(np.mean([r["wall_seconds"] for r in rows])), 4),
            "mean_cpu_seconds": round(float(np.mean([r["cpu_seconds"] for r in rows])), 4),
            "mean_visits_per_wall_second": round(
                float(np.mean([r["visits_per_wall_second"] for r in rows])), 1
            ),
            "mean_map_allocated_bytes": round(
                float(np.mean([r["map_receipt"]["allocated_bytes"] for r in rows])), 1
            ),
            "mean_map_evictions": round(float(np.mean([r["map_receipt"]["evictions"] for r in rows])), 1),
            "paired_outcomes_vs_task_reinvest": po,
        }

    # Strip per-example binary_outcomes from saved runs to keep receipt compact
    compact_runs = [{k: v for k, v in r.items() if k != "binary_outcomes"} for r in all_runs]
    mi = proc.memory_info()
    receipt = {
        "status": "PUBLIC_DEVELOPMENT_ONLY",
        "experiment": "DG-MEM-FB-01",
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "credit_cap": args.credit_cap,
        "seeds": args.seeds,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "process_rss_mb": round(mi.rss / 1e6, 2),
        "process_peak_wset_mb": round(getattr(mi, "peak_wset", mi.rss) / 1e6, 2),
        "seed_baselines": seed_summaries,
        "summary_by_arm": summary_by_arm,
        "runs": compact_runs,
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\n=== SUMMARY BY ARM ===")
    for arm_name, s in summary_by_arm.items():
        print(
            f"{arm_name:16s} ram={s['mms_ram_budget_mb']:4.2f}MB "
            f"acc={s['mean_eval_accuracy']:.4f} "
            f"gain_vs_task={s['mean_gain_vs_task_reinvest']:+.4f} "
            f"loc={s['mean_defect_localization_recall']:.2f} "
            f"evict={s['mean_map_evictions']:4.1f} "
            f"wall={s['mean_wall_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
