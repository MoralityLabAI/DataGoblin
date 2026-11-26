"""Persistent RAM-bounded mechanistic map for DG-MEM-FB-01.

Maintains an explicit byte-budgeted in-RAM mechanistic map controlled by
``mms_ram_budget_mb``. Stores eight typed mechanistic artifact families:

1. BPD component decompositions and identities.
2. Per-layer and per-visit causal contributions.
3. Activation statistics and steering directions.
4. Jacobian and gradient-alignment estimates.
5. Cross-pass transport maps.
6. Previous intervention outcomes.
7. Component stability across optimizer updates.
8. Estimated marginal learning gains and intervention costs.

Pure NumPy + standard library so base CI runs without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from datagoblin.training_gym.steering_feedback import Action, Audit, Cost, Decision, select_window

_SCALAR_OVERHEAD_BYTES = 128


def _as_f32(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("mechanistic map array contains non-finite values")
    return np.ascontiguousarray(out)


@dataclass(frozen=True)
class BPDComponent:
    component_id: str
    layer: str
    rank_idx: int
    u: np.ndarray  # [d_out] float32
    v: np.ndarray  # [d_in] float32
    singular_value: float
    weight_sha256: str
    reconstruction_cosine: float

    @property
    def nbytes(self) -> int:
        return int(self.u.nbytes + self.v.nbytes + _SCALAR_OVERHEAD_BYTES)


@dataclass(frozen=True)
class CausalContribution:
    layer: str
    visit: int
    component_id: str
    accuracy_delta: float
    logit_diff_delta: float
    specificity: float
    sample_count: int

    @property
    def key(self) -> str:
        return f"{self.layer}:v{self.visit}:{self.component_id}"

    @property
    def nbytes(self) -> int:
        return _SCALAR_OVERHEAD_BYTES


@dataclass(frozen=True)
class ActivationStats:
    layer: str
    visit: int
    mean: np.ndarray  # [d] float32
    var: np.ndarray  # [d] float32
    norm_mean: float
    steering_direction: np.ndarray  # [d] float32
    sample_count: int

    @property
    def key(self) -> str:
        return f"{self.layer}:v{self.visit}"

    @property
    def nbytes(self) -> int:
        return int(self.mean.nbytes + self.var.nbytes + self.steering_direction.nbytes + _SCALAR_OVERHEAD_BYTES)


@dataclass(frozen=True)
class JacobianGradAlignment:
    layer: str
    visit: int
    component_id: str
    jacobian_norm: float
    grad_cosine_prev: float
    grad_cosine_base: float
    directional_sensitivity: float

    @property
    def key(self) -> str:
        return f"{self.layer}:v{self.visit}:{self.component_id}"

    @property
    def nbytes(self) -> int:
        return _SCALAR_OVERHEAD_BYTES


@dataclass(frozen=True)
class CrossPassTransport:
    src_visit: int
    dst_visit: int
    layer: str
    u_factor: np.ndarray  # [d, r] float32
    v_factor: np.ndarray  # [r, d] float32
    rank: int
    r2_fidelity: float

    @property
    def key(self) -> str:
        return f"{self.layer}:v{self.src_visit}->v{self.dst_visit}"

    @property
    def nbytes(self) -> int:
        return int(self.u_factor.nbytes + self.v_factor.nbytes + _SCALAR_OVERHEAD_BYTES)


@dataclass(frozen=True)
class InterventionOutcome:
    step: int
    checkpoint_sha256: str
    target_id: str
    action_kind: str
    pre_acc: float
    post_acc: float
    delta_acc: float
    wall_seconds: float
    cost_visits: int
    accepted: bool

    @property
    def key(self) -> str:
        return f"step{self.step}:{self.target_id}:{self.action_kind}"

    @property
    def nbytes(self) -> int:
        return _SCALAR_OVERHEAD_BYTES + 64


@dataclass(frozen=True)
class ComponentStability:
    component_id: str
    step: int
    cos_u_prev: float
    cos_v_prev: float
    cos_u_base: float
    cos_v_base: float
    drift_rate: float

    @property
    def key(self) -> str:
        return self.component_id

    @property
    def nbytes(self) -> int:
        return _SCALAR_OVERHEAD_BYTES


@dataclass(frozen=True)
class MarginalGainEstimate:
    target_id: str
    layer: str
    visit: int
    component_id: str
    estimated_gain: float
    estimated_cost_visits: int
    estimated_wall_ns: int
    estimated_ram_bytes: int
    confidence: float

    @property
    def gain_per_visit(self) -> float:
        return self.estimated_gain / max(1, self.estimated_cost_visits)

    @property
    def nbytes(self) -> int:
        return _SCALAR_OVERHEAD_BYTES


@dataclass
class _EntryMeta:
    section: str
    key: str
    nbytes: int
    priority: float
    step: int


class MechanisticMapStore:
    """Persistent RAM-bounded mechanistic map controlled by ``mms_ram_budget_mb``.

    Enforces ``allocated_bytes <= budget_bytes`` at all times via deterministic
    lowest-priority (then oldest-step, then lexicographic-key) eviction.
    """

    SECTIONS = (
        "bpd_components",
        "causal_contributions",
        "activation_stats",
        "jacobian_grad_alignment",
        "cross_pass_transport",
        "intervention_outcomes",
        "component_stability",
        "marginal_gains",
    )

    def __init__(self, mms_ram_budget_mb: float) -> None:
        if not isfinite(mms_ram_budget_mb) or mms_ram_budget_mb < 0:
            raise ValueError("mms_ram_budget_mb must be finite and >= 0")
        self.mms_ram_budget_mb = float(mms_ram_budget_mb)
        self.budget_bytes = round(self.mms_ram_budget_mb * 1024 * 1024)
        self._tables: dict[str, dict[str, Any]] = {name: {} for name in self.SECTIONS}
        self._meta: dict[tuple[str, str], _EntryMeta] = {}
        self.allocated_bytes = 0
        self.peak_allocated_bytes = 0
        self.evictions = 0
        self.budget_rejections = 0
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @property
    def enabled(self) -> bool:
        return self.budget_bytes > 0

    @property
    def utilization_ratio(self) -> float:
        return 0.0 if self.budget_bytes == 0 else self.allocated_bytes / self.budget_bytes

    def _admit(self, section: str, key: str, item: Any, nbytes: int, *, priority: float, step: int) -> bool:
        if not isfinite(priority):
            raise ValueError("priority must be finite")
        if self.budget_bytes == 0 or nbytes > self.budget_bytes:
            self.budget_rejections += 1
            return False

        full_key = (section, key)
        old_meta = self._meta.get(full_key)
        old_bytes = old_meta.nbytes if old_meta is not None else 0
        needed = self.allocated_bytes - old_bytes + nbytes

        if needed > self.budget_bytes:
            # Evict strictly lower-or-equal priority entries first (deterministic order).
            candidates = [
                m for fk, m in self._meta.items() if fk != full_key
            ]
            candidates.sort(key=lambda m: (m.priority, m.step, m.section, m.key))
            freed = 0
            to_evict: list[_EntryMeta] = []
            for cand in candidates:
                if needed - freed <= self.budget_bytes:
                    break
                if cand.priority > priority:
                    # Do not evict strictly higher-priority entries for a lower-priority write.
                    break
                to_evict.append(cand)
                freed += cand.nbytes
            if needed - freed > self.budget_bytes:
                self.budget_rejections += 1
                return False
            for victim in to_evict:
                self._remove(victim.section, victim.key)
                self.evictions += 1

        if old_meta is not None:
            self.allocated_bytes -= old_bytes
        self._tables[section][key] = item
        self._meta[full_key] = _EntryMeta(section=section, key=key, nbytes=nbytes, priority=priority, step=step)
        self.allocated_bytes += nbytes
        if self.allocated_bytes > self.peak_allocated_bytes:
            self.peak_allocated_bytes = self.allocated_bytes
        self.writes += 1
        assert self.allocated_bytes <= self.budget_bytes
        return True

    def _remove(self, section: str, key: str) -> None:
        meta = self._meta.pop((section, key))
        self._tables[section].pop(key, None)
        self.allocated_bytes -= meta.nbytes

    def _get(self, section: str, key: str) -> Any | None:
        val = self._tables[section].get(key)
        if val is None:
            self.misses += 1
        else:
            self.hits += 1
        return val

    # --- 1. BPD component decompositions and identities ---
    def put_bpd_component(
        self,
        *,
        component_id: str,
        layer: str,
        rank_idx: int,
        u: np.ndarray,
        v: np.ndarray,
        singular_value: float,
        weight_sha256: str,
        reconstruction_cosine: float,
        priority: float = 1.0,
        step: int = 0,
    ) -> bool:
        item = BPDComponent(
            component_id=component_id,
            layer=layer,
            rank_idx=rank_idx,
            u=_as_f32(u),
            v=_as_f32(v),
            singular_value=float(singular_value),
            weight_sha256=weight_sha256,
            reconstruction_cosine=float(reconstruction_cosine),
        )
        return self._admit("bpd_components", component_id, item, item.nbytes, priority=priority, step=step)

    def get_bpd_component(self, component_id: str) -> BPDComponent | None:
        return self._get("bpd_components", component_id)

    # --- 2. Per-layer and per-visit causal contributions ---
    def put_causal_contribution(
        self, contrib: CausalContribution, *, priority: float | None = None, step: int = 0
    ) -> bool:
        prio = abs(contrib.accuracy_delta) + abs(contrib.logit_diff_delta) if priority is None else priority
        return self._admit("causal_contributions", contrib.key, contrib, contrib.nbytes, priority=prio, step=step)

    def get_causal_contribution(self, layer: str, visit: int, component_id: str) -> CausalContribution | None:
        return self._get("causal_contributions", f"{layer}:v{visit}:{component_id}")

    # --- 3. Activation statistics and steering directions ---
    def put_activation_stats(
        self,
        *,
        layer: str,
        visit: int,
        mean: np.ndarray,
        var: np.ndarray,
        norm_mean: float,
        steering_direction: np.ndarray,
        sample_count: int,
        priority: float = 1.0,
        step: int = 0,
    ) -> bool:
        item = ActivationStats(
            layer=layer,
            visit=visit,
            mean=_as_f32(mean),
            var=_as_f32(var),
            norm_mean=float(norm_mean),
            steering_direction=_as_f32(steering_direction),
            sample_count=int(sample_count),
        )
        return self._admit("activation_stats", item.key, item, item.nbytes, priority=priority, step=step)

    def get_activation_stats(self, layer: str, visit: int) -> ActivationStats | None:
        return self._get("activation_stats", f"{layer}:v{visit}")

    # --- 4. Jacobian and gradient-alignment estimates ---
    def put_jacobian_grad_alignment(
        self, item: JacobianGradAlignment, *, priority: float | None = None, step: int = 0
    ) -> bool:
        prio = item.jacobian_norm + max(0.0, item.grad_cosine_prev) if priority is None else priority
        return self._admit("jacobian_grad_alignment", item.key, item, item.nbytes, priority=prio, step=step)

    def get_jacobian_grad_alignment(
        self, layer: str, visit: int, component_id: str
    ) -> JacobianGradAlignment | None:
        return self._get("jacobian_grad_alignment", f"{layer}:v{visit}:{component_id}")

    # --- 5. Cross-pass transport maps ---
    def put_cross_pass_transport(
        self,
        *,
        src_visit: int,
        dst_visit: int,
        layer: str,
        matrix: np.ndarray,
        max_rank: int | None = None,
        r2_fidelity: float = 1.0,
        priority: float = 0.8,
        step: int = 0,
    ) -> bool:
        mat = _as_f32(matrix)
        if mat.ndim != 2:
            raise ValueError("cross-pass transport matrix must be 2D")
        d_out, d_in = mat.shape
        rank = min(d_out, d_in) if max_rank is None else max(1, min(int(max_rank), d_out, d_in))
        # Adaptively compress rank if the full matrix would exceed 25% of total RAM budget.
        if self.budget_bytes > 0:
            max_bytes_for_entry = max(1024, self.budget_bytes // 4)
            bytes_per_rank = 4 * (d_out + d_in)
            if bytes_per_rank > 0:
                affordable_rank = max(1, (max_bytes_for_entry - _SCALAR_OVERHEAD_BYTES) // bytes_per_rank)
                rank = min(rank, affordable_rank)
        u_svd, s_svd, vt_svd = np.linalg.svd(mat, full_matrices=False)
        u_factor = np.ascontiguousarray((u_svd[:, :rank] * s_svd[:rank]).astype(np.float32))
        v_factor = np.ascontiguousarray(vt_svd[:rank, :].astype(np.float32))
        item = CrossPassTransport(
            src_visit=src_visit,
            dst_visit=dst_visit,
            layer=layer,
            u_factor=u_factor,
            v_factor=v_factor,
            rank=rank,
            r2_fidelity=float(r2_fidelity),
        )
        return self._admit("cross_pass_transport", item.key, item, item.nbytes, priority=priority, step=step)

    def get_cross_pass_transport(self, layer: str, src_visit: int, dst_visit: int) -> CrossPassTransport | None:
        return self._get("cross_pass_transport", f"{layer}:v{src_visit}->v{dst_visit}")

    # --- 6. Previous intervention outcomes ---
    def record_intervention_outcome(
        self, outcome: InterventionOutcome, *, priority: float = 1.5
    ) -> bool:
        return self._admit(
            "intervention_outcomes", outcome.key, outcome, outcome.nbytes, priority=priority, step=outcome.step
        )

    def recent_outcomes(self, target_id: str | None = None) -> list[InterventionOutcome]:
        vals = list(self._tables["intervention_outcomes"].values())
        if target_id is not None:
            vals = [v for v in vals if v.target_id == target_id]
        vals.sort(key=lambda o: (o.step, o.target_id))
        return vals

    # --- 7. Component stability across optimizer updates ---
    def put_component_stability(self, item: ComponentStability, *, priority: float = 1.2) -> bool:
        return self._admit(
            "component_stability", item.key, item, item.nbytes, priority=priority, step=item.step
        )

    def get_component_stability(self, component_id: str) -> ComponentStability | None:
        return self._get("component_stability", component_id)

    def mean_drift_rate(self) -> float:
        vals: list[ComponentStability] = list(self._tables["component_stability"].values())
        if not vals:
            return 1.0
        return float(sum(v.drift_rate for v in vals) / len(vals))

    def should_reaudit(self, *, drift_threshold: float = 0.08, min_steps_since_audit: int = 1,
                       steps_since_audit: int = 1) -> bool:
        """Economic admission gate: skip re-audit if map is populated and component drift is low."""
        if not self.enabled or not self._tables["marginal_gains"]:
            return True
        if steps_since_audit < min_steps_since_audit:
            return False
        return self.mean_drift_rate() >= drift_threshold

    # --- 8. Estimated marginal learning gains and intervention costs ---
    def put_marginal_gain(self, item: MarginalGainEstimate, *, step: int = 0) -> bool:
        prio = 2.0 + max(0.0, item.estimated_gain)
        return self._admit("marginal_gains", item.target_id, item, item.nbytes, priority=prio, step=step)

    def select_top_targets(self, k: int = 4, *, min_gain_per_visit: float = 0.0) -> list[MarginalGainEstimate]:
        items: list[MarginalGainEstimate] = [
            g for g in self._tables["marginal_gains"].values() if g.gain_per_visit > min_gain_per_visit
        ]
        items.sort(key=lambda g: (-g.gain_per_visit, -g.estimated_gain, g.target_id))
        return items[:k]

    def select_feedback_window(
        self,
        *,
        checkpoint: str,
        feedback_cohort: str,
        step: int,
        window: int,
        remaining: Cost,
        competent: bool,
    ) -> Decision:
        """Bridge stored marginal gains into the DG-Gym ``select_window`` contract."""
        audits: list[Audit] = []
        for g in self._tables["marginal_gains"].values():
            contrib = self._tables["causal_contributions"].get(f"{g.layer}:v{g.visit}:{g.component_id}")
            spec = float(np.clip(contrib.specificity, -1.0, 1.0)) if contrib is not None else 0.0
            audits.append(
                Audit(
                    action=Action(
                        key=g.target_id,
                        site=f"{g.layer}:v{g.visit}",
                        direction=g.component_id,
                        dose=1.0,
                    ),
                    checkpoint=checkpoint,
                    feedback_cohort=feedback_cohort,
                    step=step,
                    scope="feedback",
                    specificity=spec,
                    ideal=1.0,
                    gain_lower=g.estimated_gain,
                    window_cost_upper=Cost(
                        wall_ns=max(1, g.estimated_wall_ns),
                        cpu_ns=max(1, g.estimated_wall_ns),
                        gpu_ns=0,
                        visits=max(1, g.estimated_cost_visits),
                        verifier_calls=0,
                        ram_bytes=max(0, g.estimated_ram_bytes),
                        vram_bytes=0,
                    ),
                )
            )
        return select_window(
            audits,
            checkpoint=checkpoint,
            feedback_cohort=feedback_cohort,
            step=step,
            window=window,
            remaining=remaining,
            competent=competent,
            policy="gain",
        )

    def receipt(self) -> dict[str, Any]:
        section_bytes = {s: 0 for s in self.SECTIONS}
        for meta in self._meta.values():
            section_bytes[meta.section] += meta.nbytes
        section_counts = {s: len(self._tables[s]) for s in self.SECTIONS}
        return {
            "mms_ram_budget_mb": self.mms_ram_budget_mb,
            "budget_bytes": self.budget_bytes,
            "allocated_bytes": self.allocated_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "utilization_ratio": round(self.utilization_ratio, 6),
            "evictions": self.evictions,
            "budget_rejections": self.budget_rejections,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "section_bytes": section_bytes,
            "section_counts": section_counts,
        }
