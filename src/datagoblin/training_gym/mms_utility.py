"""MMS-05/06: Utility-directed mechanistic feedback scheduler (repaired).

Estimates downstream learning value of instrumenting site ``s`` at memory tier ``r``
versus an explicit ordinary-training baseline action:

    U(s, r) = E[Delta V_dev_train | s, r] / (Delta C_CPU + lambda * Delta M)

Repairs implemented for MMS-06:
- Owned compact float32 coordinate arrays (no retained base views, immune to caller mutation).
- Zero-byte history budget rejects empty records; explicit record-count + byte bounds.
- Covariance byte accounting derived from actual configured ``cov_dim`` arrays.
- Separate accounting for owned array bytes, estimated charge bytes, and measured RSS.
- Strict input validation and explicit ordinary-training abstention when gain evidence lacks.
- Budget and live-RAM headroom filtering before ranking; explicitly budgeted research exploration.
- Full provenance binding (task/metric/direction/scale, split digest, horizon, checkpoint/opt,
  site, basis digest) with separate statistical contexts per metric unit and cross-basis guard.
- Trajectory-bound action traces replacing blanket eligibility updates.
- Separated receipt fields for empirical gain, standard error, SNR multiplier, and exploration.

Pure NumPy + stdlib so base CI runs without PyTorch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any

import numpy as np

ORDINARY_SITE = "__ordinary__"
ALLOWED_SIGNAL_KINDS = (
    "lean_proof_progress",
    "lbf_bellman_regret",
    "mcp_retrieval_recovery",
    "storyworld_repair",
)
ALLOWED_PARTITIONS = ("dev_train",)
_COORD_META_BYTES = 128
_REPAIR_META_BYTES = 256
_STATS_META_BYTES = 256


@dataclass(frozen=True)
class FeedbackContext:
    """Full provenance binding for feedback signals, trajectories, and repair retrieval."""

    task_id: str
    metric_name: str
    metric_direction: int  # +1 higher is better, -1 lower is better
    metric_scale: float  # positive scale factor normalizing metric units
    split_digest: str
    feedback_horizon: int
    checkpoint_sha256: str
    optimizer_sha256: str
    site: str
    basis_digest: str

    def __post_init__(self) -> None:
        required = (
            self.task_id,
            self.metric_name,
            self.split_digest,
            self.checkpoint_sha256,
            self.optimizer_sha256,
            self.site,
            self.basis_digest,
        )
        if not all(isinstance(s, str) and s for s in required):
            raise ValueError("FeedbackContext requires non-empty string provenance fields")
        if self.metric_direction not in (-1, 1):
            raise ValueError("metric_direction must be +1 or -1")
        if not isfinite(self.metric_scale) or self.metric_scale <= 0:
            raise ValueError("metric_scale must be finite and > 0")
        if type(self.feedback_horizon) is not int or self.feedback_horizon < 1:
            raise ValueError("feedback_horizon must be a positive integer")

    @property
    def metric_context_key(self) -> tuple[str, str, int, float]:
        """Separates statistical accumulators across distinct metric units/scales."""
        return (self.task_id, self.metric_name, self.metric_direction, float(self.metric_scale))

    def normalize_delta(self, raw_delta: float) -> float:
        """Convert raw metric delta into higher-is-better normalized units."""
        return float(self.metric_direction) * float(raw_delta) / float(self.metric_scale)


@dataclass(frozen=True)
class FeedbackSignal:
    """Dedicated training-feedback partition utility observation bound to a trajectory."""

    step: int
    trajectory_id: str
    site: str
    ram_mib: float
    partition: str
    signal_kind: str
    delta_v: float  # raw delta in metric units (oriented via context.metric_direction)
    delta_cpu_seconds: float
    delta_ram_mib: float
    context: FeedbackContext
    state_signature: np.ndarray | None = None
    component_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.partition not in ALLOWED_PARTITIONS:
            raise ValueError(f"forbidden feedback partition {self.partition!r}; only 'dev_train' allowed")
        if self.signal_kind not in ALLOWED_SIGNAL_KINDS:
            raise ValueError(f"unknown signal_kind {self.signal_kind!r}")
        if not self.site or not self.trajectory_id or type(self.step) is not int or self.step < 0:
            raise ValueError("invalid site, trajectory_id, or step")
        if self.context.site != self.site:
            raise ValueError("FeedbackSignal site must match FeedbackContext site")
        for val in (self.ram_mib, self.delta_v, self.delta_cpu_seconds, self.delta_ram_mib):
            if not isfinite(val):
                raise ValueError("non-finite numeric value in FeedbackSignal")
        if self.ram_mib < 0 or self.delta_cpu_seconds < 0 or self.delta_ram_mib < 0:
            raise ValueError("costs and RAM tiers must be non-negative")


@dataclass(frozen=True)
class RepairRecord:
    step: int
    site: str
    signal_kind: str
    normalized_delta_v: float
    component_ids: tuple[str, ...]
    signature: np.ndarray  # owned normalized float32 [d]
    context: FeedbackContext

    @property
    def owned_array_bytes(self) -> int:
        return int(self.signature.nbytes)

    @property
    def estimated_charge_bytes(self) -> int:
        return self.owned_array_bytes + _REPAIR_META_BYTES


@dataclass(frozen=True)
class AllocationChoice:
    site: str
    ram_mib: float
    utility: float
    empirical_gain: float
    uncertainty_se: float | None
    snr_multiplier: float
    exploration_bonus_term: float
    expected_cost: float
    policy: str
    reason: str
    used_exploration: bool
    retrieved_components: tuple[str, ...]


class _SiteStats:
    """Per-(metric_context, site, ram_mib) online sufficient statistics + covariance sketch."""

    def __init__(self, site: str, ram_mib: float, cov_dim: int) -> None:
        self.site = site
        self.ram_mib = ram_mib
        self.cov_dim = cov_dim
        self.count = 0
        self.ema_delta_v = 0.0
        self.mean_delta_v = 0.0
        self.m2_delta_v = 0.0
        self.ema_cpu = 1e-3
        self.ema_ram = max(0.0, ram_mib)
        self.hits = 0
        self.admitted = 0
        # Owned online component gradient mean & covariance arrays
        self.grad_n = 0
        self.grad_mean = np.zeros(cov_dim, dtype=np.float32)
        self.grad_m2 = np.zeros((cov_dim, cov_dim), dtype=np.float32)

    @property
    def owned_array_bytes(self) -> int:
        return int(self.grad_mean.nbytes + self.grad_m2.nbytes)

    @property
    def estimated_charge_bytes(self) -> int:
        return self.owned_array_bytes + _STATS_META_BYTES

    def update_gain(self, norm_delta_v: float, *, weight: float, ema_alpha: float) -> None:
        a = min(1.0, max(0.0, ema_alpha * weight))
        if self.count == 0:
            self.ema_delta_v = norm_delta_v * weight
        else:
            self.ema_delta_v = (1.0 - a) * self.ema_delta_v + a * norm_delta_v
        self.count += 1
        d1 = norm_delta_v - self.mean_delta_v
        self.mean_delta_v += d1 / self.count
        d2 = norm_delta_v - self.mean_delta_v
        self.m2_delta_v += d1 * d2

    @property
    def empirical_se(self) -> float | None:
        """Empirical standard error of normalized gain across observations (None if n < 2)."""
        if self.count < 2:
            return None
        var = max(0.0, self.m2_delta_v / (self.count - 1))
        return float(sqrt(var / self.count))

    def update_grad_cov(self, comp_grad: np.ndarray) -> None:
        vec = np.array(comp_grad, dtype=np.float32, copy=True).reshape(-1)
        if vec.size < self.cov_dim:
            padded = np.zeros(self.cov_dim, dtype=np.float32)
            padded[: vec.size] = vec
            vec = padded
        else:
            vec = vec[: self.cov_dim]
        if not np.isfinite(vec).all():
            return
        self.grad_n += 1
        delta = vec - self.grad_mean
        self.grad_mean += delta / self.grad_n
        delta2 = vec - self.grad_mean
        self.grad_m2 += np.outer(delta, delta2)

    @property
    def grad_snr(self) -> float:
        if self.grad_n < 2:
            return 1.0
        cov_trace = float(np.trace(self.grad_m2) / (self.grad_n - 1))
        mean_norm_sq = float(np.dot(self.grad_mean, self.grad_mean))
        return float(mean_norm_sq / (cov_trace + 1e-6))

    @property
    def snr_multiplier(self) -> float:
        return float(min(2.0, 0.5 + 0.5 * self.grad_snr))


class UtilityDirectedScheduler:
    """MMS-05/06 utility-directed scheduler with strict memory, provenance, and trace isolation."""

    def __init__(
        self,
        sites: tuple[str, ...],
        ram_tiers_mib: tuple[float, ...],
        *,
        lambda_ram: float = 0.01,
        eligibility_decay: float = 0.85,
        ema_alpha: float = 0.3,
        exploration_bonus: float = 0.05,
        history_budget_bytes: int = 256 * 1024,
        max_coord_records: int = 512,
        max_repair_records: int = 256,
        max_trace_length: int = 32,
        cov_dim: int = 8,
        research_mode: bool = False,
        exploration_budget: int = 0,
    ) -> None:
        if not sites or any(not isinstance(s, str) or not s or s == ORDINARY_SITE for s in sites):
            raise ValueError("sites must be non-empty strings distinct from ORDINARY_SITE")
        if len(set(sites)) != len(sites):
            raise ValueError("duplicate site identities")
        if not ram_tiers_mib:
            raise ValueError("ram_tiers_mib must be non-empty")
        tiers = tuple(float(r) for r in ram_tiers_mib)
        if any(not isfinite(r) or r < 0 for r in tiers):
            raise ValueError("ram_tiers_mib entries must be finite and >= 0")
        if len(set(tiers)) != len(tiers):
            raise ValueError("duplicate ram_tiers_mib values")
        if not isfinite(lambda_ram) or lambda_ram < 0:
            raise ValueError("lambda_ram must be finite and >= 0")
        if not isfinite(eligibility_decay) or not 0.0 <= eligibility_decay < 1.0:
            raise ValueError("eligibility_decay must be in [0, 1)")
        if not isfinite(ema_alpha) or not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not isfinite(exploration_bonus) or exploration_bonus < 0:
            raise ValueError("exploration_bonus must be finite and >= 0")
        for name, val, lo in (
            ("history_budget_bytes", history_budget_bytes, 0),
            ("max_coord_records", max_coord_records, 0),
            ("max_repair_records", max_repair_records, 0),
            ("max_trace_length", max_trace_length, 1),
            ("cov_dim", cov_dim, 1),
            ("exploration_budget", exploration_budget, 0),
        ):
            if type(val) is not int or val < lo:
                raise ValueError(f"{name} must be an integer >= {lo}")
        if type(research_mode) is not bool:
            raise ValueError("research_mode must be a bool")

        self.sites = tuple(sites)
        self.ram_tiers_mib = tiers
        self.lambda_ram = float(lambda_ram)
        self.eligibility_decay = float(eligibility_decay)
        self.ema_alpha = float(ema_alpha)
        self.exploration_bonus = float(exploration_bonus)
        self.history_budget_bytes = history_budget_bytes
        self.max_coord_records = max_coord_records
        self.max_repair_records = max_repair_records
        self.max_trace_length = max_trace_length
        self.cov_dim = cov_dim
        self.research_mode = research_mode
        self.exploration_budget = exploration_budget
        self.exploration_budget_used = 0

        # Actions include explicit ordinary training alongside memory-assisted (site, ram_mib) pairs.
        self._actions: tuple[tuple[str, float], ...] = (
            (ORDINARY_SITE, 0.0),
            *((s, r) for s in self.sites for r in self.ram_tiers_mib),
        )
        # Separate statistical tables per metric context key
        self._stats_by_ctx: dict[tuple[str, str, int, float], dict[tuple[str, float], _SiteStats]] = {}
        # Trajectory-bound action traces: trajectory_id -> list[(step, site, ram_mib)]
        self._traces: dict[str, list[tuple[int, str, float]]] = {}
        self._coord_history: list[tuple[int, str, np.ndarray, int]] = []
        self._coord_owned_bytes = 0
        self._coord_charged_bytes = 0
        self._repairs: list[RepairRecord] = []
        self._repair_owned_bytes = 0
        self._repair_charged_bytes = 0

        self.total_selections = 0
        self.ordinary_selections = 0
        self.extra_forward_passes = 0
        self.extra_backward_passes = 0
        self.scheduler_cpu_seconds = 0.0
        self.coord_rejections = 0
        self.repair_rejections = 0
        self.cross_basis_rejections = 0

    def _ctx_table(self, ctx_key: tuple[str, str, int, float]) -> dict[tuple[str, float], _SiteStats]:
        tbl = self._stats_by_ctx.get(ctx_key)
        if tbl is None:
            tbl = {act: _SiteStats(act[0], act[1], self.cov_dim) for act in self._actions}
            self._stats_by_ctx[ctx_key] = tbl
        return tbl

    @property
    def owned_array_bytes(self) -> int:
        stats_owned = sum(
            st.owned_array_bytes for tbl in self._stats_by_ctx.values() for st in tbl.values()
        )
        return stats_owned + self._coord_owned_bytes + self._repair_owned_bytes

    @property
    def estimated_charge_bytes(self) -> int:
        stats_charge = sum(
            st.estimated_charge_bytes for tbl in self._stats_by_ctx.values() for st in tbl.values()
        )
        trace_charge = sum(len(v) * 64 + 64 for v in self._traces.values())
        return stats_charge + self._coord_charged_bytes + self._repair_charged_bytes + trace_charge

    def fits_total_cap(self, bank_bytes: int, total_cap_bytes: int) -> bool:
        """Check whether scheduler estimated charges plus live bank payload fit the runner cap."""
        return (self.estimated_charge_bytes + max(0, int(bank_bytes))) <= int(total_cap_bytes)

    def record_action(self, trajectory_id: str, step: int, site: str, ram_mib: float) -> None:
        """Append an executed action to the trajectory's bounded action trace."""
        if not trajectory_id or type(step) is not int or step < 0:
            raise ValueError("invalid trajectory_id or step")
        key = (site, float(ram_mib))
        if key not in self._actions:
            raise ValueError(f"unregistered action: {key}")
        trace = self._traces.setdefault(trajectory_id, [])
        trace.append((step, site, float(ram_mib)))
        if len(trace) > self.max_trace_length:
            del trace[: len(trace) - self.max_trace_length]

    def record_coordinate_delta(self, step: int, site: str, compressed_coords: np.ndarray) -> bool:
        """Store an owned compact float32 copy of coordinate deltas under budget and record caps."""
        t0 = time.process_time()
        cap = self.history_budget_bytes // 2
        if cap == 0 or self.max_coord_records == 0:
            self.coord_rejections += 1
            self.scheduler_cpu_seconds += time.process_time() - t0
            return False
        # Force an owned compact float32 array so small views never retain large base allocations
        # and caller mutation after insertion cannot alter stored coordinates.
        arr = np.asarray(compressed_coords, dtype=np.float32).reshape(-1).copy(order="C")
        if arr.size == 0 or not np.isfinite(arr).all():
            self.coord_rejections += 1
            self.scheduler_cpu_seconds += time.process_time() - t0
            return False
        assert arr.base is None
        charge = int(arr.nbytes) + _COORD_META_BYTES
        if charge > cap:
            self.coord_rejections += 1
            self.scheduler_cpu_seconds += time.process_time() - t0
            return False
        while self._coord_history and (
            self._coord_charged_bytes + charge > cap or len(self._coord_history) >= self.max_coord_records
        ):
            _, _, old_arr, old_charge = self._coord_history.pop(0)
            self._coord_owned_bytes -= int(old_arr.nbytes)
            self._coord_charged_bytes -= old_charge
        self._coord_history.append((int(step), site, arr, charge))
        self._coord_owned_bytes += int(arr.nbytes)
        self._coord_charged_bytes += charge
        self.scheduler_cpu_seconds += time.process_time() - t0
        return True

    def record_proxy_counts(
        self,
        context: FeedbackContext,
        site: str,
        ram_mib: float,
        *,
        hits: int,
        admitted: int,
        comp_grad: np.ndarray | None = None,
    ) -> None:
        t0 = time.process_time()
        tbl = self._ctx_table(context.metric_context_key)
        st = tbl[(site, float(ram_mib))]
        st.hits += int(hits)
        st.admitted += int(admitted)
        if comp_grad is not None:
            st.update_grad_cov(comp_grad)
        self.scheduler_cpu_seconds += time.process_time() - t0

    def observe_feedback(self, sig: FeedbackSignal) -> None:
        """Credit ONLY actions on ``sig.trajectory_id`` inside ``sig.context.metric_context_key``."""
        t0 = time.process_time()
        key = (sig.site, float(sig.ram_mib))
        if key not in self._actions:
            raise ValueError(f"unregistered (site, ram_mib) pair: {key}")

        tbl = self._ctx_table(sig.context.metric_context_key)
        norm_dv = sig.context.normalize_delta(sig.delta_v)

        # Build trajectory-scoped credit weights (strictly isolated from other trajectories/sites!)
        trace = self._traces.get(sig.trajectory_id)
        if not trace:
            # No explicit trace recorded: credit only the reported action itself.
            weights: dict[tuple[str, float], float] = {key: 1.0}
        else:
            raw_w: dict[tuple[str, float], float] = {}
            n = len(trace)
            for idx, (_st_step, s_name, r_val) in enumerate(trace):
                age = (n - 1) - idx
                w = self.eligibility_decay**age
                raw_w[(s_name, r_val)] = raw_w.get((s_name, r_val), 0.0) + w
            total_w = sum(raw_w.values())
            weights = {k: v / max(total_w, 1e-12) for k, v in raw_w.items()}

        for act_key, w in weights.items():
            st = tbl[act_key]
            st.update_gain(norm_dv, weight=w, ema_alpha=self.ema_alpha)
            if act_key == key:
                st.ema_cpu = (1.0 - self.ema_alpha) * st.ema_cpu + self.ema_alpha * max(
                    1e-6, sig.delta_cpu_seconds
                )
                st.ema_ram = (1.0 - self.ema_alpha) * st.ema_ram + self.ema_alpha * sig.delta_ram_mib

        # Store positive normalized-gain repair trajectory under byte & record caps
        if norm_dv > 0 and sig.state_signature is not None:
            cap = self.history_budget_bytes - (self.history_budget_bytes // 2)
            if cap == 0 or self.max_repair_records == 0:
                self.repair_rejections += 1
            else:
                vec = np.array(sig.state_signature, dtype=np.float32, copy=True, order="C").reshape(-1)
                norm = float(np.linalg.norm(vec))
                if vec.size > 0 and norm > 1e-8 and np.isfinite(vec).all():
                    normed = np.array(vec / norm, dtype=np.float32, copy=True, order="C")
                    rec = RepairRecord(
                        step=sig.step,
                        site=sig.site,
                        signal_kind=sig.signal_kind,
                        normalized_delta_v=norm_dv,
                        component_ids=sig.component_ids,
                        signature=normed,
                        context=sig.context,
                    )
                    if rec.estimated_charge_bytes > cap:
                        self.repair_rejections += 1
                    else:
                        while self._repairs and (
                            self._repair_charged_bytes + rec.estimated_charge_bytes > cap
                            or len(self._repairs) >= self.max_repair_records
                        ):
                            old = self._repairs.pop(0)
                            self._repair_owned_bytes -= old.owned_array_bytes
                            self._repair_charged_bytes -= old.estimated_charge_bytes
                        self._repairs.append(rec)
                        self._repair_owned_bytes += rec.owned_array_bytes
                        self._repair_charged_bytes += rec.estimated_charge_bytes
                else:
                    self.repair_rejections += 1

        self.scheduler_cpu_seconds += time.process_time() - t0

    def retrieve_repairs(
        self,
        query_signature: np.ndarray | None,
        *,
        query_context: FeedbackContext,
        top_k: int = 3,
        min_cosine: float = 0.2,
        validated_transport_ok: bool = False,
    ) -> list[RepairRecord]:
        """Retrieve past successful repairs requiring compatible task/metric/site/basis provenance."""
        if query_signature is None or not self._repairs:
            return []
        q = np.asarray(query_signature, dtype=np.float32).reshape(-1)
        qn = float(np.linalg.norm(q))
        if q.size == 0 or qn <= 1e-8 or not np.isfinite(q).all():
            return []
        q = q / qn
        scored: list[tuple[float, RepairRecord]] = []
        for rec in self._repairs:
            # Strict provenance compatibility check
            if (
                rec.context.task_id != query_context.task_id
                or rec.context.metric_name != query_context.metric_name
                or rec.context.metric_direction != query_context.metric_direction
                or rec.context.site != query_context.site
            ):
                continue
            if rec.context.basis_digest != query_context.basis_digest and not validated_transport_ok:
                self.cross_basis_rejections += 1
                continue
            if rec.signature.shape != q.shape:
                continue
            cos = float(np.dot(q, rec.signature))
            if cos >= min_cosine:
                scored.append((cos * rec.normalized_delta_v, rec))
        scored.sort(key=lambda item: (-item[0], -item[1].step))
        return [rec for _, rec in scored[:top_k]]

    def _decompose_score(
        self,
        st: _SiteStats,
        *,
        allow_exploration: bool,
    ) -> tuple[float, float, float | None, float, float, float]:
        """Return (utility, empirical_gain, se, snr_mult, explore_term, cost)."""
        emp_gain = float(st.ema_delta_v)
        se = st.empirical_se
        snr_mult = 1.0 if st.site == ORDINARY_SITE else st.snr_multiplier
        explore_term = (
            float(self.exploration_bonus / sqrt(st.count + 1.0))
            if (allow_exploration and st.site != ORDINARY_SITE)
            else 0.0
        )
        num = emp_gain * snr_mult + explore_term
        cost = float(max(1e-6, st.ema_cpu + self.lambda_ram * st.ema_ram))
        return float(num / cost), emp_gain, se, snr_mult, explore_term, cost

    def select(
        self,
        *,
        context: FeedbackContext,
        policy: str = "utility",
        remaining_cpu_seconds: float | None = None,
        live_ram_headroom_bytes: int | None = None,
        query_signature: np.ndarray | None = None,
        validated_transport_ok: bool = False,
    ) -> AllocationChoice:
        """Filter feasible actions by CPU & live RAM, then select site/tier or ordinary training."""
        t0 = time.process_time()
        tbl = self._ctx_table(context.metric_context_key)

        # Feasibility filter
        feasible: list[tuple[str, float]] = []
        for s, r in self._actions:
            st = tbl[(s, r)]
            ram_bytes = round(r * 1024 * 1024)
            if live_ram_headroom_bytes is not None and ram_bytes > live_ram_headroom_bytes:
                continue
            if remaining_cpu_seconds is not None and st.count > 0 and st.ema_cpu > remaining_cpu_seconds:
                continue
            feasible.append((s, r))

        if not feasible:
            self.total_selections += 1
            self.ordinary_selections += 1
            self.scheduler_cpu_seconds += time.process_time() - t0
            return AllocationChoice(
                site=ORDINARY_SITE,
                ram_mib=0.0,
                utility=0.0,
                empirical_gain=0.0,
                uncertainty_se=None,
                snr_multiplier=1.0,
                exploration_bonus_term=0.0,
                expected_cost=0.0,
                policy=policy,
                reason="budget_exhausted",
                used_exploration=False,
                retrieved_components=(),
            )

        mem_feasible = [a for a in feasible if a[0] != ORDINARY_SITE]
        used_exploration = False
        reason = policy

        if policy == "round_robin":
            pool = mem_feasible if mem_feasible else feasible
            chosen_key = pool[self.total_selections % len(pool)]
            u, eg, se, snr, exp_t, ec = self._decompose_score(tbl[chosen_key], allow_exploration=False)
        elif policy == "hit_proxy":
            pool = mem_feasible if mem_feasible else feasible
            chosen_key = max(
                pool,
                key=lambda k: (tbl[k].admitted + 0.1 * tbl[k].hits, -k[1], k[0]),
            )
            u, eg, se, snr, exp_t, ec = self._decompose_score(tbl[chosen_key], allow_exploration=False)
        elif policy == "utility":
            can_explore = self.research_mode and (self.exploration_budget_used < self.exploration_budget)
            # First evaluate pure exploitation across memory-assisted actions
            exploit_best: tuple[float, float, float, str, float, float | None, float, float] | None = None
            for s, r in mem_feasible:
                u_e, eg_e, se_e, snr_e, _, ec_e = self._decompose_score(tbl[(s, r)], allow_exploration=False)
                cand = (u_e, eg_e, -r, s, r, se_e, snr_e, ec_e)
                if exploit_best is None or cand > exploit_best:
                    exploit_best = cand

            ord_st = tbl[(ORDINARY_SITE, 0.0)]
            ord_u, ord_eg, ord_se, ord_snr, _, ord_ec = self._decompose_score(ord_st, allow_exploration=False)

            if exploit_best is not None and exploit_best[1] > max(0.0, ord_eg) and exploit_best[0] > ord_u:
                u, eg, _, s_pick, r_pick, se, snr, ec = exploit_best
                chosen_key = (s_pick, r_pick)
                exp_t = 0.0
                reason = "utility_exploit"
            elif can_explore and mem_feasible:
                # Budgeted research exploration
                exp_best = None
                for s, r in mem_feasible:
                    u_x, eg_x, se_x, snr_x, et_x, ec_x = self._decompose_score(
                        tbl[(s, r)], allow_exploration=True
                    )
                    cand_x = (u_x, et_x, -r, s, r, eg_x, se_x, snr_x, ec_x)
                    if exp_best is None or cand_x > exp_best:
                        exp_best = cand_x
                assert exp_best is not None
                u, exp_t, _, s_pick, r_pick, eg, se, snr, ec = exp_best
                chosen_key = (s_pick, r_pick)
                self.exploration_budget_used += 1
                used_exploration = True
                reason = "budgeted_exploration"
            else:
                chosen_key = (ORDINARY_SITE, 0.0)
                u, eg, se, snr, exp_t, ec = ord_u, ord_eg, ord_se, ord_snr, 0.0, ord_ec
                reason = "no_favorable_gain_evidence"
        else:
            raise ValueError(f"unknown scheduler policy: {policy}")

        retrieved = self.retrieve_repairs(
            query_signature,
            query_context=context,
            validated_transport_ok=validated_transport_ok,
        )
        comp_ids: list[str] = []
        for rec in retrieved:
            for cid in rec.component_ids:
                if cid not in comp_ids:
                    comp_ids.append(cid)

        self.total_selections += 1
        if chosen_key[0] == ORDINARY_SITE:
            self.ordinary_selections += 1
        self.scheduler_cpu_seconds += time.process_time() - t0
        return AllocationChoice(
            site=chosen_key[0],
            ram_mib=chosen_key[1],
            utility=u,
            empirical_gain=eg,
            uncertainty_se=se,
            snr_multiplier=snr,
            exploration_bonus_term=exp_t,
            expected_cost=ec,
            policy=policy,
            reason=reason,
            used_exploration=used_exploration,
            retrieved_components=tuple(comp_ids[:4]),
        )

    def receipt(self, *, measured_process_rss_bytes: int | None = None) -> dict[str, Any]:
        contexts_out: dict[str, Any] = {}
        for ctx_key, tbl in self._stats_by_ctx.items():
            ck_str = f"{ctx_key[0]}:{ctx_key[1]}:dir={ctx_key[2]}:scale={ctx_key[3]:g}"
            contexts_out[ck_str] = {
                f"{s}@{r:g}MiB": {
                    "count": st.count,
                    "empirical_gain_ema": round(st.ema_delta_v, 6),
                    "empirical_gain_mean": round(st.mean_delta_v, 6),
                    "uncertainty_se": None if st.empirical_se is None else round(st.empirical_se, 6),
                    "snr_multiplier": round(st.snr_multiplier, 4),
                    "ema_cpu_seconds": round(st.ema_cpu, 6),
                    "hits": st.hits,
                    "admitted": st.admitted,
                }
                for (s, r), st in tbl.items()
            }
        return {
            "lambda_ram": self.lambda_ram,
            "cov_dim": self.cov_dim,
            "research_mode": self.research_mode,
            "exploration_budget": self.exploration_budget,
            "exploration_budget_used": self.exploration_budget_used,
            "total_selections": self.total_selections,
            "ordinary_selections": self.ordinary_selections,
            "extra_forward_passes": self.extra_forward_passes,
            "extra_backward_passes": self.extra_backward_passes,
            "scheduler_cpu_seconds": round(self.scheduler_cpu_seconds, 6),
            "memory": {
                "history_budget_bytes": self.history_budget_bytes,
                "owned_array_bytes": self.owned_array_bytes,
                "estimated_charge_bytes": self.estimated_charge_bytes,
                "measured_process_rss_bytes": measured_process_rss_bytes,
                "stored_coord_records": len(self._coord_history),
                "stored_repair_records": len(self._repairs),
                "coord_rejections": self.coord_rejections,
                "repair_rejections": self.repair_rejections,
                "cross_basis_rejections": self.cross_basis_rejections,
            },
            "contexts": contexts_out,
        }
