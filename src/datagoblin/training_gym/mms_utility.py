"""MMS-05: Utility-directed mechanistic feedback scheduler.

Extends hit/admission-proxy scheduling to estimate the downstream learning value
of instrumenting site ``s`` at memory allocation ``r``:

    U(s, r) = E[Delta V_dev_train | s, r] / (Delta C_CPU + lambda * Delta M)

Strictly enforces partition isolation: only ``dev_train`` feedback signals are
accepted; final development, confirmation, and sealed cohorts are rejected.

Includes:
- Historical eligibility traces for delayed credit assignment.
- Online component-level gradient covariance tracking.
- Compressed physical-coordinate history ring buffer (byte-budgeted).
- Successful-repair retrieval by cosine similarity on state signatures.

Pure NumPy + stdlib so base CI runs without PyTorch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

ALLOWED_SIGNAL_KINDS = (
    "lean_proof_progress",
    "lbf_bellman_regret",
    "mcp_retrieval_recovery",
    "storyworld_repair",
)
ALLOWED_PARTITIONS = ("dev_train",)


@dataclass(frozen=True)
class FeedbackSignal:
    """Dedicated training-feedback partition utility observation."""

    step: int
    site: str
    ram_mib: float
    partition: str
    signal_kind: str
    delta_v: float
    delta_cpu_seconds: float
    delta_ram_mib: float
    state_signature: np.ndarray | None = None
    component_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.partition not in ALLOWED_PARTITIONS:
            raise ValueError(f"forbidden feedback partition {self.partition!r}; only 'dev_train' allowed")
        if self.signal_kind not in ALLOWED_SIGNAL_KINDS:
            raise ValueError(f"unknown signal_kind {self.signal_kind!r}")
        if not self.site or self.step < 0:
            raise ValueError("invalid site or step")
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
    delta_v: float
    component_ids: tuple[str, ...]
    signature: np.ndarray  # normalized float32 [d]

    @property
    def nbytes(self) -> int:
        return int(self.signature.nbytes + 128)


@dataclass(frozen=True)
class AllocationChoice:
    site: str
    ram_mib: float
    utility: float
    expected_delta_v: float
    expected_cost: float
    policy: str
    retrieved_components: tuple[str, ...]


class _SiteStats:
    """Per-(site, ram_mib) online sufficient statistics + gradient covariance sketch."""

    def __init__(self, site: str, ram_mib: float, cov_dim: int = 8) -> None:
        self.site = site
        self.ram_mib = ram_mib
        self.count = 0
        self.ema_delta_v = 0.0
        self.ema_cpu = 1e-3
        self.ema_ram = max(0.0, ram_mib)
        self.eligibility = 0.0
        self.hits = 0
        self.admitted = 0
        # Online component gradient mean & covariance (bounded cov_dim x cov_dim)
        self.cov_dim = cov_dim
        self.grad_n = 0
        self.grad_mean = np.zeros(cov_dim, dtype=np.float32)
        self.grad_m2 = np.zeros((cov_dim, cov_dim), dtype=np.float32)

    def update_grad_cov(self, comp_grad: np.ndarray) -> None:
        vec = np.asarray(comp_grad, dtype=np.float32).reshape(-1)
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
        """Signal-to-noise ratio of component gradients from online covariance trace."""
        if self.grad_n < 2:
            return 1.0
        cov_trace = float(np.trace(self.grad_m2) / (self.grad_n - 1))
        mean_norm_sq = float(np.dot(self.grad_mean, self.grad_mean))
        return float(mean_norm_sq / (cov_trace + 1e-6))


class UtilityDirectedScheduler:
    """MMS-05 utility-directed site & memory scheduler with strict cost/partition accounting."""

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
        cov_dim: int = 8,
    ) -> None:
        if not sites or not ram_tiers_mib:
            raise ValueError("sites and ram_tiers_mib must be non-empty")
        if not isfinite(lambda_ram) or lambda_ram < 0:
            raise ValueError("lambda_ram must be finite and >= 0")
        if not 0.0 <= eligibility_decay < 1.0:
            raise ValueError("eligibility_decay must be in [0, 1)")
        if history_budget_bytes < 0:
            raise ValueError("history_budget_bytes must be >= 0")

        self.sites = tuple(sites)
        self.ram_tiers_mib = tuple(float(r) for r in ram_tiers_mib)
        self.lambda_ram = float(lambda_ram)
        self.eligibility_decay = float(eligibility_decay)
        self.ema_alpha = float(ema_alpha)
        self.exploration_bonus = float(exploration_bonus)
        self.history_budget_bytes = int(history_budget_bytes)

        self._stats: dict[tuple[str, float], _SiteStats] = {
            (s, r): _SiteStats(s, r, cov_dim=cov_dim) for s in self.sites for r in self.ram_tiers_mib
        }
        self._coord_history: list[tuple[int, str, np.ndarray]] = []
        self._coord_bytes = 0
        self._repairs: list[RepairRecord] = []
        self._repair_bytes = 0

        self.total_selections = 0
        self.extra_forward_passes = 0
        self.extra_backward_passes = 0
        self.scheduler_cpu_seconds = 0.0

    @property
    def scheduler_ram_bytes(self) -> int:
        stats_bytes = len(self._stats) * (256 + 4 * (8 + 64))
        return self._coord_bytes + self._repair_bytes + stats_bytes

    def record_coordinate_delta(self, step: int, site: str, compressed_coords: np.ndarray) -> None:
        """Store compressed physical-coordinate history under ``history_budget_bytes // 2``."""
        t0 = time.process_time()
        arr = np.ascontiguousarray(np.asarray(compressed_coords, dtype=np.float32).reshape(-1))
        cap = self.history_budget_bytes // 2
        if arr.nbytes <= cap:
            while self._coord_history and self._coord_bytes + arr.nbytes > cap:
                _, _, old = self._coord_history.pop(0)
                self._coord_bytes -= old.nbytes
            self._coord_history.append((step, site, arr))
            self._coord_bytes += arr.nbytes
        self.scheduler_cpu_seconds += time.process_time() - t0

    def record_proxy_counts(self, site: str, ram_mib: float, *, hits: int, admitted: int,
                            comp_grad: np.ndarray | None = None) -> None:
        """Record cache hits/admissions and optional component gradient vector for covariance."""
        t0 = time.process_time()
        st = self._stats[(site, float(ram_mib))]
        st.hits += int(hits)
        st.admitted += int(admitted)
        if comp_grad is not None:
            st.update_grad_cov(comp_grad)
        self.scheduler_cpu_seconds += time.process_time() - t0

    def observe_feedback(self, sig: FeedbackSignal) -> None:
        """Update utility estimates, eligibility traces, and successful-repair memory."""
        t0 = time.process_time()
        key = (sig.site, float(sig.ram_mib))
        if key not in self._stats:
            raise ValueError(f"unregistered (site, ram_mib) pair: {key}")

        # Decay all eligibility traces, then credit active site
        for st in self._stats.values():
            st.eligibility *= self.eligibility_decay
        active = self._stats[key]
        active.eligibility += 1.0

        # Credit-assigned update across sites with non-negligible eligibility
        total_e = sum(st.eligibility for st in self._stats.values())
        for st in self._stats.values():
            if st.eligibility <= 1e-4:
                continue
            weight = st.eligibility / max(total_e, 1e-12)
            a = self.ema_alpha * weight
            st.ema_delta_v = (1.0 - a) * st.ema_delta_v + a * sig.delta_v
            if st is active:
                st.count += 1
                st.ema_cpu = (1.0 - self.ema_alpha) * st.ema_cpu + self.ema_alpha * max(1e-6, sig.delta_cpu_seconds)
                st.ema_ram = (1.0 - self.ema_alpha) * st.ema_ram + self.ema_alpha * sig.delta_ram_mib

        # Store positive-utility repair trajectory for retrieval
        if sig.delta_v > 0 and sig.state_signature is not None:
            vec = np.asarray(sig.state_signature, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8 and np.isfinite(vec).all():
                rec = RepairRecord(
                    step=sig.step,
                    site=sig.site,
                    signal_kind=sig.signal_kind,
                    delta_v=sig.delta_v,
                    component_ids=sig.component_ids,
                    signature=np.ascontiguousarray(vec / norm),
                )
                cap = self.history_budget_bytes - (self.history_budget_bytes // 2)
                if rec.nbytes <= cap:
                    while self._repairs and self._repair_bytes + rec.nbytes > cap:
                        old = self._repairs.pop(0)
                        self._repair_bytes -= old.nbytes
                    self._repairs.append(rec)
                    self._repair_bytes += rec.nbytes

        self.scheduler_cpu_seconds += time.process_time() - t0

    def retrieve_repairs(
        self, query_signature: np.ndarray | None, *, top_k: int = 3, min_cosine: float = 0.2
    ) -> list[RepairRecord]:
        """Retrieve past successful repairs matching the current state signature."""
        if query_signature is None or not self._repairs:
            return []
        q = np.asarray(query_signature, dtype=np.float32).reshape(-1)
        qn = float(np.linalg.norm(q))
        if qn <= 1e-8 or not np.isfinite(q).all():
            return []
        q = q / qn
        scored: list[tuple[float, RepairRecord]] = []
        for rec in self._repairs:
            if rec.signature.shape != q.shape:
                continue
            cos = float(np.dot(q, rec.signature))
            if cos >= min_cosine:
                scored.append((cos * rec.delta_v, rec))
        scored.sort(key=lambda item: (-item[0], -item[1].step))
        return [rec for _, rec in scored[:top_k]]

    def utility_score(self, site: str, ram_mib: float) -> tuple[float, float, float]:
        """Compute U(s, r) = E[Delta V_dev_train | s, r] / (Delta C_CPU + lambda * Delta M)."""
        st = self._stats[(site, float(ram_mib))]
        # Modulate expected value gently by gradient covariance SNR when available
        snr_factor = min(2.0, 0.5 + 0.5 * st.grad_snr)
        explore = self.exploration_bonus / np.sqrt(st.count + 1.0)
        exp_v = st.ema_delta_v * snr_factor + explore
        denom = max(1e-6, st.ema_cpu + self.lambda_ram * st.ema_ram)
        return float(exp_v / denom), float(exp_v), float(denom)

    def select(
        self,
        *,
        policy: str = "utility",
        query_signature: np.ndarray | None = None,
    ) -> AllocationChoice:
        """Select (site, ram_mib) under 'utility', 'hit_proxy', or 'round_robin'."""
        t0 = time.process_time()
        keys = list(self._stats.keys())
        if policy == "round_robin":
            site, ram = keys[self.total_selections % len(keys)]
            u, ev, ec = self.utility_score(site, ram)
        elif policy == "hit_proxy":
            # Present-scheduler baseline: rank purely by admitted/hit count
            best_key = max(
                keys,
                key=lambda k: (self._stats[k].admitted + 0.1 * self._stats[k].hits, -k[1], k[0]),
            )
            site, ram = best_key
            u, ev, ec = self.utility_score(site, ram)
        elif policy == "utility":
            best = None
            for s, r in keys:
                u, ev, ec = self.utility_score(s, r)
                cand = (u, ev, -r, s, r, ec)
                if best is None or cand > best:
                    best = cand
            assert best is not None
            u, ev, _, site, ram, ec = best
        else:
            raise ValueError(f"unknown scheduler policy: {policy}")

        retrieved = self.retrieve_repairs(query_signature)
        comp_ids: list[str] = []
        for rec in retrieved:
            for cid in rec.component_ids:
                if cid not in comp_ids:
                    comp_ids.append(cid)

        self.total_selections += 1
        self.scheduler_cpu_seconds += time.process_time() - t0
        return AllocationChoice(
            site=site,
            ram_mib=ram,
            utility=u,
            expected_delta_v=ev,
            expected_cost=ec,
            policy=policy,
            retrieved_components=tuple(comp_ids[:4]),
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "lambda_ram": self.lambda_ram,
            "total_selections": self.total_selections,
            "extra_forward_passes": self.extra_forward_passes,
            "extra_backward_passes": self.extra_backward_passes,
            "scheduler_cpu_seconds": round(self.scheduler_cpu_seconds, 6),
            "scheduler_ram_bytes": self.scheduler_ram_bytes,
            "stored_repairs": len(self._repairs),
            "stored_coord_deltas": len(self._coord_history),
            "sites": {
                f"{s}@{r:g}MiB": {
                    "count": st.count,
                    "ema_delta_v": round(st.ema_delta_v, 6),
                    "ema_cpu": round(st.ema_cpu, 6),
                    "grad_snr": round(st.grad_snr, 4),
                    "hits": st.hits,
                    "admitted": st.admitted,
                    "utility": round(self.utility_score(s, r)[0], 6),
                }
                for (s, r), st in self._stats.items()
            },
        }
