"""MMS-06 Section 4: LBF-20M exact separator-query caching rebenchmark.

Compares:
- ``disabled``: native ``mdd_components`` (zero query-cache lookup/insertion overhead).
- ``zero_cap_instrumented``: ``mdd_query_cache`` with ``CachePool(0)`` (measures recursive payload accounting tax).
- ``bounded_exact``: ``mdd_query_cache`` with ``CachePool`` at 0, 2, 4, 8, 12, 16, 32, 64 KiB.

Workloads: ``cold`` (repeat=1), ``low_rep`` (repeat=2), ``high_rep`` (repeat=4) across 12 fresh CSP roots.
Randomized arm order across independent timing trials, declared boundary cache resets,
empirical clock-tick calibration, separate setup/query/accounting/verification/pipeline timings,
and paired root-level uncertainty respecting root dependence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter, OrderedDict
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import numpy as np
import psutil


def _payload_size(x: Any) -> int:
    if isinstance(x, (tuple, list)):
        return sys.getsizeof(x) + sum(_payload_size(y) for y in x)
    return sys.getsizeof(x)


class TimedCachePool:
    """LRU cache pool bounded in exact bytes with explicit cache-accounting CPU/wall timer."""

    def __init__(self, cap_bytes: int) -> None:
        if cap_bytes < 0:
            raise ValueError("cap_bytes must be >= 0")
        self.cap = int(cap_bytes)
        self.used = 0
        self.peak = 0
        self.evictions = 0
        self.rejected = 0
        self.accounting_cpu_s = 0.0
        self.accounting_wall_s = 0.0
        self.data: OrderedDict[Any, tuple[Any, int]] = OrderedDict()

    def set(self, key: Any, value: Any) -> None:
        tw, tc = time.perf_counter(), time.process_time()
        charge = _payload_size((key, value)) + 128
        if key in self.data:
            self.used -= self.data.pop(key)[1]
        if charge > self.cap:
            self.rejected += 1
        else:
            while self.used + charge > self.cap:
                _, (_, old) = self.data.popitem(last=False)
                self.used -= old
                self.evictions += 1
            self.data[key] = (value, charge)
            self.used += charge
            if self.used > self.peak:
                self.peak = self.used
        self.accounting_wall_s += time.perf_counter() - tw
        self.accounting_cpu_s += time.process_time() - tc


class TimedUnitMemo(MutableMapping[Any, Any]):
    def __init__(self, pool: TimedCachePool, site: int) -> None:
        self.pool = pool
        self.site = site

    def __getitem__(self, key: Any) -> Any:
        tw, tc = time.perf_counter(), time.process_time()
        address = (self.site, key)
        value, _ = self.pool.data[address]
        self.pool.data.move_to_end(address)
        self.pool.accounting_wall_s += time.perf_counter() - tw
        self.pool.accounting_cpu_s += time.process_time() - tc
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        self.pool.set((self.site, key), value)

    def __delitem__(self, key: Any) -> None:
        _, size = self.pool.data.pop((self.site, key))
        self.pool.used -= size

    def __iter__(self):
        return iter([key for site, key in self.pool.data if site == self.site])

    def __len__(self) -> int:
        return sum(site == self.site for site, _ in self.pool.data)

    def __contains__(self, key: object) -> bool:
        return (self.site, key) in self.pool.data

    def values(self):
        return [value for (site, _), (value, _) in self.pool.data.items() if site == self.site]


def measure_clock_quantum() -> dict[str, Any]:
    """Measure empirical minimum nonzero increment of perf_counter and process_time."""
    pc_info = time.get_clock_info("perf_counter")
    pt_info = time.get_clock_info("process_time")

    # Observe perf_counter min step
    pc_diffs: list[float] = []
    prev = time.perf_counter()
    for _ in range(5000):
        cur = time.perf_counter()
        if cur > prev:
            pc_diffs.append(cur - prev)
            if len(pc_diffs) >= 50:
                break
        prev = cur

    # Observe process_time min step (captures Windows 15.625 ms GetProcessTimes quantum)
    pt_diffs: list[float] = []
    prev_c = time.process_time()
    acc = 0
    for i in range(5_000_000):
        acc += i
        if (i & 255) == 0:
            cur_c = time.process_time()
            if cur_c > prev_c:
                pt_diffs.append(cur_c - prev_c)
                if len(pt_diffs) >= 10:
                    break
                prev_c = cur_c

    return {
        "perf_counter_info": {
            "implementation": pc_info.implementation,
            "reported_resolution_s": pc_info.resolution,
            "observed_min_increment_s": min(pc_diffs) if pc_diffs else None,
        },
        "process_time_info": {
            "implementation": pt_info.implementation,
            "reported_resolution_s": pt_info.resolution,
            "observed_min_increment_s": min(pt_diffs) if pt_diffs else None,
            "observed_median_increment_s": statistics.median(pt_diffs) if pt_diffs else None,
        },
        "_guard": acc & 1,
    }


def paired_root_summary(diffs: list[float], *, seed: int = 2026) -> dict[str, float]:
    """Paired mean, SE, and 95% percentile bootstrap CI over independent roots."""
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(2000, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n_roots": n,
        "mean": round(mean, 6),
        "se": round(se, 6),
        "ci95_low": round(float(lo), 6),
        "ci95_high": round(float(hi), 6),
    }


def run_arm_on_roots(
    native: Any,
    cases: list[tuple[Any, Any, list[dict[int, int]], list[tuple[int, ...]]]],
    *,
    arm_name: str,
    mode: str,
    cap_bytes: int,
    inner_reps: int,
    verify: bool,
) -> dict[str, Any]:
    """Execute one arm across all roots, returning per-root and aggregate timing/memory rows."""
    per_root: list[dict[str, Any]] = []
    tot_setup_c = tot_setup_w = 0.0
    tot_exec_c = tot_exec_w = 0.0
    tot_acct_c = tot_acct_w = 0.0
    tot_ver_c = tot_ver_w = 0.0

    for c, cut, queries, truth in cases:
        # Reset global helper caches at declared root boundary
        native.lift.cache_clear()
        native.project.cache_clear()

        # 1. Solver setup + factorized compilation (once per root, then query block timed over inner_reps)
        tw0, tc0 = time.perf_counter(), time.process_time()
        solver = native.Solver(c)
        dom = solver.ac(c.domains)
        assert dom is not None
        st = Counter()
        engine = native.Factorized(c, dom, cut, mode, st)
        setup_w = time.perf_counter() - tw0
        setup_c = time.process_time() - tc0

        # 2. Query execution (batched across inner_reps with fresh per-rep query cache so timing block
        # is long enough to suppress clock tick quantization while measuring the exact workload!)
        exec_w_sum = 0.0
        exec_c_sum = 0.0
        acct_w_sum = 0.0
        acct_c_sum = 0.0
        last_answers: list[tuple[Any, Any]] = []
        last_pool: TimedCachePool | None = None
        last_st = Counter()

        for _rep in range(inner_reps):
            # Reset only query memo & stats between inner timing reps; keep compiled base tables
            st_rep = Counter()
            engine.stats = st_rep
            engine.dag.stats = st_rep
            for u in engine.units:
                u.stats = st_rep
            pool: TimedCachePool | None = None
            if mode == "mdd_query_cache":
                pool = TimedCachePool(cap_bytes)
                for i, u in enumerate(engine.units):
                    u.memo = TimedUnitMemo(pool, i)
            else:
                for u in engine.units:
                    u.memo = {}

            twq, tcq = time.perf_counter(), time.process_time()
            answers = [engine.answer(q, with_witness=True) for q in queries]
            dw = time.perf_counter() - twq
            dc = time.process_time() - tcq

            exec_w_sum += dw
            exec_c_sum += dc
            if pool is not None:
                acct_w_sum += pool.accounting_wall_s
                acct_c_sum += pool.accounting_cpu_s
            last_answers = answers
            last_pool = pool
            last_st = st_rep

        # 3. Verification (separate from query execution!)
        ver_w = 0.0
        ver_c = 0.0
        if verify:
            twv, tcv = time.perf_counter(), time.process_time()
            for (ans, w), expected, q in zip(last_answers, truth, queries, strict=True):
                assert ans == expected and (w is not None) == bool(any(expected))
                if w is not None:
                    assert c.valid(w) and all(m & (1 << w[i]) for i, m in q.items())
            ver_w = time.perf_counter() - twv
            ver_c = time.process_time() - tcv

        nom_base_bits, nom_query_bits = engine.storage()
        lift_info = native.lift.cache_info()
        proj_info = native.project.cache_info()

        avg_exec_w = exec_w_sum / inner_reps
        avg_exec_c = exec_c_sum / inner_reps
        avg_acct_w = acct_w_sum / inner_reps
        avg_acct_c = acct_c_sum / inner_reps

        tot_setup_c += setup_c
        tot_setup_w += setup_w
        tot_exec_c += avg_exec_c
        tot_exec_w += avg_exec_w
        tot_acct_c += avg_acct_c
        tot_acct_w += avg_acct_w
        tot_ver_c += ver_c
        tot_ver_w += ver_w

        per_root.append(
            {
                "root": c.root,
                "queries": len(queries),
                "inner_reps": inner_reps,
                "exact_verified": verify,
                "setup_cpu_s": round(setup_c, 6),
                "setup_wall_s": round(setup_w, 6),
                "query_exec_cpu_s": round(avg_exec_c, 6),
                "query_exec_wall_s": round(avg_exec_w, 6),
                "cache_acct_cpu_s": round(avg_acct_c, 6),
                "cache_acct_wall_s": round(avg_acct_w, 6),
                "pure_query_cpu_s": round(max(0.0, avg_exec_c - avg_acct_c), 6),
                "pure_query_wall_s": round(max(0.0, avg_exec_w - avg_acct_w), 6),
                "verify_cpu_s": round(ver_c, 6),
                "verify_wall_s": round(ver_w, 6),
                "pipeline_cpu_s": round(setup_c + avg_exec_c + ver_c, 6),
                "pipeline_wall_s": round(setup_w + avg_exec_w + ver_w, 6),
                "query_payload_bytes": 0 if last_pool is None else last_pool.used,
                "peak_query_payload_bytes": 0 if last_pool is None else last_pool.peak,
                "cap_bytes": cap_bytes if mode == "mdd_query_cache" else 0,
                "evictions": 0 if last_pool is None else last_pool.evictions,
                "rejected": 0 if last_pool is None else last_pool.rejected,
                "compiled_base_bits": nom_base_bits,
                "nominal_query_bits": nom_query_bits,
                "global_lift_cache_currsize": lift_info.currsize,
                "global_project_cache_currsize": proj_info.currsize,
                "messages": int(last_st.get("messages", 0)),
                "bag_visits": int(last_st.get("bag_visits", 0)),
                "query_hits": int(last_st.get("query_hits", 0)),
                "query_misses": int(last_st.get("query_misses", 0)),
            }
        )

    return {
        "arm": arm_name,
        "mode": mode,
        "cap_bytes": cap_bytes if mode == "mdd_query_cache" else 0,
        "totals": {
            "setup_cpu_s": round(tot_setup_c, 6),
            "setup_wall_s": round(tot_setup_w, 6),
            "query_exec_cpu_s": round(tot_exec_c, 6),
            "query_exec_wall_s": round(tot_exec_w, 6),
            "cache_acct_cpu_s": round(tot_acct_c, 6),
            "cache_acct_wall_s": round(tot_acct_w, 6),
            "verify_cpu_s": round(tot_ver_c, 6),
            "verify_wall_s": round(tot_ver_w, 6),
            "pipeline_cpu_s": round(tot_setup_c + tot_exec_c + tot_ver_c, 6),
            "pipeline_wall_s": round(tot_setup_w + tot_exec_w + tot_ver_w, 6),
            "total_evictions": sum(r["evictions"] for r in per_root),
            "total_rejected": sum(r["rejected"] for r in per_root),
            "total_messages": sum(r["messages"] for r in per_root),
            "total_query_hits": sum(r["query_hits"] for r in per_root),
            "total_query_misses": sum(r["query_misses"] for r in per_root),
            "max_peak_query_payload_bytes": max(r["peak_query_payload_bytes"] for r in per_root),
            "sum_peak_query_payload_bytes": sum(r["peak_query_payload_bytes"] for r in per_root),
        },
        "per_root": per_root,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lt-repo", type=Path, required=True, help="Pinned LoopedTransformers worktree")
    p.add_argument("--roots", type=int, default=12)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--inner-reps", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    lbf_dir = args.lt_repo / "research" / "lite_braille_fresco"
    sys.path.insert(0, str(lbf_dir))
    import lbf20m_factor as native  # type: ignore[import-not-found]

    clock_meta = measure_clock_quantum()
    proc = psutil.Process(os.getpid())

    # Build immutable root fixtures & exact DFS ground-truth answers once (outside timed query loops)
    raw_roots = native.fresh()[: args.roots]
    workloads_spec = (("cold_r1", 1), ("low_rep_r2", 2), ("high_rep_r4", 4))
    arms = [
        ("disabled_mdd_components", "mdd_components", 0),
        ("zero_cap_instrumented", "mdd_query_cache", 0),
        ("cap_2KiB", "mdd_query_cache", 2 * 1024),
        ("cap_4KiB", "mdd_query_cache", 4 * 1024),
        ("cap_8KiB", "mdd_query_cache", 8 * 1024),
        ("cap_12KiB", "mdd_query_cache", 12 * 1024),
        ("cap_16KiB", "mdd_query_cache", 16 * 1024),
        ("cap_32KiB", "mdd_query_cache", 32 * 1024),
        ("cap_64KiB", "mdd_query_cache", 64 * 1024),
    ]

    # Precompute exact ground-truth supports once per unique query
    truth_by_root: list[tuple[Any, Any, list[dict[int, int]], list[tuple[int, ...]]]] = []
    for c, cut in raw_roots:
        solver = native.Solver(c)
        dom = solver.ac(c.domains)
        assert dom is not None
        base_q = native.fresh_queries(c, c.domains, cut)
        base_t = [native.dfs_support(solver, dom, q, Counter()) for q in base_q]
        truth_by_root.append((c, cut, base_q, base_t))

    # Warmup runtime consistently
    warm_cases = [(c, cut, q, t) for c, cut, q, t in truth_by_root[:2]]
    _ = run_arm_on_roots(
        native, warm_cases, arm_name="warmup", mode="mdd_components", cap_bytes=0, inner_reps=2, verify=True
    )

    results_by_workload: dict[str, Any] = {}
    rng = random.Random(2026112606)

    for wl_name, repeat in workloads_spec:
        cases = [(c, cut, q * repeat, t * repeat) for c, cut, q, t in truth_by_root]
        trials_out: list[dict[str, Any]] = []
        # Scale inner_reps inversely with repeat so each arm's query block stays well above clock quantum
        eff_reps = max(2, args.inner_reps // repeat)

        for trial_idx in range(args.trials):
            order = list(arms)
            rng.shuffle(order)
            arm_rows: dict[str, Any] = {}
            for arm_name, mode, cap_b in order:
                # Verify exact answers & witnesses on trial 0 for EVERY arm; skip redundant verify on trials 1..k-1
                # so timing blocks focus on solver + query + accounting, while keeping verify timed on trial 0.
                arm_rows[arm_name] = run_arm_on_roots(
                    native,
                    cases,
                    arm_name=arm_name,
                    mode=mode,
                    cap_bytes=cap_b,
                    inner_reps=eff_reps,
                    verify=(trial_idx == 0),
                )
            trials_out.append({"trial": trial_idx, "arm_order": [a[0] for a in order], "arms": arm_rows})

        # Aggregate across trials per root, then compute paired uncertainty vs disabled_mdd_components across roots!
        summary_arms: dict[str, Any] = {}
        # Root-level trial means for disabled baseline
        dis_root_wall = [
            statistics.mean(
                t["arms"]["disabled_mdd_components"]["per_root"][r_i]["query_exec_wall_s"]
                for t in trials_out
            )
            for r_i in range(len(cases))
        ]
        dis_root_cpu = [
            statistics.mean(
                t["arms"]["disabled_mdd_components"]["per_root"][r_i]["query_exec_cpu_s"]
                for t in trials_out
            )
            for r_i in range(len(cases))
        ]

        for arm_name, mode, cap_b in arms:
            root_wall = [
                statistics.mean(t["arms"][arm_name]["per_root"][r_i]["query_exec_wall_s"] for t in trials_out)
                for r_i in range(len(cases))
            ]
            root_cpu = [
                statistics.mean(t["arms"][arm_name]["per_root"][r_i]["query_exec_cpu_s"] for t in trials_out)
                for r_i in range(len(cases))
            ]
            root_acct_wall = [
                statistics.mean(t["arms"][arm_name]["per_root"][r_i]["cache_acct_wall_s"] for t in trials_out)
                for r_i in range(len(cases))
            ]
            t0_arm = trials_out[0]["arms"][arm_name]["totals"]
            wall_diffs = [a - b for a, b in zip(root_wall, dis_root_wall, strict=True)]
            cpu_diffs = [a - b for a, b in zip(root_cpu, dis_root_cpu, strict=True)]
            dis_tot_wall = sum(dis_root_wall)
            arm_tot_wall = sum(root_wall)
            dis_tot_cpu = sum(dis_root_cpu)
            arm_tot_cpu = sum(root_cpu)

            summary_arms[arm_name] = {
                "mode": mode,
                "cap_bytes": cap_b if mode == "mdd_query_cache" else 0,
                "mean_total_query_wall_s": round(arm_tot_wall, 6),
                "mean_total_query_cpu_s": round(arm_tot_cpu, 6),
                "mean_total_cache_acct_wall_s": round(sum(root_acct_wall), 6),
                "setup_wall_s_trial0": t0_arm["setup_wall_s"],
                "verify_wall_s_trial0": t0_arm["verify_wall_s"],
                "wall_delta_pct_vs_disabled": round(100.0 * (arm_tot_wall - dis_tot_wall) / max(dis_tot_wall, 1e-9), 2),
                "cpu_delta_pct_vs_disabled": round(100.0 * (arm_tot_cpu - dis_tot_cpu) / max(dis_tot_cpu, 1e-9), 2),
                "paired_wall_diff_per_root": paired_root_summary(wall_diffs),
                "paired_cpu_diff_per_root": paired_root_summary(cpu_diffs),
                "total_messages": t0_arm["total_messages"],
                "total_query_hits": t0_arm["total_query_hits"],
                "total_query_misses": t0_arm["total_query_misses"],
                "total_evictions": t0_arm["total_evictions"],
                "total_rejected": t0_arm["total_rejected"],
                "max_peak_query_payload_bytes": t0_arm["max_peak_query_payload_bytes"],
                "sum_peak_query_payload_bytes": t0_arm["sum_peak_query_payload_bytes"],
                "exact_verified_trial0": True,
            }

        results_by_workload[wl_name] = {
            "repeat": repeat,
            "inner_reps": eff_reps,
            "n_roots": len(cases),
            "n_trials": args.trials,
            "summary_by_arm": summary_arms,
            "trials": trials_out,
        }
        print(
            f"\n=== LBF-20M {wl_name} (roots={len(cases)}, trials={args.trials}, inner_reps={eff_reps}) ===",
            flush=True,
        )
        for arm_name, _, _ in arms:
            s = summary_arms[arm_name]
            tot_q = s["total_query_hits"] + s["total_query_misses"]
            print(
                f"  {arm_name:24s} q_wall={s['mean_total_query_wall_s'] * 1000:7.2f}ms "
                f"({s['wall_delta_pct_vs_disabled']:+6.2f}%) "
                f"acct={s['mean_total_cache_acct_wall_s'] * 1000:5.2f}ms "
                f"q_cpu={s['mean_total_query_cpu_s'] * 1000:7.2f}ms "
                f"msgs={s['total_messages']:4d} hits={s['total_query_hits']:3d}/{tot_q:3d} "
                f"evict={s['total_evictions']:3d} peakB={s['max_peak_query_payload_bytes']:6d}",
                flush=True,
            )

    mi = proc.memory_info()
    out_receipt = {
        "status": "PUBLIC_DEVELOPMENT_ONLY",
        "experiment": "MMS-06-LBF20M-REBENCHMARK",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "clock_calibration": clock_meta,
        "measured_process_rss_bytes": int(mi.rss),
        "measured_process_peak_wset_bytes": int(getattr(mi, "peak_wset", mi.rss)),
        "workloads": results_by_workload,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    # Write compact summary + full raw receipt
    (args.out / "raw_receipt.json").write_text(json.dumps(out_receipt, indent=2) + "\n", encoding="utf-8")
    compact = {
        k: v for k, v in out_receipt.items() if k != "workloads"
    }
    compact["workloads"] = {
        wl: {k2: v2 for k2, v2 in wv.items() if k2 != "trials"}
        for wl, wv in results_by_workload.items()
    }
    (args.out / "summary.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
