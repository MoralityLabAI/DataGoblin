"""DISTILL-21: exact optimum witnesses for source-revision workloads."""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datagoblin.training_gym.molt_episode_policy import PublicPolicy
from datagoblin.training_gym.molt_exclusion_atlas import (
    MAX_ATLAS_CELLS,
    NATIVE_WIDTHS,
    WIDTHS,
    WORKLOADS,
    counts,
    decode,
    dimensions,
    native_row,
    policy_for,
    public_row,
    summarize,
    update,
)
from datagoblin.training_gym.molt_goblin import digest
from datagoblin.training_gym.molt_packed_policy import Prepared, packed_select, prepare

ARMS = ("reference", "packed_dense", "atlas_delayed", "memo", "witness", "critical_atlas")
MODES = ("memo", "witness", "critical_atlas")
SPLITS = ("train", "calibration")
SEEDS = (2026120151, 2026120152)
FRESH_ARRAY_OFFSET = 1000
MEMO_LIMIT = 32
CRITICAL_DELAY = 4


@dataclass
class WitnessCounts:
    queries: int = 0
    scopes: int = 0
    direct_solves: int = 0
    full_solves: int = 0
    memo_hits: int = 0
    witness_hits: int = 0
    atlas_hits: int = 0
    critical_misses: int = 0
    builds: int = 0
    cell_rejections: int = 0
    invalidations: int = 0
    evictions: int = 0
    peak_entries: int = 0
    peak_cells: int = 0
    build_cpu_ns: int = 0


def build_critical_atlas(
    problem: Prepared, fixed: frozenset[str], critical: frozenset[str], limit: int = MAX_ATLAS_CELLS,
) -> tuple[dict[str, int], int]:
    """Exact exclusion scores with prefix snapshots only at critical IDs.

    The caller obtains critical from an exact optimum with fixed exclusions.
    Other candidates have the existing optimum as an exact witness. Only
    the selected omission scores are retained after this temporary DP.
    """
    rows, cap, _ = dimensions(problem, fixed)
    if not critical <= {row[0] for row in rows}:
        raise ValueError("critical candidates outside eligible scope")
    cells = (len(critical) + 2) * (cap + 1)
    if cells > limit:
        raise ValueError("critical atlas cell budget")
    prefix: dict[str, list[int]] = {}
    forward = [0] * (cap + 1)
    for row in rows:
        if row[0] in critical:
            prefix[row[0]] = forward.copy()
        update(forward, row)
    suffix = [0] * (cap + 1)
    omitted = {}
    for row in reversed(rows):
        name = row[0]
        if name in critical:
            left = prefix[name]
            omitted[name] = max(left[c] + suffix[cap - c] for c in range(cap + 1))
        update(suffix, row)
    if suffix[cap] != forward[cap] or omitted.keys() != critical:
        raise RuntimeError("critical atlas mismatch")
    return omitted, cells


class WitnessPlanner:
    """Per-objective, per-fixed-scope exact numeric memo and witness."""

    def __init__(
        self, problem: Prepared, mode: str, *, memo_limit: int = MEMO_LIMIT,
        cell_limit: int = MAX_ATLAS_CELLS,
    ):
        if mode not in MODES or not 1 <= memo_limit <= MEMO_LIMIT or not 1 <= cell_limit <= MAX_ATLAS_CELLS:
            raise ValueError("invalid witness planner settings")
        self.problem = problem
        self.mode = mode
        self.memo_limit = memo_limit
        self.cell_limit = cell_limit
        self.eligible = frozenset(row[0] for row in problem.rows)
        self.counts = WitnessCounts()
        self.fixed: frozenset[str] | None = None
        self.memo: OrderedDict[str | None, tuple[str, ...]] = OrderedDict()
        self.witness: tuple[str, ...] | None = None
        self.critical: frozenset[str] = frozenset()
        self.atlas: dict[str, int] | None = None
        self.scope_queries = 0
        self.scope_misses = 0
        self.blocked = False

    def reset(self, fixed: frozenset[str]) -> None:
        if self.fixed is not None:
            self.counts.invalidations += 1
        self.counts.scopes += 1
        self.fixed = fixed
        self.memo.clear()
        self.witness = None
        self.critical = frozenset()
        self.atlas = None
        self.scope_queries = 0
        self.scope_misses = 0
        self.blocked = False

    def remember(self, key: str | None, selected: tuple[str, ...]) -> tuple[str, ...]:
        if len(self.memo) >= self.memo_limit:
            self.memo.popitem(last=False)
            self.counts.evictions += 1
        self.memo[key] = selected
        self.counts.peak_entries = max(self.counts.peak_entries, len(self.memo))
        return selected

    def direct(self, fixed: frozenset[str], key: str | None) -> tuple[str, ...]:
        self.counts.direct_solves += 1
        forbidden = set(fixed)
        if key is not None:
            forbidden.add(key)
        return packed_select(self.problem, forbidden, dense=True)

    def choose(self, fixed: frozenset[str], variable: str | None) -> tuple[str, ...]:
        p, s = self.problem, self.counts
        if not fixed <= p.ids or (variable is not None and variable not in p.ids):
            raise ValueError("unknown excluded candidate")
        s.queries += 1
        if self.fixed != fixed:
            self.reset(fixed)
        self.scope_queries += 1
        # An already fixed or unaffordable exclusion is numerically inert.
        key = variable if variable in self.eligible and variable not in fixed else None
        if self.witness is not None and key not in self.critical:
            s.witness_hits += 1
            return self.witness
        if self.atlas is not None and key in self.atlas:
            s.atlas_hits += 1
            return decode(p, self.atlas[key])
        if key in self.memo:
            s.memo_hits += 1
            self.memo.move_to_end(key)
            return self.memo[key]
        if self.mode == "memo" or self.scope_queries == 1:
            selected = self.direct(fixed, key)
            # If the first requested variable is inert, this is already
            # the unrestricted fixed-scope optimum.
            if self.mode != "memo" and key is None:
                self.witness = selected
                self.critical = frozenset(selected)
            return self.remember(key, selected)

        if self.witness is None:
            # Delay full fixed-scope solving until a second novel request.
            # Exact memo hits above never trigger this additional solve.
            s.full_solves += 1
            self.witness = packed_select(p, set(fixed), dense=True)
            self.critical = frozenset(self.witness)
        if key not in self.critical:
            s.witness_hits += 1
            return self.witness

        s.critical_misses += 1
        self.scope_misses += 1
        if self.mode == "critical_atlas" and not self.blocked and self.scope_misses >= CRITICAL_DELAY:
            _, cap, _ = dimensions(p, fixed)
            cells = (len(self.critical) + 2) * (cap + 1)
            if cells > self.cell_limit:
                s.cell_rejections += 1
                self.blocked = True
            else:
                start = time.process_time_ns()
                self.atlas, peak = build_critical_atlas(p, fixed, self.critical, self.cell_limit)
                s.build_cpu_ns += time.process_time_ns() - start
                s.builds += 1
                s.peak_cells = max(s.peak_cells, peak)
                s.atlas_hits += 1
                assert key is not None
                return decode(p, self.atlas[key])
        return self.remember(key, self.direct(fixed, key))


class WitnessPolicy(PublicPolicy):
    """Keep numerical witnesses apart from current frame and receipt authority."""

    def __init__(self, mode: str):
        super().__init__("selective_refresh")
        # PublicPolicy.mode controls evidence-refresh transitions.
        self.witness_mode = mode
        self.planner: WitnessPlanner | None = None

    def action(self, obs: dict[str, Any]) -> dict[str, Any]:
        if obs["event"] == "start":
            self.planner = WitnessPlanner(prepare(obs["candidates"], obs["capacity"]), self.witness_mode)
        return super().action(obs)

    def _submit(self) -> dict[str, Any]:
        if self.planner is None:
            raise ValueError("missing public objective")
        fixed = frozenset(r["excluded"] for source, r in self.receipts.items() if source != "s0")
        variable = self.receipts["s0"]["excluded"] if "s0" in self.receipts else None
        selected = self.planner.choose(fixed, variable)
        return {
            "action": "submit",
            "frame_id": self.frame,
            "selected": list(selected),
            "receipts": sorted(r["receipt_id"] for r in self.receipts.values()),
        }


def factory(arm: str) -> PublicPolicy:
    return WitnessPolicy(arm) if arm in MODES else policy_for(arm)


def metrics(policy: PublicPolicy) -> dict[str, int]:
    if isinstance(policy, WitnessPolicy) and policy.planner is not None:
        return asdict(policy.planner.counts)
    return counts(policy)


def campaign(native_roots: int = 8, array_roots: int = 8) -> dict[str, Any]:
    if not 1 <= native_roots <= 32 or not 1 <= array_roots <= 32:
        raise ValueError("invalid root count")
    native = [
        native_row(split, seed, i, width, arm_names=ARMS, factory=factory, metrics=metrics)
        for split, seed in zip(SPLITS, SEEDS, strict=True)
        for width in NATIVE_WIDTHS for i in range(native_roots)
    ]
    public = [
        public_row(
            width, regime, i + FRESH_ARRAY_OFFSET, kind, n,
            arm_names=ARMS, factory=factory, metrics=metrics,
        )
        for width in WIDTHS for regime in ("tight", "medium")
        for kind, n in WORKLOADS for i in range(array_roots)
    ]
    return {
        "schema": "distill21-critical-witness-v1",
        "status": "development_only",
        "promotion_authorized": False,
        "training_admission": False,
        "provider_calls": 0,
        "native_roots_per_cell": native_roots,
        "public_roots_per_cell": array_roots,
        "fresh_array_offset": FRESH_ARRAY_OFFSET,
        "memo_limit": MEMO_LIMIT,
        "critical_delay": CRITICAL_DELAY,
        "max_atlas_cells": MAX_ATLAS_CELLS,
        "native_summary": {
            f"{split}/{width}": summarize([r for r in native if r["split"] == split and r["width"] == width], ARMS)
            for split in SPLITS for width in NATIVE_WIDTHS
        },
        "public_summary": {
            f"{regime}/{width}/{kind}/{n}": summarize([
                r for r in public if r["regime"] == regime and r["width"] == width
                and r["kind"] == kind and r["requests"] == n
            ], ARMS)
            for regime in ("tight", "medium") for width in WIDTHS for kind, n in WORKLOADS
        },
        "rows_digest": digest([native, public]),
        "native_rows": native,
        "public_rows": public,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-roots", type=int, default=8)
    parser.add_argument("--array-roots", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = campaign(args.native_roots, args.array_roots)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ("native_rows", "public_rows")}, sort_keys=True))


if __name__ == "__main__":
    main()
