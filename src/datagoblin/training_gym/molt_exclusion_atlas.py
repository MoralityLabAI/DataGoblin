"""DISTILL-20: bounded exact exclusion atlas for public MoLT source revisions."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datagoblin.training_gym.molt_episode import Episode, World
from datagoblin.training_gym.molt_episode_policy import PublicPolicy
from datagoblin.training_gym.molt_goblin import digest
from datagoblin.training_gym.molt_packed_policy import (
    PackedPublicPolicy,
    Prepared,
    native_world,
    packed_select,
    prepare,
)

ARMS = ("reference", "packed_dense", "atlas_eager", "atlas_delayed")
NATIVE_WIDTHS = (5, 12, 16)
WIDTHS = (16, 32, 64, 128)
SPLITS = ("train", "calibration")
SEEDS = (2026120141, 2026120142)
ARRAY_SEED = 2026120143
WORKLOADS = (("stable", 1), ("stable", 4), ("stable", 16), ("stable", 32), ("churn", 16))
MAX_ATLAS_CELLS = 150_000
DELAY = 4
Row = tuple[str, int, int, int]


@dataclass
class AtlasCounts:
    queries: int = 0
    builds: int = 0
    atlas_uses: int = 0
    fallbacks: int = 0
    invalidations: int = 0
    cell_rejections: int = 0
    peak_cells: int = 0
    build_cpu_ns: int = 0


def update(table: list[int], row: Row) -> None:
    """At-most-capacity 0/1 recurrence, descending to prevent reuse."""
    _, cost, score, _ = row
    for c in range(len(table) - 1, cost - 1, -1):
        value = table[c - cost] + score
        if value > table[c]:
            table[c] = value


@dataclass(frozen=True)
class Atlas:
    fixed: frozenset[str]
    full: int
    omitted: dict[str, int]
    cells: int

    def score(self, variable: str | None) -> int:
        return self.full if variable is None else self.omitted.get(variable, self.full)


def dimensions(problem: Prepared, fixed: frozenset[str]) -> tuple[tuple[Row, ...], int, int]:
    if not fixed <= problem.ids:
        raise ValueError("unknown fixed exclusion")
    rows = tuple(row for row in problem.rows if row[0] not in fixed)
    cap = min(problem.capacity, sum(row[1] for row in rows))
    return rows, cap, (len(rows) + 2) * (cap + 1)


def build_atlas(problem: Prepared, fixed: frozenset[str], limit: int = MAX_ATLAS_CELLS) -> Atlas:
    """Compile all one-item exclusions using prefix DP and a rolling suffix.

    Prefix and suffix candidate sets are disjoint, including their packed
    membership bits. The maximum over a capacity split is therefore exact.
    Only packed scores survive construction; the DP tables are temporary.
    """
    rows, cap, cells = dimensions(problem, fixed)
    if cells > limit:
        raise ValueError("atlas cell budget")
    prefix = [[0] * (cap + 1)]
    for row in rows:
        next_table = prefix[-1].copy()
        update(next_table, row)
        prefix.append(next_table)
    full = prefix[-1][cap]
    suffix = [0] * (cap + 1)
    omitted = {}
    for i in range(len(rows) - 1, -1, -1):
        left = prefix[i]
        omitted[rows[i][0]] = max(left[c] + suffix[cap - c] for c in range(cap + 1))
        update(suffix, rows[i])
    if suffix[cap] != full:
        raise RuntimeError("prefix/suffix mismatch")
    return Atlas(fixed, full, omitted, cells)


def decode(problem: Prepared, score: int) -> tuple[str, ...]:
    return tuple(name for name, _, _, bit in problem.rows if score & bit)


class RevisionPlanner:
    """One immutable public objective; one bounded atlas at a time."""

    def __init__(self, problem: Prepared, *, delay: int, limit: int = MAX_ATLAS_CELLS):
        if not 1 <= delay <= 32 or not 1 <= limit <= MAX_ATLAS_CELLS:
            raise ValueError("invalid atlas bounds")
        self.problem = problem
        self.delay = delay
        self.limit = limit
        self.counts = AtlasCounts()
        self.fixed: frozenset[str] | None = None
        self.since_change = 0
        self.blocked = False
        self.atlas: Atlas | None = None

    def choose(self, fixed: frozenset[str], variable: str | None) -> tuple[str, ...]:
        p = self.problem
        if not fixed <= p.ids or (variable is not None and variable not in p.ids):
            raise ValueError("unknown excluded candidate")
        stats = self.counts
        stats.queries += 1
        if fixed != self.fixed:
            if self.atlas is not None:
                stats.invalidations += 1
            self.fixed = fixed
            self.atlas = None
            self.since_change = 0
            self.blocked = False
        self.since_change += 1
        if self.atlas is None and not self.blocked and self.since_change >= self.delay:
            _, _, cells = dimensions(p, fixed)
            if cells > self.limit:
                self.blocked = True
                stats.cell_rejections += 1
            else:
                start = time.process_time_ns()
                self.atlas = build_atlas(p, fixed, self.limit)
                stats.build_cpu_ns += time.process_time_ns() - start
                stats.builds += 1
                stats.peak_cells = max(stats.peak_cells, self.atlas.cells)
        if self.atlas is not None:
            if self.atlas.fixed != fixed:
                raise RuntimeError("stale numerical scope")
            stats.atlas_uses += 1
            return decode(p, self.atlas.score(variable))
        stats.fallbacks += 1
        forbidden = set(fixed)
        if variable is not None:
            forbidden.add(variable)
        return packed_select(p, forbidden, dense=True)


class AtlasPolicy(PublicPolicy):
    """Opt-in planner; only numerical scores are compiled, never evidence authority."""

    def __init__(self, *, delay: int):
        super().__init__("selective_refresh")
        self.delay = delay
        self.planner: RevisionPlanner | None = None

    def action(self, obs: dict[str, Any]) -> dict[str, Any]:
        if obs["event"] == "start":
            self.planner = RevisionPlanner(prepare(obs["candidates"], obs["capacity"]), delay=self.delay)
        return super().action(obs)

    def _submit(self) -> dict[str, Any]:
        if self.planner is None:
            raise ValueError("missing public start objective")
        # Source s0 is the registered variable source. All other observed
        # exclusions are fixed only for the current numerical atlas.
        fixed = frozenset(r["excluded"] for source, r in self.receipts.items() if source != "s0")
        variable = self.receipts["s0"]["excluded"] if "s0" in self.receipts else None
        selected = self.planner.choose(fixed, variable)
        return {
            "action": "submit",
            "frame_id": self.frame,
            "selected": list(selected),
            "receipts": sorted(r["receipt_id"] for r in self.receipts.values()),
        }


def policy_for(arm: str) -> PublicPolicy:
    if arm == "reference":
        return PublicPolicy()
    if arm == "packed_dense":
        return PackedPublicPolicy(dense=True)
    if arm in ("atlas_eager", "atlas_delayed"):
        return AtlasPolicy(delay=1 if arm == "atlas_eager" else DELAY)
    raise ValueError("unknown arm")


def counts(policy: PublicPolicy) -> dict[str, int]:
    if isinstance(policy, AtlasPolicy) and policy.planner is not None:
        return asdict(policy.planner.counts)
    return asdict(AtlasCounts())


def rotated(i: int, arm_names: tuple[str, ...] = ARMS) -> tuple[str, ...]:
    k = i % len(arm_names)
    return arm_names[k:] + arm_names[:k]


def native_episode(
    world: World, arm: str, *, factory: Callable[[str], PublicPolicy] = policy_for,
    metrics: Callable[[PublicPolicy], dict[str, int]] = counts,
) -> tuple[Episode, dict[str, int]]:
    host = Episode(world)
    observation = host.start()
    start = time.process_time_ns()
    policy = factory(arm)
    elapsed = time.process_time_ns() - start
    submit = 0
    while not host.closed:
        public = json.loads(json.dumps(observation))
        start = time.process_time_ns()
        action = policy.action(public)
        cost = time.process_time_ns() - start
        elapsed += cost
        if action["action"] == "submit":
            submit += cost
        observation = host.step(action)
    if host.result is None or not host.result["verified"] or not host.result["optimal"]:
        raise RuntimeError("native host verification failed")
    return host, {"policy_cpu_ns": elapsed, "submit_cpu_ns": submit, **metrics(policy)}


def merge_counts(rows: list[dict[str, int]]) -> dict[str, int]:
    return {
        k: max(r[k] for r in rows) if k in {"peak_cells", "peak_entries"} else sum(r[k] for r in rows)
        for k in rows[0]
    }


def native_row(
    split: str, seed: int, i: int, width: int, *, arm_names: tuple[str, ...] = ARMS,
    factory: Callable[[str], PublicPolicy] = policy_for,
    metrics: Callable[[PublicPolicy], dict[str, int]] = counts,
) -> dict[str, Any]:
    worlds = [native_world(split, seed, i, variant, width) for variant in ("relevant", "unrelated")]
    traces = {}
    arms = {}
    for arm in rotated(i, arm_names):
        traces[arm] = []
        rows = []
        for world in worlds:
            host, timing = native_episode(world, arm, factory=factory, metrics=metrics)
            assert host.result is not None
            traces[arm].append(host.trace)
            rows.append({
                **timing,
                "episodes": 1,
                "optimal": int(host.result["optimal"]),
                "verified": int(host.result["verified"]),
            })
        arms[arm] = merge_counts(rows)
    if any(t != traces["reference"] for t in traces.values()):
        raise RuntimeError("native trace parity failure")
    return {
        "root": worlds[0].root_id,
        "split": split,
        "width": width,
        "arms": arms,
        "trace_digest": digest(traces["reference"]),
    }


def public_stream(width: int, regime: str, i: int, kind: str, n: int) -> tuple[
    str, dict[str, Any], list[dict[str, Any]]
]:
    if width not in WIDTHS or regime not in ("tight", "medium") or (kind, n) not in WORKLOADS:
        raise ValueError("unregistered public stream")
    # Same underlying objective across revision lengths; workloads are paired,
    # not additional independent roots.
    root = digest(["distill20-public-scale", ARRAY_SEED, width, regime, i])
    rng = random.Random(int(root, 16))
    candidates = [
        {"id": f"c{j:03}", "cost": rng.randint(1, 32), "utility": rng.randint(1, 100)}
        for j in range(width)
    ]
    order = [c["id"] for c in candidates]
    rng.shuffle(order)
    fixed, variable = order[-1], order[0]
    initial = {
        "event": "start", "candidates": candidates, "capacity": width * (2 if regime == "tight" else 8),
        "frame_id": root + "-start",
        "receipts": [
            {"source": "s0", "version": 1, "excluded": variable, "receipt_id": root + "-s0"},
            {"source": "s1", "version": 1, "excluded": fixed, "receipt_id": root + "-s1"},
        ],
    }
    revisions = []
    cursor = 1
    versions = {"s0": 1, "s1": 1}
    for j in range(n):
        if kind == "churn" and (j + 1) % 4 == 0:
            fixed = next(name for name in order[j % width:] + order[:j % width]
                         if name not in (fixed, variable))
            source, excluded = "s1", fixed
        else:
            while order[cursor % width] in (fixed, variable):
                cursor += 1
            variable = order[cursor % width]
            cursor += 1
            source, excluded = "s0", variable
        versions[source] += 1
        revisions.append({
            "event": "receipt", "source": source, "version": versions[source], "excluded": excluded,
            "receipt_id": root + f"-{kind}-r{j}", "frame_id": root + f"-{kind}-f{j}",
        })
    return root, initial, revisions


def public_row(
    width: int, regime: str, i: int, kind: str, n: int, *, arm_names: tuple[str, ...] = ARMS,
    factory: Callable[[str], PublicPolicy] = policy_for,
    metrics: Callable[[PublicPolicy], dict[str, int]] = counts,
) -> dict[str, Any]:
    root, initial, revisions = public_stream(width, regime, i, kind, n)
    outcomes = {}
    arms = {}
    for arm in rotated(i, arm_names):
        obs = json.loads(json.dumps(initial))
        stream = json.loads(json.dumps(revisions))
        start = time.process_time_ns()
        policy = factory(arm)
        policy.action(obs)
        setup = time.process_time_ns() - start
        solve = 0
        actions = []
        for observation in stream:
            start = time.process_time_ns()
            policy.frame = observation["frame_id"]
            action = policy.action(observation)
            solve += time.process_time_ns() - start
            actions.append(action)
        outcomes[arm] = actions
        arms[arm] = {
            "policy_cpu_ns": setup + solve, "setup_cpu_ns": setup,
            "submit_cpu_ns": solve, "requests": n, **metrics(policy),
        }
    if any(a != outcomes["reference"] for a in outcomes.values()):
        raise RuntimeError("public selection or binding parity failure")
    receipts = {r["source"]: r for r in initial["receipts"]}
    for action, obs in zip(outcomes["reference"], revisions, strict=True):
        receipts[obs["source"]] = obs
        chosen = [c for c in initial["candidates"] if c["id"] in action["selected"]]
        forbidden = {r["excluded"] for r in receipts.values()}
        if (len(chosen) != len(action["selected"])
                or sum(c["cost"] for c in chosen) > initial["capacity"]
                or {c["id"] for c in chosen} & forbidden
                or action["frame_id"] != obs["frame_id"]
                or action["receipts"] != sorted(r["receipt_id"] for r in receipts.values())):
            raise RuntimeError("public feasibility or current evidence binding failure")
    return {
        "root": root, "width": width, "regime": regime, "kind": kind, "requests": n,
        "arms": arms, "outcome_digest": digest(outcomes["reference"]),
    }


def summarize(rows: list[dict[str, Any]], arm_names: tuple[str, ...] = ARMS) -> dict[str, Any]:
    out = {}
    for arm in arm_names:
        totals = merge_counts([r["arms"][arm] for r in rows])
        totals["cpu_wins_vs_packed"] = sum(
            r["arms"][arm]["policy_cpu_ns"] < r["arms"]["packed_dense"]["policy_cpu_ns"] for r in rows
        )
        out[arm] = totals
    return out


def campaign(native_roots: int = 12, array_roots: int = 12) -> dict[str, Any]:
    if not 1 <= native_roots <= 32 or not 1 <= array_roots <= 32:
        raise ValueError("invalid root count")
    native = [
        native_row(split, seed, i, width)
        for split, seed in zip(SPLITS, SEEDS, strict=True)
        for width in NATIVE_WIDTHS for i in range(native_roots)
    ]
    public = [
        public_row(width, regime, i, kind, n)
        for width in WIDTHS for regime in ("tight", "medium")
        for kind, n in WORKLOADS for i in range(array_roots)
    ]
    return {
        "schema": "distill20-exclusion-atlas-v1",
        "status": "development_only",
        "promotion_authorized": False,
        "training_admission": False,
        "provider_calls": 0,
        "max_atlas_cells": MAX_ATLAS_CELLS,
        "delay": DELAY,
        "native_roots_per_cell": native_roots,
        "public_roots_per_cell": array_roots,
        "native_summary": {
            f"{split}/{width}": summarize([r for r in native if r["split"] == split and r["width"] == width])
            for split in SPLITS for width in NATIVE_WIDTHS
        },
        "public_summary": {
            f"{regime}/{width}/{kind}/{n}": summarize([
                r for r in public if r["regime"] == regime and r["width"] == width
                and r["kind"] == kind and r["requests"] == n
            ])
            for regime in ("tight", "medium") for width in WIDTHS for kind, n in WORKLOADS
        },
        "rows_digest": digest([native, public]),
        "native_rows": native,
        "public_rows": public,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-roots", type=int, default=12)
    parser.add_argument("--array-roots", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = campaign(args.native_roots, args.array_roots)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ("native_rows", "public_rows")}, sort_keys=True))


if __name__ == "__main__":
    main()
