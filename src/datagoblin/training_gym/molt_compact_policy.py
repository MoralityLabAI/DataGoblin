"""DISTILL-18: opt-in exact selection compactification in native MoLT episodes.

Only a numerical selection is reusable. Frame and receipt authorization always
comes from the current public observation and remains checked by the host.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datagoblin.training_gym.molt_episode import VARIANTS, Episode, World, make_world
from datagoblin.training_gym.molt_episode_policy import PublicPolicy
from datagoblin.training_gym.molt_goblin import digest

ARMS = ("reference", "compiled", "cache2", "cache8")
DEVELOPMENT_SPLITS = ("train", "calibration")
SEEDS = (2026120121, 2026120122)
REPEATS = (1, 8)
MAX_CANDIDATES = 128
MAX_CAPACITY = 4096


@dataclass(frozen=True)
class Problem:
    """Complete numerical objective, scoped to positive integer MoLT arrays."""

    rows: tuple[tuple[str, int, int], ...]
    capacity: int


@dataclass
class Counts:
    submissions: int = 0
    candidate_rows: int = 0
    receipt_rows: int = 0
    dp_visits: int = 0
    fast_paths: int = 0
    hits: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0
    peak_entries: int = 0
    key_rows: int = 0
    submit_cpu_ns: int = 0


def integer(value: Any, low: int, high: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise ValueError("integer outside supported MoLT objective")
    return value


def normalize(
    candidates: list[dict[str, Any]], capacity: int, receipts: dict[str, dict[str, Any]]
) -> Problem:
    """Discard forbidden/unaffordable candidates and saturate excess capacity.

    This is an exact quotient of the *selection* objective. It does not
    collapse source versions, frame authority, or the episode state.
    """
    cap = integer(capacity, 0, MAX_CAPACITY)
    if len(candidates) > MAX_CANDIDATES or len(receipts) > MAX_CANDIDATES:
        raise ValueError("MoLT input exceeds development limits")
    rows = []
    seen = set()
    for c in candidates:
        name = c["id"]
        if not isinstance(name, str) or not name or len(name) > 128 or name in seen:
            raise ValueError("invalid candidate ID")
        seen.add(name)
        cost = integer(c["cost"], 1, MAX_CAPACITY)
        utility = integer(c["utility"], 1, 2**31 - 1)
        rows.append((name, cost, utility))
    forbidden = {r["excluded"] for r in receipts.values()}
    if not forbidden <= seen:
        raise ValueError("unknown excluded candidate")
    eligible = tuple(sorted(row for row in rows if row[0] not in forbidden and row[1] <= cap))
    return Problem(eligible, min(cap, sum(row[1] for row in eligible)))


def solve(problem: Problem, counts: Counts) -> tuple[str, ...]:
    """Exact incumbent specialization with a sparse-DP fallback.

    All utilities are positive, so if all eligible items fit they must all
    be selected. If even the two cheapest items do not fit together, select
    the best single item. Remaining cases use the native DP and tie rule.
    """
    rows, cap = problem.rows, problem.capacity
    if sum(row[1] for row in rows) <= cap:
        counts.fast_paths += 1
        return tuple(row[0] for row in rows)
    cheapest = sorted(row[1] for row in rows)[:2]
    if sum(cheapest) > cap:
        counts.fast_paths += 1
        return (min(rows, key=lambda row: (-row[2], row[0]))[0],)
    table: dict[int, tuple[int, tuple[str, ...]]] = {0: (0, ())}
    for name, cost, utility in rows:
        for used, (value, ids) in tuple(table.items()):
            counts.dp_visits += 1
            new_cost = used + cost
            if new_cost > cap:
                continue
            candidate = (value + utility, (*ids, name))
            old = table.get(new_cost)
            if old is None or (-candidate[0], candidate[1]) < (-old[0], old[1]):
                table[new_cost] = candidate
    return min(table.values(), key=lambda x: (-x[0], x[1]))[1]


class SelectionMemory:
    """Bounded lazy exact memo; stores selected IDs only, never evidence."""

    def __init__(self, capacity: int, counts: Counts):
        self.capacity = integer(capacity, 0, 128)
        self.counts = counts
        self.entries: OrderedDict[Problem, tuple[str, ...]] = OrderedDict()

    def choose(self, problem: Problem) -> tuple[str, ...]:
        stats = self.counts
        if not self.capacity:
            return solve(problem, stats)
        stats.key_rows += len(problem.rows)
        if problem in self.entries:
            stats.hits += 1
            self.entries.move_to_end(problem)
            return self.entries[problem]
        stats.misses += 1
        selected = solve(problem, stats)
        if len(self.entries) == self.capacity:
            self.entries.popitem(last=False)
            stats.evictions += 1
        self.entries[problem] = selected
        stats.writes += 1
        stats.peak_entries = max(stats.peak_entries, len(self.entries))
        return selected


class CompiledPublicPolicy(PublicPolicy):
    """Preserve the public policy transition rules and specialize only submit."""

    def __init__(self, memory: SelectionMemory | None = None):
        super().__init__("selective_refresh")
        self.memory = memory if memory is not None else SelectionMemory(0, Counts())

    def _submit(self) -> dict[str, Any]:
        start = time.process_time_ns()
        stats = self.memory.counts
        stats.submissions += 1
        stats.candidate_rows += len(self.candidates)
        stats.receipt_rows += len(self.receipts)
        problem = normalize(self.candidates, self.capacity, self.receipts)
        selected = self.memory.choose(problem)
        # Rebind authorization from current observations on *every* call.
        result = {
            "action": "submit",
            "frame_id": self.frame,
            "selected": list(selected),
            "receipts": sorted(r["receipt_id"] for r in self.receipts.values()),
        }
        stats.submit_cpu_ns += time.process_time_ns() - start
        return result


def reference_dp_visits(policy: PublicPolicy) -> int:
    """Offline accounting of the native DP iteration count, independent of utilities."""
    forbidden = {r["excluded"] for r in policy.receipts.values()}
    costs = {0}
    visits = 0
    for c in sorted(policy.candidates, key=lambda x: x["id"]):
        if c["id"] in forbidden:
            continue
        visits += len(costs)
        costs.update(used + c["cost"] for used in tuple(costs) if used + c["cost"] <= policy.capacity)
    return visits


class MeasuredReference(PublicPolicy):
    def __init__(self, counts: Counts):
        super().__init__("selective_refresh")
        self.counts = counts

    def _submit(self) -> dict[str, Any]:
        start = time.process_time_ns()
        result = super()._submit()
        self.counts.submit_cpu_ns += time.process_time_ns() - start
        # Accounting only, after the timed native call.
        self.counts.submissions += 1
        self.counts.candidate_rows += len(self.candidates)
        self.counts.receipt_rows += len(self.receipts)
        self.counts.dp_visits += reference_dp_visits(self)
        return result


def episode(world: World, policy: PublicPolicy) -> Episode:
    host = Episode(world)
    observation = host.start()
    while not host.closed:
        action = policy.action(json.loads(json.dumps(observation)))
        observation = host.step(action)
    if host.result is None or not host.result["verified"] or not host.result["optimal"]:
        raise RuntimeError("native host rejected planner output")
    return host


def run_root(split: str, seed: int, index: int, repeats: int) -> dict[str, Any]:
    # Root and stream scoped banks. Repetitions are an explicit workload,
    # each with a new host and freshly observed receipts.
    if split not in DEVELOPMENT_SPLITS or not 1 <= repeats <= 32:
        raise ValueError("development-only workload")
    worlds = [make_world(split, seed, index, variant) for variant in VARIANTS]
    stream = worlds * repeats
    rows: dict[str, dict[str, Any]] = {}
    traces: list[list[dict[str, Any]]] = []
    for arm in ARMS:
        stats = Counts()
        memory = SelectionMemory({"reference": 0, "compiled": 0, "cache2": 2, "cache8": 8}[arm], stats)
        if memory.entries:
            raise RuntimeError("bank not empty")
        optimal = verified = calls = utility = 0
        for i, world in enumerate(stream):
            policy: PublicPolicy
            if arm == "reference":
                policy = MeasuredReference(stats)
            else:
                policy = CompiledPublicPolicy(memory)
            host = episode(world, policy)
            if arm == "reference":
                traces.append(host.trace)
            elif host.trace != traces[i]:
                raise RuntimeError("native action or observation parity failed")
            assert host.result is not None
            optimal += int(host.result["optimal"])
            verified += int(host.result["verified"])
            calls += host.result["verifier_calls"]
            utility += host.result["utility"]
        rows[arm] = {
            **asdict(stats),
            "episodes": len(stream),
            "optimal": optimal,
            "verified": verified,
            "verifier_calls": calls,
            "utility": utility,
            "trace_digest": digest(traces),
        }
        if stats.submissions != len(stream) or stats.peak_entries > memory.capacity:
            raise RuntimeError("memory accounting failed")
    return {"root_id": worlds[0].root_id, "split": split, "repeats": repeats, "arms": rows}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in ARMS:
        arms = [r["arms"][arm] for r in rows]
        totals = {
            k: sum(x[k] for x in arms)
            for k in arms[0]
            if k not in {"trace_digest", "peak_entries"}
        }
        totals["peak_entries"] = max(x["peak_entries"] for x in arms)
        totals["dp_visit_wins"] = sum(r["arms"][arm]["dp_visits"] < r["arms"]["reference"]["dp_visits"] for r in rows)
        totals["submit_cpu_wins"] = sum(
            r["arms"][arm]["submit_cpu_ns"] < r["arms"]["reference"]["submit_cpu_ns"] for r in rows
        )
        result[arm] = totals
    return result


def campaign(roots: int = 64, repeats: tuple[int, ...] = REPEATS) -> dict[str, Any]:
    integer(roots, 1, 256)
    if not repeats or len(set(repeats)) != len(repeats):
        raise ValueError("invalid repeat counts")
    rows = [
        run_root(split, seed, i, n)
        for split, seed in zip(DEVELOPMENT_SPLITS, SEEDS, strict=True)
        for i in range(roots)
        for n in repeats
    ]
    return {
        "schema": "distill18-native-molt-compact-v1",
        "status": "development_only",
        "promotion_authorized": False,
        "training_admission": False,
        "provider_calls": 0,
        "roots_per_split": roots,
        "seeds": list(SEEDS),
        "repeats": list(repeats),
        "summary": {
            f"{split}/{n}": summarize([r for r in rows if r["split"] == split and r["repeats"] == n])
            for split in DEVELOPMENT_SPLITS for n in repeats
        },
        "rows_digest": digest(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = campaign(args.roots)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
