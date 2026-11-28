"""DISTILL-22: exact relaxation witnesses over changing public exclusions."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datagoblin.training_gym.molt_critical_witness import factory as previous_factory
from datagoblin.training_gym.molt_critical_witness import metrics as previous_metrics
from datagoblin.training_gym.molt_episode_policy import PublicPolicy
from datagoblin.training_gym.molt_exclusion_atlas import native_row, rotated, summarize
from datagoblin.training_gym.molt_goblin import digest
from datagoblin.training_gym.molt_packed_policy import Prepared, packed_select, prepare

MODES = ("lattice_memo", "lattice_relax", "lattice_guarded")
ARMS = ("reference", "packed_dense", "atlas_delayed", "critical_atlas", *MODES)
SPLITS = ("train", "calibration")
SEEDS = (2026120161, 2026120162)
ARRAY_SEED = 2026120163
NATIVE_WIDTHS = (5, 12, 16)
WIDTHS = (32, 64, 128)
WORKLOADS = (("stable", 4), ("stable", 32), ("churn", 16), ("rotate4", 32), ("overlap4", 32))
MEMO_LIMIT = 32
RECENT_LIMIT = 8


@dataclass(frozen=True)
class Entry:
    selected: tuple[str, ...]
    membership: int


@dataclass
class LatticeCounts:
    queries: int = 0
    direct_solves: int = 0
    relaxed_solves: int = 0
    exact_hits: int = 0
    subset_hits: int = 0
    relaxed_hits: int = 0
    subset_checks: int = 0
    relaxation_checks: int = 0
    writes: int = 0
    evictions: int = 0
    peak_entries: int = 0
    peak_cells: int = 0


class LatticePlanner:
    """Exact cache scoped to one immutable numerical problem.

    K subset F and optimum(K) feasible under F imply optimum(K)=optimum(F).
    Keys and membership are bitsets of affordable candidate IDs. Evidence
    identity, versions and frame authority never enter the cache.
    """

    def __init__(self, problem: Prepared, mode: str, *, limit: int = MEMO_LIMIT):
        if mode not in MODES or not 1 <= limit <= MEMO_LIMIT:
            raise ValueError("invalid lattice settings")
        self.problem = problem
        self.mode = mode
        self.limit = limit
        self.bits = {name: bit for name, _, _, bit in problem.rows}
        self.cache: OrderedDict[int, Entry] = OrderedDict()
        self.recent: OrderedDict[int, None] = OrderedDict()
        self.counts = LatticeCounts()

    def key(self, forbidden: set[str]) -> int:
        if not forbidden <= self.problem.ids:
            raise ValueError("unknown excluded candidate")
        return sum(self.bits.get(name, 0) for name in forbidden)

    def solve(self, key: int, *, relaxed: bool) -> Entry:
        forbidden = {name for name, bit in self.bits.items() if key & bit}
        selected = packed_select(self.problem, forbidden, dense=True)
        entry = Entry(selected, sum(self.bits[name] for name in selected))
        if entry.membership & key:
            raise RuntimeError("infeasible solved witness")
        if relaxed:
            self.counts.relaxed_solves += 1
        else:
            self.counts.direct_solves += 1
        if key in self.cache:
            if self.cache[key] != entry:
                raise RuntimeError("inconsistent exact witness")
            self.cache.move_to_end(key)
            return entry
        if len(self.cache) >= self.limit:
            self.cache.popitem(last=False)
            self.counts.evictions += 1
        self.cache[key] = entry
        self.counts.writes += 1
        self.counts.peak_entries = max(self.counts.peak_entries, len(self.cache))
        return entry

    def lookup(self, key: int) -> Entry | None:
        if key in self.cache:
            self.counts.exact_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        # Enumerate the exclusion submasks only when there are few. For
        # broader queries scan the bounded cache instead.
        if key.bit_count() <= 5:
            subset = key
            candidates = []
            while subset:
                subset = (subset - 1) & key
                candidates.append(subset)
        else:
            candidates = list(reversed(self.cache))
        for scope in candidates:
            self.counts.subset_checks += 1
            entry = self.cache.get(scope)
            if entry is not None and scope & key == scope and entry.membership & key == 0:
                self.counts.subset_hits += 1
                self.cache.move_to_end(scope)
                return entry
        return None

    def relaxation(self, key: int, history: tuple[int, ...]) -> int | None:
        """One prospective relaxation at most, using only previous requests.

        Prefer the largest proper intersection with a past exclusion scope.
        Its support is the number of distinct recent actual queries that
        contain it. Guarded mode requires two such previous queries.
        An already solved but infeasible relaxation triggers direct solving.
        """
        if self.mode == "lattice_memo" or not history:
            return None
        candidates = {key & old for old in history if key & old != key}
        if not candidates:
            return None
        best = None
        best_rank = (-1, -1, -1)
        threshold = 2 if self.mode == "lattice_guarded" else 1
        for scope in candidates:
            support = 0
            for old in history:
                self.counts.relaxation_checks += 1
                support += int(scope & old == scope)
            rank = (scope.bit_count(), support, -scope)
            if support >= threshold and rank > best_rank:
                best, best_rank = scope, rank
        return best if best is not None and best not in self.cache else None

    def choose(self, forbidden: set[str]) -> tuple[str, ...]:
        key = self.key(forbidden)
        self.counts.queries += 1
        history = tuple(self.recent)
        if key in self.recent:
            self.recent.move_to_end(key)
        else:
            if len(self.recent) >= RECENT_LIMIT:
                self.recent.popitem(last=False)
            self.recent[key] = None
        cached = self.lookup(key)
        if cached is not None:
            return cached.selected
        scope = self.relaxation(key, history)
        if scope is not None:
            witness = self.solve(scope, relaxed=True)
            if witness.membership & key == 0:
                self.counts.relaxed_hits += 1
                return witness.selected
        return self.solve(key, relaxed=False).selected


class LatticePolicy(PublicPolicy):
    """Refresh evidence exactly as the native policy; optimize selection only."""

    def __init__(self, mode: str):
        super().__init__("selective_refresh")
        self.lattice_mode = mode
        self.planner: LatticePlanner | None = None

    def action(self, obs: dict[str, Any]) -> dict[str, Any]:
        if obs["event"] == "start":
            self.planner = LatticePlanner(prepare(obs["candidates"], obs["capacity"]), self.lattice_mode)
        return super().action(obs)

    def _submit(self) -> dict[str, Any]:
        if self.planner is None:
            raise ValueError("missing public start objective")
        selected = self.planner.choose({r["excluded"] for r in self.receipts.values()})
        return {
            "action": "submit",
            "frame_id": self.frame,
            "selected": list(selected),
            "receipts": sorted(r["receipt_id"] for r in self.receipts.values()),
        }


def factory(arm: str) -> PublicPolicy:
    return LatticePolicy(arm) if arm in MODES else previous_factory(arm)


def metrics(policy: PublicPolicy) -> dict[str, int]:
    if isinstance(policy, LatticePolicy) and policy.planner is not None:
        return asdict(policy.planner.counts)
    return previous_metrics(policy)


def stream(width: int, regime: str, i: int, kind: str, n: int) -> tuple[
    str, dict[str, Any], list[dict[str, Any]]
]:
    if width not in WIDTHS or regime not in ("tight", "medium") or (kind, n) not in WORKLOADS:
        raise ValueError("unregistered stream")
    root = digest(["distill22-lattice", ARRAY_SEED, width, regime, i])
    rng = random.Random(int(root, 16))
    candidates = [
        {"id": f"c{j:03}", "cost": rng.randint(1, 32), "utility": rng.randint(1, 100)}
        for j in range(width)
    ]
    order = [c["id"] for c in candidates]
    rng.shuffle(order)
    sources = [f"s{j}" for j in range(4 if kind.endswith("4") else 2)]
    active = dict(zip(sources, order[:len(sources)], strict=True))
    versions = dict.fromkeys(sources, 1)
    initial = {
        "event": "start", "candidates": candidates, "capacity": width * (2 if regime == "tight" else 8),
        "frame_id": root + "-start",
        "receipts": [
            {"source": name, "version": 1, "excluded": active[name], "receipt_id": root + "-" + name}
            for name in sources
        ],
    }
    revisions = []
    cursor = len(sources)
    for j in range(n):
        source = "s0"
        if kind == "churn" and (j + 1) % 4 == 0:
            source = "s1"
        elif kind == "rotate4":
            source = sources[j % len(sources)]
        elif kind == "overlap4" and (j + 1) % 4 == 0:
            source = sources[1 + (j // 4) % 3]
        if kind == "overlap4" and (j + 1) % 3 == 0:
            # Two sources can exclude the same candidate; their receipt
            # authority is distinct, but the numerical constraint is one.
            other = next(name for name in sources if name != source)
            excluded = active[other]
        else:
            while order[cursor % width] in active.values():
                cursor += 1
            excluded = order[cursor % width]
            cursor += 1
        active[source] = excluded
        versions[source] += 1
        revisions.append({
            "event": "receipt", "source": source, "version": versions[source], "excluded": excluded,
            "receipt_id": root + f"-{kind}-r{j}", "frame_id": root + f"-{kind}-f{j}",
        })
    return root, initial, revisions


def public_row(width: int, regime: str, i: int, kind: str, n: int) -> dict[str, Any]:
    root, initial, revisions = stream(width, regime, i, kind, n)
    outcomes = {}
    arms = {}
    for arm in rotated(i, ARMS):
        obs = json.loads(json.dumps(initial))
        observations = json.loads(json.dumps(revisions))
        start = time.process_time_ns()
        policy = factory(arm)
        policy.action(obs)
        setup = time.process_time_ns() - start
        elapsed = 0
        actions = []
        for observation in observations:
            start = time.process_time_ns()
            policy.frame = observation["frame_id"]
            action = policy.action(observation)
            elapsed += time.process_time_ns() - start
            actions.append(action)
        outcomes[arm] = actions
        arms[arm] = {"policy_cpu_ns": setup + elapsed, "setup_cpu_ns": setup,
                     "submit_cpu_ns": elapsed, "requests": n, **metrics(policy)}
    if any(a != outcomes["reference"] for a in outcomes.values()):
        raise RuntimeError("public action or evidence parity failed")
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
            raise RuntimeError("public feasibility or current binding failed")
    return {"root": root, "width": width, "regime": regime, "kind": kind, "requests": n,
            "arms": arms, "outcome_digest": digest(outcomes["reference"])}


def campaign(native_roots: int = 8, array_roots: int = 8) -> dict[str, Any]:
    if not 1 <= native_roots <= 32 or not 1 <= array_roots <= 32:
        raise ValueError("invalid root count")
    native = [
        native_row(split, seed, i, width, arm_names=ARMS, factory=factory, metrics=metrics)
        for split, seed in zip(SPLITS, SEEDS, strict=True)
        for width in NATIVE_WIDTHS for i in range(native_roots)
    ]
    public = [
        public_row(width, regime, i, kind, n)
        for width in WIDTHS for regime in ("tight", "medium")
        for kind, n in WORKLOADS for i in range(array_roots)
    ]
    return {
        "schema": "distill22-constraint-lattice-v1", "status": "development_only",
        "promotion_authorized": False, "training_admission": False, "provider_calls": 0,
        "memo_limit": MEMO_LIMIT, "recent_limit": RECENT_LIMIT,
        "native_roots_per_cell": native_roots, "public_roots_per_cell": array_roots,
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
        "rows_digest": digest([native, public]), "native_rows": native, "public_rows": public,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-roots", type=int, default=8)
    parser.add_argument("--array-roots", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = campaign(args.native_roots, args.array_roots)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(report, file, sort_keys=True, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ("native_rows", "public_rows")}, sort_keys=True))


if __name__ == "__main__":
    main()
