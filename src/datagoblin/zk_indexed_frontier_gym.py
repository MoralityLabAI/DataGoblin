"""ZKL-15: exact incremental monomial-frontier ranking, CPU-only development."""
from __future__ import annotations

import argparse
import heapq
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from neighborhood_search import BN254_FR, Search
from poly_certificate import verify

from datagoblin.zk_connected_search_gym import TEST, describe, dynamic, fixture, overlap

BUDGETS = (24, 96)
ROOT_BLOCKS = (15101, 15201)
MAX_HEAP_FACTOR = 4


def score(base, residual_count, frontier_count, size):
    # Preserve the operation order of ZKL-14B (including float roundoff).
    den = max(1, size)
    return base + 4*(residual_count/den) + frontier_count/den - 0.5*((size-frontier_count)/den)


class IndexedFrontier:
    """An exact heap ranker over the same symbolic supports as the scan control."""

    def __init__(self, keys, supports, base, tie, initial):
        self.supports, self.base, self.tie = supports, base, tie
        self.remaining, self.frontier = set(initial), set(initial)
        self.active = set(keys)
        self.postings = defaultdict(list)
        self.rc, self.fc, self.versions = {}, {}, dict.fromkeys(keys, 0)
        self.heap = []
        self.posting_visits = 0
        self.score_evals = 0
        self.heap_pushes = 0
        self.stale_pops = 0
        self.heap_rebuilds = 0
        self.peak_heap = 0
        for key in keys:
            support = supports[key]
            for mon in support:
                self.postings[mon].append(key)
            self.rc[key] = len(support & self.remaining)
            self.fc[key] = len(support & self.frontier)
            self.heap.append(self.entry(key))
        heapq.heapify(self.heap)
        self.peak_heap = len(self.heap)
        self.index_entries = sum(map(len, self.postings.values()))

    def entry(self, key):
        self.score_evals += 1
        s = score(self.base[key], self.rc[key], self.fc[key], len(self.supports[key]))
        return (-s, -self.tie[key], key, self.versions[key])

    def take(self):
        while self.heap:
            _, _, key, version = heapq.heappop(self.heap)
            if key not in self.active or version != self.versions[key]:
                self.stale_pops += 1
                continue
            self.active.remove(key)
            return key
        raise AssertionError("heap exhausted with active candidates")

    def update(self, remaining, added_support):
        remaining = set(remaining)
        added_frontier = added_support - self.frontier
        touched = set()
        for changes, counts, delta in (
            (self.remaining - remaining, self.rc, -1),
            (remaining - self.remaining, self.rc, 1),
            (added_frontier, self.fc, 1),
        ):
            for mon in sorted(changes):
                for key in self.postings.get(mon, ()):
                    self.posting_visits += 1
                    if key in self.active:
                        counts[key] += delta
                        touched.add(key)
        self.remaining = remaining
        self.frontier.update(added_frontier)
        for key in sorted(touched):
            self.versions[key] += 1
            heapq.heappush(self.heap, self.entry(key))
            self.heap_pushes += 1
        self.peak_heap = max(self.peak_heap, len(self.heap))
        # Stale entries cannot grow indefinitely as the search proceeds.
        if len(self.heap) > MAX_HEAP_FACTOR*max(1, len(self.active)):
            self.heap = [self.entry(key) for key in sorted(self.active)]
            heapq.heapify(self.heap)
            self.heap_rebuilds += 1

    def metrics(self):
        return {name: getattr(self, name) for name in (
            "posting_visits", "score_evals", "heap_pushes", "stale_pops",
            "heap_rebuilds", "peak_heap", "index_entries",
        )}


def run(p, family, seed, budget, indexed):
    start = time.process_time_ns()
    problem, scans = fixture(p, family, seed)
    state = Search(problem)
    keys = list(problem.keys)
    rng = random.Random(seed)
    tie = {key: rng.random() for key in keys}
    # The previous scan used a unique seeded float tie breaker.
    if len(set(tie.values())) != len(keys):
        raise ValueError("tie collision outside paired protocol")
    supports, static = describe(problem)
    base = {key: overlap(static[key]) for key in keys}
    pending = set(keys)
    frontier = set(problem.target_support)
    visited = set()
    trace = []
    score_evals = 0
    index = IndexedFrontier(keys, supports, base, tie, frontier) if indexed else None
    for step in range(min(budget, len(keys))):
        if state.solved:
            break
        if index is not None:
            key = index.take()
        else:
            remaining = set(state.remaining)

            def rank(key, rem=remaining, front=frontier, rows=visited):
                d = dynamic(key, supports, rem, front, rows)
                return base[key] + 4*d[0] + d[1] - 0.5*d[2], tie[key]

            score_evals += len(pending)
            key = max(pending, key=rank)
            pending.remove(key)
        trace.append(key)
        state.offer(key)
        if index is not None:
            if not state.solved and step+1 < min(budget, len(keys)):
                index.update(state.remaining, supports[key])
        else:
            frontier.update(supports[key])
            visited.add(key[0])
    cert = state.certificate()
    if cert is not None:
        verify(problem.original, problem.replacement, cert, *problem.hashes)
    metrics = index.metrics() if index is not None else {
        "posting_visits": 0, "score_evals": score_evals, "heap_pushes": 0,
        "stale_pops": 0, "heap_rebuilds": 0, "peak_heap": 0, "index_entries": 0,
    }
    return {**state.counts(), **metrics, "base_score_evals": len(keys),
            "prepared_columns": problem.prepared_columns, "prepared_terms": problem.pool_terms,
            "support_terms": sum(map(len, supports.values())), "peel_scans": scans,
            "cpu_ns": time.process_time_ns()-start, "certificate": cert, "trace": trace,
            "exact_identity_checked": cert is not None, "lean_replayed": False, "production_accepted": False}


def campaign(roots=8, test_start=15101):
    if not 1 <= roots <= 16 or test_start not in ROOT_BLOCKS:
        raise ValueError("reserved root block or root cap")
    tasks = []
    for family in TEST:
        for seed in range(test_start, test_start+roots):
            p = 11 if seed % 2 else BN254_FR
            pairs = {}
            for budget in BUDGETS:
                # Alternate execution order deterministically to reduce ordering bias.
                order = (False, True) if seed % 2 else (True, False)
                records = {mode: run(p, family, seed, budget, mode) for mode in order}
                a, b = records[False], records[True]
                for key in ("trace", "certificate", "columns", "input_terms", "reductions", "coefficient_work",
                            "inversions", "prepared_columns", "prepared_terms", "support_terms"):
                    if a[key] != b[key]:
                        raise AssertionError(f"paired frontier changed {family}/{seed}/{budget}: {key}")
                assert a["exact_identity_checked"] == (family != "essential")
                assert a["columns"] <= budget and a["prepared_columns"] == a["columns"]
                pairs[str(budget)] = {"scan": a, "indexed": b}
            tasks.append({"family": family, "seed": seed, "modulus": str(p), "pairs": pairs})
    return {"schema": "zkl-indexed-frontier-v1", "test_start": test_start, "tasks": tasks,
            "lean_replayed": False, "production_accepted": False}


def summarize(result):
    fields = ("columns", "prepared_terms", "coefficient_work", "base_score_evals", "score_evals",
              "posting_visits", "heap_pushes", "stale_pops", "heap_rebuilds", "index_entries", "cpu_ns")
    return [
        {"budget": budget, "arm": arm, "certified": sum(r["exact_identity_checked"] for r in recs),
         "max_heap_entries": max(r["peak_heap"] for r in recs),
         **{key: sum(r[key] for r in recs) for key in fields}}
        for budget in BUDGETS for arm in ("scan", "indexed")
        for recs in [[t["pairs"][str(budget)][arm] for t in result["tasks"]]]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--test-start", type=int, choices=ROOT_BLOCKS, default=ROOT_BLOCKS[0])
    args = parser.parse_args()
    result = campaign(args.roots, args.test_start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, sort_keys=True)
    print(json.dumps({"summary": summarize(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
