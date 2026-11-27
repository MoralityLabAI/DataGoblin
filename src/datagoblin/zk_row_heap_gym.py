"""ZKL-20: exact per-row priority heaps for the recurrent exploration gate."""
from __future__ import annotations

import argparse
import heapq
import json
import random
from collections import defaultdict
from pathlib import Path

from neighborhood_search import BN254_FR

from datagoblin.zk_advantage_gym import FAMILIES, SHIFT
from datagoblin.zk_counterfactual_gym import collect
from datagoblin.zk_exploration_gym import Hybrid, fit
from datagoblin.zk_exploration_gym import run as run_gate
from datagoblin.zk_indexed_frontier_gym import IndexedFrontier
from datagoblin.zk_neighborhood_gym import PROTOCOLS, full_reference

ROOT_BLOCKS = (20101, 20201)
ARMS = ("indexed", "row_random", "periodic", "random_gate",
        "scan_recurrent", "heap_recurrent", "scan_reset", "heap_reset")
ROW_FACTOR = 4


class RowFrontier(IndexedFrontier):
    """Share scores with the global heap. Row entries are invalidated by version."""

    def __init__(self, keys, supports, base, tie, initial):
        self.row_heaps = defaultdict(list)
        self.row_keys = defaultdict(list)
        for key in keys:
            self.row_keys[key[0]].append(key)
        self.dirty_rows = set()
        self.row_pushes = 0
        self.row_stale_pops = 0
        self.row_rebuilds = 0
        self.row_rebuild_scans = 0
        self.row_queries = 0
        self.row_peak = 0
        super().__init__(keys, supports, base, tie, initial)
        self.compact_rows()

    def entry(self, key):
        entry = super().entry(key)
        row = key[0]
        heapq.heappush(self.row_heaps[row], entry)
        self.row_pushes += 1
        self.dirty_rows.add(row)
        self.row_peak = max(self.row_peak, len(self.row_heaps[row]))
        return entry

    def valid(self, entry):
        _, _, key, version = entry
        return key in self.active and version == self.versions[key]

    def compact_rows(self):
        for row in sorted(self.dirty_rows):
            heap = self.row_heaps[row]
            if len(heap) > ROW_FACTOR*len(self.row_keys[row]):
                # Deduplicate existing live entries without evaluating any score again.
                self.row_rebuild_scans += len(heap)
                heap = list({entry for entry in heap if self.valid(entry)})
                heapq.heapify(heap)
                self.row_heaps[row] = heap
                self.row_rebuilds += 1
        self.dirty_rows.clear()

    def update(self, remaining, added_support):
        super().update(remaining, added_support)
        self.compact_rows()

    def row_max(self, row):
        self.row_queries += 1
        heap = self.row_heaps[row]
        while heap and not self.valid(heap[0]):
            heapq.heappop(heap)
            self.row_stale_pops += 1
        if not heap:
            raise AssertionError("selected nonempty row has no current heap entry")
        return heap[0][2]

    def row_metrics(self):
        return {
            "row_pushes": self.row_pushes, "row_stale_pops": self.row_stale_pops,
            "row_rebuilds": self.row_rebuilds, "row_rebuild_scans": self.row_rebuild_scans,
            "row_queries": self.row_queries, "row_peak": self.row_peak,
            "row_entries": sum(map(len, self.row_heaps.values())),
        }


class HeapHybrid(Hybrid):
    def __init__(self, problem, seed):
        super().__init__(problem, seed, index_factory=RowFrontier)

    def explore(self):
        rows = sorted(i for i, keys in self.pending.items() if keys)
        row = self.rng.choice(rows)
        key = self.index.row_max(row)
        if key not in self.pending[row]:
            raise AssertionError("row index returned unavailable candidate")
        return key


def train(roots):
    # Recreate the strongest unweighted sign model from ZKL-18/19.
    sequences, counts = collect(roots)
    model, linear, updates = fit(sequences)
    counts.update({"updates": updates, "recurrent_parameters": sum(v.size for v in model.values())})
    return model, linear, counts


def evaluate(p, family, seed, arm, budget, limit, model, linear):
    if arm.startswith(("scan_", "heap_")):
        kind, variant = arm.split("_", 1)
        factory = HeapHybrid if kind == "heap" else Hybrid
        return run_gate(p, family, seed, variant, budget, limit, model, linear,
                        hybrid_factory=factory, record_trace=True)
    return run_gate(p, family, seed, arm, budget, limit, model, linear)


def check_pair(a, b):
    for key in ("trace", "certificate", "columns", "input_terms", "reductions", "coefficient_work",
                "inversions", "prepared_columns", "prepared_terms", "gate_decisions", "explore_actions"):
        if a[key] != b[key]:
            raise AssertionError(f"row heap changed {key}")
    assert b["local_scores"] == 0
    assert b["row_index_metrics"]["row_queries"] == a["gate_decisions"]
    assert b["column_scores"] <= a["column_scores"]


def campaign(roots=8, test_start=20101, train_roots=8):
    if not 1 <= roots <= 16 or not 1 <= train_roots <= 16 or test_start not in ROOT_BLOCKS:
        raise ValueError("reserved root blocks and caps")
    model, linear, training = train(range(18001, 18001+train_roots))
    tasks = []
    for family in FAMILIES:
        for seed in range(test_start, test_start+roots):
            p = 11 if seed % 2 else BN254_FR
            essential = family.startswith("essential_")
            ref, ref_cols = full_reference(p, family, seed)
            assert (ref is not None) == (not essential)
            arms = {}
            order = list(ARMS)
            random.Random(seed).shuffle(order)
            for protocol, (budget, limit) in PROTOCOLS.items():
                recs = {arm: evaluate(p, family, seed, arm, budget, limit, model, linear)
                        for arm in order}
                if limit is None:
                    for variant in ("recurrent", "reset"):
                        check_pair(recs[f"scan_{variant}"], recs[f"heap_{variant}"])
                else:
                    for variant in ("recurrent", "reset"):
                        a, b = recs[f"scan_{variant}"]["trace"], recs[f"heap_{variant}"]["trace"]
                        length = min(len(a), len(b))
                        assert a[:length] == b[:length]
                for rec in recs.values():
                    assert rec["columns"] <= budget
                    assert rec["prepared_columns"] == rec["columns"]
                    assert not rec["lean_replayed"] and not rec["production_accepted"]
                    if essential:
                        assert not rec["exact_identity_checked"]
                    if limit is not None and rec["credited"]:
                        assert rec["cpu_ns"] <= limit
                arms[protocol] = recs
            tasks.append({"family": family, "seed": seed, "modulus": str(p),
                          "shift": family in SHIFT, "essential": essential,
                          "reference": ref, "reference_columns": ref_cols, "arms": arms})
    return {"schema": "zkl-row-heap-v1", "test_start": test_start, "training": training,
            "model": {k: v.tolist() for k, v in model.items()},
            "linear": linear.tolist(), "tasks": tasks, "lean_replayed": False, "production_accepted": False}


def summarize(result):
    fields = ("columns", "column_scores", "gate_decisions", "explore_actions",
              "coefficient_work", "setup_ns", "search_ns", "verify_ns", "cpu_ns")
    rows = []
    for protocol in PROTOCOLS:
        for arm in ARMS:
            recs = [t["arms"][protocol][arm] for t in result["tasks"]]
            rows.append({"protocol": protocol, "arm": arm,
                         "certified": sum(q["exact_identity_checked"] for q in recs),
                         "credited": sum(q["credited"] for q in recs),
                         "deadline_hits": sum(q["deadline_hit"] for q in recs),
                         "shift_credited": sum(t["arms"][protocol][arm]["credited"]
                                               for t in result["tasks"] if t["shift"]),
                         "local_scores": sum(q.get("local_scores", 0) for q in recs),
                         "row_pushes": sum(q.get("row_index_metrics", {}).get("row_pushes", 0) for q in recs),
                         "row_stale_pops": sum(q.get("row_index_metrics", {}).get("row_stale_pops", 0) for q in recs),
                         "row_rebuilds": sum(q.get("row_index_metrics", {}).get("row_rebuilds", 0) for q in recs),
                         "row_rebuild_scans": sum(q.get("row_index_metrics", {}).get("row_rebuild_scans", 0)
                                                  for q in recs),
                         "max_row_peak": max(q.get("row_index_metrics", {}).get("row_peak", 0) for q in recs),
                         **{key: sum(q[key] for q in recs) for key in fields}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--train-roots", type=int, default=8)
    parser.add_argument("--test-start", type=int, choices=ROOT_BLOCKS, default=ROOT_BLOCKS[0])
    args = parser.parse_args()
    result = campaign(args.roots, args.test_start, args.train_roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, sort_keys=True)
    print(json.dumps({"training": result["training"], "summary": summarize(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
