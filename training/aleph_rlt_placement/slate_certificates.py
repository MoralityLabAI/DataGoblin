"""Replay exact minimal-cost proof-slate comparisons on frozen public Lean grades."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from collections import defaultdict
from functools import cache
from pathlib import Path

from corpus import identity, load_corpus
from lean_acquisition import bucket


def outcome(a, b, positive, free):
    """Return the difference if every completion of the free bits agrees."""
    hit_a, hit_b = bool(a & positive), bool(b & positive)
    open_a = 0 if hit_a else a & free
    open_b = 0 if hit_b else b & free
    if not open_a and not open_b:
        return int(hit_a) - int(hit_b)
    if not hit_a and not hit_b and open_a == open_b:
        return 0
    return None


def census():
    checked = 0
    for a, b in itertools.product(range(8), repeat=2):
        for states in itertools.product(range(3), repeat=3):
            positive = sum(1 << i for i, v in enumerate(states) if v == 1)
            free = sum(1 << i for i, v in enumerate(states) if v == 2)
            possible = {
                int(bool(a & (positive | (s & free)))) - int(bool(b & (positive | (s & free)))) for s in range(8)
            }
            assert outcome(a, b, positive, free) == (next(iter(possible)) if len(possible) == 1 else None)
            checked += 1
    return checked


def run_goal(slate_a, slate_b, priors, oracle, seed):
    union = list(dict.fromkeys(slate_a + slate_b))
    n = len(union)
    full = (1 << n) - 1
    a, b = [sum(1 << union.index(x) for x in slate) for slate in (slate_a, slate_b)]
    p = [priors[x] for x in union]

    @cache
    def plan(positive, free, flat):
        if outcome(a, b, positive, free) is not None:
            return 0.0, None
        choices = []
        for i in range(n):
            bit = 1 << i
            if bit & free:
                tail = free ^ bit
                yes, _ = plan(positive | bit, tail, flat)
                no, _ = plan(positive, tail, flat)
                probability = 0.5 if flat else p[i]
                choices.append((1 + probability * yes + (1 - probability) * no, i))
        return min(choices)

    shuffle = list(range(n))
    random.Random(seed).shuffle(shuffle)
    common = sorted(range(n), key=lambda i: (-int(bool((a & b) & (1 << i))), -p[i], i))
    exclusive = sorted(range(n), key=lambda i: (int(bool((a & b) & (1 << i))), i))
    methods = {}
    for policy, order in (
        ("lexical_stop", range(n)),
        ("random_stop", shuffle),
        ("common_first", common),
        ("exclusive_first", exclusive),
        ("flat_prior", None),
        ("optimal_prior", None),
    ):
        positive, free = 0, full
        log = []
        while (d := outcome(a, b, positive, free)) is None:
            i = (
                plan(positive, free, policy == "flat_prior")[1]
                if order is None
                else next(j for j in order if (1 << j) & free)
            )
            grade = oracle(union[i])
            if grade not in (0, 1):
                raise ValueError("unavailable grade in selected public pilot; cannot certify with binary replay")
            log.append((union[i], grade))
            free ^= 1 << i
            positive |= grade << i
        methods[policy] = {"d": d, "checks": len(log), "receipt": log}

    gold = {x: oracle(x) for x in union}
    if any(value not in (0, 1) for value in gold.values()):
        raise ValueError("ungraded union in pilot")
    delta = int(any(gold[x] for x in slate_a)) - int(any(gold[x] for x in slate_b))
    if any(v["d"] != delta for v in methods.values()):
        raise AssertionError("invalid certificate")
    methods["union"] = {"d": delta, "checks": n}
    # Release the potentially thousands of dynamic programming states for this one goal.
    plan.cache_clear()
    return {"identical_sets": a == b, "lexical": slate_a, "candidate": slate_b, "methods": methods}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--acquisition", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    started = time.process_time()
    train, dev, source = load_corpus(args.pack)
    origin = json.loads(args.acquisition.read_text(encoding="utf-8"))
    if source != origin["source"]:
        raise ValueError("acquisition receipt belongs to different data")
    lookup = {(r.root, r.order): r for r in train["lean"]}
    by_root = defaultdict(list)
    for r in dev["lean"]:
        by_root[r.root].append(r)
    exact = census()
    results = {}
    for seed, trial in origin["results"]["target"].items():
        acquired = trial["96"]["query_ids"]
        if len(set(map(tuple, acquired))) != 96 or identity(acquired) != trial["96"]["query_digest"]:
            raise ValueError("inconsistent training evidence")
        counts = defaultdict(lambda: [1, 2])
        for root, rank in acquired:
            row = lookup[root, rank]
            if row.good:
                cell = counts[bucket(row)]
                cell[0] += row.good[0]
                cell[1] += 1
        goals = {}
        for root, rows in by_root.items():
            p = {r.order: counts[bucket(r)][0] / counts[bucket(r)][1] for r in rows}
            grades = {r.order: r.good[0] if r.good else None for r in rows}
            sa = sorted(p)[:4]
            sb = sorted(p, key=lambda i: (-p[i], i))[:4]
            goals[root] = run_goal(sa, sb, p, grades.__getitem__, f"{seed}-{root}")
        totals = {
            key: sum(g["methods"][key]["checks"] for g in goals.values())
            for key in next(iter(goals.values()))["methods"]
        }
        delta = sum(g["methods"]["union"]["d"] for g in goals.values())
        if delta != trial["0"]["proved"] - trial["96"]["proved"]:
            raise AssertionError("wrong student or slate")
        results[seed] = {
            "checks": totals,
            "net_lexical_minus_candidate": delta,
            "goals": goals,
            "identical_slates": sum(g["identical_sets"] for g in goals.values()),
        }
        print(seed, totals, "D", delta, "identical", results[seed]["identical_slates"])
    args.out.write_text(
        json.dumps(
            {
                "source": source,
                "scope": "public kernel receipt replay; counts debits to independent policies",
                "predicate_census": exact,
                "results": results,
                "cpu_seconds": time.process_time() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
