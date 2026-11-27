"""ZKL-14B: connected nonlinear decoys, lazy columns, and feedback search."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from audit import canonical, check, digest
from neighborhood_search import BN254_FR, Problem, Search
from poly_certificate import verify

from datagoblin.zk_search_gym import dot, peel

TRAIN = ("linear_chain", "multiplicative_zero", "scaled_product", "inverse_square")
TEST = ("quadratic_sum", "inverse_transport", "square_transport", "essential")
ARMS = ("degree", "random", "peel", "overlap", "frontier", "learned", "feedback", "shuffled")
BUDGETS = (24, 96)


def fixture(p, family, seed):
    rng = random.Random(seed)
    k, h = (rng.randrange(1, p) for _ in range(2))

    def lc(*terms):
        acc = {}
        for i, c in terms:
            acc[i] = (acc.get(i, 0) + c) % p
        return [[i, str(c)] for i, c in sorted(acc.items()) if c]

    one = lc((0, 1))
    x, y, z = (lc((i, 1)) for i in (1, 2, 3))
    if family == "linear_chain":
        rows = [[lc((2, 1), (1, -k)), one, []], [lc((3, 1), (2, -h)), one, []]]
        target = [lc((3, 1), (1, -k*h)), one, []]
    elif family == "multiplicative_zero":
        rows, target = [[x, one, []]], [x, lc((2, k)), []]
    elif family == "scaled_product":
        rows = [[x, y, z]]
        target = [lc((2, k)), lc((1, h)), lc((3, k*h))]
    elif family == "inverse_square":
        rows = [[x, y, one], [x, z, one]]
        target = [lc((2, 1), (3, -1)), lc((2, k), (3, k)), []]
    elif family == "quadratic_sum":
        rows = [[x, y, z], [x, z, y]]
        target = [x, lc((2, k), (3, h)), lc((3, k), (2, h))]
    elif family == "inverse_transport":
        rows, target = [[x, y, one], [y, z, one]], [lc((1, k), (3, -k)), one, []]
    elif family == "square_transport":
        rows, target = [[x, y, z], [x, x, []]], [z, lc((3, k)), []]
    elif family == "essential":
        rows, target = [[x, y, z]], [z, lc((3, k)), []]
    else:
        raise ValueError(family)

    # Every core assignment extends to the decoy subnetwork by setting d_i=0.
    # Every d_i occurs nonlinearly, and every row joins the same core via c_i.
    # The test cycle has a different chord and linear-return topology.
    for i in range(6):
        d = 4 + i
        nxt = 4 + (i + (1 if family in TRAIN else 2)) % 6
        ret = 4 + (i + (5 if family in TRAIN else 1)) % 6
        c = 1 + i % 3
        rows.append([lc((d, 1)), lc((nxt, 1), (c, rng.randrange(1, p))), lc((ret, 1))])

    perm = list(range(1, 10))
    rng.shuffle(perm)
    renaming = [0, *perm]

    def rename(row):
        return [sorted([[renaming[i], c] for i, c in comb]) for comb in row]

    tagged = [(False, rename(row)) for row in rows] + [(True, rename(target))]
    rng.shuffle(tagged)
    idx = next(i for i, (flag, _) in enumerate(tagged) if flag)
    original = canonical({"format": "zkl-r1cs-v1", "modulus": str(p), "wires": 10,
                          "public": list(range(1, 10)), "rows": [row for _, row in tagged]})
    replacement = dict(original, rows=original["rows"][:idx] + original["rows"][idx+1:])
    # The essential fixture has a concrete assignment satisfying exactly the retained rows.
    if family == "essential":
        w = [0]*10
        for i in range(4):
            w[renaming[i]] = 1
        values = list(map(str, w))
        assert check(replacement, values, values[1:]) == []
        assert check(original, values, values[1:]) == [idx]
    problem = Problem(original, replacement, idx, (digest(original), digest(replacement)), eager=False)
    active, _, scans = peel(problem)
    if len(active) != len(problem.polys):
        raise AssertionError("connected nonlinear fixture was peelable")
    return problem, scans


def describe(problem):
    """Symbolic support only: no field-column materialization while scoring."""
    row_supports = [{m for m, c in poly.items() if c % problem.p} for poly in problem.polys]
    target = problem.target_support
    target_wires = {i for m in target for i in m}
    rw = [{i for m in support for i in m} for support in row_supports]
    freq = Counter(i for wires in rw for i in wires)
    supports, static = {}, {}
    for key in problem.keys:
        i, mon = key
        support = frozenset(tuple(sorted((*m, *mon))) for m in row_supports[i])
        supports[key] = support
        wires = {j for m in support for j in m}
        static[key] = (
            1.0, float(len(mon) == 0), float(len(mon) == 1), float(len(mon) == 2),
            len(support & target)/max(1, len(support)),
            len(rw[i] & target_wires)/max(1, len(rw[i])),
            len(wires & target_wires)/max(1, len(wires)),
            len(set(mon) & target_wires)/max(1, len(mon)),
            sum(freq[j] == 1 for j in rw[i])/max(1, len(rw[i])),
            min(len(support), 8)/8, min(len(rw[i]), 6)/6,
            sum(freq[j] for j in rw[i])/max(1, 16*len(rw[i])),
        )
    return supports, static


def dynamic(key, supports, remaining, frontier, visited_rows):
    s = supports[key]
    return (len(s & remaining)/max(1, len(s)), len(s & frontier)/max(1, len(s)),
            len(s - frontier)/max(1, len(s)), float(key[0] in visited_rows))


def overlap(f):
    return 3*f[4] + f[5] + f[6] + 0.5*f[7] - f[8] + 0.7*f[1] + 0.2*f[2]


def run(p, family, seed, arm, budget, weights=None):
    start = time.process_time_ns()
    problem, scans = fixture(p, family, seed)
    state = Search(problem)
    keys = list(problem.keys)
    rng = random.Random(seed)
    tie = {key: rng.random() for key in keys}
    supports, static = {}, {}
    if arm not in ("degree", "random", "peel"):
        supports, static = describe(problem)
    score_evals = 0
    if arm in ("degree", "peel"):
        scores = {key: -len(key[1]) for key in keys}
    elif arm == "random":
        scores = dict.fromkeys(keys, 0)
    elif arm in ("overlap", "frontier"):
        scores = {key: overlap(static[key]) for key in keys}
    elif arm in ("learned", "shuffled"):
        scores = {key: dot(weights["static"], static[key]) for key in keys}
        if arm == "shuffled":
            vals = list(scores.values())
            rng.shuffle(vals)
            scores = dict(zip(keys, vals, strict=True))
    elif arm == "feedback":
        scores = {}
    else:
        raise ValueError(arm)
    if arm not in ("degree", "random", "peel", "feedback"):
        score_evals += len(keys)

    if arm in ("frontier", "feedback"):
        pending = set(keys)
        frontier = set(problem.target_support)
        visited_rows = set()
        for _ in range(min(budget, len(keys))):
            if state.solved:
                break
            remaining = set(state.remaining)

            def rank(key, rem=remaining, active=frontier, visited=visited_rows):
                d = dynamic(key, supports, rem, active, visited)
                value = (scores[key] + 4*d[0] + d[1] - 0.5*d[2]
                         if arm == "frontier" else dot(weights["feedback"], (*static[key], *d)))
                return value, tie[key]

            score_evals += len(pending)
            key = max(pending, key=rank)
            pending.remove(key)
            state.offer(key)
            frontier.update(supports[key])
            visited_rows.add(key[0])
    else:
        ordered = sorted(keys, key=lambda key: (scores[key], tie[key]), reverse=True)
        for key in ordered[:budget]:
            if state.solved:
                break
            state.offer(key)

    cert = state.certificate()
    if cert is not None:
        verify(problem.original, problem.replacement, cert, *problem.hashes)
    return {**state.counts(), "prepared_columns": problem.prepared_columns,
            "prepared_terms": problem.pool_terms, "pool_columns": len(keys),
            "support_terms": sum(map(len, supports.values())),
            "score_evals": score_evals, "peel_scans": scans,
            "cpu_ns": time.process_time_ns()-start, "certificate": cert,
            "exact_identity_checked": cert is not None, "lean_replayed": False,
            "production_accepted": False}


def train(roots):
    rng = random.Random(140014)
    static_examples, feedback_examples = [], []
    references, positives = 0, 0
    start = time.process_time_ns()
    for family in TRAIN:
        for seed in roots:
            p = 7 if seed % 2 else 11
            problem, _ = fixture(p, family, seed)
            ref = run(p, family, seed, "degree", len(problem.keys))
            if not ref["exact_identity_checked"]:
                raise AssertionError("training reference failed")
            references += ref["columns"]
            oracle = {(i, tuple(mon)) for i, mon, _ in ref["certificate"]["multipliers"]}
            positives += len(oracle)
            supports, static = describe(problem)
            negative = [key for key in problem.keys if key not in oracle]
            for key in sorted(oracle):
                static_examples.append((static[key], 1))
                static_examples.append((static[rng.choice(negative)], 0))
            state = Search(problem)
            todo = set(oracle)
            frontier = set(problem.target_support)
            visited = set()
            while todo and not state.solved:
                remaining = set(state.remaining)

                def teacher(key, rem=remaining, active=frontier, rows=visited):
                    d = dynamic(key, supports, rem, active, rows)
                    return (4*d[0] + d[1] - d[2], key)

                key = max(todo, key=teacher)
                todo.remove(key)
                for candidate, label in ((key, 1), (rng.choice(negative), 0)):
                    d = dynamic(candidate, supports, remaining, frontier, visited)
                    feedback_examples.append(((*static[candidate], *d), label))
                state.offer(key)
                frontier.update(supports[key])
                visited.add(key[0])
            if not state.solved:
                raise AssertionError("reference certificate support failed replay")

    def fit(examples):
        weights = [0.0]*len(examples[0][0])
        for _ in range(32):
            rng.shuffle(examples)
            for f, label in examples:
                score = max(-30.0, min(30.0, dot(weights, f)))
                err = label - 1/(1+math.exp(-score))
                weights = [w + 0.08*(err*v - 0.001*w) for w, v in zip(weights, f, strict=True)]
        return weights

    model = {"static": fit(static_examples), "feedback": fit(feedback_examples)}
    return model, {"reference_columns": references, "positive_labels": positives,
                   "static_updates": 32*len(static_examples), "feedback_updates": 32*len(feedback_examples),
                   "cpu_ns": time.process_time_ns()-start}


def campaign(roots=8, test_start=14101):
    if not 1 <= roots <= 16 or test_start not in (14101, 14201):
        raise ValueError("reserved root block or root cap")
    weights, fit = train(range(14001, 14001+roots))
    tasks = []
    for family in TEST:
        for seed in range(test_start, test_start+roots):
            p = 11 if seed % 2 else BN254_FR
            problem, _ = fixture(p, family, seed)
            full = run(p, family, seed, "degree", len(problem.keys))
            assert full["exact_identity_checked"] == (family != "essential")
            arms = {str(b): {a: run(p, family, seed, a, b, weights) for a in ARMS} for b in BUDGETS}
            for budget, recs in arms.items():
                for rec in recs.values():
                    assert rec["columns"] <= int(budget)
                    assert rec["prepared_columns"] == rec["columns"]
                    if family == "essential":
                        assert not rec["exact_identity_checked"]
            assert arms["24"]["degree"]["certificate"] == arms["24"]["peel"]["certificate"]
            assert arms["96"]["degree"]["certificate"] == arms["96"]["peel"]["certificate"]
            tasks.append({"seed": seed, "family": family, "modulus": str(p), "full": full, "arms": arms})
    return {"schema": "zkl-connected-search-v1", "test_start": test_start, "tasks": tasks,
            "weights": weights, "fit": fit, "lean_replayed": False, "production_accepted": False}


def summarize(result):
    metrics = ("columns", "prepared_terms", "coefficient_work", "score_evals", "support_terms", "cpu_ns")
    out = []
    for b in BUDGETS:
        for a in ARMS:
            recs = [t["arms"][str(b)][a] for t in result["tasks"]]
            by_family = {f: sum(t["arms"][str(b)][a]["exact_identity_checked"]
                                for t in result["tasks"] if t["family"] == f) for f in TEST}
            out.append({"budget": b, "arm": a, "certified": sum(by_family.values()), "by_family": by_family,
                        **{k: sum(r[k] for r in recs) for k in metrics}})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--test-start", type=int, choices=(14101, 14201), default=14101)
    args = parser.parse_args()
    result = campaign(args.roots, args.test_start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, sort_keys=True)
    print(json.dumps({"fit": result["fit"], "summary": summarize(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
