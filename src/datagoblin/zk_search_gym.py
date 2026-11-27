"""ZKL-13: CPU ranker pilot, paired column budgets, independent identity checking."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from audit import canonical, check, digest
from neighborhood_search import BN254_FR, Problem, Search
from poly_certificate import verify

TRAIN = ("linear_chain", "multiplicative_zero", "scaled_product")
TEST = ("quadratic_sum", "inverse_transport", "square_transport", "essential")
ARMS = ("degree", "random", "heuristic", "adaptive_heuristic", "learned", "adaptive_learned",
        "shuffled", "peel_local")
BUDGETS = (24, 48, 96)


def fixture(p, family, seed):
    rng = random.Random(seed)
    k, scale2 = (rng.randrange(1, p) for _ in range(2))

    def lc(*terms):
        acc = {}
        for i, c in terms:
            acc[i] = (acc.get(i, 0) + c) % p
        return [[i, str(c)] for i, c in sorted(acc.items()) if c]

    one = lc((0, 1))
    x, y, z = (lc((i, 1)) for i in (1, 2, 3))
    if family == "linear_chain":
        rows = [[lc((2, 1), (1, -k)), one, []], [lc((3, 1), (2, -scale2)), one, []]]
        target = [lc((3, 1), (1, -k*scale2)), one, []]
    elif family == "multiplicative_zero":
        rows, target = [[x, one, []]], [x, lc((2, k)), []]
    elif family == "scaled_product":
        rows, target = [[x, y, z]], [lc((2, k)), lc((1, scale2)), lc((3, k*scale2))]
    elif family == "quadratic_sum":
        rows = [[x, y, z], [x, z, y]]
        target = [x, lc((2, k), (3, scale2)), lc((3, k), (2, scale2))]
    elif family == "inverse_transport":
        rows, target = [[x, y, one], [y, z, one]], [lc((1, k), (3, -k)), one, []]
    elif family == "square_transport":
        rows, target = [[x, y, z], [x, x, []]], [z, lc((3, k)), []]
    elif family == "essential":
        rows, target = [[x, y, z]], [z, lc((3, k)), []]
    else:
        raise ValueError(family)
    # Each fresh decoy wire is determined by earlier wires, so the decoys impose
    # no restriction on the core. They share core variables and some monomials.
    for i in range(4, 10):
        a = rng.randrange(1, 4)
        b = rng.randrange(1, i)
        rows.append([lc((a, rng.randrange(1, p))), lc((b, 1)), lc((i, 1))])
    witness = None
    if family == "essential":
        witness = [1]*4
        for a, b, _ in rows[1:]:
            witness.append(int(a[0][1])*witness[a[0][0]]*witness[b[0][0]] % p)
    perm = list(range(1, 10))
    rng.shuffle(perm)
    renaming = [0, *perm]

    def rename(row):
        return [sorted([[renaming[i], c] for i, c in comb]) for comb in row]

    tagged = [(False, rename(row)) for row in rows]
    tagged.append((True, rename(target)))
    rng.shuffle(tagged)
    idx = next(i for i, (is_target, _) in enumerate(tagged) if is_target)
    original = canonical({"format": "zkl-r1cs-v1", "modulus": str(p), "wires": 10,
                          "public": list(range(1, 10)), "rows": [row for _, row in tagged]})
    replacement = dict(original, rows=original["rows"][:idx] + original["rows"][idx+1:])
    if witness is not None:
        renamed = [0]*10
        for i, value in enumerate(witness):
            renamed[renaming[i]] = value
        w = list(map(str, renamed))
        assert check(replacement, w, w[1:]) == [] and check(original, w, w[1:]) == [idx]
    return Problem(original, replacement, idx, (digest(original), digest(replacement)))


def features(problem):
    target = problem.target_support
    tw = {i for mon in target for i in mon}
    row_wires = [{i for mon in poly for i in mon} for poly in problem.polys]
    freq = Counter(i for wires in row_wires for i in wires)
    out = {}
    for key, col in problem.columns.items():
        i, mon = key
        support = set(col)
        rw = row_wires[i]
        cw = {j for m in support for j in m}
        out[key] = (1.0, float(len(mon) == 0), float(len(mon) == 1), float(len(mon) == 2),
                    len(support & target)/max(1, len(support)), len(rw & tw)/max(1, len(rw)),
                    len(cw & tw)/max(1, len(cw)), len(set(mon) & tw)/max(1, len(mon)),
                    sum(freq[j] == 1 for j in rw)/max(1, len(rw)),
                    sum(freq[j] == 1 for j in set(mon))/max(1, len(set(mon))),
                    min(len(support), 8)/8, min(len(rw), 6)/6)
    return out


def dot(weights, values):
    return sum(w*v for w, v in zip(weights, values, strict=True))


def heuristic(f):
    return 3*f[4] + f[5] + f[6] + 0.5*f[7] - f[8] - f[9] + 0.7*f[1] + 0.2*f[2]


def peel(problem):
    """Search heuristic: remove rows defining an unshared linear nuisance wire.

    Only changes the column order, never the certified replacement matrix.
    """
    active = set(range(len(problem.polys)))
    target_wires = {i for mon in problem.target_support for i in mon}
    row_wires = [{i for mon in poly for i in mon} for poly in problem.polys]
    scans = 0
    while True:
        freq = Counter(i for r in active for i in row_wires[r])
        drop = set()
        for r in sorted(active):
            scans += 1
            poly = problem.polys[r]
            for i in row_wires[r] - target_wires:
                if (freq[i] == 1 and poly.get((i,), 0) % problem.p
                        and all(i not in mon or mon == (i,) for mon in poly)):
                    drop.add(r)
                    break
        if not drop:
            break
        active -= drop
    core = target_wires | {i for r in active for i in row_wires[r]}
    return active, core, scans


def run(problem, arm, budget, weights=None, seed=0):
    rng = random.Random(seed)
    state = Search(problem)
    keys = list(problem.keys)
    tie = {key: rng.random() for key in keys}
    fs = features(problem) if arm not in ("degree", "random", "peel_local") else {}
    graph_scans = 0
    if arm == "peel_local":
        active, core, graph_scans = peel(problem)
    if arm in ("heuristic", "adaptive_heuristic"):
        scores = {key: heuristic(fs[key]) for key in keys}
    elif arm in ("learned", "adaptive_learned", "shuffled"):
        scores = {key: dot(weights, fs[key]) for key in keys}
        if arm == "shuffled":
            vals = list(scores.values())
            rng.shuffle(vals)
            scores = dict(zip(keys, vals, strict=True))
    elif arm == "peel_local":
        scores = {key: 10*int(key[0] in active and set(key[1]) <= core) - len(key[1]) for key in keys}
    else:
        scores = {key: -len(key[1]) if arm == "degree" else 0.0 for key in keys}
    score_evals = len(keys) if fs or arm == "peel_local" else 0
    adaptive = arm in ("adaptive_heuristic", "adaptive_learned")
    if adaptive:
        pending = set(keys)
        for _ in range(min(budget, len(keys))):
            if state.solved:
                break
            remaining = set(state.remaining)

            def rank(key, remaining=remaining):
                support = problem.columns[key].keys()
                overlap = len(support & remaining)/max(1, len(support))
                return scores[key] + 3*overlap, tie[key]

            score_evals += len(pending)
            key = max(pending, key=rank)
            pending.remove(key)
            state.offer(key)
    else:
        ordered = sorted(keys, key=lambda key: (scores[key], tie[key]), reverse=True)
        for key in ordered[:budget]:
            if state.solved:
                break
            state.offer(key)
    cert = state.certificate()
    if cert is not None:
        verify(problem.original, problem.replacement, cert, *problem.hashes)
    return {**state.counts(), "pool_columns": len(keys), "pool_terms": problem.pool_terms,
            "feature_evals": len(fs), "score_evals": score_evals, "graph_row_scans": graph_scans, "certificate": cert,
            "exact_identity_checked": cert is not None, "lean_replayed": False,
            "production_accepted": False}


def train(roots):
    positives, negatives, costs = [], [], []
    for family in TRAIN:
        for seed in roots:
            p = 7 if seed % 2 else 11
            problem = fixture(p, family, seed)
            receipt = run(problem, "degree", len(problem.keys), seed=seed)
            cert = receipt["certificate"]
            if cert is None:
                raise AssertionError("reference failed training fixture")
            costs.append(receipt["columns"])
            labels = {(i, tuple(mon)) for i, mon, _ in cert["multipliers"]}
            for key, f in features(problem).items():
                (positives if key in labels else negatives).append(f)
    rng = random.Random(130013)
    weights = [0.0]*len(positives[0])
    # Balanced sampled SGD; the reference certificate support is a nonunique label.
    for _ in range(24):
        batch = [(f, 1) for f in positives] + [(rng.choice(negatives), 0) for _ in positives]
        rng.shuffle(batch)
        for f, label in batch:
            score = max(-30.0, min(30.0, dot(weights, f)))
            err = label - 1/(1+math.exp(-score))
            weights = [w + 0.08*(err*v - 0.001*w) for w, v in zip(weights, f, strict=True)]
    return weights, {"positive_labels": len(positives), "negative_labels": len(negatives),
                     "training_reference_columns": sum(costs), "sgd_updates": 48*len(positives)}


def campaign(train_roots=8, test_roots=8, test_start=13101):
    if not 1 <= train_roots <= 32 or not 1 <= test_roots <= 32:
        raise ValueError("root cap")
    if test_start not in (13101, 13201):
        raise ValueError("use a reserved evaluation root block")
    weights, fit = train(range(13001, 13001+train_roots))
    tasks = []
    for family in TEST:
        for seed in range(test_start, test_start+test_roots):
            p = 11 if seed % 2 else BN254_FR
            problem = fixture(p, family, seed)
            full = run(problem, "degree", len(problem.keys), seed=seed)
            if full["exact_identity_checked"] != (family != "essential"):
                raise AssertionError("full bounded reference disagrees with fixture")
            arms = {str(b): {arm: run(problem, arm, b, weights, seed) for arm in ARMS} for b in BUDGETS}
            tasks.append({"family": family, "seed": seed, "modulus": str(p), "full": full, "arms": arms})
    return {"schema": "zkl-search-gym-v1", "train_roots": train_roots, "test_roots": test_roots,
            "test_start": test_start, "weights": weights, "fit": fit, "tasks": tasks, "lean_replayed": False,
            "production_accepted": False}


def summarize(result):
    rows = []
    for budget in BUDGETS:
        for arm in ARMS:
            recs = [t["arms"][str(budget)][arm] for t in result["tasks"]]
            rows.append({"budget": budget, "arm": arm,
                         "certified": sum(r["exact_identity_checked"] for r in recs),
                         "columns": sum(r["columns"] for r in recs),
                         "work": sum(r["coefficient_work"] for r in recs),
                         "score_evals": sum(r["score_evals"] for r in recs),
                         "graph_row_scans": sum(r["graph_row_scans"] for r in recs)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-roots", type=int, default=8)
    parser.add_argument("--test-roots", type=int, default=8)
    parser.add_argument("--test-start", type=int, choices=(13101, 13201), default=13101)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = campaign(args.train_roots, args.test_roots, args.test_start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True)
    print(json.dumps({"fit": result["fit"], "summary": summarize(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
