"""ZKL-16: CPU-bounded neighborhood policies over exact polynomial search."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from audit import canonical, check, digest
from neighborhood_search import BN254_FR, Problem, Search
from poly_certificate import verify

from datagoblin.zk_connected_search_gym import describe, overlap
from datagoblin.zk_indexed_frontier_gym import IndexedFrontier, score
from datagoblin.zk_search_gym import dot

TRAIN = ("linear_chain3", "product_zero", "inverse_square", "square_transport")
TEST = ("product_chain_zero", "inverse_chain", "coupled_sum", "essential_chain")
ARMS = ("indexed", "row_heuristic", "row_static", "row_feedback", "row_shuffled", "row_random")
ROOT_BLOCKS = (16101, 16201)
# The same deadline applies to all arms, including setup and verification.
PROTOCOLS = {"c24": (24, None), "c96": (96, None), "cpu10": (96, 10_000_000)}


def fixture(p, family, seed):
    rng = random.Random(seed)
    k, h = (rng.randrange(1, p) for _ in range(2))

    def lc(*terms):
        acc = {}
        for i, c in terms:
            acc[i] = (acc.get(i, 0) + c) % p
        return [[i, str(c)] for i, c in sorted(acc.items()) if c]

    one = lc((0, 1))
    x, y, z, u, t = (lc((i, 1)) for i in range(1, 6))

    def linear(*terms):
        return [lc(*terms), one, []]

    if family == "linear_chain3":
        rows = [linear((2, 1), (1, -k)), linear((3, 1), (2, -h)), linear((4, 1), (3, -1))]
        target = linear((4, 1), (1, -k*h))
    elif family == "product_zero":
        rows, target = [[x, one, []], [x, y, z]], [z, one, []]
    elif family == "inverse_square":
        rows = [[x, y, one], [x, z, one]]
        target = [lc((2, 1), (3, -1)), lc((2, k), (3, k)), []]
    elif family == "square_transport":
        rows, target = [[x, y, z], [x, x, []]], [z, lc((3, k)), []]
    elif family == "product_chain_zero":
        rows = [[x, y, z], [z, u, t], [x, one, []]]
        target = [t, lc((0, k)), []]
    elif family == "inverse_chain":
        rows = [[x, y, one], [y, z, one], [z, u, one]]
        target = [x, lc((4, k)), lc((0, k))]
    elif family == "coupled_sum":
        rows = [[x, y, z], [x, u, t]]
        target = [x, lc((2, k), (4, h)), lc((3, k), (5, h))]
    elif family == "essential_chain":
        rows, target = [[x, y, z], [z, u, t]], [t, lc((0, k)), []]
    else:
        raise ValueError(family)
    # A different connected cycle/chord topology, with two core variables per
    # distractor. All core assignments extend by setting nuisance wires to zero.
    for i in range(5):
        d = 6+i
        nxt = 6+(i+(1 if family in TRAIN else 2)) % 5
        ret = 6+(i+(4 if family in TRAIN else 1)) % 5
        c, e = 1+i, 1+(i+2) % 5
        rows.append([lc((d, 1)), lc((nxt, 1), (c, rng.randrange(1, p)),
                                    (e, rng.randrange(1, p))), lc((ret, 1))])
    perm = list(range(1, 11))
    rng.shuffle(perm)
    rename = [0, *perm]

    def transform(row):
        factor = rng.randrange(1, p)
        a, bb, cc = row
        scaled = (lc(*((i, factor*int(v)) for i, v in a)), bb,
                  lc(*((i, factor*int(v)) for i, v in cc)))
        return [sorted([[rename[i], v] for i, v in comb]) for comb in scaled]

    tagged = [(False, transform(row)) for row in rows] + [(True, transform(target))]
    rng.shuffle(tagged)
    idx = next(i for i, (flag, _) in enumerate(tagged) if flag)
    a = canonical({"format": "zkl-r1cs-v1", "modulus": str(p), "wires": 11,
                   "public": list(range(1, 11)), "rows": [row for _, row in tagged]})
    b = dict(a, rows=a["rows"][:idx] + a["rows"][idx+1:])
    if family == "essential_chain":
        w = [0]*11
        for i in range(6):
            w[rename[i]] = 1
        values = list(map(str, w))
        assert check(b, values, values[1:]) == []
        assert check(a, values, values[1:]) == [idx]
    return Problem(a, b, idx, (digest(a), digest(b)), eager=False)


def verified_certificate(problem, state):
    candidate = state.certificate()
    start = time.process_time_ns()
    if candidate is not None:
        verify(problem.original, problem.replacement, candidate, *problem.hashes)
    return candidate, time.process_time_ns()-start


class Neighborhood:
    """Rank retained-row neighborhoods, then scan only the chosen row's columns."""

    def __init__(self, problem, supports, base):
        self.supports, self.base = supports, base
        self.pending = {}
        self.raw = {}
        for key in problem.keys:
            self.pending.setdefault(key[0], set()).add(key)
        self.union = {i: set().union(*(supports[key] for key in keys))
                      for i, keys in self.pending.items()}
        self.raw = {i: set(supports[i, ()]) for i in self.pending}
        target = problem.target_support
        tw = {j for mon in target for j in mon}
        self.static = {}
        for i, keys in self.pending.items():
            raw = self.raw[i]
            wires = {j for mon in raw for j in mon}
            self.static[i] = (
                1.0, len(raw & target)/max(1, len(raw)),
                len(wires & tw)/max(1, len(wires)),
                len(self.union[i] & target)/max(1, len(target)),
                max(base[key] for key in keys)/8, min(len(raw), 8)/8, min(len(wires), 8)/8,
            )
        self.frontier = set(target)
        self.visits = dict.fromkeys(self.pending, 0)
        self.memory = dict.fromkeys(self.pending, 0.0)
        self.row_scores = 0
        self.local_scores = 0
        self.memory_updates = 0
        self.index_terms = sum(map(len, self.union.values()))

    def features(self, row, remaining):
        raw, union = self.raw[row], self.union[row]
        d = (
            len(raw & remaining)/max(1, len(raw)),
            len(union & remaining)/max(1, len(remaining)),
            len(raw & self.frontier)/max(1, len(raw)),
            len(union & self.frontier)/max(1, len(self.frontier)),
            min(self.visits[row], 8)/8, self.memory[row],
        )
        return self.static[row], d

    def choose(self, arm, remaining, tie, rng, weights):
        rows = sorted(i for i, keys in self.pending.items() if keys)
        if arm == "row_random":
            row = rng.choice(rows)
        else:
            vals = []
            for i in rows:
                s, d = self.features(i, remaining)
                if arm == "row_heuristic":
                    v = 2*s[1] + s[2] + s[3] + 2*d[0] + 3*d[1] + d[2] + d[3] - d[4] + d[5]
                elif arm == "row_static":
                    v = dot(weights["static"], s) - 0.5*d[4]
                else:
                    v = dot(weights["feedback"], (*s, *d))
                vals.append(v)
            self.row_scores += len(rows)
            if arm == "row_shuffled":
                rng.shuffle(vals)
            row = max(zip(vals, rows, strict=True))[1]
        keys = self.pending[row]
        self.local_scores += len(keys)

        def rank(key):
            support = self.supports[key]
            return (score(self.base[key], len(support & remaining),
                          len(support & self.frontier), len(support)), tie[key])

        return max(keys, key=rank)

    def observe(self, key, before, after):
        row = key[0]
        self.pending[row].remove(key)
        self.frontier.update(self.supports[key])
        self.visits[row] += 1
        # Weight-tied, explicit recurrent state; no gradient through elimination.
        gain = max(-1.0, min(1.0, (len(before)-len(after))/max(1, len(before))))
        self.memory[row] = 0.75*self.memory[row] + 0.25*gain
        self.memory_updates += 1


def prepare(problem, seed):
    supports, static = describe(problem)
    base = {key: overlap(static[key]) for key in problem.keys}
    rng = random.Random(seed)
    tie = {key: rng.random() for key in problem.keys}
    if len(set(tie.values())) != len(tie):
        raise ValueError("tie collision")
    return supports, base, tie, rng


def full_reference(p, family, seed):
    problem = fixture(p, family, seed)
    state = Search(problem)
    for key in problem.keys:
        if state.solved:
            break
        state.offer(key)
    cert, _ = verified_certificate(problem, state)
    return cert, state.counts()["columns"]


def train(roots):
    start = time.process_time_ns()
    rng = random.Random(160016)
    static_examples, feedback_examples = [], []
    refs, positives = 0, 0
    for family in TRAIN:
        for seed in roots:
            p = 7 if seed % 2 else 11
            cert, cols = full_reference(p, family, seed)
            if cert is None:
                raise AssertionError("training reference failed")
            refs += cols
            oracle = {(i, tuple(mon)) for i, mon, _ in cert["multipliers"]}
            positives += len(oracle)
            problem = fixture(p, family, seed)
            supports, base, tie, _ = prepare(problem, seed)
            policy = Neighborhood(problem, supports, base)
            state = Search(problem)
            todo = set(oracle)
            while todo and not state.solved:
                remaining = set(state.remaining)

                def teacher(key, rem=remaining, front=policy.frontier, sup=supports, bs=base, ts=tie):
                    s = sup[key]
                    return score(bs[key], len(s & rem), len(s & front), len(s)), ts[key]

                key = max(todo, key=teacher)
                relevant = {k[0] for k in todo}
                negative = [i for i, keys in policy.pending.items() if keys and i not in relevant]
                if negative:
                    for i, label in ((key[0], 1), (rng.choice(negative), 0)):
                        s, d = policy.features(i, remaining)
                        static_examples.append((s, label))
                        feedback_examples.append(((*s, *d), label))
                todo.remove(key)
                state.offer(key)
                policy.observe(key, remaining, set(state.remaining))
            if not state.solved:
                raise AssertionError("oracle support replay failed")

    def fit(examples):
        if not examples:
            raise ValueError("no training examples")
        w = [0.0]*len(examples[0][0])
        for _ in range(32):
            rng.shuffle(examples)
            for f, label in examples:
                pred = 1/(1+math.exp(-max(-30.0, min(30.0, dot(w, f)))))
                w = [v+0.08*((label-pred)*x-0.001*v) for v, x in zip(w, f, strict=True)]
        return w

    model = {"static": fit(static_examples), "feedback": fit(feedback_examples)}
    return model, {"reference_columns": refs, "oracle_terms": positives,
                   "examples": len(static_examples), "updates_per_ranker": 32*len(static_examples),
                   "cpu_ns": time.process_time_ns()-start}


def run(p, family, seed, arm, budget, cpu_limit, weights):
    start = time.process_time_ns()
    problem = fixture(p, family, seed)
    supports, base, tie, rng = prepare(problem, seed)
    state = Search(problem)
    index = None
    policy = None
    if arm == "indexed":
        index = IndexedFrontier(problem.keys, supports, base, tie, problem.target_support)
    else:
        policy = Neighborhood(problem, supports, base)
    setup_ns = time.process_time_ns()-start

    def expired():
        return cpu_limit is not None and time.process_time_ns()-start >= cpu_limit

    timed_out = False
    for step in range(min(budget, len(problem.keys))):
        if state.solved:
            break
        if expired():
            timed_out = True
            break
        before = set(state.remaining)
        key = index.take() if index is not None else policy.choose(arm, before, tie, rng, weights)
        if expired():
            timed_out = True
            break
        state.offer(key)
        if index is not None:
            if not state.solved and step+1 < min(budget, len(problem.keys)):
                index.update(state.remaining, supports[key])
        else:
            policy.observe(key, before, set(state.remaining))
    search_ns = time.process_time_ns()-start-setup_ns
    cert, verify_ns = verified_certificate(problem, state)
    elapsed = time.process_time_ns()-start
    late = cpu_limit is not None and elapsed > cpu_limit
    metrics = ({"column_scores": index.score_evals, "row_scores": 0,
                "postings": index.posting_visits, "heap_entries": index.peak_heap,
                "index_terms": index.index_entries, "memory_updates": 0}
               if index is not None else
               {"column_scores": policy.local_scores, "row_scores": policy.row_scores,
                "postings": 0, "heap_entries": 0, "index_terms": policy.index_terms,
                "memory_updates": policy.memory_updates})
    return {**state.counts(), **metrics, "setup_ns": setup_ns, "search_ns": search_ns,
            "verify_ns": verify_ns, "cpu_ns": elapsed,
            "support_terms": sum(map(len, supports.values())), "base_scores": len(problem.keys),
            "prepared_columns": problem.prepared_columns, "prepared_terms": problem.pool_terms,
            "certificate": cert, "exact_identity_checked": cert is not None,
            "credited": cert is not None and not late, "deadline_hit": timed_out or late,
            "lean_replayed": False, "production_accepted": False}


def campaign(roots=8, test_start=16101):
    if not 1 <= roots <= 16 or test_start not in ROOT_BLOCKS:
        raise ValueError("reserved root block or root cap")
    weights, fit = train(range(16001, 16001+roots))
    tasks = []
    for family in TEST:
        for seed in range(test_start, test_start+roots):
            p = 11 if seed % 2 else BN254_FR
            ref, ref_cols = full_reference(p, family, seed)
            assert (ref is not None) == (family != "essential_chain")
            order = list(ARMS)
            random.Random(seed).shuffle(order)
            arms = {}
            for protocol, (budget, limit) in PROTOCOLS.items():
                recs = {a: run(p, family, seed, a, budget, limit, weights) for a in order}
                for rec in recs.values():
                    assert rec["columns"] <= budget
                    assert rec["prepared_columns"] == rec["columns"]
                    if family == "essential_chain":
                        assert not rec["exact_identity_checked"]
                arms[protocol] = recs
            tasks.append({"family": family, "seed": seed, "modulus": str(p), "reference_columns": ref_cols,
                          "reference": ref, "arms": arms})
    return {"schema": "zkl-neighborhood-v1", "test_start": test_start, "tasks": tasks,
            "weights": weights, "fit": fit, "lean_replayed": False, "production_accepted": False}


def summarize(result):
    fields = ("columns", "column_scores", "row_scores", "postings", "index_terms", "coefficient_work",
              "prepared_terms", "support_terms", "setup_ns", "search_ns", "verify_ns", "cpu_ns")
    rows = []
    for protocol in PROTOCOLS:
        for arm in ARMS:
            recs = [t["arms"][protocol][arm] for t in result["tasks"]]
            rows.append({"protocol": protocol, "arm": arm,
                         "certified": sum(r["exact_identity_checked"] for r in recs),
                         "credited": sum(r["credited"] for r in recs),
                         "deadline_hits": sum(r["deadline_hit"] for r in recs),
                         "by_family": {f: sum(t["arms"][protocol][arm]["credited"]
                                              for t in result["tasks"] if t["family"] == f) for f in TEST},
                         **{k: sum(r[k] for r in recs) for k in fields}})
    return rows


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
    print(json.dumps({"fit": result["fit"], "summary": summarize(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
