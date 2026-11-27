"""ZKL-17: tiny trained recurrent gate for exploration on a shared exact basis."""
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from neighborhood_search import BN254_FR, Search

from datagoblin.zk_indexed_frontier_gym import IndexedFrontier, score
from datagoblin.zk_neighborhood_gym import (
    PROTOCOLS,
    TEST,
    TRAIN,
    fixture,
    full_reference,
    prepare,
    verified_certificate,
)
from datagoblin.zk_neighborhood_gym import run as legacy_run

ARMS = ("indexed", "row_random", "periodic", "stagnation", "random_gate",
        "linear_gate", "recurrent", "reset")
ROOT_BLOCKS = (17101, 17201)
DIM = 12
HIDDEN = 8
TRAIN_BUDGET = 48


class Hybrid:
    """Both actions use one search basis; exploration never restarts elimination."""

    def __init__(self, problem, seed):
        self.state = Search(problem)
        self.supports, self.base, self.tie, self.rng = prepare(problem, seed)
        self.index = IndexedFrontier(problem.keys, self.supports, self.base, self.tie,
                                     problem.target_support)
        self.pending = {}
        for key in problem.keys:
            self.pending.setdefault(key[0], set()).add(key)
        self.initial = max(1, len(problem.target_support))
        self.visits = dict.fromkeys(self.pending, 0)
        self.local_scores = 0
        self.explore_actions = 0
        self.gate_decisions = 0
        self.stall = 0
        self.last_gain = 0.0
        self.last_explore = False

    def exploit(self):
        while self.index.heap:
            _, _, key, version = self.index.heap[0]
            if key in self.index.active and version == self.index.versions[key]:
                return key
            heapq.heappop(self.index.heap)
            self.index.stale_pops += 1
        raise AssertionError("active index exhausted")

    def explore(self):
        rows = sorted(i for i, keys in self.pending.items() if keys)
        row = self.rng.choice(rows)
        keys = self.pending[row]
        self.local_scores += len(keys)

        def rank(key):
            return self.value(key), self.tie[key]

        return max(keys, key=rank)

    def value(self, key):
        return score(self.base[key], self.index.rc[key], self.index.fc[key],
                     len(self.supports[key]))

    def features(self, a, b):
        residual = self.index.remaining
        av, bv = self.value(a), self.value(b)
        return np.array((
            1.0,
            min(len(self.state.offered), 96)/96,
            min(len(residual)/self.initial, 4.0)/4,
            min(self.stall, 12)/12,
            self.last_gain,
            float(self.last_explore),
            len(self.supports[a] & residual)/max(1, len(self.supports[a])),
            len(self.supports[b] & residual)/max(1, len(self.supports[b])),
            max(-1.0, min(1.0, (bv-av)/8)),
            min(self.visits[a[0]], 12)/12,
            min(self.visits[b[0]], 12)/12,
            float(a[0] == b[0]),
        ), dtype=np.float64)

    def advance(self, key, explore, more=True):
        before = len(self.state.remaining)
        self.index.active.remove(key)
        self.pending[key[0]].remove(key)
        self.state.offer(key)
        after = len(self.state.remaining)
        self.last_gain = max(-1.0, min(1.0, (before-after)/max(1, before)))
        self.stall = 0 if after < before else self.stall+1
        self.last_explore = explore
        self.explore_actions += int(explore)
        self.visits[key[0]] += 1
        if more and not self.state.solved:
            self.index.update(self.state.remaining, self.supports[key])


def sigmoid(value):
    return 1/(1+math.exp(-max(-30.0, min(30.0, float(value)))))


def parameters(seed):
    rng = np.random.default_rng(seed)
    return {
        "wx": rng.normal(0, 0.2, (HIDDEN, DIM)),
        "wh": rng.normal(0, 0.15, (HIDDEN, HIDDEN)),
        "bh": np.zeros(HIDDEN),
        "wy": rng.normal(0, 0.2, HIDDEN),
        "by": np.zeros(1),
    }


def recurrent_step(model, x, h):
    new = np.tanh(model["wx"] @ x + model["wh"] @ h + model["bh"])
    return sigmoid(model["wy"] @ new + model["by"][0]), new


def fit(sequences):
    """Masked teacher-label BPTT. All search actions and verifiers stay external."""
    model = parameters(170017)
    linear = np.zeros(DIM)
    rng = random.Random(170019)
    labels = sum(label is not None for seq in sequences for _, label in seq)
    if not labels:
        raise ValueError("no unambiguous teacher labels")
    updates = 0
    for _ in range(16):
        order = list(range(len(sequences)))
        rng.shuffle(order)
        for idx in order:
            seq = sequences[idx]
            count = sum(label is not None for _, label in seq)
            if not count:
                continue
            states = [np.zeros(HIDDEN)]
            probs = []
            gl = np.zeros_like(linear)
            for x, label in seq:
                prob, h = recurrent_step(model, x, states[-1])
                states.append(h)
                probs.append(prob)
                if label is not None:
                    gl += (sigmoid(linear @ x)-label)*x/count
            grad = {k: np.zeros_like(v) for k, v in model.items()}
            dh = np.zeros(HIDDEN)
            for t in reversed(range(len(seq))):
                x, label = seq[t]
                err = 0.0 if label is None else (probs[t]-label)/count
                grad["wy"] += err*states[t+1]
                grad["by"][0] += err
                dz = (dh+err*model["wy"])*(1-states[t+1]**2)
                grad["wx"] += np.outer(dz, x)
                grad["wh"] += np.outer(dz, states[t])
                grad["bh"] += dz
                dh = model["wh"].T @ dz
            norm = math.sqrt(sum(float(np.sum(g*g)) for g in grad.values()))
            factor = min(1.0, 2.0/max(norm, 1e-12))
            for name in model:
                model[name] -= 0.06*(factor*grad[name]+0.0005*model[name])
            linear -= 0.06*(gl+0.0005*linear)
            updates += 1
    if any(not np.isfinite(v).all() for v in [*model.values(), linear]):
        raise AssertionError("nonfinite trained parameters")
    return model, linear, updates


def train(roots):
    start = time.process_time_ns()
    sequences = []
    refs = 0
    teacher_solves = 0
    explore_labels = 0
    exploit_labels = 0
    for family in TRAIN:
        for seed in roots:
            p = 7 if seed % 2 else 11
            cert, columns = full_reference(p, family, seed)
            if cert is None:
                raise AssertionError("reference failed")
            refs += columns
            oracle = {(i, tuple(mon)) for i, mon, _ in cert["multipliers"]}
            for repeat in range(2):
                problem = fixture(p, family, seed)
                policy = Hybrid(problem, seed+100000*repeat)
                seq = []
                for t in range(min(TRAIN_BUDGET, len(problem.keys))):
                    if policy.state.solved:
                        break
                    a, b = policy.exploit(), policy.explore()
                    x = policy.features(a, b)
                    # A known certificate is only one witness to redundancy.
                    # Ambiguous labels are masked, not treated as negatives.
                    label = int(b in oracle) if (a in oracle) != (b in oracle) else None
                    explore_labels += int(label == 1)
                    exploit_labels += int(label == 0)
                    action = bool(label) if label is not None else policy.rng.random() < 0.35
                    seq.append((x, label))
                    policy.advance(b if action else a, action, t+1 < TRAIN_BUDGET)
                produced, _ = verified_certificate(problem, policy.state)
                teacher_solves += int(produced is not None)
                sequences.append(seq)
    model, linear, updates = fit(sequences)
    return model, linear, {
        "sequences": len(sequences), "reference_columns": refs,
        "teacher_solves": teacher_solves, "explore_labels": explore_labels,
        "exploit_labels": exploit_labels, "trajectory_steps": sum(map(len, sequences)),
        "updates": updates, "recurrent_parameters": sum(v.size for v in model.values()),
        "linear_parameters": linear.size, "cpu_ns": time.process_time_ns()-start,
    }


def run(p, family, seed, arm, budget, cpu_limit, model, linear):
    if arm in ("indexed", "row_random"):
        rec = legacy_run(p, family, seed, arm, budget, cpu_limit, {})
        return {**rec, "gate_decisions": 0, "explore_actions": rec["columns"] if arm == "row_random" else 0}
    start = time.process_time_ns()
    problem = fixture(p, family, seed)
    policy = Hybrid(problem, seed)
    h = np.zeros(HIDDEN)
    setup_ns = time.process_time_ns()-start

    def expired():
        return cpu_limit is not None and time.process_time_ns()-start >= cpu_limit

    timed_out = False
    for t in range(min(budget, len(problem.keys))):
        if policy.state.solved:
            break
        if expired():
            timed_out = True
            break
        a = policy.exploit()
        if arm == "periodic":
            action = t % 3 == 2
        elif arm == "stagnation":
            action = policy.stall >= 2
        elif arm == "random_gate":
            action = policy.rng.random() < 0.5
        else:
            b = policy.explore()
            x = policy.features(a, b)
            if arm == "linear_gate":
                prob = sigmoid(linear @ x)
            elif arm == "recurrent":
                prob, h = recurrent_step(model, x, h)
            elif arm == "reset":
                prob, _ = recurrent_step(model, x, np.zeros(HIDDEN))
            else:
                raise ValueError(arm)
            action = prob >= 0.5
            policy.gate_decisions += 1
        if arm in ("periodic", "stagnation", "random_gate") and action:
            b = policy.explore()
        if expired():
            timed_out = True
            break
        policy.advance(b if action else a, action, t+1 < min(budget, len(problem.keys)))
    search_ns = time.process_time_ns()-start-setup_ns
    cert, verify_ns = verified_certificate(problem, policy.state)
    elapsed = time.process_time_ns()-start
    late = cpu_limit is not None and elapsed > cpu_limit
    return {**policy.state.counts(), "column_scores": policy.index.score_evals+policy.local_scores,
            "row_scores": 0, "postings": policy.index.posting_visits,
            "index_terms": policy.index.index_entries, "gate_decisions": policy.gate_decisions,
            "explore_actions": policy.explore_actions, "setup_ns": setup_ns,
            "search_ns": search_ns, "verify_ns": verify_ns, "cpu_ns": elapsed,
            "prepared_columns": problem.prepared_columns, "prepared_terms": problem.pool_terms,
            "certificate": cert, "exact_identity_checked": cert is not None,
            "credited": cert is not None and not late, "deadline_hit": timed_out or late,
            "lean_replayed": False, "production_accepted": False}


def campaign(roots=8, test_start=17101, train_roots=12):
    if not 1 <= roots <= 16 or not 1 <= train_roots <= 24 or test_start not in ROOT_BLOCKS:
        raise ValueError("reserved root block or root cap")
    model, linear, training = train(range(17001, 17001+train_roots))
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
                recs = {a: run(p, family, seed, a, budget, limit, model, linear) for a in order}
                for rec in recs.values():
                    assert rec["columns"] <= budget
                    assert rec["prepared_columns"] == rec["columns"]
                    if family == "essential_chain":
                        assert not rec["exact_identity_checked"]
                    if limit is not None and rec["credited"]:
                        assert rec["cpu_ns"] <= limit
                arms[protocol] = recs
            tasks.append({"family": family, "seed": seed, "modulus": str(p),
                          "reference_columns": ref_cols, "reference": ref, "arms": arms})
    return {"schema": "zkl-hybrid-exploration-v1", "test_start": test_start, "tasks": tasks,
            "training": training, "model": {k: v.tolist() for k, v in model.items()},
            "linear": linear.tolist(), "lean_replayed": False, "production_accepted": False}


def summarize(result):
    fields = ("columns", "column_scores", "row_scores", "gate_decisions", "explore_actions",
              "postings", "coefficient_work", "prepared_terms", "setup_ns", "search_ns",
              "verify_ns", "cpu_ns")
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
                         **{key: sum(r[key] for r in recs) for key in fields}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--train-roots", type=int, default=12)
    parser.add_argument("--test-start", type=int, choices=ROOT_BLOCKS, default=ROOT_BLOCKS[0])
    args = parser.parse_args()
    result = campaign(args.roots, args.test_start, args.train_roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, sort_keys=True)
    print(json.dumps({"training": result["training"], "summary": summarize(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
