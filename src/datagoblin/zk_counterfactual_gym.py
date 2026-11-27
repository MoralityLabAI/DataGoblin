"""ZKL-18: paired counterfactual exploration teacher and recurrent ablation."""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

from neighborhood_search import BN254_FR

from datagoblin.zk_exploration_gym import Hybrid, fit
from datagoblin.zk_exploration_gym import run as run_gate
from datagoblin.zk_neighborhood_gym import (
    PROTOCOLS,
    TEST,
    fixture,
    full_reference,
    verified_certificate,
)

# One hard family is deliberately shared with test, at disjoint roots and fields.
# No topology-holdout claim is made for that family.
TRAIN = ("linear_chain3", "product_zero", "inverse_square", "square_transport", "product_chain_zero")
ROOT_BLOCKS = (18101, 18201)
TRAIN_LIMIT = 48
HORIZON = 24
ARMS = ("indexed", "row_random", "periodic", "random_gate",
        "cf_linear", "cf_recurrent", "cf_reset", "cf_unweighted")


def fork(policy):
    """Copy mutable state; immutable symbolic supports and basis vectors may be shared."""
    out = copy.copy(policy)
    out.rng = random.Random()
    out.rng.setstate(policy.rng.getstate())
    out.pending = {i: set(keys) for i, keys in policy.pending.items()}
    out.visits = dict(policy.visits)
    out.state = copy.copy(policy.state)
    out.state.basis = dict(policy.state.basis)
    out.state.remaining = dict(policy.state.remaining)
    out.state.weights = dict(policy.state.weights)
    out.state.offered = set(policy.state.offered)
    problem = copy.copy(policy.state.problem)
    problem.columns = copy.copy(problem.columns)
    problem.columns.cache = dict(problem.columns.cache)
    out.state.problem = problem
    out.index = copy.copy(policy.index)
    for name in ("active", "remaining", "frontier"):
        setattr(out.index, name, set(getattr(policy.index, name)))
    for name in ("rc", "fc", "versions"):
        setattr(out.index, name, dict(getattr(policy.index, name)))
    out.index.heap = list(policy.index.heap)
    return out


def paired_schedule(seed, horizon, repeat):
    """Fixed continuation flags shared by the two action branches."""
    if repeat == 0:
        return [j % 3 == 1 for j in range(horizon-1)]
    rng = random.Random(seed)
    return [rng.random() < 0.5 for _ in range(horizon-1)]


def rollout(policy, key, explore, flags, seed):
    """Independent fork; all successful teacher rewards require exact verification."""
    clone = fork(policy)
    clone.rng.seed(seed)
    before = clone.state.counts()
    clone.advance(key, explore, bool(flags))
    for j, action in enumerate(flags):
        if clone.state.solved or not clone.index.active:
            break
        candidate = clone.explore() if action else clone.exploit()
        clone.advance(candidate, action, j+1 < len(flags))
    cert, verify_ns = verified_certificate(clone.state.problem, clone.state)
    after = clone.state.counts()
    columns = after["columns"]-before["columns"]
    horizon = 1+len(flags)
    if not 1 <= columns <= horizon:
        raise AssertionError("invalid rollout accounting")
    # Success first, then fewer additional offered columns. Both failures score zero.
    return {
        "utility": horizon+1-columns if cert is not None else 0,
        "solved": cert is not None,
        "columns": columns,
        "work": after["coefficient_work"]-before["coefficient_work"],
        "verify_ns": verify_ns,
        "certificate_terms": 0 if cert is None else len(cert["multipliers"]),
    }


def counterfactual(policy, a, b, horizon, seed):
    if a == b:
        return None, []
    receipts = []
    for repeat in range(2):
        pair_seed = seed+repeat*100003
        flags = paired_schedule(pair_seed, horizon, repeat)
        left = rollout(policy, a, False, flags, pair_seed)
        right = rollout(policy, b, True, flags, pair_seed)
        receipts.append({"exploit": left, "explore": right})
    delta = sum(q["explore"]["utility"]-q["exploit"]["utility"] for q in receipts)
    return (int(delta > 0) if delta else None), receipts


def collect(roots, *, include_advantage=False):
    start = time.process_time_ns()
    sequences = []
    counts = {"reference_columns": 0, "trajectory_steps": 0, "teacher_solves": 0,
              "fork_pairs": 0, "fork_columns": 0, "fork_work": 0, "fork_verifier_calls": 0,
              "fork_verify_ns": 0, "exploit_labels": 0, "explore_labels": 0,
              "masked": 0, "explore_only_pairs": 0, "exploit_only_pairs": 0}
    for family in TRAIN:
        for root in roots:
            p = 7 if root % 2 else 11
            cert, cols = full_reference(p, family, root)
            assert cert is not None
            counts["reference_columns"] += cols
            for repeat in range(2):
                seed = root+100000*repeat
                behavior = random.Random(seed+180018)
                problem = fixture(p, family, root)
                policy = Hybrid(problem, seed)
                seq = []
                for t in range(min(TRAIN_LIMIT, len(problem.keys))):
                    if policy.state.solved:
                        break
                    a, b = policy.exploit(), policy.explore()
                    x = policy.features(a, b)
                    label, receipts = None, []
                    if t % 2 == 0:
                        horizon = min(HORIZON, TRAIN_LIMIT-t, len(policy.index.active))
                        label, receipts = counterfactual(policy, a, b, horizon, seed*101+t)
                    for pair in receipts:
                        counts["fork_pairs"] += 1
                        counts["explore_only_pairs"] += int(pair["explore"]["solved"] and not pair["exploit"]["solved"])
                        counts["exploit_only_pairs"] += int(pair["exploit"]["solved"] and not pair["explore"]["solved"])
                        for rec in pair.values():
                            counts["fork_columns"] += rec["columns"]
                            counts["fork_work"] += rec["work"]
                            counts["fork_verifier_calls"] += int(rec["solved"])
                            counts["fork_verify_ns"] += rec["verify_ns"]
                    counts["explore_labels"] += int(label == 1)
                    counts["exploit_labels"] += int(label == 0)
                    counts["masked"] += int(label is None)
                    if include_advantage:
                        advantage = (sum(q["explore"]["utility"]-q["exploit"]["utility"] for q in receipts)
                                     / (len(receipts)*horizon) if receipts else None)
                        seq.append((x, label, advantage))
                    else:
                        seq.append((x, label))
                    # The behavior trajectory is independent of the teacher decision.
                    action = behavior.random() < 0.5
                    policy.advance(b if action else a, action, t+1 < TRAIN_LIMIT)
                accepted, _ = verified_certificate(problem, policy.state)
                counts["teacher_solves"] += int(accepted is not None)
                counts["trajectory_steps"] += len(seq)
                sequences.append(seq)
    counts["sequences"] = len(sequences)
    counts["collection_cpu_ns"] = time.process_time_ns()-start
    return sequences, counts


def train(roots):
    sequences, counts = collect(roots)
    n0, n1 = counts["exploit_labels"], counts["explore_labels"]
    if not n0 or not n1:
        raise ValueError("counterfactual teacher missing a label class")
    weights = ((n0+n1)/(2*n0), (n0+n1)/(2*n1))
    start = time.process_time_ns()
    balanced, linear, updates = fit(sequences, weights)
    unweighted, _, unweighted_updates = fit(sequences)
    counts.update({"class_weights": list(weights), "updates": updates,
                   "unweighted_updates": unweighted_updates, "fit_cpu_ns": time.process_time_ns()-start,
                   "recurrent_parameters": sum(v.size for v in balanced.values()),
                   "linear_parameters": linear.size})
    return balanced, linear, unweighted, counts


def evaluate(p, family, seed, arm, budget, limit, balanced, linear, unweighted):
    aliases = {"cf_linear": "linear_gate", "cf_recurrent": "recurrent",
               "cf_reset": "reset", "cf_unweighted": "recurrent"}
    model = unweighted if arm == "cf_unweighted" else balanced
    return run_gate(p, family, seed, aliases.get(arm, arm), budget, limit, model, linear)


def campaign(roots=8, test_start=18101, train_roots=8):
    if not 1 <= roots <= 16 or not 1 <= train_roots <= 16 or test_start not in ROOT_BLOCKS:
        raise ValueError("reserved evaluation roots and caps")
    balanced, linear, unweighted, training = train(range(18001, 18001+train_roots))
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
                records = {a: evaluate(p, family, seed, a, budget, limit, balanced, linear, unweighted)
                           for a in order}
                for rec in records.values():
                    assert rec["columns"] <= budget and rec["prepared_columns"] == rec["columns"]
                    assert not rec["production_accepted"]
                    if family == "essential_chain":
                        assert not rec["exact_identity_checked"]
                    if limit is not None and rec["credited"]:
                        assert rec["cpu_ns"] <= limit
                arms[protocol] = records
            tasks.append({"family": family, "seed": seed, "modulus": str(p),
                          "reference": ref, "reference_columns": ref_cols, "arms": arms})
    return {"schema": "zkl-counterfactual-gate-v1", "test_start": test_start, "training": training,
            "balanced": {k: v.tolist() for k, v in balanced.items()},
            "unweighted": {k: v.tolist() for k, v in unweighted.items()}, "linear": linear.tolist(),
            "tasks": tasks, "lean_replayed": False, "production_accepted": False}


def summarize(result):
    fields = ("columns", "explore_actions", "gate_decisions", "column_scores", "row_scores",
              "coefficient_work", "verify_ns", "cpu_ns")
    rows = []
    for protocol in PROTOCOLS:
        for arm in ARMS:
            recs = [t["arms"][protocol][arm] for t in result["tasks"]]
            rows.append({"protocol": protocol, "arm": arm,
                         "certified": sum(q["exact_identity_checked"] for q in recs),
                         "credited": sum(q["credited"] for q in recs),
                         "deadline_hits": sum(q["deadline_hit"] for q in recs),
                         "by_family": {f: sum(t["arms"][protocol][arm]["credited"]
                                              for t in result["tasks"] if t["family"] == f) for f in TEST},
                         **{k: sum(q[k] for q in recs) for k in fields}})
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
