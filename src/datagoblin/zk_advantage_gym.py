"""ZKL-19: signed counterfactual advantage and the missing reset ablation."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from neighborhood_search import BN254_FR

from datagoblin.zk_counterfactual_gym import collect
from datagoblin.zk_exploration_gym import DIM, HIDDEN, fit, parameters
from datagoblin.zk_exploration_gym import run as run_gate
from datagoblin.zk_neighborhood_gym import PROTOCOLS, TEST, full_reference

SHIFT = ("inverse_product_transport", "idempotent_transport", "zero_pair_transport",
         "essential_inverse_transport")
FAMILIES = (*TEST, *SHIFT)
ROOT_BLOCKS = (19101, 19201)
ARMS = ("indexed", "row_random", "periodic", "random_gate",
        "sign_linear", "sign_recurrent", "sign_reset",
        "value_linear", "value_recurrent", "value_reset")


def value_step(model, x, h):
    new = np.tanh(model["wx"] @ x + model["wh"] @ h + model["bh"])
    return float(np.tanh(model["wy"] @ new + model["by"][0])), new


def huber_derivative(error, delta=0.25):
    return max(-delta, min(delta, float(error)))


def fit_value(sequences):
    """Regress normalized signed advantage. Observed ties are zero, unobserved steps masked."""
    model = parameters(170017)
    linear = np.zeros(DIM)
    rng = random.Random(170019)
    if not any(target is not None for seq in sequences for _, target in seq):
        raise ValueError("no observed counterfactual values")
    updates = 0
    for _ in range(16):
        order = list(range(len(sequences)))
        rng.shuffle(order)
        for idx in order:
            seq = sequences[idx]
            count = sum(target is not None for _, target in seq)
            if not count:
                continue
            states = [np.zeros(HIDDEN)]
            values = []
            gl = np.zeros_like(linear)
            for x, target in seq:
                value, h = value_step(model, x, states[-1])
                states.append(h)
                values.append(value)
                if target is not None:
                    pred = float(np.tanh(linear @ x))
                    gl += huber_derivative(pred-target)*(1-pred*pred)*x/count
            grad = {k: np.zeros_like(v) for k, v in model.items()}
            dh = np.zeros(HIDDEN)
            for t in reversed(range(len(seq))):
                x, target = seq[t]
                pred = values[t]
                err = (0.0 if target is None else
                       huber_derivative(pred-target)*(1-pred*pred)/count)
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
    # Identical observations and rollouts feed both objectives.
    sequences, counts = collect(roots, include_advantage=True)
    sign = [[(x, label) for x, label, _ in seq] for seq in sequences]
    value = [[(x, advantage) for x, _, advantage in seq] for seq in sequences]
    observed = [v for seq in value for _, v in seq if v is not None]
    if not observed or not all(-1 <= v <= 1 for v in observed):
        raise AssertionError("invalid normalized advantages")
    for seq in sequences:
        for _, label, v in seq:
            if v is not None:
                assert label == (int(v > 0) if v else None)
            else:
                assert label is None
    start = time.process_time_ns()
    sign_model, sign_linear, sign_updates = fit(sign)
    sign_ns = time.process_time_ns()-start
    start = time.process_time_ns()
    value_model, value_linear, value_updates = fit_value(value)
    counts.update({
        "observed_advantages": len(observed), "observed_ties": sum(v == 0 for v in observed),
        "positive_advantages": sum(v > 0 for v in observed),
        "negative_advantages": sum(v < 0 for v in observed),
        "mean_abs_advantage": sum(map(abs, observed))/len(observed),
        "sign_updates": sign_updates, "value_updates": value_updates,
        "sign_fit_ns": sign_ns, "value_fit_ns": time.process_time_ns()-start,
        "recurrent_parameters": sum(v.size for v in sign_model.values()),
    })
    return {"sign": (sign_model, sign_linear), "value": (value_model, value_linear)}, counts


def evaluate(p, family, seed, arm, budget, limit, models):
    if arm.startswith(("sign_", "value_")):
        objective, variant = arm.split("_", 1)
        model, linear = models[objective]
        aliases = {"linear": "linear_gate", "recurrent": "recurrent", "reset": "reset"}
        # sign(tanh(logit)) and sign(sigmoid(logit)-0.5) are identical.
        # Both objectives use the exact same runtime gate and threshold.
        return run_gate(p, family, seed, aliases[variant], budget, limit, model, linear)
    model, linear = models["sign"]
    return run_gate(p, family, seed, arm, budget, limit, model, linear)


def campaign(roots=8, test_start=19101, train_roots=8):
    if not 1 <= roots <= 16 or not 1 <= train_roots <= 16 or test_start not in ROOT_BLOCKS:
        raise ValueError("reserved root blocks and caps")
    # Reuse ZKL-18 training roots so its strongest unweighted policy is
    # reproduced on the same dataset, then evaluated at new disjoint roots.
    models, training = train(range(18001, 18001+train_roots))
    tasks = []
    for family in FAMILIES:
        for seed in range(test_start, test_start+roots):
            p = 11 if seed % 2 else BN254_FR
            essential = family.startswith("essential_")
            ref, ref_cols = full_reference(p, family, seed)
            assert (ref is not None) == (not essential)
            order = list(ARMS)
            random.Random(seed).shuffle(order)
            arms = {}
            for protocol, (budget, limit) in PROTOCOLS.items():
                records = {a: evaluate(p, family, seed, a, budget, limit, models) for a in order}
                for rec in records.values():
                    assert rec["columns"] <= budget and rec["prepared_columns"] == rec["columns"]
                    assert not rec["production_accepted"] and not rec["lean_replayed"]
                    if essential:
                        assert not rec["exact_identity_checked"]
                    if limit is not None and rec["credited"]:
                        assert rec["cpu_ns"] <= limit
                arms[protocol] = records
            tasks.append({"family": family, "seed": seed, "modulus": str(p),
                          "essential": essential, "shift": family in SHIFT,
                          "reference_columns": ref_cols, "reference": ref, "arms": arms})
    encoded = {k: {"recurrent": {n: v.tolist() for n, v in m.items()}, "linear": lin.tolist()}
               for k, (m, lin) in models.items()}
    return {"schema": "zkl-advantage-v1", "test_start": test_start, "training": training,
            "models": encoded, "tasks": tasks, "lean_replayed": False, "production_accepted": False}


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
                         "shift_credited": sum(t["arms"][protocol][arm]["credited"]
                                               for t in result["tasks"] if t["shift"]),
                         "deadline_hits": sum(q["deadline_hit"] for q in recs),
                         "by_family": {f: sum(t["arms"][protocol][arm]["credited"]
                                              for t in result["tasks"] if t["family"] == f) for f in FAMILIES},
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
