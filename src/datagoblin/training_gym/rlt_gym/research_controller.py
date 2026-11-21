"""Fit a small recurrent research correction controller and run paired fork trials."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .hypergame_taste import _file_sha256, _json
from .research_campaign import _write_json, case, source_hashes
from .research_world import DEPTHS, FAMILIES, Game

POLICIES = (
    "fixed_depth", "greedy", "heuristic_repair", "learned_order", "check_every_visit",
    "learned_deferral", "first_valid", "matched_damping", "random", "rlt",
)
CHECK_BUDGET, VISIT_BUDGET = 3, 4
REGRET_LIMIT = 0.5
HIDDEN = 12
START_NATIVE = {"heuristic_repair", "check_every_visit", "learned_deferral", "matched_damping", "rlt"}


def abstract(game):
    """Public evidence only. No lookahead values or evaluator state enter the controller."""
    state = game.initial
    public = game.public
    actions = tuple(game.legal(state))
    masses = np.array(game.initial_masses, dtype=np.float64)
    p = masses / masses.sum()
    correct, wrong = game.rewards(state.turns)
    next_correct, next_wrong = game.rewards(state.turns - 1)
    present = max(0, max(p * correct + (1 - p) * wrong))
    route = FAMILIES.index(public["family"])
    features, heuristic = [], []
    for name in actions:
        publishes = name.startswith("publish:")
        experiment = game.experiments.get(name)
        mode = int(name[-1]) if publishes else public["extension"]["mode"]
        success = p[mode]
        terminal_value = (success * correct + (1 - success) * wrong) if publishes else 0
        cost, single_probe, information = 0, 0, 0
        if experiment:
            cost = experiment["cost"]
            signal = np.array(experiment["p16"]) / 16
            prediction = float(p @ signal)
            for weights in (p * signal, p * (1 - signal)):
                payoff = weights * next_correct + (weights.sum() - weights) * next_wrong
                single_probe += max(0, float(payoff.max()))
            single_probe -= cost
            information = float(p @ (signal - prediction) ** 2)
        heuristic.append(single_probe if experiment else terminal_value)
        features.append([
            1, publishes, bool(experiment), name == "abstain", name == "pilot", name == "replicate",
            name not in game.native, success, terminal_value / 20, cost / 4, single_probe / 10,
            (single_probe - present) / 10 if experiment else 0, information * 4, max(p),
            public["K"] / 16, state.turns / 4, state.credits / 4, public["goal_change"] / 2,
            next_correct / 10, next_wrong / 20,
            *(float(route == i) for i in range(len(FAMILIES))),
        ])
    return actions, np.array(features), np.array(heuristic)


def verified(values, action):
    value = values[action]
    return value >= 0 and max(values.values()) - value <= REGRET_LIMIT


def training_data(key, per_cell):
    examples = []
    for depth in DEPTHS:
        for ordinal in range(per_cell):
            # Namespace and root custody are independent from the exported campaign.
            task, _ = case("train", depth, ordinal, 0, hashlib.sha256(key + b"/fit").digest())
            game = Game(task["public"])
            actions, features, _ = abstract(game)
            values = game.values(game.initial)
            failed = [i for i, name in enumerate(actions) if not verified(values, name)]
            rng = random.Random(task["root"])
            rng.shuffle(failed)
            targets = np.array([values[name] / 10 for name in actions])
            for visit in range(min(3, len(failed)) + 1):
                keep = [i for i in range(len(actions)) if i not in failed[:visit]]
                examples.append((features, features[failed[:visit]], targets, keep))
            game._solve.cache_clear()
    return examples


def fit(key, per_cell):
    examples = training_data(key, per_cell)
    direct = [example for example in examples if not len(example[1])]
    x = np.concatenate([item[0] for item in direct])
    targets = np.concatenate([item[2] for item in direct])
    width = x.shape[1]
    linear = np.linalg.solve(x.T @ x + 0.1 * np.eye(width), x.T @ targets)
    rng = np.random.default_rng(811)
    input_weights = rng.normal(0, 0.15, (width, HIDDEN))
    recurrence = np.eye(HIDDEN) * 0.5 + rng.normal(0, 0.05, (HIDDEN, HIDDEN))
    correction = rng.normal(0, 0.04, (HIDDEN, width))
    for epoch in range(16):
        rate = 0.013 / (1 + 0.12 * epoch)
        for index in rng.permutation(len(examples)):
            features, history, target, keep = examples[index]
            if not len(history):
                continue
            state_stack = [np.zeros(HIDDEN)]
            for failed in history:
                state_stack.append(np.tanh(failed @ input_weights + state_stack[-1] @ recurrence))
            available = features[keep]
            error = (available @ (linear + state_stack[-1] @ correction) - target[keep]) / len(keep)
            action_gradient = error @ available
            hidden_gradient = correction @ action_gradient
            input_gradient = np.zeros_like(input_weights)
            recurrence_gradient = np.zeros_like(recurrence)
            for i in range(len(history) - 1, -1, -1):
                internal = hidden_gradient * (1 - state_stack[i + 1] ** 2)
                input_gradient += np.outer(history[i], internal)
                recurrence_gradient += np.outer(state_stack[i], internal)
                hidden_gradient = recurrence @ internal
            correction -= rate * np.outer(state_stack[-1], action_gradient)
            input_weights -= rate * input_gradient
            recurrence -= rate * recurrence_gradient
    return (linear, input_weights, recurrence, correction), len(examples)


def trial(policy, actions, features, heuristic, native, values, weights, root):
    linear, inputs, recurrent, correction = weights
    available = set(range(len(actions)))
    rng = random.Random(root + "/random")
    hidden = np.zeros(HIDDEN)
    base = features @ linear
    calls, log = 0, []
    chosen = 0
    for visit in range(VISIT_BUDGET):
        candidates = sorted(available)
        if not candidates:
            break
        if visit == 0 and policy in START_NATIVE:
            candidates = [i for i in candidates if actions[i] in native]
        if policy in {"greedy", "heuristic_repair", "first_valid"}:
            if policy == "greedy":
                candidates = [i for i in candidates if actions[i] in native and not features[i, 2]]
            if policy == "first_valid":
                candidates = [i for i in candidates if actions[i] in native]
                if not candidates:
                    break
            index = max(candidates, key=lambda i: heuristic[i])
        elif policy == "random":
            index = rng.choice(candidates)
        else:
            scores = (
                base + features @ (hidden @ correction)
                if policy in {"fixed_depth", "matched_damping", "rlt"} else base
            )
            index = max(candidates, key=lambda i: scores[i])
        action = actions[index]
        check = True
        if policy == "fixed_depth":
            check = visit == VISIT_BUDGET - 1 or len(available) == 1
        elif policy == "learned_deferral":
            check = base[index] > 0 or visit == VISIT_BUDGET - 1 or len(available) == 1
        accepted = verified(values, action) if check else None
        calls += int(check)
        log.append({"action": action, "check": accepted})
        if accepted or policy in {"greedy", "learned_order"} or (policy == "fixed_depth" and check):
            chosen = index
            break
        if policy == "fixed_depth":
            hidden = np.tanh(features[index] @ inputs + hidden @ recurrent)
            continue
        if check and policy in {"matched_damping", "rlt"}:
            hidden = np.tanh(features[index] @ inputs + (0.5 * hidden if policy == "matched_damping"
                                                       else hidden @ recurrent))
        available.remove(index)
        if calls >= CHECK_BUDGET:
            break
    output = actions[chosen]
    miss = max(values.values()) > 0 and not any(verified(values, a) for a in actions if a in native)
    return {
        "action": output, "history": log, "verifier_calls": calls, "visits": len(log),
        "expected_utility": values[output], "native_miss": miss,
        "solved": verified(values, output),
        "repaired_miss": miss and output not in native and verified(values, output),
    }


def summarize(groups):
    for (split, axis, label, policy), rows in sorted(groups.items()):
        n = len(rows)
        delta = np.array([pair[0]["expected_utility"] - pair[1]["expected_utility"] for pair in rows])
        mean = float(delta.mean())
        error = float(1.96 * delta.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        yield {
            "split": split, "axis": axis, "label": label, "policy": policy, "roots": n,
            "paired_expected_gain_vs_heuristic": mean,
            "paired_normal_95": [mean - error, mean + error] if error is not None else None,
            **{field: sum(pair[0][field] for pair in rows) for field in (
                "solved", "native_miss", "repaired_miss", "visits", "verifier_calls", "expected_utility",
            )},
        }


def run(out, key, train_per_cell, eval_per_cell):
    if not key or min(train_per_cell, eval_per_cell) <= 0:
        raise ValueError("positive cohort sizes and a private key are required")
    out.mkdir(parents=True)
    started = perf_counter()
    weights, presentations = fit(key, train_per_cell)
    _write_json(out / "weights.json", [weight.tolist() for weight in weights])
    fit_seconds = perf_counter() - started
    groups = defaultdict(list)
    trace_path = out / "forks.jsonl"
    scores_started = perf_counter()
    with trace_path.open("xb") as trace:
        for split in ("calibration", "ordinary_test", "ood_test"):
            for depth in DEPTHS:
                for ordinal in range(eval_per_cell):
                    task, _ = case(split, depth, ordinal, 0, hashlib.sha256(key + b"/evaluate").digest())
                    game = Game(task["public"])
                    actions, features, heuristic = abstract(game)
                    values = game.values(game.initial)
                    results = {
                        policy: trial(policy, actions, features, heuristic, game.native, values, weights, task["root"])
                        for policy in POLICIES
                    }
                    axes = ("K", "T", "dimension", "family", "condition")
                    public = task["public"]
                    strata = [("overall", "all"), *((axis, str(public[axis])) for axis in axes)]
                    for policy, row in results.items():
                        for axis, label in strata:
                            groups[split, axis, label, policy].append((row, results["heuristic_repair"]))
                    trace.write((_json({
                        "root": task["root"], "split": split, "strata": dict(strata),
                        "results": results, "evaluation_owned_values": values,
                    }) + "\n").encode())
                    game._solve.cache_clear()
    report = {
        "evidence_class": "trained_cpu_recurrent_head_on_synthetic_decision_forks",
        "source_sha256": {**source_hashes(), Path(__file__).name: _file_sha256(Path(__file__))},
        "private_seed_commitment": hashlib.sha256(key).hexdigest(),
        "train_per_depth": train_per_cell, "train_roots": train_per_cell * len(DEPTHS),
        "train_presentations": presentations, "eval_per_depth_split": eval_per_cell,
        "verifier_regret_tolerance": REGRET_LIMIT, "check_budget": CHECK_BUDGET, "visit_budget": VISIT_BUDGET,
        "fit_wall_seconds": fit_seconds, "evaluation_wall_seconds": perf_counter() - scores_started,
        "weights_sha256": _file_sha256(out / "weights.json"), "traces_sha256": _file_sha256(trace_path),
        "cells": list(summarize(groups)),
    }
    _write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--private-seed-file", type=Path, required=True)
    parser.add_argument("--train-per-depth", type=int, default=192)
    parser.add_argument("--eval-per-depth", type=int, default=108)
    args = parser.parse_args()
    result = run(args.out, args.private_seed_file.read_bytes(), args.train_per_depth, args.eval_per_depth)
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}))


if __name__ == "__main__":
    main()
