"""Synthetic K-depth belief and research-taste gym for verifier-grounded RLT training.

This is a *nested subjective signal-game proxy*, not a claim of realistic theory
of mind or a neural result. Public policy observations and sealed sampling are
separate. The independent reference solves expected value with exact Fractions.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import random
from collections import defaultdict
from contextlib import ExitStack
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA = "datagoblin.hypergame_taste.v1"
SPLITS = ("train", "calibration", "ordinary_test", "ood_test", "sealed_poison")
POLICIES = ("bellman", "greedy", "ignore_nested", "forced_probe", "proxy", "random")
TESTS = (("pilot", Fraction(2, 3), 1), ("replicate", Fraction(9, 10), 2))
ROLES = ("scout", "reviewer", "rival", "auditor", "committee", "archivist")
CORRECT, WRONG = 10, -15


def _fraction(value: Fraction | str | int) -> str:
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _draw(rng: random.Random, probability: Fraction) -> bool:
    return rng.randrange(probability.denominator) < probability.numerator


def generate(depth: int, split: str, rng: random.Random, ordinal: int) -> tuple[dict, dict]:
    """Build a private world and the strictly predecision public observation."""
    if depth < 0 or depth > 16 or split not in SPLITS:
        raise ValueError("K must be 0..16 and split must be registered")
    prior = rng.choice((Fraction(1, 3), Fraction(1, 2), Fraction(2, 3)))
    truth = int(_draw(rng, prior))
    base_accuracy = rng.choice((Fraction(3, 4), Fraction(4, 5)))
    report = truth if _draw(rng, base_accuracy) else 1 - truth
    layers = []
    for layer in range(depth):
        accuracy = rng.choice((Fraction(3, 5), Fraction(2, 3), Fraction(3, 4))) if split == "ood_test" else (
            rng.choice((Fraction(2, 3), Fraction(3, 4), Fraction(4, 5)))
        )
        invert = bool(rng.getrandbits(1))
        report = (report ^ int(invert)) if _draw(rng, accuracy) else (report ^ int(invert) ^ 1)
        layers.append({
            "role": f"{ROLES[layer % len(ROLES)]}_{layer + 1}",
            "belief_about": "origin" if layer == 0 else layers[-1]["role"],
            "inverts_earlier_endorsement": invert,
            "fidelity": _fraction(accuracy),
            "subjective_game": "challenge the previous actor" if invert else "defend the previous actor",
        })
    budget = rng.choice((2, 3, 4))
    turns = rng.choice((2, 3))
    public = {
        "schema": SCHEMA, "scenario": "research-council-with-nested-subjective-beliefs",
        "question": "Which of hypotheses H0 and H1 is supported, and which experiment is worth its cost?",
        "K": depth, "prior_h1": _fraction(prior), "base_sensor_fidelity": _fraction(base_accuracy),
        "nested_views": layers, "outer_endorsement": report,
        "available_experiments": [
            {"action": name, "fidelity": _fraction(accuracy), "cost": cost}
            for name, accuracy, cost in TESTS
        ],
        "credits": budget, "turn_limit": turns,
        "payoffs": {"correct_publication": CORRECT, "wrong_publication": WRONG, "abstain": 0},
        "response_contract": "Choose one legal action; experiment results become visible at the next visit.",
    }
    # The identifier binds the public observation; the ordinal prevents accidental
    # reuse of equivalent visible scenarios within one frozen run.
    root = hashlib.sha256(f"{_digest(public)}:{ordinal}:{split}".encode()).hexdigest()
    stream = {
        name: [truth if _draw(rng, accuracy) else 1 - truth for _ in range(turns)]
        for name, accuracy, _ in TESTS
    }
    private = {"root": root, "truth_h1": truth, "test_outcomes": stream}
    return {"root": root, "split": split, "public": public}, private


def posterior(public: dict, observations: list[dict] | tuple = (), *, ignore_nested: bool = False) -> Fraction:
    """Exact Bayes update from the visible outer report and observed experiments."""
    p = Fraction(public["prior_h1"])
    fidelity = Fraction(public["base_sensor_fidelity"])
    inverted = False
    if not ignore_nested:
        for layer in public["nested_views"]:
            fidelity = (1 + (2 * fidelity - 1) * (2 * Fraction(layer["fidelity"]) - 1)) / 2
            inverted ^= layer["inverts_earlier_endorsement"]
    signal = public["outer_endorsement"] ^ int(inverted)
    p = _bayes(p, fidelity, signal)
    tests = {name: accuracy for name, accuracy, _ in TESTS}
    for result in observations:
        p = _bayes(p, tests[result["experiment"]], result["result"])
    return p


def _bayes(p: Fraction, accuracy: Fraction, signal: int) -> Fraction:
    likelihood_one = accuracy if signal == 1 else 1 - accuracy
    likelihood_zero = 1 - likelihood_one
    numerator = p * likelihood_one
    denominator = numerator + (1 - p) * likelihood_zero
    if denominator == 0:
        raise ValueError("impossible observation")
    return numerator / denominator


def _experiment_result(p: Fraction, accuracy: Fraction) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    prob_one = p * accuracy + (1 - p) * (1 - accuracy)
    prob_zero = 1 - prob_one
    after_one = p * accuracy / prob_one if prob_one else p
    after_zero = p * (1 - accuracy) / prob_zero if prob_zero else p
    return prob_one, after_one, prob_zero, after_zero


@lru_cache(maxsize=200_000)
def q_values(p: Fraction, credits: int, turns: int) -> tuple[tuple[str, Fraction], ...]:
    """Independent Bayes/Bellman values. Returned order fixes ties."""
    if credits < 0 or turns < 1:
        raise ValueError("invalid finite horizon")
    options = [
        ("abstain", Fraction(0)),
        ("submit_0", (1 - p) * CORRECT + p * WRONG),
        ("submit_1", p * CORRECT + (1 - p) * WRONG),
    ]
    if turns > 1:
        for name, accuracy, cost in TESTS:
            if credits >= cost:
                prob_one, p_one, prob_zero, p_zero = _experiment_result(p, accuracy)
                future = prob_one * optimal_value(p_one, credits - cost, turns - 1)
                future += prob_zero * optimal_value(p_zero, credits - cost, turns - 1)
                options.append((name, future - cost))
    return tuple(options)


def optimal_value(p: Fraction, credits: int, turns: int) -> Fraction:
    return max(value for _, value in q_values(p, credits, turns))


def best_action(p: Fraction, credits: int, turns: int) -> str:
    q = q_values(p, credits, turns)
    maximum = max(v for _, v in q)
    return next(action for action, value in q if value == maximum)


def choose(policy: str, public: dict, observations: list[dict],
           credits: int, turns: int, rng: random.Random) -> str:
    p = posterior(public, observations, ignore_nested=(policy == "ignore_nested"))
    options = q_values(p, credits, turns)
    if policy in {"bellman", "ignore_nested"}:
        return best_action(p, credits, turns)
    if policy == "random":
        return rng.choice([name for name, _ in options])
    if policy == "proxy":
        return f"submit_{public['outer_endorsement']}"
    if policy == "forced_probe" and turns > 1 and credits >= 1:
        return "pilot"
    terminals = options[:3]
    return max(terminals, key=lambda pair: pair[1])[0]


def rollout(task: dict, sealed: dict, policy: str) -> dict:
    """Matched-world simulation. Grader labels are recorded after each choice."""
    if policy not in POLICIES or task["root"] != sealed["root"]:
        raise ValueError("unknown policy or mismatched sealed world")
    public = task["public"]
    credits, turns = public["credits"], public["turn_limit"]
    observations: list[dict] = []
    used: dict[str, int] = defaultdict(int)
    events = []
    spent = 0
    rng = random.Random(int(_digest({"root": task["root"], "policy": policy})[:16], 16))
    while turns > 0:
        p = posterior(public, observations)
        values = dict(q_values(p, credits, turns))
        action = choose(policy, public, observations, credits, turns, rng)
        if action not in values:
            raise ValueError("policy chose an illegal action")
        best = max(values.values())
        event = {
            "before": {"history": list(observations), "credits": credits, "turns_left": turns},
            "proposal": action, "post_choice_oracle_values": {k: str(v) for k, v in values.items()},
            "post_choice_regret": str(best - values[action]),
            "oracle_preferred": best_action(p, credits, turns),
        }
        if action in {"pilot", "replicate"}:
            cost = next(c for name, _, c in TESTS if name == action)
            answer = sealed["test_outcomes"][action][used[action]]
            used[action] += 1
            observations.append({"experiment": action, "result": answer})
            event["experiment_observation"] = answer
            credits -= cost
            spent += cost
            events.append(event)
            turns -= 1
            continue
        success = action == f"submit_{sealed['truth_h1']}"
        utility = (CORRECT if success else WRONG) if action != "abstain" else 0
        events.append(event)
        break
    else:
        action, success, utility = "abstain", False, 0
    return {
        "schema": SCHEMA, "root": task["root"], "split": task["split"], "K": public["K"],
        "policy": policy, "events": events, "terminal": action, "correct_publication": success,
        "realized_utility": utility - spent, "verifier_calls": len(events),
        "total_research_cost": spent, "evidence_class": "synthetic_exact_oracle",
    }


def fork_rows(task: dict, row: dict) -> list[dict]:
    """Outcome-blind exact preferences at each visited decision point."""
    public = task["public"]
    forks = []
    for i, step in enumerate(row["events"]):
        values = {key: Fraction(value) for key, value in step["post_choice_oracle_values"].items()}
        preferred = step["oracle_preferred"]
        for candidate, score in values.items():
            if score < values[preferred]:
                forks.append({
                    "schema": SCHEMA, "root": task["root"], "K": public["K"],
                    "step": i, "public": public, "context": step["before"],
                    "preferred_action": preferred, "dispreferred_action": candidate,
                    "expected_regret": str(values[preferred] - score),
                    "source_policy": row["policy"], "evidence_class": "synthetic_exact_oracle",
                    "split": "train",
                })
    return forks


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(out: Path, *, depths: tuple[int, ...] = (0, 1, 2, 4, 8),
            per_cell: int = 16, seed: int = 23, private_seed_file: Path | None = None) -> dict:
    """Freeze cases and stream paired controls with bounded working memory."""
    if out.exists() or per_cell < 1 or per_cell > 10_000 or len(set(depths)) != len(depths):
        raise ValueError("output must be fresh; per-cell must be 1..10000; depths must be unique")
    if not depths or any(k < 0 or k > 16 for k in depths):
        raise ValueError("depths must be in 0..16")
    key = private_seed_file.read_bytes() if private_seed_file else None
    if key is not None and len(key) < 16:
        raise ValueError("private seed file must contain at least 16 bytes")
    out.mkdir(parents=True)
    files = (
        "public_tasks.jsonl", "private_sealed.jsonl", "rollouts_scored.jsonl",
        "train_forks.jsonl", "train_teacher_requests.jsonl",
    )
    counts = {name: 0 for name in files}
    hashers = {name: hashlib.sha256() for name in files}
    metrics: dict[tuple[str, int, str], dict] = defaultdict(
        lambda: {"episodes": 0, "correct_publications": 0, "verifier_calls": 0,
                 "utility": 0, "cost": 0, "regret": Fraction(0)}
    )
    seen = set()
    with ExitStack() as stack:
        outputs = {name: stack.enter_context((out / name).open("xb")) for name in files}

        def emit(name: str, item: dict) -> None:
            payload = (_json(item) + "\n").encode("utf-8")
            outputs[name].write(payload)
            hashers[name].update(payload)
            counts[name] += 1

        for split in SPLITS:
            for depth in depths:
                for ordinal in range(per_cell):
                    selector = f"{split}:{depth}:{ordinal}"
                    material = (
                        hmac.new(key, selector.encode(), hashlib.sha256).digest() if key is not None
                        else hashlib.sha256(f"{seed}:{selector}".encode()).digest()
                    )
                    rng = random.Random(int.from_bytes(material, "big"))
                    task, sealed = generate(depth, split, rng, ordinal)
                    if task["root"] in seen:
                        raise AssertionError("duplicate task lineage")
                    seen.add(task["root"])
                    emit("public_tasks.jsonl", task)
                    emit("private_sealed.jsonl", sealed)
                    if split == "train":
                        emit("train_teacher_requests.jsonl", {
                            "request_id": task["root"], "public": task["public"],
                            "visible_history": [], "answer_format": {"action": "one legal action"},
                        })
                    root_forks = []
                    for policy in POLICIES:
                        row = rollout(task, sealed, policy)
                        emit("rollouts_scored.jsonl", row)
                        cell = metrics[(split, depth, policy)]
                        cell["episodes"] += 1
                        cell["correct_publications"] += int(row["correct_publication"])
                        cell["verifier_calls"] += row["verifier_calls"]
                        cell["utility"] += row["realized_utility"]
                        cell["cost"] += row["total_research_cost"]
                        cell["regret"] += Fraction(row["events"][0]["post_choice_regret"])
                        if split == "train":
                            root_forks.extend(fork_rows(task, row))
                    if split == "train":
                        # One teacher request + three distinct forks is the shared
                        # four-view root budget across every policy and export.
                        unique: dict[tuple[str, str, str], dict] = {}
                        for item in root_forks:
                            identity = (
                                _json(item["context"]), item["preferred_action"],
                                item["dispreferred_action"],
                            )
                            unique.setdefault(identity, item)
                        ranked = sorted(
                            unique.values(),
                            key=lambda item: (-Fraction(item["expected_regret"]),
                                              _json(item["context"]), item["dispreferred_action"]),
                        )
                        for item in ranked[:3]:
                            emit("train_forks.jsonl", item)
    artifacts = {
        name: {"rows": counts[name], "sha256": hashers[name].hexdigest()}
        for name in files
    }
    cells = [
        {
            "split": split, "K": depth, "policy": policy, "episodes": cell["episodes"],
            "correct_publications": cell["correct_publications"],
            "verifier_calls": cell["verifier_calls"],
            "mean_true_utility": cell["utility"] / cell["episodes"],
            "mean_research_cost": cell["cost"] / cell["episodes"],
            "initial_action_regret": str(cell["regret"] / cell["episodes"]),
        }
        for (split, depth, policy), cell in sorted(metrics.items())
    ]
    manifest = {
        "schema": SCHEMA, "profile": "private_seeded" if key is not None else "public_seed_smoke",
        "seed_commitment": hashlib.sha256(key).hexdigest() if key is not None else None,
        "K_values": list(depths), "per_depth_per_split": per_cell, "splits": list(SPLITS),
        "policies": list(POLICIES), "artifacts": artifacts, "cells": cells,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "architecture_promotion_eligible": False, "max_train_presentations_per_root": 4,
        "holdout_rule": "train_forks and train_teacher_requests contain training roots only",
        "visibility": "rollouts_scored and private_sealed are evaluation-owned; never send to a teacher",
    }
    (out / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    return manifest


def replay_archive(folder: Path) -> dict:
    """Verify file hashes and replay every policy against its sealed world."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA:
        raise ValueError("unexpected archive schema")
    for name, data in manifest["artifacts"].items():
        if _file_sha256(folder / name) != data["sha256"]:
            raise ValueError(f"archive hash mismatch: {name}")
    roots = actions = episodes = 0
    with (folder / "public_tasks.jsonl").open(encoding="utf-8") as tasks, (
        folder / "private_sealed.jsonl"
    ).open(encoding="utf-8") as secrets, (
        folder / "rollouts_scored.jsonl"
    ).open(encoding="utf-8") as traces:
        for raw_task, raw_secret in zip(tasks, secrets, strict=True):
            task, secret = json.loads(raw_task), json.loads(raw_secret)
            roots += 1
            for policy in POLICIES:
                line = traces.readline()
                if not line:
                    raise ValueError("truncated trace archive")
                got = json.loads(line)
                expected = rollout(task, secret, policy)
                if expected != got:
                    raise ValueError(f"policy trace mismatch: {task['root']}/{policy}")
                episodes += 1
                actions += len(got["events"])
        if traces.readline():
            raise ValueError("unexpected extra scored trace")
    return {"verified_roots": roots, "verified_rollouts": episodes, "verified_actions": actions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replay", type=Path, help="Verify and replay a frozen run archive")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--per-cell", type=int, default=16)
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--private-seed-file", type=Path)
    args = parser.parse_args()
    if args.replay:
        if args.out:
            parser.error("--replay and --out are mutually exclusive")
        print(_json(replay_archive(args.replay)))
        return
    if args.out is None:
        parser.error("--out is required for collection")
    manifest = collect(
        args.out, seed=args.seed, per_cell=args.per_cell,
        depths=tuple(args.depths), private_seed_file=args.private_seed_file,
    )
    print(_json({"profile": manifest["profile"], "artifacts": manifest["artifacts"]}))


if __name__ == "__main__":
    main()
