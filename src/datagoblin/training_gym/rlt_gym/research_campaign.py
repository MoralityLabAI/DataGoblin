"""Collect and independently audit strategic research-game episodes."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import random
from collections import defaultdict
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from time import perf_counter

from . import research_world
from .hypergame_taste import SPLITS, _file_sha256, _json
from .research_world import DENOMINATOR, DEPTHS, Game, enumerated_likelihood, generate

SCHEMA = "datagoblin.research_campaign.v1"
POLICIES = ("exact", "greedy", "native", "two_ply", "ignore_views", "random")
FORMATS = ("sft", "transition", "repair", "preference", "outcome_prediction", "research_selection")
FILES = (
    "public.jsonl", "sealed.jsonl", "scored.jsonl", "training_ledger.jsonl",
    *(f"{name}.jsonl" for name in FORMATS),
)


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def case(split, depth, ordinal, seed, key):
    selector = f"{split}/{depth}/{ordinal}".encode()
    material = hmac.digest(key, selector, "sha256") if key else hashlib.sha256(str(seed).encode() + selector).digest()
    return generate(depth, split, random.Random(int.from_bytes(material, "big")), ordinal)


def reference(game, state, masses, lookahead):
    """Independent small-horizon rational enumeration over terminal plans."""
    total = sum(masses)
    prior = tuple(Fraction(mass, total) for mass in masses)
    correct, wrong = game.rewards(state.turns)
    result = {"abstain": Fraction(0), **{
        f"publish:{i}": prior[i] * correct + (1 - prior[i]) * wrong for i in range(3)
    }}
    if lookahead <= 1:
        return result
    for name, spec in game.experiments.items():
        if name not in game.legal(state):
            continue
        expectation = Fraction(-spec["cost"])
        for answer in (0, 1):
            weights = tuple(
                mass * (p if answer else DENOMINATOR - p)
                for mass, p in zip(masses, spec["p16"], strict=True)
            )
            p_answer = Fraction(sum(weights), DENOMINATOR * total)
            successor = game.after(state, name, answer)
            expectation += p_answer * max(reference(game, successor, weights, lookahead - 1).values())
        result[name] = expectation
    return result


def best(values):
    return max(values, key=values.__getitem__)


def choose(policy, game, state, values, rng):
    if policy == "exact":
        return best(values)
    if policy == "random":
        return rng.choice(tuple(values))
    if policy == "native":
        return best(game.values(state, native_only=True))
    if policy == "ignore_views":
        return best(game.values(state, ignore_views=True))
    return best(game.values(state, lookahead={"greedy": 1, "two_ply": 2}[policy]))


def episode(task, sealed, game, policy):
    rng = random.Random(f"{task['root']}:{policy}")
    state = game.initial
    events, utility, spent = [], 0, 0
    while state is not None:
        values = game.values(state)
        action = choose(policy, game, state, values, rng)
        future, reward = game.outcome(state, action, sealed)
        masses = game.masses(state) if policy != "ignore_views" else game.prior
        if policy == "ignore_views":
            for name, answer in state.history:
                masses = game.observe(masses, name, answer)
        probabilities = [mass / sum(masses) for mass in masses]
        events.append({
            "action": action, "observation": future.history[-1][1] if future is not None else None,
            "reward": reward, "regret": round(values[best(values)] - values[action], 10),
            "causal_prediction": best(dict(enumerate(probabilities))),
            "brier": round(sum((p - int(i == sealed["truth"])) ** 2 for i, p in enumerate(probabilities)), 10),
        })
        utility += reward
        spent += game.experiments[action]["cost"] if future is not None else 0
        state = future
    return {
        "root": task["root"], "policy": policy, "evidence_class": "synthetic_exact_reference",
        "events": events, "utility": utility, "cost": spent, "visits": len(events),
        "verifier_calls": len(events), "published_correctly": action == f"publish:{sealed['truth']}",
        "causal_identified": events[-1]["causal_prediction"] == sealed["truth"],
    }


def context(task, state, game):
    return {
        "public": task["public"], "history": list(state.history), "credits": state.credits,
        "turns": state.turns, "legal_actions": list(game.legal(state)),
    }


def training_views(task, game, rows):
    """One row per presentation. Alternatives and outcomes remain in targets."""
    if task["split"] != "train":
        return
    states = [game.initial]
    for policy in ("exact", "random"):
        state = game.initial
        for event in rows[policy]["events"]:
            if event["observation"] is None:
                break
            state = game.after(state, event["action"], event["observation"])
            if state not in states:
                states.append(state)
    offset = int(task["root"][:8], 16)
    for i, state in enumerate(states[:4]):
        values = game.values(state)
        chosen = best(values)
        alternative = min(values, key=values.__getitem__)
        mode = FORMATS[(offset + i) % len(FORMATS)]
        if mode == "outcome_prediction" and not any(a in game.experiments for a in values):
            mode = "research_selection"
        visible = context(task, state, game)
        target = {"action": chosen}
        if mode in ("preference", "repair"):
            if mode == "repair" and values[chosen] == values[alternative]:
                mode = "research_selection"
            else:
                visible["candidate"] = alternative
                target.update({"rejected_action": alternative, "expected_gain": values[chosen] - values[alternative]})
        elif mode == "transition":
            spec = game.experiments.get(chosen)
            target["legal_successors"] = [
                {"observation": bit, "credits": state.credits - spec["cost"], "turns": state.turns - 1}
                for bit in (0, 1)
            ] if spec else []
        elif mode == "outcome_prediction":
            action = max((a for a in values if a in game.experiments), key=values.__getitem__)
            masses = game.masses(state)
            probability = sum(game.observe(masses, action, 1)) / (DENOMINATOR * sum(masses))
            visible["candidate"] = action
            target = {"action": action, "probability_signal_one": probability}
        elif mode == "research_selection":
            target["expected_utilities"] = values
        yield mode, {
            "root": task["root"], "split": "train", "presentation": i,
            "input": visible, "target": target, "evidence_class": "synthetic_exact_reference",
        }


class Scorecard:
    """Aggregate within-root paired utility differences and resource use."""

    def __init__(self):
        self.cells = defaultdict(lambda: defaultdict(float))

    def add(self, task, rows):
        public = task["public"]
        axes = ("K", "T", "dimension", "family", "condition")
        strata = [("overall", "all"), *((axis, str(public[axis])) for axis in axes)]
        reference_utility = rows["exact"]["utility"]
        for policy, row in rows.items():
            for axis, label in strata:
                cell = self.cells[task["split"], axis, label, policy]
                difference = row["utility"] - reference_utility
                cell["n"] += 1
                cell["utility"] += row["utility"]
                cell["squared_delta"] += difference * difference
                cell["delta"] += difference
                cell["correct"] += row["published_correctly"]
                cell["identified"] += row["causal_identified"]
                cell["brier"] += row["events"][0]["brier"]
                cell["initial_regret"] += row["events"][0]["regret"]
                for metric in ("cost", "visits", "verifier_calls"):
                    cell[metric] += row[metric]

    def report(self):
        for (split, axis, label, policy), cell in sorted(self.cells.items()):
            n = int(cell["n"])
            mean_delta = cell["delta"] / n
            deviation = math.sqrt(max(0, (cell["squared_delta"] - n * mean_delta**2) / (n - 1) / n)) if n > 1 else None
            yield {
                "split": split, "axis": axis, "label": label, "policy": policy, "n": n,
                "paired_utility_delta": mean_delta,
                "paired_normal_95": [mean_delta - 1.96 * deviation, mean_delta + 1.96 * deviation]
                if deviation is not None else None,
                **{k: cell[k] / n for k in (
                    "utility", "cost", "visits", "verifier_calls", "brier", "identified", "correct", "initial_regret",
                )},
            }


def source_hashes():
    return {path.name: _file_sha256(path) for path in (Path(__file__), Path(research_world.__file__))}


def _write_json(path, value):
    path.write_bytes((_json(value) + "\n").encode())


def collect(out, per_cell, depths, seed, key=None):
    if per_cell < 1 or not depths or len(set(depths)) != len(depths) or not set(depths) <= set(DEPTHS):
        raise ValueError("invalid campaign grid")
    out.mkdir(parents=True)
    started = perf_counter()
    totals = dict.fromkeys(FILES, 0)
    score = Scorecard()
    with ExitStack() as stack:
        outputs = {name: stack.enter_context((out / name).open("xb")) for name in FILES}

        def write(name, row):
            outputs[name].write((_json(row) + "\n").encode())
            totals[name] += 1

        for split in SPLITS:
            for depth in depths:
                for ordinal in range(per_cell):
                    task, sealed = case(split, depth, ordinal, seed, key)
                    game = Game(task["public"])
                    write("public.jsonl", task)
                    write("sealed.jsonl", sealed)
                    rows = {policy: episode(task, sealed, game, policy) for policy in POLICIES}
                    score.add(task, rows)
                    for row in rows.values():
                        write("scored.jsonl", row)
                    for name, view in training_views(task, game, rows):
                        write(f"{name}.jsonl", view)
                        write("training_ledger.jsonl", {
                            "root": task["root"], "presentation": view["presentation"], "format": name,
                        })
                    game._solve.cache_clear()
    _write_json(out / "scorecard.json", list(score.report()))
    manifest = {
        "schema": SCHEMA, "sources": source_hashes(), "depths": depths, "per_cell": per_cell,
        "splits": SPLITS, "policies": POLICIES, "public_seed": None if key else seed,
        "private_seed_commitment": hashlib.sha256(key).hexdigest() if key else None,
        "max_train_presentations_per_root": 4, "scorecard_sha256": _file_sha256(out / "scorecard.json"),
        "artifacts": {name: {"rows": totals[name], "sha256": _file_sha256(out / name)} for name in FILES},
        "collection_wall_seconds": round(perf_counter() - started, 3),
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def replay(folder, key=None):
    started = perf_counter()
    manifest = json.loads((folder / "manifest.json").read_bytes())
    if manifest["schema"] != SCHEMA or manifest["sources"] != source_hashes():
        raise ValueError("frozen campaign sources differ")
    if manifest["private_seed_commitment"] != (hashlib.sha256(key).hexdigest() if key else None):
        raise ValueError("private seed commitment differs")
    if set(manifest["artifacts"]) != set(FILES):
        raise ValueError("campaign artifact list differs")
    for name, item in manifest["artifacts"].items():
        if _file_sha256(folder / name) != item["sha256"]:
            raise ValueError(f"artifact digest differs: {name}")
    streams = {name: records(folder / name) for name in FILES}
    counts = dict.fromkeys(FILES, 0)
    audited = 0
    roots = set()
    score = Scorecard()

    def match(name, expected):
        if _json(next(streams[name], None)) != _json(expected):
            raise ValueError(f"replayed record differs: {name} row {counts[name] + 1}")
        counts[name] += 1

    for split in SPLITS:
        for depth in manifest["depths"]:
            for ordinal in range(manifest["per_cell"]):
                task, sealed = case(split, depth, ordinal, manifest["public_seed"], key)
                if task["root"] in roots:
                    raise ValueError("duplicate case lineage")
                roots.add(task["root"])
                match("public.jsonl", task)
                match("sealed.jsonl", sealed)
                game = Game(task["public"])
                if ordinal % 11 == 0:
                    actual = game.values(game.initial)
                    rational = reference(game, game.initial, game.initial_masses, game.initial.turns)
                    if actual != {action: float(value) for action, value in rational.items()}:
                        raise ValueError("independent Bellman solver disagrees")
                    if depth <= 4 and (
                        research_world.message_likelihood(game.public) != enumerated_likelihood(game.public)
                    ):
                        raise ValueError("independent signal-path enumeration disagrees")
                    audited += 1
                rows = {policy: episode(task, sealed, game, policy) for policy in POLICIES}
                score.add(task, rows)
                for row in rows.values():
                    match("scored.jsonl", row)
                for name, view in training_views(task, game, rows):
                    match(f"{name}.jsonl", view)
                    match("training_ledger.jsonl", {
                        "root": task["root"], "presentation": view["presentation"], "format": name,
                    })
                game._solve.cache_clear()
    if any(next(stream, None) is not None for stream in streams.values()) or counts != {
        name: item["rows"] for name, item in manifest["artifacts"].items()
    }:
        raise ValueError("unexpected campaign row counts")
    scorecard = folder / "scorecard.json"
    if (_file_sha256(scorecard) != manifest["scorecard_sha256"]
            or _json(list(score.report())) != scorecard.read_text(encoding="utf-8").strip()):
        raise ValueError("scorecard differs")
    return {
        "verified_roots": len(roots), "verified_rollouts": counts["scored.jsonl"],
        "independent_exact_checks": audited, "verified_training_presentations": counts["training_ledger.jsonl"],
        "replay_wall_seconds": round(perf_counter() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    dest = parser.add_mutually_exclusive_group(required=True)
    dest.add_argument("--out", type=Path)
    dest.add_argument("--replay", type=Path)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--per-cell", type=int, default=16)
    parser.add_argument("--depths", type=int, nargs="+", default=DEPTHS[:5])
    parser.add_argument("--private-seed-file", type=Path)
    args = parser.parse_args()
    key = args.private_seed_file.read_bytes() if args.private_seed_file else None
    if key is not None and len(key) < 16:
        parser.error("private seed must contain at least 16 bytes")
    if args.replay:
        receipt = replay(args.replay, key)
        _write_json(args.replay / "replay.json", receipt)
        print(_json(receipt))
    else:
        result = collect(args.out, args.per_cell, args.depths, args.seed, key)
        print(_json(result["artifacts"]))


if __name__ == "__main__":
    main()
