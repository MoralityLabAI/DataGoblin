"""Development-only interactive research-array episodes with versioned evidence."""
from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagoblin.training_gym.molt_goblin import digest

VARIANTS = ("relevant", "unrelated")
SPLITS = ("train", "calibration", "ordinary_test", "ood_test")


@dataclass(frozen=True)
class Candidate:
    id: str
    cost: int
    utility: int


@dataclass(frozen=True)
class Fact:
    source: str
    version: int
    excluded: str


@dataclass(frozen=True)
class World:
    root_id: str
    split: str
    variant: str
    candidates: tuple[Candidate, ...]
    initial: tuple[Fact, ...]
    replacement: Fact
    cap: int
    distractors: tuple[int, ...]

    @property
    def task_id(self) -> str:
        return digest([self.root_id, self.variant])

    @property
    def final(self) -> tuple[Fact, ...]:
        if self.variant == "unrelated":
            return self.initial
        return tuple(self.replacement if f.source == self.replacement.source else f for f in self.initial)


def make_world(split: str, seed: int, index: int, variant: str) -> World:
    if split not in SPLITS or variant not in VARIANTS or index < 0:
        raise ValueError("invalid split, variant or index")
    root = digest(["molt-episode-v2", split, seed, index])
    rng = random.Random(int(root, 16))
    n = 8 if split == "ood_test" else 5
    candidates = tuple(Candidate(f"c{i}", rng.randint(1, 4), rng.randint(1, 30)) for i in range(n))
    excluded = rng.sample([c.id for c in candidates], 3)
    facts = (Fact("s0", 1, excluded[0]), Fact("s1", 1, excluded[1]))
    replacement = Fact("s0", 2, excluded[2])
    distractors = tuple(rng.randrange(1000) for _ in range(5 if split == "ood_test" else 2))
    return World(root, split, variant, candidates, facts, replacement, 5, distractors)


def optimum(world: World, facts: tuple[Fact, ...]) -> tuple[tuple[str, ...], int]:
    forbidden = {f.excluded for f in facts}
    eligible = [c for c in world.candidates if c.id not in forbidden]
    best_ids: tuple[str, ...] = ()
    best_utility = 0
    for k in range(1, len(eligible) + 1):
        for group in itertools.combinations(eligible, k):
            if sum(c.cost for c in group) > world.cap:
                continue
            ids = tuple(sorted(c.id for c in group))
            utility = sum(c.utility for c in group)
            if utility > best_utility or (utility == best_utility and ids < best_ids):
                best_ids, best_utility = ids, utility
    return best_ids, best_utility


class Episode:
    """Host-owned state. Models receive only observations and submit typed actions."""

    def __init__(self, world: World):
        self.world = world
        self.facts = world.initial
        self.issued: dict[str, Fact] = {}
        self.cursor = 0
        self.frame = 0
        self.calls = 0
        self.turns = 0
        self.closed = False
        self.trace: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None

    @property
    def turn_cap(self) -> int:
        return len(self.world.distractors) + 3

    @property
    def frame_id(self) -> str:
        return digest([self.world.task_id, self.frame, [(f.source, f.version) for f in self.facts]])

    def _issue(self, fact: Fact) -> dict[str, Any]:
        rid = digest([self.world.task_id, fact.source, fact.version, fact.excluded])
        self.issued[rid] = fact
        return {"receipt_id": rid, "source": fact.source, "version": fact.version, "excluded": fact.excluded}

    def start(self) -> dict[str, Any]:
        if self.trace:
            raise ValueError("episode already started")
        obs = {
            "event": "start",
            "task_id": self.world.task_id,
            "frame_id": self.frame_id,
            "candidates": [{"id": c.id, "cost": c.cost, "utility": c.utility} for c in self.world.candidates],
            "capacity": self.world.cap,
            "receipts": [self._issue(f) for f in self.facts],
            "verifier_calls_remaining": 1,
            "turns_remaining": self.turn_cap,
            "objective": "Select maximum-utility admissible array; ties use smallest sorted ID tuple.",
            "actions": ["advance", "request", "submit", "abstain"],
        }
        self.trace.append({"observation": obs})
        return json.loads(json.dumps(obs))

    def _finish(self, reason: str, *, valid: bool = False, utility: int = 0,
                optimal: bool = False) -> dict[str, Any]:
        self.closed = True
        self.result = {"event": "terminal", "reason": reason, "verified": valid, "optimal": optimal,
                       "utility": utility, "turns": self.turns, "verifier_calls": self.calls,
                       "frame_id": self.frame_id}
        return dict(self.result)

    def _transition(self, action: dict[str, Any]) -> dict[str, Any]:
        kind = action.get("action")
        if kind == "abstain" and set(action) == {"action"}:
            return self._finish("abstained")
        if self.cursor <= len(self.world.distractors):
            if kind != "advance" or set(action) != {"action"}:
                return self._finish("expected_advance")
            if self.cursor < len(self.world.distractors):
                obs = {"event": "distractor", "value": self.world.distractors[self.cursor]}
            else:
                old_frame = self.frame_id
                self.facts = self.world.final
                relevant = self.world.variant == "relevant"
                self.frame += int(relevant)
                obs = {"event": "source_change", "source": "s0" if relevant else "metadata",
                       "version": 2, "prior_frame_id": old_frame, "frame_id": self.frame_id}
            self.cursor += 1
            return obs
        if kind == "request" and set(action) == {"action", "source"}:
            if self.calls >= 1:
                return self._finish("verifier_budget_exhausted")
            self.calls += 1
            for fact in self.facts:
                if fact.source == action["source"]:
                    return {"event": "receipt", **self._issue(fact), "frame_id": self.frame_id}
            return self._finish("unknown_source")
        if kind != "submit" or set(action) != {"action", "selected", "receipts", "frame_id"}:
            return self._finish("invalid_action")
        if action["frame_id"] != self.frame_id:
            return self._finish("stale_frame")
        ids, receipts = action["selected"], action["receipts"]
        if not isinstance(ids, list) or not isinstance(receipts, list):
            return self._finish("invalid_schema")
        if any(not isinstance(x, str) for x in ids + receipts):
            return self._finish("invalid_schema")
        if len(set(ids)) != len(ids) or len(set(receipts)) != len(receipts):
            return self._finish("duplicate_ids")
        if any(r not in self.issued for r in receipts):
            return self._finish("unissued_receipt")
        if {self.issued[r] for r in receipts} != set(self.facts):
            return self._finish("stale_or_missing_evidence")
        selected = [c for c in self.world.candidates if c.id in ids]
        if len(selected) != len(ids) or set(ids) & {f.excluded for f in self.facts}:
            return self._finish("inadmissible_array")
        if sum(c.cost for c in selected) > self.world.cap:
            return self._finish("capacity_exceeded")
        utility = sum(c.utility for c in selected)
        best, _ = optimum(self.world, self.facts)
        return self._finish("submitted", valid=True, utility=utility, optimal=tuple(sorted(ids)) == best)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.closed or not self.trace:
            raise ValueError("episode is not active")
        self.turns += 1
        if not isinstance(action, dict):
            obs = self._finish("invalid_schema")
        else:
            obs = self._transition(action)
        if not self.closed and self.turns >= self.turn_cap:
            obs = self._finish("turn_budget_exhausted")
        self.trace.append({"action": json.loads(json.dumps(action)), "observation": dict(obs)})
        return json.loads(json.dumps(obs))


def exact_trace(world: World) -> Episode:
    """Reference policy: carries initial receipts and requests only changed evidence."""
    ep = Episode(world)
    start = ep.start()
    receipts = {r["source"]: r["receipt_id"] for r in start["receipts"]}
    for _ in world.distractors:
        ep.step({"action": "advance"})
    change = ep.step({"action": "advance"})
    if change["source"] in receipts:
        new = ep.step({"action": "request", "source": change["source"]})
        receipts[change["source"]] = new["receipt_id"]
    selected, _ = optimum(world, world.final)
    ep.step({"action": "submit", "frame_id": change["frame_id"], "selected": list(selected),
             "receipts": sorted(receipts.values())})
    return ep


def replay(world: World, actions: list[dict[str, Any]]) -> Episode:
    ep = Episode(world)
    ep.start()
    for action in actions:
        ep.step(action)
    if not ep.closed:
        raise ValueError("incomplete episode")
    return ep


def export_train(seed: int, count: int, destination: Path) -> None:
    if not 1 <= count <= 2000:
        raise ValueError("count outside bounds")
    with destination.open("x", encoding="utf-8") as stream:
        for i in range(count):
            for variant in VARIANTS:
                world = make_world("train", seed, i, variant)
                ep = exact_trace(world)
                assert ep.result is not None and ep.result["optimal"]
                stream.write(json.dumps({"schema": "molt-episode-v2", "split": "train",
                                        "root_id": world.root_id, "task_id": world.task_id,
                                        "trace": ep.trace, "result": ep.result}, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--train-out", type=Path, required=True)
    args = parser.parse_args()
    export_train(args.seed, args.roots, args.train_out)
    print(json.dumps({"train_roots": args.roots, "episodes": 2 * args.roots,
                      "model_executed": False, "promotion_authorized": False}))


if __name__ == "__main__":
    main()
