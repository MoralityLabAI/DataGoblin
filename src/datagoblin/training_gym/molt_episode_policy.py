"""Public-observation policies and TRAIN chat export for MOLT-GOBLIN-02."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from datagoblin.training_gym.molt_episode import VARIANTS, Episode, World, make_world

POLICIES = ("selective_refresh", "stale", "always_refresh", "forget_receipts")
SYSTEM = (
    "You are an experiment-array curator. Read the observations, keep valid receipt IDs, "
    "request current evidence when needed, and select the maximum-utility admissible array "
    "within capacity. Break ties by lexicographically smallest sorted ID tuple. "
    "Output one JSON action only. No execution or promotion authority."
)


class PublicPolicy:
    """Reference planner consumes only serialized observations, never the World."""

    def __init__(self, mode: str = "selective_refresh"):
        if mode not in POLICIES:
            raise ValueError("unknown policy")
        self.mode = mode
        self.candidates: list[dict[str, Any]] = []
        self.receipts: dict[str, dict[str, Any]] = {}
        self.capacity = 0
        self.frame = ""

    def _submit(self) -> dict[str, Any]:
        forbidden = {r["excluded"] for r in self.receipts.values()}
        eligible = [c for c in self.candidates if c["id"] not in forbidden]
        # Deliberately separate algorithm from the environment's combinations oracle:
        # exact dynamic program over capacity, with a deterministic tie rule.
        table: dict[int, tuple[int, tuple[str, ...]]] = {0: (0, ())}
        for c in sorted(eligible, key=lambda x: x["id"]):
            for cost, (utility, ids) in tuple(table.items()):
                new_cost = cost + c["cost"]
                if new_cost > self.capacity:
                    continue
                candidate = (utility + c["utility"], (*ids, c["id"]))
                old = table.get(new_cost)
                if old is None or candidate[0] > old[0] or (candidate[0] == old[0] and candidate[1] < old[1]):
                    table[new_cost] = candidate
        best = min(table.values(), key=lambda x: (-x[0], x[1]))
        return {"action": "submit", "frame_id": self.frame, "selected": list(best[1]),
                "receipts": sorted(r["receipt_id"] for r in self.receipts.values())}

    def action(self, obs: dict[str, Any]) -> dict[str, Any]:
        event = obs["event"]
        if event == "start":
            self.candidates = list(obs["candidates"])
            self.capacity = obs["capacity"]
            self.frame = obs["frame_id"]
            self.receipts = {r["source"]: r for r in obs["receipts"]}
            return {"action": "advance"}
        if event == "distractor":
            return {"action": "advance"}
        if event == "source_change":
            self.frame = obs["frame_id"]
            source = obs["source"]
            if self.mode == "forget_receipts":
                self.receipts.clear()
                return {"action": "request", "source": "s0"}
            if self.mode == "always_refresh":
                return {"action": "request", "source": "s0"}
            if self.mode == "selective_refresh" and source in self.receipts:
                return {"action": "request", "source": source}
            return self._submit()
        if event == "receipt":
            self.receipts[obs["source"]] = obs
            return self._submit()
        raise ValueError("unknown observation")


def run_public(world: World, mode: str) -> Episode:
    policy = PublicPolicy(mode)
    ep = Episode(world)
    obs = ep.start()
    while not ep.closed:
        # Serialization keeps host state and policy input detached.
        action = policy.action(json.loads(json.dumps(obs)))
        obs = ep.step(action)
    return ep


def report(seed: int, roots: int) -> dict[str, Any]:
    if not 1 <= roots <= 2000:
        raise ValueError("roots outside bounds")
    rows = []
    for i in range(roots):
        for variant in VARIANTS:
            world = make_world("train", seed, i, variant)
            for mode in POLICIES:
                ep = run_public(world, mode)
                rows.append({"root_id": world.root_id, "task_id": world.task_id,
                             "variant": variant, "policy": mode, **ep.result})
    summaries = {}
    for mode in POLICIES:
        per_mode = [r for r in rows if r["policy"] == mode]
        summaries[mode] = {"episodes": len(per_mode),
                           "optimal": sum(r["optimal"] for r in per_mode),
                           "verified": sum(r["verified"] for r in per_mode),
                           "verifier_calls": sum(r["verifier_calls"] for r in per_mode)}
    return {"status": "exact_train_fixture", "roots": roots, "summaries": summaries,
            "rows": rows, "model_executed": False, "promotion_authorized": False}


def train_messages(world: World) -> dict[str, Any]:
    if world.split != "train":
        raise ValueError("only train worlds may be exported")
    ep = run_public(world, "selective_refresh")
    if not ep.result["optimal"]:
        raise RuntimeError("reference policy failed")
    messages = [{"role": "system", "content": SYSTEM}]
    for event in ep.trace:
        if "action" in event:
            messages.append({"role": "assistant", "content": json.dumps(event["action"], sort_keys=True)})
        obs = event["observation"]
        if obs["event"] != "terminal":
            messages.append({"role": "user", "content": json.dumps(obs, sort_keys=True)})
    return {"schema": "molt-episode-chat-v2", "root_id": world.root_id, "task_id": world.task_id,
            "split": "train", "messages": messages, "history_policy": "full_transcript"}


def export(seed: int, roots: int, path: Path) -> None:
    if not 1 <= roots <= 2000:
        raise ValueError("roots outside bounds")
    with path.open("x", encoding="utf-8") as stream:
        for i in range(roots):
            for variant in VARIANTS:
                row = train_messages(make_world("train", seed, i, variant))
                stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--roots", type=int, default=64)
    parser.add_argument("--train-out", type=Path)
    args = parser.parse_args()
    if args.train_out:
        export(args.seed, args.roots, args.train_out)
    result = report(args.seed, args.roots)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
