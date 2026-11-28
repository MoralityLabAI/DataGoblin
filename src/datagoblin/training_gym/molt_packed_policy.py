"""DISTILL-19: packed exact MoLT optimization, development-only scale study."""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from datagoblin.training_gym.molt_compact_policy import (
    CompiledPublicPolicy,
    normalize,
)
from datagoblin.training_gym.molt_episode import Candidate, Episode, Fact, World, make_world
from datagoblin.training_gym.molt_episode_policy import PublicPolicy
from datagoblin.training_gym.molt_goblin import digest

ARMS = ("reference", "compact", "packed_sparse", "packed_dense")
NATIVE_WIDTHS = (5, 8, 12, 16)
ARRAY_WIDTHS = (16, 32, 64, 128)
SPLITS = ("train", "calibration")
SEEDS = (2026120131, 2026120132)
ARRAY_SEED = 2026120133
REVISIONS = 4


@dataclass(frozen=True)
class Prepared:
    """Immutable objective prepared once from a public start observation."""

    ids: frozenset[str]
    rows: tuple[tuple[str, int, int, int], ...]
    capacity: int
    width: int


def prepare(candidates: list[dict[str, Any]], capacity: int) -> Prepared:
    # Reuse DISTILL-18's strict positive-integer and size checks once.
    problem = normalize(candidates, capacity, {})
    width = len(problem.rows)
    rows = []
    for i, (name, cost, utility) in enumerate(problem.rows):
        bit = 1 << (width - i - 1)
        rows.append((name, cost, (utility << width) | bit, bit))
    return Prepared(frozenset(c["id"] for c in candidates), tuple(rows), problem.capacity, width)


def packed_select(problem: Prepared, forbidden: set[str], *, dense: bool) -> tuple[str, ...]:
    """Exact knapsack. Higher packed integer implements the native tie rule.

    Equal utility subsets cannot be strict subsets because every utility is
    positive. The smallest differing ID decides their lexical order; the
    subset including it has the higher high-to-low membership bit mask.
    Descending dense updates or a sparse snapshot prevent item reuse.
    """
    if not forbidden <= problem.ids:
        raise ValueError("unknown excluded candidate")
    rows = tuple(row for row in problem.rows if row[0] not in forbidden)
    total = sum(row[1] for row in rows)
    if total <= problem.capacity:
        return tuple(row[0] for row in rows)
    cap = problem.capacity
    if dense:
        table = [-1] * (cap + 1)
        table[0] = 0
        high = 0
        for _, cost, score, _ in rows:
            new_high = min(cap, high + cost)
            for used in range(new_high, cost - 1, -1):
                prev = table[used - cost]
                if prev >= 0:
                    value = prev + score
                    if value > table[used]:
                        table[used] = value
            high = new_high
        best = max(table)
    else:
        sparse = {0: 0}
        for _, cost, score, _ in rows:
            for used, prev in tuple(sparse.items()):
                new_cost = used + cost
                if new_cost <= cap:
                    value = prev + score
                    if value > sparse.get(new_cost, -1):
                        sparse[new_cost] = value
        best = max(sparse.values())
    return tuple(name for name, _, _, bit in rows if best & bit)


class PackedPublicPolicy(PublicPolicy):
    """Opt-in; only the numerical objective is prepared, never authorization."""

    def __init__(self, *, dense: bool):
        super().__init__("selective_refresh")
        self.dense = dense
        self.prepared: Prepared | None = None

    def action(self, obs: dict[str, Any]) -> dict[str, Any]:
        if obs["event"] == "start":
            self.prepared = prepare(obs["candidates"], obs["capacity"])
        return super().action(obs)

    def _submit(self) -> dict[str, Any]:
        if self.prepared is None:
            raise ValueError("missing public start objective")
        forbidden = {r["excluded"] for r in self.receipts.values()}
        selected = packed_select(self.prepared, forbidden, dense=self.dense)
        return {
            "action": "submit",
            "frame_id": self.frame,
            "selected": list(selected),
            "receipts": sorted(r["receipt_id"] for r in self.receipts.values()),
        }


def policy_for(arm: str) -> PublicPolicy:
    if arm == "reference":
        return PublicPolicy()
    if arm == "compact":
        return CompiledPublicPolicy()
    if arm in ("packed_sparse", "packed_dense"):
        return PackedPublicPolicy(dense=arm == "packed_dense")
    raise ValueError("unknown arm")


def rotated_arms(i: int) -> tuple[str, ...]:
    k = i % len(ARMS)
    return ARMS[k:] + ARMS[:k]


def native_world(split: str, seed: int, index: int, variant: str, width: int) -> World:
    if split not in SPLITS or width not in NATIVE_WIDTHS:
        raise ValueError("development-only native dimensions")
    world = make_world(split, seed, index, variant)
    if width == 5:
        return world
    root = digest(["distill19-native-scale", world.root_id, width])
    rng = random.Random(int(root, 16))
    candidates = tuple(Candidate(f"c{i:03}", rng.randint(1, 9), rng.randint(1, 30)) for i in range(width))
    excluded = rng.sample([c.id for c in candidates], 3)
    facts = (Fact("s0", 1, excluded[0]), Fact("s1", 1, excluded[1]))
    return replace(
        world, root_id=root, candidates=candidates, initial=facts,
        replacement=Fact("s0", 2, excluded[2]), cap=width * 2,
    )


def native_episode(world: World, arm: str) -> tuple[Episode, dict[str, int]]:
    host = Episode(world)
    obs = host.start()
    start = time.process_time_ns()
    policy = policy_for(arm)
    setup = time.process_time_ns() - start
    elapsed = setup
    submit = 0
    actions = 0
    while not host.closed:
        # Same detached public observation in every arm; host work untimed.
        public = json.loads(json.dumps(obs))
        start = time.process_time_ns()
        a = policy.action(public)
        cost = time.process_time_ns() - start
        elapsed += cost
        if a["action"] == "submit":
            submit += cost
        actions += 1
        obs = host.step(a)
    if host.result is None or not host.result["verified"] or not host.result["optimal"]:
        raise RuntimeError("native host verification failed")
    return host, {"policy_cpu_ns": elapsed, "submit_cpu_ns": submit, "actions": actions}


def native_row(split: str, seed: int, i: int, width: int) -> dict[str, Any]:
    worlds = [native_world(split, seed, i, v, width) for v in ("relevant", "unrelated")]
    traces: dict[str, list[Any]] = {}
    arms: dict[str, dict[str, int]] = {}
    for arm in rotated_arms(i):
        stats = {"policy_cpu_ns": 0, "submit_cpu_ns": 0, "actions": 0, "episodes": 0,
                 "verified": 0, "optimal": 0, "utility": 0, "verifier_calls": 0}
        traces[arm] = []
        for world in worlds:
            host, timing = native_episode(world, arm)
            assert host.result is not None
            for k, v in timing.items():
                stats[k] += v
            stats["episodes"] += 1
            for k in ("verified", "optimal", "utility", "verifier_calls"):
                stats[k] += int(host.result[k])
            traces[arm].append(host.trace)
        arms[arm] = stats
    if any(t != traces["reference"] for t in traces.values()):
        raise RuntimeError("native trace mismatch")
    return {"root": worlds[0].root_id, "split": split, "width": width, "arms": arms,
            "trace_digest": digest(traces["reference"])}


def public_array(width: int, regime: str, i: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    if width not in ARRAY_WIDTHS or regime not in ("tight", "medium"):
        raise ValueError("unsupported public-array dimensions")
    root = digest(["distill19-public-scale", ARRAY_SEED, width, regime, i])
    rng = random.Random(int(root, 16))
    candidates = [
        {"id": f"c{j:03}", "cost": rng.randint(1, 32), "utility": rng.randint(1, 100)}
        for j in range(width)
    ]
    cap = width * (2 if regime == "tight" else 8)
    excluded = rng.sample([c["id"] for c in candidates], REVISIONS + 1)
    initial = {"event": "start", "candidates": candidates, "capacity": cap, "frame_id": root + "-0",
               "receipts": [{"source": "s0", "excluded": excluded[0], "receipt_id": root + "-0"},
                            {"source": "s1", "excluded": excluded[-1], "receipt_id": root + "-static"}]}
    revisions = [
        {"event": "receipt", "source": "s0", "version": j + 1, "excluded": excluded[j],
         "receipt_id": root + f"-r{j}", "frame_id": root + f"-f{j}"}
        for j in range(REVISIONS)
    ]
    return root, initial, revisions


def public_row(width: int, regime: str, i: int) -> dict[str, Any]:
    root, initial, revisions = public_array(width, regime, i)
    outcomes: dict[str, list[Any]] = {}
    arms: dict[str, dict[str, int]] = {}
    for arm in rotated_arms(i):
        obs = json.loads(json.dumps(initial))
        observations = json.loads(json.dumps(revisions))
        start = time.process_time_ns()
        policy = policy_for(arm)
        policy.action(obs)
        setup = time.process_time_ns() - start
        solve = 0
        actions = []
        for observation in observations:
            # This is an isolated public objective/receipt workload, not a
            # native Episode. The caller supplies each new public frame.
            start = time.process_time_ns()
            policy.frame = observation["frame_id"]
            a = policy.action(observation)
            solve += time.process_time_ns() - start
            actions.append(a)
        outcomes[arm] = actions
        arms[arm] = {"policy_cpu_ns": setup + solve, "setup_cpu_ns": setup,
                     "submit_cpu_ns": solve, "queries": len(observations)}
    if any(a != outcomes["reference"] for a in outcomes.values()):
        raise RuntimeError("wide public objective parity failure")
    for a, observation in zip(outcomes["reference"], revisions, strict=True):
        chosen = [c for c in initial["candidates"] if c["id"] in a["selected"]]
        if (len(chosen) != len(a["selected"]) or
                sum(c["cost"] for c in chosen) > initial["capacity"] or
                {c["id"] for c in chosen} & {observation["excluded"], initial["receipts"][1]["excluded"]} or
                a["frame_id"] != observation["frame_id"] or
                a["receipts"] != sorted([observation["receipt_id"], initial["receipts"][1]["receipt_id"]])):
            raise RuntimeError("public objective or receipt binding mismatch")
    return {"root": root, "width": width, "regime": regime, "arms": arms,
            "outcome_digest": digest(outcomes["reference"])}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for arm in ARMS:
        records = [r["arms"][arm] for r in rows]
        totals = {k: sum(r[k] for r in records) for k in records[0]}
        totals["cpu_wins"] = sum(
            r["arms"][arm]["policy_cpu_ns"] < r["arms"]["reference"]["policy_cpu_ns"] for r in rows
        )
        out[arm] = totals
    return out


def campaign(native_roots: int = 24, array_roots: int = 16) -> dict[str, Any]:
    if not 1 <= native_roots <= 64 or not 1 <= array_roots <= 64:
        raise ValueError("invalid root count")
    native = [
        native_row(split, seed, i, width)
        for split, seed in zip(SPLITS, SEEDS, strict=True)
        for width in NATIVE_WIDTHS
        for i in range(native_roots)
    ]
    arrays = [
        public_row(width, regime, i)
        for width in ARRAY_WIDTHS
        for regime in ("tight", "medium")
        for i in range(array_roots)
    ]
    return {
        "schema": "distill19-packed-molt-v1",
        "status": "development_only",
        "promotion_authorized": False,
        "training_admission": False,
        "provider_calls": 0,
        "native_roots_per_cell": native_roots,
        "array_roots_per_cell": array_roots,
        "native_summary": {
            f"{split}/{width}": summarize([r for r in native if r["split"] == split and r["width"] == width])
            for split in SPLITS for width in NATIVE_WIDTHS
        },
        "public_summary": {
            f"{regime}/{width}": summarize([r for r in arrays if r["regime"] == regime and r["width"] == width])
            for regime in ("tight", "medium") for width in ARRAY_WIDTHS
        },
        "rows_digest": digest([native, arrays]),
        "native_rows": native,
        "public_rows": arrays,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-roots", type=int, default=24)
    parser.add_argument("--array-roots", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = campaign(args.native_roots, args.array_roots)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ("native_rows", "public_rows")}, sort_keys=True))


if __name__ == "__main__":
    main()
