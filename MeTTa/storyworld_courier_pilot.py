"""Test whether shortest-action supervision transfers to native Storyworld navigation."""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from metta_research_bench.looped.__main__ import revision
from metta_research_bench.looped.action_ties import save_lines
from metta_research_bench.looped.model import BATCH_SIZE, BUCKETS, FIELDS, LOOPS, LoopedPointer, fit, upstream_revision

from .evidence_budget_confirmation import primary_intervals
from .ram_hillclimb import canonical, digest, install_memory_ceiling, peak_rss_mib

ARMS = ("canonical", "hashed", "action_set")
PROTOCOL = {
    "schema": "metta-storyworld-courier-v1", "train_roots_per_family": 40, "frozen_roots_per_family": 24,
    "draw_limit_per_family": 500, "train_seeds": (1101, 1102, 1103), "optimizer_updates": 180,
    "primary": "action_set_minus_canonical", "meaningful_completion_gain": 0.02,
    "storyworld_commit": "4cfe1713d73b4e508833b668477921adfbe570a6",
    "max_pres_per_root": 2, "evidence_class": "frozen_synthetic_domain_transfer_pilot",
}


def arcs(open_edges: tuple, gated_edges: tuple, one_way: tuple = ()) -> tuple:
    return tuple(sorted([(a, b, gate) for group, gate in ((open_edges, False), (gated_edges, True))
                         for x, y in group for a, b in ((x, y), (y, x))] + list(one_way)))


TRAIN = {
    "joined_loops": arcs(((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (6, 4)), ((2, 3),)),
    "diamond_fork": arcs(((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6)), ((3, 4),)),
    "ring_crossings": arcs(((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)), ((0, 3), (2, 5))),
    "forked_bridge": arcs(((0, 1), (2, 3), (2, 4), (5, 6), (6, 4)), ((1, 2), (1, 5))),
    "arrow_cycle": arcs(((0, 1), (1, 2), (3, 4), (5, 6)), ((1, 4),), ((2, 3, False), (4, 5, False), (6, 0, True))),
}
FROZEN = {
    "gated_spokes": arcs(((0, 1), (0, 3), (0, 4), (0, 6), (1, 2), (4, 5)), ((0, 2), (0, 5))),
    "twin_turns": arcs(((0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 6)), ((1, 5), (2, 6))),
    "ladder_cut": arcs(((0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (2, 5), (5, 6)), ((1, 4), (2, 6))),
    "one_way_locks": arcs(((0, 1), (2, 3), (4, 5)), ((0, 6),),
                         ((1, 2, True), (3, 4, False), (5, 6, True), (6, 2, False))),
    "permit_corridor": arcs(((0, 1), (1, 2), (3, 4), (5, 6)), ((2, 3), (4, 5))),
    "double_fan": arcs(((0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (5, 6)), ((0, 3), (2, 6))),
}


def load_courier(checkout: Path):
    root = checkout.resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    tracked = ("bridge/courier_planner.py", "engine/courier.metta")
    if actual != PROTOCOL["storyworld_commit"] or subprocess.check_output(
        ["git", "-C", str(root), "diff", "HEAD", "--", *tracked]
    ):
        raise ValueError("Use the registered Storyworld commit and unedited courier sources")
    spec = importlib.util.spec_from_file_location("metta_pinned_courier_planner", root / tracked[0])
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    spec.loader.exec_module(candidate)
    return candidate, {name: digest((root / name).read_bytes()) for name in tracked}


def allocate(courier) -> tuple[dict, dict]:
    structures = {courier.topology(route) for route in (*TRAIN.values(), *FROZEN.values())}
    if len(structures) != len(TRAIN) + len(FROZEN):
        raise ValueError("Courier topology strata must be non-isomorphic")
    cohorts, excluded = {}, {}
    used = set()
    for split, templates, count in (("train", TRAIN, PROTOCOL["train_roots_per_family"]),
                                    ("frozen", FROZEN, PROTOCOL["frozen_roots_per_family"])):
        group = []
        rejected = Counter()
        for family, route in templates.items():
            accepted = 0
            for ordinal in range(PROTOCOL["draw_limit_per_family"]):
                world, roles = courier.source_world(route, family, ordinal, PROTOCOL["schema"])
                key = family, roles
                if key in used:
                    rejected["isomorphic_roles"] += 1
                    continue
                _, distance = courier.graph_and_distances(world, courier.python_successors)
                length = distance.get(world.initial, courier.MAX_MOVES + 1)
                if length > courier.MAX_MOVES:
                    rejected["unreachable_within_cap"] += 1
                    continue
                used.add(key)
                open_only = replace(world, arcs=tuple(edge for edge in world.arcs if not edge[2]))
                _, unpermitted = courier.graph_and_distances(open_only, courier.python_successors)
                group.append({"family": family, "ordinal": ordinal, "source_sha256": world.source_sha256,
                              "role_signature": roles, "shortest_moves": length, "world": world,
                              "gates_necessary_under_cap": unpermitted.get(world.initial, courier.MAX_MOVES + 1)
                              > courier.MAX_MOVES})
                accepted += 1
                if accepted == count:
                    break
            if accepted != count:
                raise ValueError(f"The locked public generation cap was insufficient for {family}")
        cohorts[split], excluded[split] = group, rejected
    hashes = [row["source_sha256"] for group in cohorts.values() for row in group]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Identical courier source across the allocation")
    return cohorts, excluded


def tokenize(courier, world, state, legal) -> tuple:
    rows = np.zeros((courier.STOP + 1, FIELDS))
    for i in range(courier.PLACES):
        rows[i, :9] = (1, i == state.place, i in world.offices, i in world.depots, i == world.desk,
                       state.permit, state.cargo / 2, i / (courier.PLACES - 1), 0)
    for source, destination, gate in world.arcs:
        rows[source, 9 + destination + courier.PLACES * gate] = 1
    for kind in range(courier.PLACES, courier.STOP + 1):
        rows[kind, :9] = (2, 0, 0, 0, 0, state.permit, state.cargo / 2,
                          state.place / (courier.PLACES - 1), (kind - courier.PLACES) / 3)
    codes = np.rint((rows + 2) * 8).astype(np.int64) + np.arange(FIELDS) * BUCKETS
    allowed = np.zeros(courier.STOP + 1, dtype=np.bool_)
    allowed[[*legal, courier.STOP]] = True
    return torch.from_numpy(codes), torch.from_numpy(allowed)


def training_data(courier, native, sources):
    tokens, masks, targets, frames, traces = [], [], defaultdict(list), [], []
    for source in sources:
        world = source["world"]
        graph, distance = courier.graph_and_distances(world, native.checked_successors)
        if distance[world.initial] != source["shortest_moves"]:
            raise ValueError("Native training graph differs from the registered public route")
        seen = set()
        for view in (0, 1):
            state = world.initial
            actions = []
            while True:
                options = courier.shortest_actions(state, graph, distance)
                if state not in seen:
                    hashed = int(digest([world.source_sha256, asdict(state)])[:16], 16) % len(options)
                    labels = {"canonical": options[:1], "hashed": options[hashed:hashed + 1], "action_set": options}
                    code, mask = tokenize(courier, world, state, graph[state])
                    tokens.append(code)
                    masks.append(mask)
                    for arm, actions_for_arm in labels.items():
                        target = torch.zeros(courier.STOP + 1)
                        target[list(actions_for_arm)] = 1 / len(actions_for_arm)
                        targets[arm].append(target)
                    frames.append({"source_sha256": world.source_sha256, "state": asdict(state),
                                   "labels": labels, "legal": sorted(graph[state]), "remaining": distance[state]})
                    seen.add(state)
                action = options[0] if view == 0 else options[-1]
                if action == courier.STOP:
                    break
                actions.append(action)
                state = graph[state][action]
            receipt = native.verify_plan(world, actions)
            if not receipt["delivered"] or len(actions) != source["shortest_moves"]:
                raise ValueError("A shortest training presentation failed native replay")
            traces.append({"source_sha256": world.source_sha256, "presentation": view, "moves": actions, **receipt})
    return torch.stack(tokens), torch.stack(masks), targets, frames, traces


def fit_all(tokens, masks, targets) -> tuple[dict, list]:
    models, fits = {}, []
    fields = ("initial_sha256", "parameters", "supervised_examples", "optimizer_updates", "supervised_action_tokens")
    for seed in PROTOCOL["train_seeds"]:
        torch.manual_seed(seed)
        initial = LoopedPointer()
        group = []
        for arm in ARMS:
            batch = tokens, masks, torch.stack(targets[arm])
            models[arm, seed], receipt = fit(initial, batch, seed, PROTOCOL["optimizer_updates"])
            group.append({**receipt, "arm": arm})
        if any(len({row[field] for row in group}) != 1 for field in fields):
            raise ValueError("Matched courier fits differ in initial weights or optimizer allocation")
        fits.extend(group)
    return models, fits


@torch.inference_mode()
def rollout(courier, model, world, graph) -> tuple:
    state = world.initial
    visited = {state}
    actions = []
    for decision in range(courier.MAX_MOVES + 1):
        legal = [action for action, after in graph[state].items() if after not in visited]
        code, mask = tokenize(courier, world, state, legal)
        action = int(model(code[None], mask[None])[0].argmax().item())
        if action == courier.STOP:
            return actions, state, True, (decision + 1) * LOOPS
        if action not in legal:
            raise ValueError("Courier policy bypassed its preverified action mask")
        if decision == courier.MAX_MOVES:
            break
        actions.append(action)
        state = graph[state][action]
        visited.add(state)
    return actions, state, False, (courier.MAX_MOVES + 1) * LOOPS


def evaluate(courier, native, sources, models) -> list:
    rows = []
    for source in sources:
        world = source["world"]
        graph, distance = courier.graph_and_distances(world, native.checked_successors)
        if distance[world.initial] != source["shortest_moves"]:
            raise ValueError("Native frozen route differs from its manifest")
        common = {key: value for key, value in source.items() if key != "world"}
        shortest = courier.shortest_actions(world.initial, graph, distance)
        for (arm, seed), model in models.items():
            actions, state, stopped, visits = rollout(courier, model, world, graph)
            receipt = native.verify_plan(world, actions)
            if receipt["native_states"] != [courier.state_atom(state)]:
                raise ValueError("Native replay contradicted the proposed courier trajectory")
            rows.append({**common, "arm": arm, "seed": seed, "moves": actions, "stopped": stopped, **receipt,
                         "verified_completion": stopped and receipt["delivered"], "recurrent_visits": visits,
                         "shortest_first": bool(actions) and actions[0] in shortest})
        route, state = [], world.initial
        while distance[state]:
            action = courier.shortest_actions(state, graph, distance)[0]
            route.append(action)
            state = graph[state][action]
        receipt = native.verify_plan(world, route)
        if not receipt["delivered"] or len(route) != source["shortest_moves"]:
            raise ValueError("Native exact route failed its manifest")
        rows.append({**common, "arm": "exact", "seed": None, "moves": route, "stopped": True, **receipt,
                     "verified_completion": True, "shortest_first": True, "recurrent_visits": 0})
    return rows


def analyze(rows: list) -> dict:
    outcomes = defaultdict(Counter)
    pairs = defaultdict(dict)
    for row in rows:
        solved, moves = row["verified_completion"], len(row["moves"])
        outcomes[row["arm"]].update(
            trials=1,
            completed=int(solved),
            moves=moves,
            shortest_first=int(row["shortest_first"]),
            premature_stop=int(row["stopped"] and not row["delivered"]),
            move_cap_exhausted=int(not row["stopped"]),
            invalid_routes=int(not row["route_valid"]),
            recurrent_visits=row["recurrent_visits"],
            successful_route_overhead=(moves - row["shortest_moves"]) * int(solved),
        )
        if row["arm"] in ARMS:
            pairs[row["source_sha256"]][row["arm"], row["seed"]] = row
    contrasts = {}
    for baseline, candidate in (("canonical", "action_set"), ("hashed", "action_set"), ("canonical", "hashed")):
        strata = defaultdict(list)
        for group in pairs.values():
            delta = []
            for seed in PROTOCOL["train_seeds"]:
                left, right = group[baseline, seed], group[candidate, seed]
                delta.append((len(left["moves"]) - len(right["moves"]),
                              right["verified_completion"] - left["verified_completion"]))
            strata[left["family"]].append(np.mean(delta, axis=0).tolist())
        intervals = primary_intervals(strata)
        contrasts[f"{candidate}_minus_{baseline}"] = {"moves_saved": intervals["calls_saved"],
                                                        "completion_difference": intervals["completion_difference"]}
    return {"outcomes": outcomes, "contrasts": contrasts}


def run(storyworld: Path, out: Path) -> dict:
    memory = install_memory_ceiling(8192)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    commit, dirty = revision()
    if dirty:
        raise ValueError("Commit the courier protocol before scoring frozen roots")
    upstream = upstream_revision()
    courier, source_pins = load_courier(storyworld)
    native = courier.NativeCourier()
    cohorts, rejected = allocate(courier)
    out.mkdir(parents=True, exist_ok=False)
    lock = {"commit": commit, "upstream": upstream, "torch": str(torch.__version__), "batch_size": BATCH_SIZE,
            "protocol": PROTOCOL, "max_moves": courier.MAX_MOVES, "architecture_visits": LOOPS,
            "storyworld_sources": source_pins, "train_topologies": TRAIN, "frozen_topologies": FROZEN,
            "rejected_public_draws": rejected,
            "cohorts": {split: [{**row, "world": asdict(row["world"])} for row in group]
                        for split, group in cohorts.items()}}
    lock_file = out / "locked_protocol.json"
    lock_file.write_bytes((canonical(lock) + "\n").encode())
    tokens, masks, targets, frames, traces = training_data(courier, native, cohorts["train"])
    frame_sha = save_lines(out / "training_frames.jsonl", frames)
    trace_sha = save_lines(out / "training_trajectories.jsonl", traces)
    train_calls = {"successor": native.successor_calls, "replay": native.replay_calls}
    models, fits = fit_all(tokens, masks, targets)
    fit_file = out / "locked_fits.json"
    fit_file.write_bytes((canonical(fits) + "\n").encode())
    rows = evaluate(courier, native, cohorts["frozen"], models)
    report = {"protocol_sha256": digest(lock_file.read_bytes()), "fits_sha256": digest(fit_file.read_bytes()),
              "train_frame_sha256": frame_sha, "train_trajectory_sha256": trace_sha,
              "receipts_sha256": save_lines(out / "frozen_receipts.jsonl", rows), **analyze(rows),
              "training_ties": sum(len(row["labels"]["action_set"]) > 1 for row in frames),
              "native_successor_calls": native.successor_calls, "native_replay_calls": native.replay_calls,
              "train_native_calls": train_calls,
              "frozen_native_calls": {"successor": native.successor_calls - train_calls["successor"],
                                      "replay": native.replay_calls - train_calls["replay"]},
              "wall_seconds": time.perf_counter() - started, "peak_mib": peak_rss_mib(), "memory": memory,
              "training_ready": False}
    (out / "courier_report.json").write_bytes((canonical(report) + "\n").encode())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("storyworld", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    options = parser.parse_args()
    print(canonical(run(options.storyworld, options.out)))


if __name__ == "__main__":
    main()
