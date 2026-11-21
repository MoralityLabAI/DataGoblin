"""Bounded edits to public drafts and matched, verifier-trained check scheduling."""

from __future__ import annotations

import copy
import hashlib
import math
import secrets
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .aleph_proxy import CHECK_BUDGET, DEPTH, WIDTH, features, proposals
from .corpus import records
from .dynamics import NativeRanker
from .tasks import FAMILIES, generate_tasks, load_suite, public_task, verify

EDIT_WIDTH = 64
EDITS_PER_VISIT = 12
POLICIES = ("fixed_8", "heuristic_repair", "learned_order", "learned_defer")
COUNTS = {"train": 16, "calibration": 8, "ordinary_test": 16, "ood_test": 16}


def edits(world, draft):
    """Propose local public edits without consulting the task checker or solver."""
    family, data = world["family"], world["input"]
    if family == "graph_route":
        path = draft["path"]
        edges = sorted(data["edges"], key=lambda edge: (edge[2], edge[0], edge[1]))
        for a, b, _ in edges:
            if path and a == path[-1] and b not in path:
                yield "extend_path", {"path": [*path, b]}
        for i in range(len(path) - 1):
            for a, b, _ in edges:
                if a == path[i] and b not in path:
                    yield "insert_hop", {"path": [*path[:i + 1], b, *path[i + 1:]]}
        if path:
            yield "reverse_path", {"path": path[::-1]}
        for i in range(1, len(path) - 1):
            yield "drop_hop", {"path": path[:i] + path[i + 1:]}
    elif family == "arithmetic_program":
        for delta in (-1, 1, -2, 2, -3, 3, -4, 4):
            label = "offset_negative" if delta < 0 else "offset_positive"
            yield label, {"value": draft["value"] + delta}
    elif family == "constraint_assignment":
        assignment = draft["assignment"]
        for key, value in data["fixed"].items():
            if assignment.get(key) != value:
                yield "pin_variable", {"assignment": {**assignment, key: value}}
        for key in data["variables"]:
            colors = range(data["colors"])
            for value in colors:
                if assignment.get(key) != value:
                    yield "edit_color", {"assignment": {**assignment, key: value}}
    elif family == "evidence_retrieval":
        docs = sorted(data["documents"], key=lambda doc: (
            not doc["active"], doc["epoch"] != data["query"]["epoch"], doc["id"]
        ))
        for doc in docs:
            if doc["attribute"] == data["query"]["attribute"]:
                yield "copy_fact_value", {**draft, "value": doc["value"]}
                yield "replace_citation", {**draft, "evidence_ids": [doc["id"]]}
                for alias in docs:
                    if alias["attribute"] == "alias_of":
                        yield "cite_fact_and_alias", {**draft, "evidence_ids": [alias["id"], doc["id"]]}
    elif family == "table_transform":
        for field in ("total", "count"):
            for i, row in enumerate(draft["rows"]):
                for delta in (-1, 1, -2, 2):
                    rows = copy.deepcopy(draft["rows"])
                    rows[i][field] = row[field] + delta
                    yield f"edit_{field}", {"rows": rows}
    elif family == "experiment_plan":
        for key, value in (
            ("independent_verifier", True), ("selection_split", "calibration"),
            ("tokens_per_run", data["max_tokens_per_run"]), ("arms", data["required_arms"]),
            ("metrics", data["required_metrics"]), ("replicates", data["minimum_replicates"]),
        ):
            if draft.get(key) != value:
                yield f"edit_{key}", {**draft, key: value}
    else:
        actions = draft["actions"]
        size = len(data["capacities"])
        for i in range(size):
            for action in (f"fill:{i}", f"empty:{i}", *(f"pour:{i}:{j}" for j in range(size) if i != j)):
                if len(actions) < data["max_actions"]:
                    yield "append_action", {"actions": [*actions, action]}
                if actions:
                    yield "replace_last_action", {"actions": [*actions[:-1], action]}


def candidate_features(world, candidate, operation):
    extra = np.zeros(EDIT_WIDTH, dtype=float)
    for category in (operation, f"{world['family']}:{operation}"):
        key = hashlib.blake2s(category.encode(), digest_size=4).digest()
        extra[int.from_bytes(key, "little") % EDIT_WIDTH] += 1
    return np.concatenate((features(world, candidate, []), extra))


def snapshots(model, task):
    """Expose at most one new draft neighborhood per visit, shared by all repair arms."""
    pool = task["candidates"]
    world = public_task(task)
    frontier = {sha256_json(candidate): (candidate, "native", index)
                for index, candidate in enumerate(pool)}
    expanded = set()
    for visit, step in enumerate(proposals(model, task), 1):
        order = sorted(range(len(pool)), key=lambda i: (-step["candidate_scores"][i], i))
        anchor = next((i for i in order if i not in expanded), order[(visit - 1) % len(order)])
        skip = 0 if anchor not in expanded else EDITS_PER_VISIT
        expanded.add(anchor)
        for operation, candidate in islice(edits(world, pool[anchor]), skip, skip + EDITS_PER_VISIT):
            digest = sha256_json(candidate)
            frontier.setdefault(digest, (candidate, operation, anchor))
        yield step, dict(frontier)


def native_snapshots(model, task):
    for step in proposals(model, task):
        candidate = step["candidate"]
        yield step, {sha256_json(candidate): (candidate, "native", step["candidate_index"])}


def ridge(xs, ys, prior, penalty):
    x, y = np.asarray(xs), np.asarray(ys)
    size = len(prior)
    return np.linalg.solve(x.T @ x + penalty * np.eye(size), x.T @ y + penalty * prior)


def train_rank(model, tasks, previous_weights, guard):
    xs, ys, calls = [], [], 0
    for item in tasks:
        guard()
        task = item["task"]
        world = public_task(task)
        _, frontier = list(snapshots(model, task))[-1]
        weight = 1 / math.sqrt(len(frontier))
        for candidate, operation, _ in frontier.values():
            receipt = verify(task, candidate)
            calls += 1
            xs.append(candidate_features(world, candidate, operation) * weight)
            ys.append((1 if receipt["passed"] else -1) * weight)
    prior = np.concatenate((previous_weights, np.zeros(EDIT_WIDTH)))
    return ridge(xs, ys, prior, 4), calls


def gate_features(world, operation, score, visit, new, failure_count):
    family = [float(world["family"] == f) for f in FAMILIES]
    return np.asarray([1, np.clip(score, -2, 2), visit / DEPTH, float(operation != "native"),
                       min(new, 16) / 16, failure_count / CHECK_BUDGET, *family], dtype=float)


def score_candidate(world, record, rank):
    return float(candidate_features(world, record[0], record[1]) @ rank)


def fit_gate(model, tasks, rank, guard):
    xs, ys, calls = [], [], 0
    for item in tasks:
        guard()
        task = item["task"]
        world = public_task(task)
        checked = set()
        scores = {}
        for visit, (step, frontier) in enumerate(snapshots(model, task), 1):
            fresh = [digest for digest in frontier if digest not in scores]
            scores.update((d, score_candidate(world, frontier[d], rank)) for d in fresh)
            options = [d for d in frontier if d not in checked]
            if not options:
                continue
            native = sha256_json(step["candidate"])
            choice = max(options, key=lambda d: (scores[d], d == native))
            candidate, operation, _ = frontier[choice]
            xs.append(gate_features(world, operation, scores[choice], visit, len(fresh), len(checked)))
            receipt = verify(task, candidate)
            calls += 1
            ys.append(1 if receipt["passed"] else -1)
            checked.add(choice)
            if receipt["passed"] or len(checked) == CHECK_BUDGET:
                break
    return ridge(xs, ys, np.zeros(len(xs[0])), 4), calls


def play(item, trace, policy, rank, gate, threshold):
    task = item["task"]
    world = public_task(task)
    digests = {sha256_json(c) for c in task["candidates"]}
    checks, decisions, scores = [], [], {}
    checked = set()
    out = {"checks": checks, "decisions": decisions, "policy": policy, "condition": item["condition"],
           "family": task["family"], "split": task["split"], "root": task["lineage_id"],
           "solved": False, "out_of_pool_solve": False, "deferred": 0, "visits": 0,
           "rank_scores": 0, "gate_scores": 0}
    for visit, (step, frontier) in enumerate(trace, 1):
        out["visits"] = visit
        if policy == "fixed_8" and visit < DEPTH:
            continue
        native = sha256_json(step["candidate"])
        fresh = [digest for digest in frontier if digest not in scores]
        options = [digest for digest in frontier if digest not in checked]
        if not options:
            continue
        if policy in {"learned_order", "learned_defer"}:
            scores.update((d, score_candidate(world, frontier[d], rank)) for d in fresh)
            out["rank_scores"] += len(fresh)
            choice = max(options, key=lambda d: (scores[d], d == native))
        elif policy == "heuristic_repair":
            choice = next((d for d in options if d not in digests), native if native in options else options[0])
        else:
            choice = native
        candidate, operation, _ = frontier[choice]
        gate_score = None
        if policy == "learned_defer":
            gate_score = float(gate_features(world, operation, scores[choice], visit, len(fresh), len(checked)) @ gate)
            out["gate_scores"] += 1
            if gate_score < threshold and visit < DEPTH:
                out["deferred"] += 1
                decisions.append({"visit": visit, "choice": choice, "action": "defer", "gate_score": gate_score})
                continue
        decisions.append({"visit": visit, "choice": choice, "action": "check", "gate_score": gate_score})
        receipt = verify(task, candidate)
        checks.append({"visit": visit, "candidate": candidate, "operation": operation, "receipt": receipt,
                       "out_of_pool": choice not in digests})
        checked.add(choice)
        if receipt["passed"]:
            out["solved"] = True
            out["out_of_pool_solve"] = choice not in digests
            break
        if len(checks) == CHECK_BUDGET:
            break
    return out


def tally(rows):
    return {"episodes": len(rows), "solved": sum(row["solved"] for row in rows),
            "out_of_pool_solves": sum(row["out_of_pool_solve"] for row in rows),
            "verifier_calls": sum(len(row["checks"]) for row in rows),
            **{key: sum(row[key] for row in rows) for key in ("deferred", "visits", "rank_scores", "gate_scores")}}


def freeze(previous, seed, guard):
    seen = set(previous)
    groups = {(split, family): [] for split in COUNTS for family in FAMILIES}
    construction_calls = 0
    for draw in range(4):
        guard()
        draw_seed = int(sha256_json([seed, draw])[:16], 16) % (2**63)
        for task in generate_tasks(draw_seed, per_family=48):
            key, root = (task["split"], task["family"]), task["lineage_id"]
            if key not in groups or root in seen or len(groups[key]) == COUNTS[key[0]]:
                continue
            positives = []
            negatives = []
            for candidate in task["candidates"]:
                receipt = verify(task, candidate)
                construction_calls += 1
                (positives if receipt["passed"] else negatives).append(candidate)
            miss = len(groups[key]) % 4 != 3
            if not positives or (miss and len(negatives) < 2):
                continue
            original_digest = task["task_digest"]
            if miss:
                task = {**task, "candidates": negatives}
                task.pop("task_digest")
                task["task_digest"] = sha256_json(task)
            group = groups[key]
            group.append({"task": task, "condition": "miss" if miss else "intact",
                          "original_task_digest": original_digest,
                          "original_pool_size": len(positives) + len(negatives),
                          "removed_passing_candidates": len(positives) if miss else 0})
            seen.add(root)
        if all(len(items) == COUNTS[split] for (split, _), items in groups.items()):
            return groups, construction_calls
    raise ValueError("insufficient unused fresh roots")


def load_reservoir(campaign):
    snapshot = campaign / "terminal" / "models" / "seed-11-rlt.json"
    model = NativeRanker.load(snapshot)
    manifest = load_structured(campaign / "terminal" / "manifest.json")
    if manifest["models"][model.model_digest]["sha256"] != sha256_file(snapshot):
        raise ValueError("the source reservoir changed")
    return model


def run(campaign, previous_proxy, out, guard):
    previous = set()
    for name in load_structured(campaign / "source-spec.json")["suites"]:
        guard()
        _, tasks = load_suite(Path(name))
        previous.update(task["lineage_id"] for task in tasks)
    old_summary = load_structured(previous_proxy / "summary.json")
    for name in ("fresh-tasks.json", "proxy.json"):
        if sha256_file(previous_proxy / name) != old_summary["artifacts"][name]:
            raise ValueError("the source proxy artifact changed")
    previous.update(t["lineage_id"] for t in load_structured(previous_proxy / "fresh-tasks.json")["tasks"])
    prior = np.asarray(load_structured(previous_proxy / "proxy.json")["weights"])
    if prior.shape != (WIDTH,):
        raise ValueError("wrong Aleph proxy width")
    model = load_reservoir(campaign)

    out.mkdir(parents=True, exist_ok=False)
    seed = secrets.randbits(63)
    thresholds = (-0.35, -0.1, 0.15, 0.35, 0.55)
    dump_structured(out / "plan.json", {"seed": seed, "counts_per_family": COUNTS,
                    "checks_per_policy": CHECK_BUDGET, "depth": DEPTH, "model_digest": model.model_digest,
                    "edit_budget_per_visit": EDITS_PER_VISIT, "thresholds": list(thresholds),
                    "selection": "calibration maximum solved, then fewest verifier calls and visits"})
    groups, construction = freeze(previous, seed, guard)
    dump_structured(out / "fresh-worlds.json", {"worlds": [item for group in groups.values() for item in group]})
    rank_train = [item for (split, _), group in groups.items() if split == "train" for item in group[:8]]
    gate_train = [item for (split, _), group in groups.items() if split == "train" for item in group[8:]]
    rank, rank_calls = train_rank(model, rank_train, prior, guard)
    gate, gate_calls = fit_gate(model, gate_train, rank, guard)
    dump_structured(out / "weights.json", {"rank": rank.tolist(), "gate": gate.tolist()})
    setup = {"construction_verifier_calls": construction, "rank_fitting_checks": rank_calls,
             "gate_fitting_checks": gate_calls}
    dump_structured(out / "fitting-receipt.json", setup)
    print(f"Fitted repair ranker and check gate on {len(rank_train)} + {len(gate_train)} roots", flush=True)

    calibration = [(item, list(snapshots(model, item["task"])))
                   for (split, _), group in groups.items() if split == "calibration" for item in group]
    sweep = []
    for threshold in thresholds:
        guard()
        rows = [play(item, trace, "learned_defer", rank, gate, threshold) for item, trace in calibration]
        sweep.append({"threshold": threshold, **tally(rows)})
    selected = max(sweep, key=lambda row: (row["solved"], -row["verifier_calls"], -row["visits"]))
    dump_structured(out / "calibration.json", {"selection": selected, "sweep": sweep})
    print(f"Selected check threshold {selected['threshold']}", flush=True)
    return evaluate(model, out, guard, setup)


def evaluate(model, out, guard, setup):
    plan = load_structured(out / "plan.json")
    if plan["model_digest"] != model.model_digest or plan["checks_per_policy"] != CHECK_BUDGET:
        raise ValueError("evaluation plan and reservoir differ")
    items = load_structured(out / "fresh-worlds.json")["worlds"]
    held = {item["task"]["lineage_id"]: item for item in items
            if item["task"]["split"] in {"ordinary_test", "ood_test"}}
    if len(held) != 2 * COUNTS["ood_test"] * len(FAMILIES):
        raise ValueError("held-out cohort differs from the registered matrix")
    weights = load_structured(out / "weights.json")
    rank, gate = np.asarray(weights["rank"]), np.asarray(weights["gate"])
    calibration = load_structured(out / "calibration.json")
    selected = calibration["selection"]
    cells = defaultdict(list)
    replay_calls = 0
    completed = set()

    def retain(row, item):
        nonlocal replay_calls
        task = item["task"]
        key = (row["root"], row["policy"])
        if (key in completed or row["policy"] not in POLICIES or len(row["checks"]) > CHECK_BUDGET
                or row["condition"] != item["condition"] or row["family"] != task["family"]
                or row["split"] != task["split"]):
            raise ValueError("saved comparison row differs from the registered matrix")
        for check in row["checks"]:
            if verify(task, check["candidate"]) != check["receipt"]:
                raise ValueError("held-out checker receipt replay mismatch")
            replay_calls += 1
        completed.add(key)
        cells[row["policy"], row["condition"], row["split"], row["family"]].append(row)

    trace_path = out / "held-rollouts.jsonl"
    if trace_path.exists():
        for row in records(trace_path):
            guard()
            retain(row, held[row["root"]])
        print(f"Replayed {len(completed)} completed policy rows before resuming", flush=True)
    with trace_path.open("a", encoding="utf-8", newline="\n") as stream:
        for item in held.values():
            guard()
            task = item["task"]
            for policy in POLICIES:
                if (task["lineage_id"], policy) in completed:
                    continue
                trace = native_snapshots(model, task) if policy == "fixed_8" else snapshots(model, task)
                row = play(item, trace, policy, rank, gate, selected["threshold"])
                retain(row, item)
                stream.write(canonical_json(row) + "\n")
    rows = [row for group in cells.values() for row in group]
    names = ("plan.json", "fresh-worlds.json", "weights.json", "calibration.json", "held-rollouts.jsonl")
    report = {
        "schema": "rlt_gym.hard_repair.v1", **setup,
        "calibration_checks": sum(row["verifier_calls"] for row in calibration["sweep"]),
        "receipt_replays": replay_calls,
        "policies": {policy: tally([row for row in rows if row["policy"] == policy]) for policy in POLICIES},
        "pool_miss": {policy: tally([row for row in rows if row["policy"] == policy and row["condition"] == "miss"])
                      for policy in POLICIES},
        "cells": [{"policy": p, "condition": c, "split": s, "family": f, **tally(group)}
                  for (p, c, s, f), group in sorted(cells.items())],
        "artifacts": {name: sha256_file(out / name) for name in names},
    }
    dump_structured(out / "summary.json", report)
    return report


def resume(campaign, out, guard):
    if (out / "summary.json").exists():
        raise ValueError("the comparison is already complete")
    receipt = out / "fitting-receipt.json"
    setup = load_structured(receipt) if receipt.exists() else {
        "construction_verifier_calls": None, "rank_fitting_checks": None, "gate_fitting_checks": None,
    }
    return evaluate(load_reservoir(campaign), out, guard, setup)
