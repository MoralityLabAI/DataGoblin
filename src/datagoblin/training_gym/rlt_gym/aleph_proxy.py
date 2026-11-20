"""Fit and evaluate a small verifier-aware controller over frozen RLT proposals."""

from __future__ import annotations

import argparse
import hashlib
import math
import secrets
import time
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .corpus import records
from .dynamics import NativeRanker
from .tasks import FAMILIES, generate_tasks, load_suite, public_task, verify

WIDTH = 192
DEPTH = 8
CHECK_BUDGET = 4
POLICIES = ("fixed_4", "fixed_8", "first_valid", "heuristic_repair", "learned")
COHORT_COUNTS = {"calibration": 12, "ordinary_test": 24, "ood_test": 24}


def features(world, candidate, failed):
    """Cheap public features. The controller never calls the task solver."""
    family, data = world["family"], world["input"]
    result = np.zeros(WIDTH, dtype=np.float64)
    result[0] = 1

    def add(key, weight=1):
        digest = hashlib.blake2s(key.encode(), digest_size=4).digest()
        position = 1 + int.from_bytes(digest, "little") % (WIDTH - 1)
        result[position] += weight

    def feature(key, value):
        add(f"{family}:{key}:{value}")

    feature("family", True)
    feature("failures", min(len(failed), 4))
    feature("previously_failed", candidate in failed)
    if not isinstance(candidate, dict):
        feature("answer_type", type(candidate).__name__)
        return result
    feature("keys", ",".join(sorted(candidate)))
    for key, value in candidate.items():
        feature(key + "_type", type(value).__name__)
        if isinstance(value, (list, dict)):
            feature(key + "_length", min(len(value), 24))
        elif type(value) is int:
            feature(key + "_size", min(abs(value).bit_length(), 20))
            feature(key + "_sign", (value > 0) - (value < 0))
        elif isinstance(value, bool):
            feature(key + "_value", value)

    if family == "graph_route" and isinstance(candidate.get("path"), list):
        path = candidate["path"]
        if path and all(isinstance(node, str) for node in path):
            edges = {(a, b): cost for a, b, cost in data["edges"]}
            hops = list(pairwise(path))
            feature("endpoints", path[0] == data["start"] and path[-1] == data["goal"])
            feature("unknown_hops", min(sum(hop not in edges for hop in hops), 3))
            feature("revisits", len(path) - len(set(path)))
            feature("edge_cost", min(sum(edges.get(hop, 0) for hop in hops) // 5, 24))
    if family == "constraint_assignment" and isinstance(candidate.get("assignment"), dict):
        assignment = candidate["assignment"]
        feature("missing_variables", min(len(set(data["variables"]) - assignment.keys()), 4))
        feature("fixed_count", sum(assignment.get(key) == value for key, value in data["fixed"].items()))
    if family == "experiment_plan":
        for field, required in (("arms", "required_arms"), ("metrics", "required_metrics")):
            value = candidate.get(field)
            if isinstance(value, list):
                feature(field + "_overlap", sum(item in value for item in data[required]))
        tokens = candidate.get("tokens_per_run")
        if type(tokens) is int:
            feature("budget_crossing", tokens > data["max_tokens_per_run"])
    return result


def observed_failures(context):
    return [step["candidate"] for step in context["history"] if step["feedback"]["status"] == "failed"]


def fit(pack, tasks, guard):
    manifest_path = pack / "manifest.json"
    manifest = load_structured(manifest_path)
    if manifest["schema_version"] != "rlt_gym.aleph_corpus.v1" or manifest["split"] != "train":
        raise ValueError("expected the verified Aleph training pack")
    by_root = defaultdict(list)
    checks = 0
    for kind in ("outcomes", "repairs", "preferences"):
        stream = pack / f"{kind}.jsonl"
        if sha256_file(stream) != manifest["streams"][kind]["sha256"]:
            raise ValueError("training stream digest changed")
        for index, row in enumerate(records(stream)):
            if index % 64 == 0:
                guard()
            if not row["loss"]["eligible"]:
                continue
            task = tasks[row["task_digest"]]
            world = public_task(task)
            context, target = row["input"], row["target"]
            if (row["split"] != task["split"] or task["split"] != "train"
                    or row["root"] != task["lineage_id"] or context["task"] != world):
                raise ValueError("training row differs from its frozen public world")
            failed = observed_failures(context)
            if kind == "outcomes":
                candidate = context["candidate"]
                receipt = verify(task, candidate)
                checks += 1
                if receipt != row["verification"] or receipt["passed"] is not target["passed"]:
                    raise ValueError("outcome training receipt failed replay")
                label = 1 if target["passed"] else -1
                values = features(world, candidate, failed)
                lane = "pass" if label == 1 else "fail"
            else:
                if kind == "repairs":
                    if not failed:
                        raise ValueError("repair has no visible failed attempt")
                    good, bad = target["candidate"], failed[-1]
                else:
                    good, bad = target["chosen"]["candidate"], target["rejected"]["candidate"]
                positive, negative = verify(task, good), verify(task, bad)
                checks += 2
                if positive != row["verification"] or negative["passed"] is not False:
                    raise ValueError("repair pair did not replay")
                if kind == "preferences" and negative != row["rejected_provenance"]["verification"]:
                    raise ValueError("rejected preference receipt failed replay")
                label, lane = 1, kind
                values = features(world, good, failed) - features(world, bad, failed)
            by_root[row["root"]].append((lane, row["id"], values, label))

    cap = int(manifest["sampling_plan"]["max_presentations_per_root_across_streams"])
    lanes = ("pass", "fail", "repairs", "preferences")
    design, target, selected, selected_roots = [], [], Counter(), set()
    for root, rows in by_root.items():
        ordered = sorted(rows, key=lambda row: row[1])
        representatives = [next((row for row in ordered if row[0] == lane), None) for lane in lanes]
        head = [row for row in representatives if row is not None]
        seen = {row[1] for row in head}
        keep = (head + [row for row in ordered if row[1] not in seen])[:cap]
        selected_roots.add(root)
        weight = 1 / math.sqrt(len(keep))
        for lane, _, x, y in keep:
            selected[lane] += 1
            strength = weight * (0.5 if lane in {"repairs", "preferences"} else 1)
            design.append(x * strength)
            target.append(y * strength)
    x, y = np.asarray(design), np.asarray(target)
    w = np.linalg.solve(x.T @ x + 8 * np.eye(WIDTH), x.T @ y)
    receipt = {"parameters": WIDTH, "train_examples": dict(selected), "train_roots": len(selected_roots),
               "root_view_cap": cap, "verifier_rechecks": checks, "pack_manifest_sha256": sha256_file(manifest_path)}
    return w, receipt, selected_roots


def rollout(task, steps, mode, *, weights=None, threshold=0.2, margin=0.1):
    limit = 4 if mode == "fixed_4" else DEPTH
    seen, checks, failed = set(), [], []
    outcome = {"solved": False, "visits": 0, "verifier_calls": 0, "repair_checks": 0,
               "repair_solves": 0, "rerank_checks": 0, "controller_scores": 0}
    world = public_task(task)
    candidates = task["candidates"]
    digests = [sha256_json(candidate) for candidate in candidates]
    cached_scores = None
    for visit, step in enumerate(steps, 1):
        outcome["visits"] = visit
        current = step["candidate_index"]
        final = visit == limit
        selected = current
        if mode.startswith("fixed") and not final:
            continue
        if mode in {"learned", "heuristic_repair"}:
            eligible = [i for i, digest in enumerate(digests) if digest not in seen]
            if not eligible:
                break
            if mode == "heuristic_repair":
                if digests[current] in seen:
                    selected = max(eligible, key=lambda i: step["candidate_scores"][i])
            else:
                if cached_scores is None:
                    cached_scores = [float(features(world, candidate, failed) @ weights) for candidate in candidates]
                    outcome["controller_scores"] += len(candidates)
                best = max(eligible, key=lambda i: (cached_scores[i], step["candidate_scores"][i]))
                current_score = cached_scores[current]
                if (digests[current] in seen or cached_scores[best] - current_score >= margin
                        or (final and current_score < threshold)):
                    selected = best
                if not final and cached_scores[selected] < threshold:
                    continue
        digest = digests[selected]
        if digest in seen:
            continue
        seen.add(digest)
        repair = selected != current and bool(failed)
        candidate = candidates[selected]
        receipt = verify(task, candidate)
        checks.append({"visit": visit, "candidate": candidate, "receipt": receipt, "repair": repair,
                       "native_candidate_index": current, "checked_candidate_index": selected})
        outcome["verifier_calls"] += 1
        outcome["repair_checks"] += int(repair)
        outcome["rerank_checks"] += int(selected != current and not failed)
        if receipt["passed"]:
            outcome["solved"] = True
            outcome["repair_solves"] += int(repair)
            break
        failed.append(candidate)
        cached_scores = None
        if final or (mode in {"learned", "heuristic_repair"} and len(checks) == CHECK_BUDGET):
            break
    return {"task_digest": task["task_digest"], "family": task["family"], "split": task["split"],
            "root": task["lineage_id"], "policy": mode, "checks": checks, **outcome}


def proposals(model, task, depth=DEPTH):
    return model.visits(task, "rlt", depth)


def summarize(rows):
    keys = ("solved", "visits", "verifier_calls", "repair_checks", "repair_solves",
            "rerank_checks", "controller_scores")
    counts = {key: sum(row[key] for row in rows) for key in keys}
    return {"episodes": len(rows), **counts, "checks_per_solve":
            counts["verifier_calls"] / counts["solved"] if counts["solved"] else None}


def freeze_fresh(previous, seed, guard):
    groups = {(split, family): [] for split in COHORT_COUNTS for family in FAMILIES}
    seen = set(previous)
    for draw in range(4):
        guard()
        draw_seed = int(sha256_json([seed, draw])[:16], 16) % (2**63)
        for task in generate_tasks(draw_seed, per_family=48):
            key = (task["split"], task["family"])
            root = task["lineage_id"]
            if key in groups and root not in seen and len(groups[key]) < COHORT_COUNTS[key[0]]:
                groups[key].append(task)
                seen.add(root)
        if all(len(rows) == COHORT_COUNTS[split] for (split, _), rows in groups.items()):
            return [task for group in groups.values() for task in group]
    shortage = [f"{split}/{family}={len(rows)}" for (split, family), rows in groups.items()
                if len(rows) != COHORT_COUNTS[split]]
    raise ValueError(f"insufficient fresh roots: {', '.join(shortage)}")


def run(campaign, out, guard, *, cohort=None):
    manifest = load_structured(campaign / "terminal" / "manifest.json")
    weights_path = campaign / "terminal" / "models" / "seed-11-rlt.json"
    model = NativeRanker.load(weights_path)
    if sha256_file(weights_path) != manifest["models"][model.model_digest]["sha256"]:
        raise ValueError("proposal checkpoint differs from the replayed campaign")
    task_index, prior = {}, set()
    for path in load_structured(campaign / "source-spec.json")["suites"]:
        guard()
        _, tasks = load_suite(Path(path))
        task_index.update((task["task_digest"], task) for task in tasks)
        prior.update(task["lineage_id"] for task in tasks)
    out.mkdir(parents=True, exist_ok=False)
    seed = secrets.randbits(63) if cohort is None else None
    dump_structured(out / "plan.json", {"seed": seed, "model": model.model_digest, "runtime": "rlt",
                    "fixed_depths": [4, DEPTH], "adaptive_depth": DEPTH, "repair_check_budget": CHECK_BUDGET,
                    "frozen_cohort_sha256": sha256_file(cohort) if cohort is not None else None,
                    "thresholds": [-0.15, 0.1, 0.35, 0.6], "repair_margins": [0.1, 0.35],
                    "selection": "highest calibration coverage, then fewer verifier calls and visits"})
    fresh = freeze_fresh(prior, seed, guard) if cohort is None else load_structured(cohort)["tasks"]
    fresh_roots = {task["lineage_id"] for task in fresh}
    counts = Counter((task["split"], task["family"]) for task in fresh)
    expected = {(split, family): count for split, count in COHORT_COUNTS.items() for family in FAMILIES}
    if fresh_roots & prior or len(fresh_roots) != len(fresh) or counts != expected:
        raise ValueError("fresh cohort has repeated roots or differs from the registered split balance")
    dump_structured(out / "fresh-tasks.json", {"tasks": fresh})
    w, train, roots = fit(campaign / "aleph-pack", task_index, guard)
    if roots & fresh_roots:
        raise ValueError("fresh evaluation overlaps proxy fitting")
    dump_structured(out / "proxy.json", {"weights": w.tolist(), "training": train})
    print(f"Fitted {WIDTH} weights on {train['train_roots']} verified training roots", flush=True)

    calibration = [task for task in fresh if task["split"] == "calibration"]
    cached = [(task, list(proposals(model, task))) for task in calibration]
    sweep = []
    for threshold in (-0.15, 0.1, 0.35, 0.6):
        for margin in (0.1, 0.35):
            guard()
            rows = [rollout(task, steps, "learned", weights=w, threshold=threshold, margin=margin)
                    for task, steps in cached]
            sweep.append({"threshold": threshold, "margin": margin, **summarize(rows)})
    selected = max(sweep, key=lambda row: (row["solved"], -row["verifier_calls"], -row["visits"]))
    dump_structured(out / "calibration.json", {"selection": selected, "sweep": sweep})
    print(f"Selected confidence {selected['threshold']} and repair margin {selected['margin']}", flush=True)

    held = [task for task in fresh if task["split"] != "calibration"]
    eval_groups = defaultdict(list)
    replays = 0
    with (out / "held-rollouts.jsonl").open("x", encoding="utf-8", newline="\n") as trace:
        for task in held:
            guard()
            for mode in POLICIES:
                started = time.perf_counter()
                depth = 4 if mode == "fixed_4" else DEPTH
                row = rollout(task, proposals(model, task, depth), mode, weights=w,
                              threshold=selected["threshold"], margin=selected["margin"])
                row["wall_seconds"] = time.perf_counter() - started
                for check in row["checks"]:
                    if verify(task, check["candidate"]) != check["receipt"]:
                        raise ValueError("held-out verifier receipt mismatch")
                    replays += 1
                trace.write(canonical_json(row) + "\n")
                eval_groups[task["split"], task["family"], mode].append(row)
    per_policy = {
        mode: summarize([row for (_, _, policy), rows in eval_groups.items() if policy == mode for row in rows])
        for mode in POLICIES
    }
    pairs = {}
    learned = {row["root"]: row for key, rows in eval_groups.items() if key[2] == "learned" for row in rows}
    for mode in POLICIES[:-1]:
        compared = [row for key, rows in eval_groups.items() if key[2] == mode for row in rows]
        pairs[mode] = {"learned_only": sum(learned[r["root"]]["solved"] and not r["solved"] for r in compared),
                       "baseline_only": sum(r["solved"] and not learned[r["root"]]["solved"] for r in compared)}
    report = {"schema": "rlt_gym.aleph_proxy.v1", "training": train, "calibration": selected,
              "training_checks": train["verifier_rechecks"],
              "calibration_checks": sum(r["verifier_calls"] for r in sweep),
              "holdout_roots": len(held), "receipt_replays": replays, "policies": per_policy, "paired": pairs,
              "cells": [{"split": s, "family": f, "policy": mode, **summarize(rows)}
                        for (s, f, mode), rows in sorted(eval_groups.items())],
              "artifacts": {name: sha256_file(out / name)
                            for name in ("plan.json", "fresh-tasks.json", "proxy.json", "calibration.json",
                                         "held-rollouts.jsonl")}}
    dump_structured(out / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.campaign, args.out, lambda: None)
    print(canonical_json(result["policies"]))


if __name__ == "__main__":
    main()
