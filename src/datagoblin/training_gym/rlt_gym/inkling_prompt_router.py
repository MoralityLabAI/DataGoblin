"""Distill the trace-trained RLT depth selector to public prompt features."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured

from .inkling import _jsonl, _manifest, _new_dir
from .inkling_holdout import SPLITS, _metrics, _policy, _prelude, audit_holdout_plan
from .inkling_routing import routing_strength
from .inkling_sae import _encode
from .inkling_trace import WIDTH
from .runner import read_jsonl

FEATURE_WIDTH = 256
WORDS = re.compile(r"[a-z_]+|[0-9]+|[^\s\w]")
STUDENT_SCHEMA = "rlt_gym.inkling_prompt_router_fit.v1"
ROUTES_SCHEMA = "rlt_gym.inkling_prompt_router_choices.v1"


def _features(path: Path, digest: str) -> np.ndarray:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("public prompt changed after staging")
    words = ["<number>" if word.isdigit() else word for word in WORDS.findall(data.decode("utf-8").lower())]
    terms = [*words, *(left + " " + right for left, right in pairwise(words))]
    counts = np.zeros(FEATURE_WIDTH, dtype=np.float64)
    for term in terms:
        hashed = hashlib.blake2b(term.encode("utf-8"), digest_size=8, person=b"inkling-rlt").digest()
        counts[int.from_bytes(hashed, "little") % FEATURE_WIDTH] += 1
    counts = np.log1p(counts)
    return np.r_[counts / max(np.linalg.norm(counts), 1), np.log1p(len(words))]


def fit_prompt_router(suite: Path, holdout: Path, reference: Path, fit_prompts: Path, out: Path) -> dict:
    plan, _, _ = audit_holdout_plan(suite, holdout)
    sealed, teacher, dictionary, heads = _policy(plan, reference, fit_prompts)
    trace = reference / "p03-raw"
    if (sha256_file(trace / "rows.tsv") != sealed["files"]["rows.tsv"]
            or (trace / "activations.f32").stat().st_size != sealed["rows"] * WIDTH * 4):
        raise ValueError("the teacher trace differs from its sealed index or shape")
    samples = _manifest(fit_prompts / "manifest.json")["samples"]
    train = [row for row in samples if row["split"] == "train"]
    calibration = [row for row in samples if row["split"] == "calibration"]
    roots = {row["lineage_id"] for row in samples}
    if min(len(train), len(calibration)) < 4 or len(roots) != len(samples):
        raise ValueError("the prompt router needs disjoint training and calibration roots")
    ordered = train + calibration
    state = _prelude(trace, [
        {**row, "prompt_tokens": sealed["tokens_by_sample"][row["sample_key"]]} for row in ordered
    ])
    with np.load(dictionary, allow_pickle=False) as w:
        latent = _encode(np.stack([state[row["sample_key"]] for row in ordered]) - w["pre_bias"],
                         w["encoder_weight"], w["encoder_bias"], int(w["top_k"]))
    with np.load(heads, allow_pickle=False) as w:
        target = routing_strength(latent, w["rlt"], "rlt")

    features = np.stack([
        _features(fit_prompts / (row["sample_key"] + ".txt"), row["input_sha256"]) for row in ordered
    ])
    size = len(train)
    x_train, x_cal = features[:size], features[size:]
    t_train, t_cal = target[:size], target[size:]
    center, intercept = x_train.mean(axis=0), float(t_train.mean())
    centered = x_train - center
    gram = centered @ centered.T
    options = {}
    for ridge in (0.01, 0.1, 1.0, 10.0):
        weight = centered.T @ np.linalg.solve(gram + ridge * np.eye(size), t_train - intercept)
        error = (x_cal - center) @ weight + intercept - t_cal
        options[ridge] = (weight, float(np.mean(error ** 2)))
    ridge = min(options, key=lambda value: (options[value][1], -value))
    _new_dir(out)
    model = out / "student.npz"
    np.savez(model, weight=options[ridge][0], center=center, intercept=intercept,
             threshold=teacher["fitted"]["rlt"]["threshold"])
    result = {
        "schema_version": STUDENT_SCHEMA, "suite_sha256": plan["suite_sha256"],
        "teacher_policy_sha256": plan["frozen_depth_policy_sha256"], "teacher_trace_sha256": plan["prior_trace_sha256"],
        "selected_prelude_sha256": {key: hashlib.sha256(vector).hexdigest() for key, vector in state.items()},
        "train_roots": [row["lineage_id"] for row in train],
        "calibration_roots": [row["lineage_id"] for row in calibration],
        "ridge": ridge, "calibration_mse": {str(key): value[1] for key, value in options.items()},
        "student_sha256": sha256_file(model),
    }
    dump_structured(out / "manifest.json", result)
    return result


def stage_prompt_routes(suite: Path, holdout: Path, student: Path, out: Path) -> dict:
    plan, _, _ = audit_holdout_plan(suite, holdout)
    fit_path, model = student / "manifest.json", student / "student.npz"
    fit = _manifest(fit_path)
    seen = set(fit["train_roots"]) | set(fit["calibration_roots"])
    if (fit["schema_version"] != STUDENT_SCHEMA or fit["suite_sha256"] != plan["suite_sha256"]
            or fit["teacher_policy_sha256"] != plan["frozen_depth_policy_sha256"]
            or fit["teacher_trace_sha256"] != plan["prior_trace_sha256"]
            or fit["student_sha256"] != sha256_file(model)
            or seen.intersection(row["lineage_id"] for row in plan["rows"])):
        raise ValueError("invalid fitted prompt router or a reused held-out root")
    with np.load(model, allow_pickle=False) as w:
        weight, center = w["weight"], w["center"]
        intercept, threshold = float(w["intercept"]), float(w["threshold"])
    if (weight.shape != center.shape or weight.shape != (FEATURE_WIDTH + 1,)
            or not np.isfinite(weight).all() or not np.isfinite(center).all()
            or not np.isfinite([intercept, threshold]).all()):
        raise ValueError("invalid prompt router weights")
    rows = []
    start = time.perf_counter_ns()
    for row in plan["rows"]:
        path = holdout / row["batch"] / (row["sample_key"] + ".txt")
        estimate = float((_features(path, row["prompt_sha256"]) - center) @ weight + intercept)
        rows.append({**row, "chosen_depth": 3 if estimate >= threshold else 1, "estimated_strength": estimate})
    latency_ms = (time.perf_counter_ns() - start) / 1e6
    _new_dir(out)
    choices = out / "choices.jsonl"
    _jsonl(choices, rows)
    result = {
        "schema_version": ROUTES_SCHEMA, "holdout_plan_sha256": sha256_file(holdout / "manifest.json"),
        "fit_manifest_sha256": sha256_file(fit_path), "student_sha256": fit["student_sha256"],
        "choices_sha256": sha256_file(choices), "planned_roots": len(rows),
        "three_pass_roots": sum(row["chosen_depth"] == 3 for row in rows), "selection_wall_ms": latency_ms,
    }
    dump_structured(out / "manifest.json", result)
    return result


def replay_prompt_routes(routes: Path, audit: Path, out: Path) -> dict:
    plan_path, audit_path = routes / "manifest.json", audit / "manifest.json"
    plan, report = _manifest(plan_path), _manifest(audit_path)
    choices_path, outcome_path = routes / "choices.jsonl", audit / "outcomes.jsonl"
    if (plan["schema_version"] != ROUTES_SCHEMA or plan["holdout_plan_sha256"] != report["holdout_plan_sha256"]
            or plan["choices_sha256"] != sha256_file(choices_path)
            or report["files"]["outcomes.jsonl"] != sha256_file(outcome_path)):
        raise ValueError("prompt routes and audited native outcomes do not share the frozen plan")
    choices = {row["lineage_id"]: row for row in read_jsonl(choices_path)}
    if len(choices) != plan["planned_roots"] or any(row["chosen_depth"] not in (1, 3) for row in choices.values()):
        raise ValueError("invalid frozen prompt choices")
    grouped = defaultdict(dict)
    for row in read_jsonl(outcome_path):
        choice = choices[row["lineage_id"]]
        if any(choice[key] != row[key] for key in ("task_digest", "prompt_sha256", "split")):
            raise ValueError("prompt choice and verified outcome disagree on the root")
        group = grouped[row["lineage_id"]]
        if row["passes"] in group:
            raise ValueError("duplicate verified depth")
        group[row["passes"]] = row
    results = {}
    for split in (*SPLITS, "all"):
        paired = [(key, group) for key, group in grouped.items()
                  if 1 in group and 3 in group and (split == "all" or choices[key]["split"] == split)]
        selected = [group[choices[key]["chosen_depth"]] for key, group in paired]
        fixed = [group[3] for _, group in paired]
        results[split] = {"student": _metrics(selected, selected), "fixed_3": _metrics(fixed, fixed)}
    _new_dir(out)
    result = {
        "schema_version": "rlt_gym.inkling_prompt_router_replay.v1",
        "scope": "prompt_only_choices_replayed_on_paired_verified_native_outcomes",
        "choices_manifest_sha256": sha256_file(plan_path), "outcome_manifest_sha256": sha256_file(audit_path),
        "results": results,
    }
    dump_structured(out / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    for name, handler, parameters in (
        ("fit", fit_prompt_router, ("suite", "holdout", "reference", "fit-prompts", "out")),
        ("stage", stage_prompt_routes, ("suite", "holdout", "student", "out")),
        ("replay", replay_prompt_routes, ("routes", "audit", "out")),
    ):
        subcommand = commands.add_parser(name)
        subcommand.set_defaults(handler=handler)
        for parameter in parameters:
            subcommand.add_argument(f"--{parameter}", type=Path, required=True)
    args = vars(parser.parse_args())
    handler = args.pop("handler")
    print(json.dumps(handler(**args), sort_keys=True))


if __name__ == "__main__":
    main()
