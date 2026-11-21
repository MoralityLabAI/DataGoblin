"""Verifier-grounded value-of-another-check controller for DataGoblin's seven gyms.

Uses source-root paired FULL fixed-depth RLT trajectories. The learned policy
observes the task, its failed current candidate and checker code, then decides
whether another recurrent visit and independent verification are worth paying
for. This is offline counterfactual replay, not measured inference-time savings.

A future Lean/MCP/hypergame adapter must supply its own independent verifier.
No foreign or historical-only certificates are promoted by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .aleph_proxy import WIDTH, features
from .runner import read_jsonl, replay_run
from .tasks import FAMILIES, load_suite, public_task

SCHEMA = "rlt_gym.postfailure_exit.v1"
FAMILIES_SET = frozenset(FAMILIES)
PUBLIC_SPLITS = ("train", "calibration", "ordinary_test", "ood_test")
EXCLUDED_SPLITS = frozenset({"sealed_poison"})
MAX_ROOTS = 20000


def _first_valid(episode: dict[str, Any]) -> int | None:
    return next(
        (i for i, step in enumerate(episode["steps"]) if step["verification"]["passed"]), None
    )


def _case(task: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    if (
        episode["config"]["verified_stop"] is not False
        or episode["outcome"]["censored"] is not False
        or episode["backend"]["kind"] != "frozen_random_reservoir"
    ):
        raise ValueError("requires uncensored, continuous fixed-depth RLT rollout")
    steps = episode["steps"]
    if len(steps) < 2:
        raise ValueError("at least two fixed-depth visits required")
    first = _first_valid(episode)
    if len(steps) != episode["config"]["planned_depth"]:
        raise ValueError("fixed-depth run did not finish")
    # Only information already available AFTER verifying the first candidate
    # can enter the post-failure decision. Later check results are labels only.
    fail = steps[0]
    if fail["verification"]["passed"]:
        observed = None
    else:
        observed = features(public_task(task), fail["candidate"], [fail["candidate"]])
        # The first verifier failure code is already public when the post-check
        # controller decides; hashed features never include future receipts.
        for signal in (
            f"known_check_code:{fail['verification']['code']}",
            f"remaining_budget:{min(len(steps) - 1, 16)}",
        ):
            digest = hashlib.blake2s(signal.encode(), digest_size=4).digest()
            observed[1 + int.from_bytes(digest, "little") % (WIDTH - 1)] += 1.0
        if observed.shape != (WIDTH,) or not np.isfinite(observed).all():
            raise ValueError("bad pre-decision feature vector")
    return {
        "root_id": task["lineage_id"], "task_digest": task["task_digest"],
        "family": task["family"], "split": task["split"],
        "model_digest": episode["backend"]["model_digest"],
        "verification_code": fail["verification"]["code"],
        "visit_budget": len(steps), "first_passed": first == 0,
        "future_rescue": first is not None and first > 0,
        "first_valid_visit_zero_based": first,
        "postfailure_features": observed.tolist() if observed is not None else None,
        "oracle_full_depth_checks": 1 + (first if first is not None else len(steps) - 1),
        "evidence_class": "verified_native_reservoir_development_control",
        "trainable": task["split"] == "train",
    }


def extract_cases(
    suite: Path, run: Path, *, seed: int = 7,
    mode: str = "direct", training_mode: str = "direct", planned_depth: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently replay original models and every candidate before labeling."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be uint32")
    if mode not in {"direct", "rlt", "matched_damping"} or training_mode not in {
        "direct", "rlt", "matched_damping",
    }:
        raise ValueError("unregistered schedule")
    if type(planned_depth) is not int or not 2 <= planned_depth <= 16:
        raise ValueError("planned depth must be 2..16")
    run_manifest = load_structured(run / "manifest.json")
    if EXCLUDED_SPLITS & set(run_manifest.get("configuration", {}).get("splits", [])):
        raise ValueError("this research controller cannot access sealed_poison")
    replay = replay_run(suite, run)
    if replay.get("passed") is not True or replay.get("receipt_only_aborted_episodes"):
        raise ValueError("incomplete independent rollout replay")
    _, tasks = load_suite(suite)
    lookup = {task["task_id"]: task for task in tasks}
    cases: list[dict[str, Any]] = []
    observed = set()
    for episode in read_jsonl(run / "rollouts.jsonl"):
        cfg, backend = episode["config"], episode["backend"]
        if (
            backend["kind"] != "frozen_random_reservoir"
            or backend.get("training", {}).get("training_mode") != training_mode
            or cfg["seed"] != seed or cfg["mode"] != mode
            or cfg["planned_depth"] != planned_depth
        ):
            continue
        if episode["split"] not in PUBLIC_SPLITS:
            raise ValueError("unexpected split in selected fixed-depth matrix")
        task = lookup[episode["task_id"]]
        c = _case(task, episode)
        identity = (c["split"], c["root_id"])
        if identity in observed:
            raise ValueError("multiple model presentations for one source root")
        observed.add(identity)
        cases.append(c)
    if len(cases) > MAX_ROOTS or not cases:
        raise ValueError("missing or oversized source-root set")
    counts = Counter(c["split"] for c in cases)
    if any(counts.get(split, 0) == 0 for split in PUBLIC_SPLITS):
        raise ValueError("need full train/calibration/ordinary/OOD source custody")
    if len({c["root_id"] for c in cases}) != len(cases):
        raise ValueError("root reused across splits")
    provenance = {
        "schema_version": SCHEMA,
        "suite_manifest_sha256": sha256_file(suite / "manifest.json" if suite.is_dir() else suite),
        "run_manifest_sha256": sha256_file(run / "manifest.json"),
        "run_replay_digest": sha256_json(replay),
        "selected": {"seed": seed, "mode": mode, "training_mode": training_mode,
                     "planned_depth": planned_depth},
        "split_counts": dict(sorted(counts.items())),
        "source_roots": len(cases),
        "candidate_check_replay": replay["independently_rechecked_candidates"],
    }
    return cases, provenance


def _design(cases: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    failed = [c for c in cases if not c["first_passed"]]
    if not failed:
        raise ValueError("no initial failures to learn a continuation rule")
    x = np.array([c["postfailure_features"] for c in failed], dtype=np.float64)
    y = np.array([float(c["future_rescue"]) for c in failed], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != WIDTH or not np.isfinite(x).all():
        raise ValueError("incompatible observation features")
    return x, y, np.array([c["root_id"] for c in failed])


def fit(cases: list[dict[str, Any]], *, ridge: float = 12.0) -> dict[str, Any]:
    """Fit only TRAIN roots with a shrinkage-ridge continuation score."""
    if not cases or any(c["split"] != "train" for c in cases):
        raise ValueError("TRAIN split required for fitted controller")
    if type(ridge) not in (int, float) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    x, y, roots = _design(cases)
    if len(set(roots)) != len(roots):
        raise ValueError("duplicate source-root training presentations")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1.0)
    xx = (x - mean) / scale
    prior = float(y.mean())
    # Ridge regularization avoids singularity when train roots are fewer than
    # 192 dimensions. Zero classes are allowed, with a constant predictor.
    weights = np.linalg.solve(
        xx.T @ xx + ridge * np.eye(WIDTH), xx.T @ (y - prior),
    )
    return {
        "schema_version": "rlt_gym.postfailure_ridge.v1",
        "feature_size": WIDTH, "mean": mean.tolist(), "scale": scale.tolist(),
        "weights": weights.tolist(), "bias": prior,
        "train_failure_roots": len(roots), "train_rescues": int(y.sum()),
        "ridge": ridge, "source": "train_only_verified_future_rescue",
        "measured_neural_recurrent_training": False,
    }


def predict(case: dict[str, Any], model: dict[str, Any]) -> float:
    if case["first_passed"]:
        return 1.0
    if model.get("schema_version") != "rlt_gym.postfailure_ridge.v1":
        raise ValueError("unknown controller model")
    values = np.array(case["postfailure_features"], dtype=np.float64)
    mu = np.array(model["mean"], dtype=np.float64)
    scale = np.array(model["scale"], dtype=np.float64)
    weight = np.array(model["weights"], dtype=np.float64)
    if any(v.shape != (WIDTH,) for v in (values, mu, scale, weight)):
        raise ValueError("controller features do not match training")
    if not np.isfinite(values).all():
        raise ValueError("controller saw nonfinite values")
    return float(np.clip(((values - mu) / scale) @ weight + model["bias"], 0.0, 1.0))


def _policy(
    cases: list[dict[str, Any]], continuation: dict[str, bool],
) -> dict[str, Any]:
    solved = calls = rescued = missed = 0
    per_family = defaultdict(lambda: [0, 0, 0])
    for c in cases:
        root = c["root_id"]
        if root not in continuation:
            raise ValueError("missing continuation decision")
        wants_more = continuation[root]
        if c["first_passed"]:
            spent, passes = 1, True
        elif wants_more:
            spent = c["oracle_full_depth_checks"]
            passes = bool(c["future_rescue"])
            rescued += int(passes)
        else:
            spent, passes = 1, False
            missed += int(c["future_rescue"])
        calls += spent
        solved += int(passes)
        summary = per_family[c["family"]]
        summary[0] += 1
        summary[1] += int(passes)
        summary[2] += spent
    return {
        "solved": solved, "roots": len(cases), "verifier_calls": calls,
        "rescued_after_first_failure": rescued, "missed_future_rescues": missed,
        "by_family": {
            name: {"roots": row[0], "solved": row[1], "verifier_calls": row[2]}
            for name, row in sorted(per_family.items())
        },
    }


def choose_threshold(
    cases: list[dict[str, Any]], model: dict[str, Any], *,
    missed_rescue_allowance: int = 0,
) -> dict[str, Any]:
    if not cases or any(c["split"] != "calibration" for c in cases):
        raise ValueError("frozen calibration roots required")
    if type(missed_rescue_allowance) is not int or not 0 <= missed_rescue_allowance <= 10:
        raise ValueError("registered missed rescue allowance out of bounds")
    scores = {c["root_id"]: predict(c, model) for c in cases}
    choices = sorted({-0.001, 1.001, *scores.values()})
    feasible = []
    for threshold in choices:
        decisions = {
            c["root_id"]: c["first_passed"] or scores[c["root_id"]] >= threshold for c in cases
        }
        result = _policy(cases, decisions)
        if result["missed_future_rescues"] <= missed_rescue_allowance:
            feasible.append((result["verifier_calls"], -result["solved"], -threshold, threshold, result))
    if not feasible:
        raise ValueError("no policy within calibration rescue budget")
    selected = min(feasible)
    baseline = _policy(cases, {c["root_id"]: True for c in cases})
    continuation_rate = sum(
        not c["first_passed"] and scores[c["root_id"]] >= selected[3] for c in cases
    ) / max(1, sum(not c["first_passed"] for c in cases))
    return {
        "schema_version": "rlt_gym.postfailure_calibration.v1",
        "threshold": selected[3], "missed_rescue_allowance": missed_rescue_allowance,
        "calibration_policy": selected[4], "calibration_first_valid": baseline,
        "random_continuation_probability_among_failures": continuation_rate,
        "calibration_roots": len(cases),
        "scientific_confirmation": False,
    }


def evaluate(
    cases: list[dict[str, Any]], model: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    if not cases or any(c["split"] not in {"ordinary_test", "ood_test"} for c in cases):
        raise ValueError("only untouched ordinary/OOD development root families allowed")
    scores = {c["root_id"]: predict(c, model) for c in cases}
    threshold = selection["threshold"]
    p_random = selection["random_continuation_probability_among_failures"]
    random_decisions = {
        c["root_id"]: c["first_passed"] or (
            random.Random(int(hashlib.sha256(
                f"rlt_postfailure_random:{c['root_id']}".encode()
            ).hexdigest()[:16], 16)).random() < p_random
        ) for c in cases
    }
    learned = {
        c["root_id"]: c["first_passed"] or scores[c["root_id"]] >= threshold
        for c in cases
    }
    always = {c["root_id"]: True for c in cases}
    never = {c["root_id"]: False for c in cases}
    return {
        "schema_version": "rlt_gym.postfailure_evaluation.v1",
        "evidence_class": "offline_verified_prefix_replay",
        "roots": len(cases),
        "model_is_reexecuted_with_dynamic_stop": False,
        "policies": {
            "first_valid": _policy(cases, always),
            "fixed_one": _policy(cases, never),
            "learned_continue": _policy(cases, learned),
            "calibration_rate_random": _policy(cases, random_decisions),
        },
        "decision_threshold": threshold,
        "no_label_lookahead_in_inference_features": True,
    }


def campaign(
    suite: Path, run: Path, out: Path, *, seed: int = 7,
    mode: str = "direct", training_mode: str = "direct",
    planned_depth: int = 4, max_missed_calibration_rescues: int = 0,
) -> dict[str, Any]:
    cases, provenance = extract_cases(
        suite, run, seed=seed, mode=mode, training_mode=training_mode,
        planned_depth=planned_depth,
    )
    by_split = {s: [c for c in cases if c["split"] == s] for s in PUBLIC_SPLITS}
    model = fit(by_split["train"])
    selected = choose_threshold(
        by_split["calibration"], model,
        missed_rescue_allowance=max_missed_calibration_rescues,
    )
    observed = {
        s: evaluate(by_split[s], model, selected) for s in ("ordinary_test", "ood_test")
    }
    if out.exists():
        raise ValueError("output directory exists")
    out.mkdir(parents=True)
    model_path = out / "controller.json"
    select_path = out / "selection.json"
    model_path.write_text(canonical_json(model) + "\n", encoding="utf-8")
    select_path.write_text(canonical_json(selected) + "\n", encoding="utf-8")
    counts = Counter((c["split"], c["family"]) for c in cases)
    report = {
        "schema_version": SCHEMA, "evidence_class": "source_replayed_CPU_proxy_development",
        "provenance": provenance, "selection": selected, "evaluations": observed,
        "root_family_counts": {
            f"{s}:{family}": counts[(s, family)] for s in PUBLIC_SPLITS for family in FAMILIES
        },
        "files_sha256": {
            "controller.json": sha256_file(model_path),
            "selection.json": sha256_file(select_path),
        },
        "claim_boundary": (
            "The RLT reservoir produced all candidate trajectories ahead of time. "
            "Savings are offline counterfactual counts of independent verifier calls, "
            "not measured native recurrent execution. The controller fits post-failure "
            "continuation on TRAIN and uses CAL for threshold selection; ordinary/OOD "
            "roots are descriptive development confirmation without sealed testing."
        ),
        "lean_mcp_transfer_status": "awaiting_verifier_backed_independent_source_adapters",
    }
    dump_structured(out / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=("direct", "rlt", "matched_damping"), default="direct")
    parser.add_argument("--training-mode", choices=("direct", "rlt", "matched_damping"), default="direct")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-missed-calibration-rescues", type=int, default=0)
    args = parser.parse_args()
    result = campaign(
        args.suite, args.run, args.out, seed=args.seed, mode=args.mode,
        training_mode=args.training_mode, planned_depth=args.depth,
        max_missed_calibration_rescues=args.max_missed_calibration_rescues,
    )
    print(json.dumps({
        "out": str(args.out),
        "source_roots": result["provenance"]["source_roots"],
        "evaluation": result["evaluations"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
