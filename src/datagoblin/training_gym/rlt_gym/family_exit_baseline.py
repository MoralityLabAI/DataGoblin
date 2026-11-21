"""Family-conditional post-failure continuation baseline with separate CAL selection.

This is a low-capacity control for the 192D ridge stop head. It uses only the
source task family and TRAIN future-rescue frequency after an initial failure.
Future checker outcomes remain labels, never inference observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured

from .postfailure_exit import FAMILIES, PUBLIC_SPLITS, _policy, extract_cases

SCHEMA = "rlt_gym.family_postfailure.v1"
ALPHA = 1.0
BETA = 9.0


def fit_family(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases or any(row["split"] != "train" for row in cases):
        raise ValueError("family rescue probabilities require TRAIN roots only")
    roots = [row["root_id"] for row in cases]
    if len(roots) != len(set(roots)):
        raise ValueError("duplicate family-training source roots")
    counts: Counter[str] = Counter()
    rescued: Counter[str] = Counter()
    for row in cases:
        family = row["family"]
        if family not in FAMILIES:
            raise ValueError("unregistered task family")
        if not row["first_passed"]:
            counts[family] += 1
            rescued[family] += int(row["future_rescue"])
    probabilities = {
        f: (rescued[f] + ALPHA) / (counts[f] + ALPHA + BETA)
        for f in FAMILIES
    }
    return {
        "schema_version": "rlt_gym.family_rescue_prior.v1",
        "alpha": ALPHA, "beta": BETA,
        "family_probabilities": probabilities,
        "family_failures": {f: counts[f] for f in FAMILIES},
        "family_future_rescues": {f: rescued[f] for f in FAMILIES},
        "train_roots": len(cases),
        "training_inputs": "public_family_only_after_first_failed_verifier",
    }


def score(row: dict[str, Any], model: dict[str, Any]) -> float:
    if model.get("schema_version") != "rlt_gym.family_rescue_prior.v1":
        raise ValueError("unrecognized fitted family controller")
    family = row["family"]
    if family not in FAMILIES:
        raise ValueError("unknown family")
    probability = model["family_probabilities"][family]
    if type(probability) not in (float, int) or not math.isfinite(probability):
        raise ValueError("nonfinite family rescue estimate")
    return float(probability)


def choose_family_threshold(
    cases: list[dict[str, Any]], model: dict[str, Any], *,
    missed_rescue_allowance: int = 0,
) -> dict[str, Any]:
    if not cases or any(row["split"] != "calibration" for row in cases):
        raise ValueError("family threshold requires disjoint CALIBRATION source roots")
    if type(missed_rescue_allowance) is not int or not 0 <= missed_rescue_allowance <= 10:
        raise ValueError("invalid missed-rescue allowance")
    estimates = {row["root_id"]: score(row, model) for row in cases}
    thresholds = sorted({-0.001, 1.001, *estimates.values()})
    chosen: list[tuple[int, int, float, dict[str, Any]]] = []
    for threshold in thresholds:
        decisions = {
            row["root_id"]: row["first_passed"] or estimates[row["root_id"]] >= threshold
            for row in cases
        }
        values = _policy(cases, decisions)
        if values["missed_future_rescues"] <= missed_rescue_allowance:
            chosen.append((values["verifier_calls"], -values["solved"], -threshold, values))
    if not chosen:
        raise ValueError("no family policy fits CAL rescue budget")
    optimum = min(chosen, key=lambda row: row[:3])
    selected_threshold = -optimum[2]
    failed = [row for row in cases if not row["first_passed"]]
    continuation_rate = (
        sum(score(row, model) >= selected_threshold for row in failed)
        / max(1, len(failed))
    )
    return {
        "schema_version": "rlt_gym.family_threshold_calibration.v1",
        "threshold": selected_threshold,
        "calibration_policy": optimum[3],
        "calibration_first_valid": _policy(cases, {row["root_id"]: True for row in cases}),
        "missed_rescue_allowance": missed_rescue_allowance,
        "random_continuation_probability_among_failures": continuation_rate,
        "calibration_roots": len(cases),
        "verification_source": "CAL only; evaluation roots remain unobserved",
    }


def evaluate_family(
    cases: list[dict[str, Any]], model: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    if not cases or any(row["split"] not in {"ordinary_test", "ood_test"} for row in cases):
        raise ValueError("family policy only scores untouched ordinary/OOD roots")
    cutoff = selection["threshold"]
    probability = selection["random_continuation_probability_among_failures"]
    learned = {
        row["root_id"]: row["first_passed"] or score(row, model) >= cutoff
        for row in cases
    }
    random_controls = {
        row["root_id"]: row["first_passed"] or (
            random.Random(int(hashlib.sha256(
                f"family_rate_random:{row['root_id']}".encode()
            ).hexdigest()[:16], 16)).random() < probability
        )
        for row in cases
    }
    return {
        "schema_version": "rlt_gym.family_postfailure_evaluation.v1",
        "split": cases[0]["split"], "roots": len(cases),
        "evidence_class": "offline_root_paired_fixed_depth_replay",
        "policies": {
            "fixed_one": _policy(cases, {row["root_id"]: False for row in cases}),
            "first_valid": _policy(cases, {row["root_id"]: True for row in cases}),
            "family_continue": _policy(cases, learned),
            "family_rate_random": _policy(cases, random_controls),
        },
        "future_verifier_lookahead_for_selection": False,
        "measured_runtime_savings": False,
    }


def run(
    suite: Path, source: Path, out: Path, *, seed: int = 7,
    planned_depth: int = 4, missed_rescue_allowance: int = 0,
) -> dict[str, Any]:
    cases, provenance = extract_cases(
        suite, source, seed=seed, mode="direct",
        training_mode="direct", planned_depth=planned_depth,
    )
    by_split = {
        split: [item for item in cases if item["split"] == split]
        for split in PUBLIC_SPLITS
    }
    model = fit_family(by_split["train"])
    selection = choose_family_threshold(
        by_split["calibration"], model, missed_rescue_allowance=missed_rescue_allowance,
    )
    results = {
        split: evaluate_family(by_split[split], model, selection)
        for split in ("ordinary_test", "ood_test")
    }
    if out.exists():
        raise ValueError("family control output directory exists")
    out.mkdir(parents=True)
    for filename, value in (("model.json", model), ("calibration.json", selection)):
        (out / filename).write_text(canonical_json(value) + "\n", encoding="utf-8")
    report = {
        "schema_version": SCHEMA,
        "provenance": provenance, "model": model, "selection": selection,
        "evaluation": results,
        "files_sha256": {
            name: sha256_file(out / name) for name in ("model.json", "calibration.json")
        },
        "claim_boundary": (
            "Family-only TRAIN prior, CAL stopping threshold and disjoint development "
            "evaluation. All model trajectories were generated at fixed depth beforehand. "
            "Verifier calls are offline prefix counts, not online inference time."
        ),
    }
    dump_structured(out / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--missed-rescue-allowance", type=int, default=0)
    args = parser.parse_args()
    report = run(
        args.suite, args.source, args.out, seed=args.seed,
        planned_depth=args.depth, missed_rescue_allowance=args.missed_rescue_allowance,
    )
    print(json.dumps({
        "out": str(args.out), "family_probabilities": report["model"]["family_probabilities"],
        "evaluation": {
            split: {
                p: (r["solved"], r["verifier_calls"])
                for p, r in panel["policies"].items()
            } for split, panel in report["evaluation"].items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
