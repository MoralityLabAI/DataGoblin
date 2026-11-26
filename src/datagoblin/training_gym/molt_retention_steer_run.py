"""Run the fixed MOLT-05 development panel; preserve every paired outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from datagoblin.training_gym.molt_goblin import digest
from datagoblin.training_gym.molt_retention import VARIANTS, episode, make_world
from datagoblin.training_gym.molt_retention_steer import (
    POLICIES,
    TOL,
    choose_components,
    factorize,
    interventions,
)
from datagoblin.training_gym.molt_retention_train import features, fit

ACTIVE = ("targeted", "equal_rank", "norm_sham", "jacobian_radius_sham")


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["variant"], row["condition"], row["policy"])].append(row)
    summaries = []
    for (split, variant, condition, policy), group in sorted(groups.items()):
        eligible = [r for r in group if r["intact_correct"]]
        broken = [r for r in eligible if not r["corrupt_correct"]]
        summaries.append({
            "split": split, "variant": variant, "condition": condition, "policy": policy,
            "rows": len(group), "world_roots": len({r["root_id"] for r in group}),
            "model_roots": len({r["training_seed"] for r in group}),
            "verified": sum(r["verified"] for r in group),
            "success": sum(r["verified"] for r in group) / len(group),
            "intact_eligible": len(eligible), "headroom": len(broken),
            "recovered": sum(r["verified"] for r in broken) if condition == "corrupt" else None,
            "clean_damaged": sum(not r["verified"] for r in eligible) if condition == "clean" else None,
            "mean_absolute_logit_error": sum(r["mean_abs_logit_error"] for r in group) / len(group),
        })
    return summaries


def contrasts(rows):
    index = {(r["root_id"], r["training_seed"], r["variant"], r["condition"], r["policy"]): r
             for r in rows}
    groups = defaultdict(list)
    for r in rows:
        if r["condition"] != "corrupt" or r["policy"] != "targeted":
            continue
        key = (r["root_id"], r["training_seed"], r["variant"])
        clean = index[(*key, "clean", "targeted")]
        target = 0.5 * (int(r["verified"]) + int(clean["verified"]))
        for sham in ("equal_rank", "norm_sham", "jacobian_radius_sham"):
            a, b = index[(*key, "corrupt", sham)], index[(*key, "clean", sham)]
            delta = target - 0.5 * (int(a["verified"]) + int(b["verified"]))
            groups[(r["split"], r["variant"], sham)].append(
                {"root_id": r["root_id"], "seed": r["training_seed"], "delta": delta,
                 "match_valid": a["match_valid"] and b["match_valid"]})
    result = []
    for (split, variant, sham), items in sorted(groups.items()):
        by_seed = defaultdict(list)
        by_root = defaultdict(list)
        for item in items:
            by_seed[item["seed"]].append(item["delta"])
            by_root[item["root_id"]].append(item["delta"])
        result.append({"split": split, "variant": variant, "sham": sham,
                       "balanced_success_delta": sum(sum(v) / len(v) for v in by_root.values()) / len(by_root),
                       "per_model_delta": {str(k): sum(v) / len(v) for k, v in sorted(by_seed.items())},
                       "matched_pairs": sum(x["match_valid"] for x in items), "pairs": len(items)})
    return result


def valid_match(audit, condition, policy):
    a = audit["conditions"][condition]
    if policy == "jacobian_radius_sham":
        return a["strong"]["all_matched"]
    if policy == "norm_sham":
        return a["norm"]["all_matched"]
    if policy == "equal_rank":
        return a["equal_rank_valid"]
    return True


def solve(world, scores, variant):
    # Stabilize floating-point ties in the host's deterministic top-k rule.
    return episode(world, tuple(float(v) for v in np.round(scores, 10)), 2, 1, variant)


def run(train_roots: int, dev_roots: int, steps: int, seeds: tuple[int, ...]):
    if not 1 <= train_roots <= 4096 or not 1 <= dev_roots <= 256:
        raise ValueError("invalid root counts")
    if not 1 <= len(seeds) <= 8 or len(set(seeds)) != len(seeds) or not all(0 <= s <= 100000 for s in seeds):
        raise ValueError("invalid model seeds")
    train = [make_world("train", 41, i) for i in range(train_roots)]
    dev = [make_world(split, 991, i)
           for split in ("calibration_short", "calibration_long") for i in range(dev_roots)]
    rows, audits, fits = [], [], []
    for seed in seeds:
        model, receipt = fit(train, "tied3", seed, steps)
        factors = factorize(model)
        ids, importance = choose_components(model, train, factors)
        checkpoint = model.identity()
        fits.append({**receipt, "factorization_digest": digest([a.tolist() for a in factors]),
                     "train_component_scores": importance.tolist(), "selected": list(ids)})
        for world in dev:
            conditions, audit = interventions(model, features(world), world.root_id, factors, ids)
            if audit["full_restore_max_abs"] > TOL:
                raise RuntimeError("full restoration parity failed")
            audits.append({"root_id": world.root_id, "split": world.split, "training_seed": seed, **audit})
            reference = conditions["corrupt"]["intact"]
            corrupted = conditions["corrupt"]["corrupted"]
            for variant in VARIANTS:
                intact_result = solve(world, reference, variant)
                intact_correct = intact_result["verified"]
                corrupt_correct = solve(world, corrupted, variant)["verified"]
                for condition, scores in conditions.items():
                    policies = POLICIES if condition == "corrupt" else ("intact", "reconstruction", *ACTIVE)
                    for policy in policies:
                        result = solve(world, scores[policy], variant)
                        if (policy in {"reconstruction", "full_restore"}
                                and result["stored"] != intact_result["stored"]):
                            raise RuntimeError("reconstructed retention decision changed")
                        rows.append({**result, "training_seed": seed, "condition": condition, "policy": policy,
                                     "intact_correct": intact_correct, "corrupt_correct": corrupt_correct,
                                     "match_valid": valid_match(audit, condition, policy),
                                     "mean_abs_logit_error": float(np.abs(scores[policy] - reference).mean()),
                                     "scores": scores[policy].tolist()})
        if model.identity() != checkpoint:
            raise RuntimeError("frozen checkpoint changed")
    strong = [a["conditions"][condition]["strong"] for a in audits for condition in ("clean", "corrupt")]
    dose = {"active": sum(a["active"] for a in strong), "matched": sum(a["matched"] for a in strong),
            "unmatched": sum(a["unmatched"] for a in strong), "zero_dose": sum(a["zero_dose"] for a in strong),
            **{k: max(a[k] for a in strong)
               for k in ("max_norm_error", "max_signed_j_error", "max_radius_error")}}
    parity = {k: max(a[k] for a in audits) for k in
              ("write_reconstruction_max_abs", "score_reconstruction_max_abs", "full_restore_max_abs")}
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ("molt_goblin.py", "molt_retention.py", "molt_retention_train.py",
                            "molt_retention_steer.py", "molt_retention_steer_run.py")}
    return {"schema": "molt-steer-v5", "status": "development_only", "numpy": np.__version__,
            "train_root_count": len(train), "world_root_count": len(dev), "model_seeds": list(seeds),
            "train_manifest_digest": digest([asdict(w) for w in train]),
            "dev_manifest_digest": digest([asdict(w) for w in dev]), "sources": sources, "fits": fits,
            "parity": parity, "strong_sham_audit": dose, "summary": summarize(rows), "contrasts": contrasts(rows),
            "rows_digest": digest(rows), "audit_digest": digest(audits), "rows": rows, "audits": audits,
            "score_rounding_decimals": 10, "confirmation_ready": False,
            "promotion_authorized": False, "native_bpd": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-roots", type=int, default=1024)
    parser.add_argument("--dev-roots", type=int, default=128)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[53, 59, 61, 67])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refuse to overwrite")
    report = run(args.train_roots, args.dev_roots, args.steps, tuple(args.seeds))
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"status": report["status"], "rows": len(report["rows"]),
                      "rows_digest": report["rows_digest"], "audit_digest": report["audit_digest"],
                      "parity": report["parity"], "strong_sham_audit": report["strong_sham_audit"],
                      "summary": report["summary"], "contrasts": report["contrasts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
