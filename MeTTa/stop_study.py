"""Matched-compute variant comparison on fresh roots (INK-FP-AUDIT-03 stop study).

Every row trains on the same 24-root bundle TRAIN presentations with the same
optimizer steps, learning rate, hidden width and unroll depth, and is scored on
fresh development roots disjoint from the bundle's train, dev and public-test
seed bands. Rows marked `unmatched_data` change the TRAIN data and are reported
separately.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .ram_hillclimb import DEFAULTS, make_bundle
from .stop_diagnostics import diagnose, fit_existing, fresh_dev_rows, train_accuracy
from .student_variants import VARIANTS, attach_contexts, extra_train_rows, fit_variant

SUM_FIELDS = (
    "cases", "initial_solved", "ever_reached_optimum", "solved_oracle_stop",
    "solved_and_stopped", "premature_stops", "own_never_stopped",
    "unnecessary_visits_never", "unnecessary_visits_own",
)


def peak_working_set_mib() -> float:
    import psutil

    info = psutil.Process().memory_info()
    return round(getattr(info, "peak_wset", info.rss) / 1048576, 2)


def merge(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {f: sum(r[f] for r in results) for f in SUM_FIELDS}
    for f in ("onpolicy_correct", "teacher_forced_correct", "teacher_forced_progress"):
        vals = [r[f] for r in results if r[f] is not None]
        out[f] = round(sum(vals) / len(vals), 4) if vals else None
    for key in ("stop_calibration_canonical", "stop_calibration_onpolicy"):
        tot = Counter()
        for r in results:
            tot.update({k: r[key][k] for k in ("tp", "fn", "fp", "tn")})
        out[key] = {
            **dict(tot),
            "stop_rate_at_solved": round(tot["tp"] / (tot["tp"] + tot["fn"]), 4)
            if tot["tp"] + tot["fn"] else None,
            "stop_rate_at_unsolved": round(tot["fp"] / (tot["fp"] + tot["tn"]), 4)
            if tot["fp"] + tot["tn"] else None,
        }
    steps: dict[str, list[int]] = {}
    for r in results:
        for k, (a, b) in r["teacher_forced_by_step"].items():
            cur = steps.setdefault(str(k), [0, 0])
            cur[0] += a
            cur[1] += b
    out["teacher_forced_by_step"] = dict(sorted(steps.items(), key=lambda kv: int(kv[0])))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle = make_bundle(24, 8)
    fresh = fresh_dev_rows(args.fresh_per_family)
    report: dict[str, Any] = {"fresh_cases": len(fresh), "seeds": args.seeds, "results": {}}
    extra = {
        "unmatched_data_100roots_4views": extra_train_rows(100, full_trajectory=False),
    } if args.extra_data else {}
    for steps in args.steps:
        rows: dict[str, list[dict]] = {}
        train_acc: dict[str, list[float]] = {}
        for seed in args.seeds:
            policies: dict[str, Any] = {}
            cfg = {**DEFAULTS["trm"], "steps": steps}
            policies["original_trm"] = fit_existing("trm", cfg, bundle, seed=seed)
            policies["existing_stop_head_trm"] = fit_existing(
                "trm", {**cfg, "stop_policy": "train_calibrated"}, bundle, seed=seed)
            policies["original_ldt_proxy"] = fit_existing(
                "ldt_proxy", {**DEFAULTS["ldt_proxy"], "steps": steps}, bundle, seed=seed)
            for name, spec in VARIANTS.items():
                policies[name] = fit_variant(spec, bundle["train"], seed=seed, steps=steps)
            for name, tr in extra.items():
                for vname in ("plain_class_stop", "sep_stop_head"):
                    policies[f"{name}|{vname}"] = fit_variant(
                        VARIANTS[vname], tr, seed=seed, steps=steps)
            for name, policy in policies.items():
                rows.setdefault(name, []).append(diagnose(policy, fresh))
            for name in ("original_trm", "plain_class_stop", "ldt_trm_hybrid"):
                train_acc.setdefault(name, []).append(
                    train_accuracy(policies[name], attach_contexts(bundle["train"])))
            print(f"steps={steps} seed={seed} done", flush=True)
        report["results"][str(steps)] = {
            "variants": {name: merge(rs) for name, rs in rows.items()},
            "train_edit_accuracy": {k: round(sum(v) / len(v), 4) for k, v in train_acc.items()},
        }
    report["peak_working_set_mib"] = peak_working_set_mib()
    return report


LEVER_GRID = (
    # (roots_per_family, full_trajectory, steps, hidden, loops)
    (24, False, 1000, 12, 4),
    (100, False, 1000, 12, 4),
    (100, False, 1000, 48, 4),
    (100, True, 1000, 12, 4),
    (300, False, 1000, 12, 4),
)


def run_levers(args: argparse.Namespace) -> dict[str, Any]:
    """Data volume, trajectory coverage, steps and width, one variant per row."""
    fresh = fresh_dev_rows(args.fresh_per_family)
    out: dict[str, Any] = {"fresh_cases": len(fresh), "rows": {}}
    cache: dict[tuple[int, bool], list[dict]] = {}
    for roots, full, steps, hidden, loops in LEVER_GRID:
        train = cache.setdefault((roots, full), extra_train_rows(roots, full_trajectory=full))
        for vname in ("plain_class_stop", "sep_stop_head"):
            results = [
                diagnose(fit_variant(VARIANTS[vname], train, seed=seed, steps=steps,
                                     hidden=hidden, loops=loops), fresh)
                for seed in args.seeds
            ]
            key = f"roots{roots}|full{int(full)}|steps{steps}|hidden{hidden}|loops{loops}|{vname}"
            out["rows"][key] = {"train_presentations": len(train), **merge(results)}
            print(key, out["rows"][key]["solved_and_stopped"], flush=True)
    out["peak_working_set_mib"] = peak_working_set_mib()
    return out


CLIMB_GRID = {
    "steps": (16, 32, 64, 128, 200),
    "lr": (0.04, 0.08, 0.12, 0.20, 0.30),
    "hidden": (8, 12, 24, 48),
    "loops": (2, 4, 6),
}


def run_climb(args: argparse.Namespace) -> dict[str, Any]:
    """Coordinate hill climb of one variant on the bundle's public dev roots.

    Selection reads the bundle dev roots (seed band 30000); the winner is then
    re-scored on the fresh roots with seeds that never entered selection.
    """
    spec = VARIANTS[args.climb_variant]
    bundle = make_bundle(24, 8)
    fresh = fresh_dev_rows(args.fresh_per_family)
    select_seeds = args.seeds[:2]
    final_seeds = args.seeds[2:] or args.seeds

    def score(cfg: dict[str, Any], rows: list[dict], seeds: list[int]) -> dict[str, Any]:
        return merge([
            diagnose(fit_variant(spec, bundle["train"], seed=s, steps=cfg["steps"], lr=cfg["lr"],
                                 hidden=cfg["hidden"], loops=cfg["loops"]), rows)
            for s in seeds
        ])

    def key(m: dict[str, Any]) -> tuple:
        return (m["solved_and_stopped"], m["solved_oracle_stop"], -m["premature_stops"],
                m["teacher_forced_correct"])

    current = {"steps": 16, "lr": 0.20, "hidden": 12, "loops": 4}
    best = score(current, bundle["dev"], select_seeds)
    history = [{"config": dict(current), "dev": best, "accepted": True}]
    for _ in range(2):
        for field, values in CLIMB_GRID.items():
            for value in values:
                if value == current[field]:
                    continue
                trial = {**current, field: value}
                metrics = score(trial, bundle["dev"], select_seeds)
                accepted = key(metrics) > key(best)
                history.append({"config": trial, "dev": metrics, "accepted": accepted})
                print(field, value, metrics["solved_and_stopped"], metrics["solved_oracle_stop"],
                      accepted, flush=True)
                if accepted:
                    current, best = trial, metrics
    return {
        "variant": args.climb_variant, "start": history[0]["config"], "final": current,
        "history": history, "final_on_fresh_unseen_seeds": score(current, fresh, final_seeds),
        "start_on_fresh_unseen_seeds": score(history[0]["config"], fresh, final_seeds),
        "peak_working_set_mib": peak_working_set_mib(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/metta/stop_study.json"))
    parser.add_argument("--fresh-per-family", type=int, default=12)
    parser.add_argument("--steps", type=int, nargs="+", default=[16, 200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261122, 20261123, 20261124])
    parser.add_argument("--extra-data", action="store_true")
    parser.add_argument("--levers", action="store_true")
    parser.add_argument("--climb", action="store_true")
    parser.add_argument("--climb-variant", default="ldt_trm_hybrid", choices=sorted(VARIANTS))
    args = parser.parse_args()
    report = run_climb(args) if args.climb else run_levers(args) if args.levers else run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"peak_working_set_mib": report["peak_working_set_mib"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
