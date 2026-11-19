"""Verify the completed RLT-TRM pilot and export paired DataGoblin evidence."""

from __future__ import annotations

import argparse
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .rlt_revision import NODES, SPLITS, read_rows, score


def outcome(guess, target):
    return {
        "before": guess["before"], "after": guess["after"],
        "before_ok": guess["before"] == target["before"],
        "after_ok": guess["after"] == target["after"],
    }


def _collect(pack: Path, run: Path, out: Path):
    receipt = load_structured(run / "run.json")
    protocol = load_structured(run / "protocol.json")
    if receipt["status"] != "complete" or (run / "active.lock").exists():
        raise ValueError("inference must finish before verification")
    if receipt["peak_job_commit_bytes"] > receipt["hard_job_commit_limit_bytes"]:
        raise ValueError("reported job memory exceeds its cap")
    manifest_path = pack / "manifest.json"
    if sha256_file(manifest_path) != protocol["rule_manifest_sha256"]:
        raise ValueError("rule manifest differs from the training protocol")
    modes, crossover = protocol["modes"], protocol["crossover_depth"]
    splits = SPLITS[1:]
    summary = {
        "schema": "datagoblin.rlt-trm-rule-change.v1",
        "evaluation_scope": "retrospective frozen rule-revision benchmark",
        "protocol": protocol, "rule_pack": load_structured(manifest_path), "resources": receipt,
        "training": [], "cells": [], "native_pairs": [],
    }
    out.mkdir(parents=True, exist_ok=False)
    with (
        (out / "item-outcomes.jsonl").open("w", encoding="utf-8", newline="\n") as items,
        (out / "loop-traces.jsonl").open("w", encoding="utf-8", newline="\n") as traces,
    ):
        for seed in protocol["seeds"]:
            matched = set()
            for mode in modes:
                arm = run / str(seed) / mode
                fit = load_structured(arm / "train.json")
                for key, value in {
                    "seed": seed, "mode": mode, "steps": protocol["steps"], "batch": protocol["batch"],
                    "depth": protocol["train_depth"], "training_manifest_sha256": protocol["rule_manifest_sha256"],
                    "weights_sha256": sha256_file(arm / "weights.npz"),
                    "updates_sha256": sha256_file(arm / "updates.jsonl"),
                }.items():
                    if fit[key] != value:
                        raise ValueError(f"training receipt mismatch for {seed}/{mode}/{key}")
                matched.add((fit["init_sha256"], fit["schedule_sha256"], fit["parameters"], fit["cell_calls"]))
                summary["training"].append(fit)
            if len(matched) != 1:
                raise ValueError(f"unmatched training conditions for seed {seed}")
            for depth, split in product(protocol["depths"], splits):
                targets = read_rows(pack / f"{split}.targets.jsonl", {"id", "lineage", "before", "after"})
                tasks = read_rows(pack / f"{split}.tasks.jsonl", {"id", "lineage", "links", "start", "steps", "patch"})
                cells = product(modes, repeat=2) if depth == crossover else ((mode, mode) for mode in modes)
                guesses = {}
                for train, runtime in cells:
                    stem = run / str(seed) / train / f"{split}-{runtime}-k{depth}"
                    pred_path = stem.with_suffix(".jsonl")
                    meta = {"seed": seed, "split": split, "depth": depth, "trained": train, "runtime": runtime}
                    verified = score(pack, split, pred_path)
                    pred = read_rows(pred_path, {"id", "before", "after"})
                    if any(p["before"] not in NODES or p["after"] not in NODES for p in pred.values()):
                        raise ValueError("predicted node is outside the rule vocabulary")
                    before_by_lineage = {}
                    for key, target in targets.items():
                        before = pred[key]["before"]
                        if before_by_lineage.setdefault(target["lineage"], before) != before:
                            raise ValueError("first-turn answers disagree within a patch lineage")
                    guesses[train, runtime] = pred
                    trace_path = stem.with_suffix(".trace.jsonl")
                    loop = read_rows(trace_path, {"id", "before", "after"})
                    if list(loop) != list(tasks)[:8]:
                        raise ValueError("trace sample differs from the protocol")
                    for key, trace in loop.items():
                        for turn in ("before", "after"):
                            visits = trace[turn]
                            if len(visits) != depth or NODES[visits[-1]["argmax"]] != pred[key][turn]:
                                raise ValueError("terminal trace and answer disagree")
                        traces.write(canonical_json({**trace, **meta, "hops": tasks[key]["steps"]}) + "\n")
                    summary["cells"].append({**meta, **verified, "trace_sha256": sha256_file(trace_path)})
                deltas = {pair: [0, 0, 0] for pair in combinations(modes, 2)}
                for key, target in targets.items():
                    results = {
                        f"{train}/{runtime}": outcome(pred[key], target) for (train, runtime), pred in guesses.items()
                    }
                    changed = target["before"] != target["after"]
                    if changed:
                        for (a, b), counts in deltas.items():
                            diff = results[f"{a}/{a}"]["after_ok"] - results[f"{b}/{b}"]["after_ok"]
                            counts[diff + 1] += 1
                    items.write(canonical_json({
                        "seed": seed, "depth": depth, "split": split, "id": key, "lineage": target["lineage"],
                        "hops": tasks[key]["steps"], "changed": changed, "results": results,
                    }) + "\n")
                for (left, right), (right_only, tie, left_only) in deltas.items():
                    summary["native_pairs"].append({
                        "seed": seed, "depth": depth, "split": split, "metric": "changed_after_exact",
                        "left": left, "right": right, "left_only": left_only, "right_only": right_only, "tie": tie,
                    })
    summary["artifact_sha256"] = {
        name: sha256_file(out / name) for name in ("item-outcomes.jsonl", "loop-traces.jsonl")
    }
    dump_structured(out / "evidence.json", summary)
    return summary


def collect(pack: Path, run: Path, out: Path):
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as workspace:
        staged = Path(workspace) / "evidence"
        summary = _collect(pack, run, staged)
        staged.rename(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collect(args.pack, args.run, args.out)


if __name__ == "__main__":
    main()
