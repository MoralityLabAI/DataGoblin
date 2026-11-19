"""Independently score the frozen carry predictions with integer arithmetic."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from itertools import product
from pathlib import Path

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured, load_structured

from .rlt_revision import read_rows


def score_cell(tasks, path):
    fields = {"id", "before", "reset", "carry", "carry_delta_rms"}
    predictions = read_rows(path, fields)
    if predictions.keys() != tasks.keys():
        raise ValueError("predictions must cover each task once")
    counts = defaultdict(lambda: {
        "n": 0, "bits": 0, "before_exact": 0, "reset_exact": 0, "carry_exact": 0,
        "reset_bits_correct": 0, "carry_bits_correct": 0, "carry_out_exact": 0,
        "stale_when_before_correct": 0, "reset_disagrees_with_carry": 0, "sum_carry_delta_rms": 0.0,
    })
    for key, task in tasks.items():
        guess = predictions[key]
        width = task["width"]
        if any(type(guess[arm]) is not int or not 0 <= guess[arm] < 1 << (width + 1)
               for arm in ("reset", "carry", "before")):
            raise ValueError("invalid predicted sum")
        delta = guess["carry_delta_rms"]
        if not (math.isfinite(delta) and 0 <= delta <= 1):
            raise ValueError("invalid carry delta")
        before = task["a"] + task["b"]
        answer = task["a"] + task["edited"]
        group = (task["family"], task["status"], width - task["anchor"] if task["family"] == "chain" else None)
        count = counts[group]
        count["n"] += 1
        count["bits"] += width + 1
        count["before_exact"] += guess["before"] == before
        count["reset_exact"] += guess["reset"] == answer
        count["carry_exact"] += guess["carry"] == answer
        count["reset_bits_correct"] += width + 1 - (guess["reset"] ^ answer).bit_count()
        count["carry_bits_correct"] += width + 1 - (guess["carry"] ^ answer).bit_count()
        count["carry_out_exact"] += (guess["carry"] >> width) == (answer >> width)
        count["stale_when_before_correct"] += (
            task["status"] == "changed" and before == guess["before"] == guess["carry"]
        )
        count["reset_disagrees_with_carry"] += guess["reset"] != guess["carry"]
        count["sum_carry_delta_rms"] += delta
    return [
        {"family": family, "status": status, "span": span, **counts[family, status, span]}
        for family, status, span in sorted(counts)
    ]


def report(run, out):
    resources = load_structured(run / "resources.json")
    if (run / "active.lock").exists() or resources["segments"][-1]["status"] != "complete":
        raise ValueError("run must finish before verifying predictions")
    if any(segment["peak_commit_bytes"] > segment["hard_commit_cap_bytes"] for segment in resources["segments"]):
        raise ValueError("memory cap exceeded")
    protocol = load_structured(run / "protocol.json")
    tasks_path = run / "tasks.jsonl"
    if sha256_file(tasks_path) != protocol["tasks_sha256"]:
        raise ValueError("frozen evaluation tasks changed")
    fields = {"id", "family", "status", "width", "anchor", "a", "b", "edited"}
    tasks = read_rows(tasks_path, fields)
    modes = protocol["arms"]
    fits, metrics = [], []
    for seed in protocol["seeds"]:
        matches = set()
        for mode in modes:
            arm = run / str(seed) / mode
            fit = load_structured(arm / "fit.json")
            if (fit["seed"] != seed or fit["arm"] != mode
                    or fit["weights_sha256"] != sha256_file(arm / "weights.npz")):
                raise ValueError("fit receipt mismatch")
            matches.add(tuple(fit[k] for k in ("init_sha256", "batches_sha256", "updates", "params", "slow_visits",
                                                "fast_per_slow", "training_width", "batch")))
            fits.append(fit)
            cells = [(mode, depth) for depth in protocol["slow_depths"]]
            if mode == "direct":
                cells += list(product((arm for arm in modes if arm != "direct"), (4, 8)))
            for runtime, depth in cells:
                path = arm / f"{runtime}-{depth}.jsonl"
                metrics.append({
                    "seed": seed, "trained": mode, "runtime": runtime, "slow": depth,
                    "fast": depth * protocol["fast_per_slow"], "rows": score_cell(tasks, path),
                    "predictions_sha256": sha256_file(path), "predictions_file": path.relative_to(run).as_posix(),
                })
        if len(matches) != 1:
            raise ValueError("training conditions do not match")
    dump_structured(out, {
        "schema": "datagoblin.rlt-hrm-carry.v1", "protocol": protocol, "resources": resources,
        "eval_items": len(tasks), "fits": fits, "cells": metrics,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.out)


if __name__ == "__main__":
    main()
