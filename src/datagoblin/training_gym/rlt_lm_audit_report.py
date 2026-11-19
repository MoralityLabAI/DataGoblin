"""Check frozen LM crossover coverage and summarize paired effects."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import median

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured, load_structured

from .rlt_lm_audit import CLOCKS, DEPTHS, bits_per_byte, model_receipts
from .rlt_pilot import MODES
from .rlt_pilot_data import PUBLIC, TEXT, jsonl


def read_cell(path, expected):
    predictions = defaultdict(dict)
    for row in jsonl(path):
        group = predictions[row["benchmark"]]
        if row["id"] in group:
            raise ValueError(f"duplicate prediction in {path.name}")
        group[row["id"]] = row
    if predictions.keys() != expected.keys() or any(rows.keys() != expected[key] for key, rows in predictions.items()):
        raise ValueError(f"bad item coverage in {path.name}")
    return predictions


def verify_native(root, audit, corpus_ids):
    for mode, depth in product(MODES, (2, 4, 8)):
        original = read_cell(root / "run" / mode / f"known-k{depth}.jsonl", corpus_ids)
        for name in TEXT if depth == 2 else PUBLIC:
            found = audit[mode, mode, depth, "native"][name]
            for key, row in found.items():
                prior = original[name][key]
                if name in TEXT:
                    if prior["bytes"] != row["bytes"] or not math.isclose(
                        row["nll_nats"], prior["nll_nats"], rel_tol=2e-6, abs_tol=2e-5,
                    ):
                        raise ValueError("native text prediction differs from the prior evaluation")
                elif row != prior:
                    raise ValueError("native known-benchmark prediction differs from prior evaluation")


def summarize_pulse(path, ids):
    groups = defaultdict(dict)
    for row in jsonl(path):
        if row["id"] in groups[row["perturbation"]] or not math.isclose(row["input_rms"], 0.01, rel_tol=1e-5):
            raise ValueError("bad pulse sampling or dose")
        groups[row["perturbation"]][row["id"]] = row
    if groups.keys() != {"initial", "early_proposal", "late_proposal"} or any(g.keys() != ids for g in groups.values()):
        raise ValueError("bad pulse coverage")
    fields = ("endpoint_rms", "endpoint_centered_rms", "endpoint_gain", "normalized_endpoint_gain", "logit_gain")
    return [
        {
            "kind": kind, "items": len(rows), "eta": next(iter(rows.values()))["feedback_weight"],
            "median": {k: median(r[k] for r in rows.values()) for k in fields},
            "top1_flips": sum(r["top1_flip_count"] for r in rows.values()),
        }
        for kind, rows in groups.items()
    ]


def paired_effects(audit):
    result = []
    for depth, corpus, alternative in product(DEPTHS, TEXT, MODES[1:]):
        base = audit["direct", "direct", depth, "native"][corpus]
        fitted = audit[alternative, "direct", depth, "native"][corpus]
        runtime = audit["direct", alternative, depth, "native"][corpus]
        both = audit[alternative, alternative, depth, "native"][corpus]
        scores = [bits_per_byte(x.values()) for x in (base, fitted, runtime, both)]
        b, f, r, j = scores
        result.append({
            "alternative": alternative, "depth": depth, "corpus": corpus,
            "training_effect_bpb": f - b, "runtime_effect_bpb": r - b,
            "interaction_bpb": j - f - r + b, "joint_effect_bpb": j - b,
            "joint_better_document_windows": sum(both[k]["nll_nats"] < base[k]["nll_nats"] for k in base),
            "training_better_document_windows": sum(fitted[k]["nll_nats"] < base[k]["nll_nats"] for k in base),
            "document_windows": len(base),
        })
    return result


def report(root: Path, audit: Path, resources_path: Path, out: Path):
    resources = load_structured(resources_path)
    if resources["segments"][-1]["status"] != "complete" or (resources_path.parent / "active.lock").exists():
        raise ValueError("crossover must complete before verification")
    if any(s["peak_commit_bytes"] > s["hard_commit_cap_bytes"] for s in resources["segments"]):
        raise ValueError("resource cap exceeded")
    protocol = load_structured(audit / "protocol.json")
    training_hash, known_hash, fits, known = model_receipts(root)
    if (protocol["training_protocol_sha256"], protocol["known_sha256"], protocol["fits"]) != (
        training_hash, known_hash, fits,
    ):
        raise ValueError("frozen source checkpoints or public tasks changed")
    ids = {name: {item["id"] for item in items} for name, items in known.items()}
    cache = {}
    summaries, pulse = [], []
    for trained, mode, depth in product(MODES, MODES, DEPTHS):
        for clock in CLOCKS if depth in (8, 16) else ("native",):
            path = audit / f"{trained}--{mode}--{depth}--{clock}.jsonl"
            expected = ids if depth in (4, 8) and clock == "native" else {name: ids[name] for name in TEXT}
            groups = read_cell(path, expected)
            cache[trained, mode, depth, clock] = groups
            scores = {}
            for name, rows in groups.items():
                values = rows.values()
                if name in TEXT:
                    if any(r["bytes"] <= 0 or not math.isfinite(r["nll_nats"]) or r["nll_nats"] < 0 for r in values):
                        raise ValueError("invalid byte loss")
                    scores[name] = {"bpb": bits_per_byte(values), "items": len(rows)}
                else:
                    if name != "lambada":
                        labels = {item["id"]: item["gold"] for item in known[name]}
                        for key, row in rows.items():
                            if (row["gold"] != labels[key]
                                    or row["correct"] != (row["predicted"] == labels[key])
                                    or row["correct_normalized"] != (row["predicted_normalized"] == labels[key])):
                                raise ValueError("the frozen choice label or prediction differs")
                    metric = "correct_normalized" if name == "hellaswag" else "correct"
                    scores[name] = {"correct": sum(r[metric] for r in values), "items": len(rows)}
            summaries.append({
                "trained": trained, "runtime": mode, "depth": depth, "clocks": clock,
                "metrics": scores, "file": path.name, "sha256": sha256_file(path),
            })
        if depth in (4, 8, 16):
            path = audit / f"pulse--{trained}--{mode}--{depth}.jsonl"
            pulse.append({
                "trained": trained, "runtime": mode, "depth": depth,
                "measurements": summarize_pulse(path, {i["id"] for i in known[TEXT[0]][:8]}),
                "file": path.name, "sha256": sha256_file(path),
            })
    verify_native(root, cache, ids)
    for trained in MODES:
        if cache[trained, "direct", 1, "native"] != cache[trained, "rlt", 1, "native"]:
            raise ValueError("one visit should have identical runtime feedback")
        if cache[trained, "rlt", 1, "native"] != cache[trained, "matched_damping", 1, "native"]:
            raise ValueError("one visit should have identical runtime feedback")
        if cache[trained, "rlt", 2, "native"] != cache[trained, "matched_damping", 2, "native"]:
            raise ValueError("RLT and matched damping should coincide at two visits")
    dump_structured(out, {
        "schema": "datagoblin.rlt-lm-crossover-report.v1", "protocol": protocol, "resources": resources,
        "cells": summaries, "pulses": pulse, "paired_effects": paired_effects(cache),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.audit, args.resources, args.out)


if __name__ == "__main__":
    main()
