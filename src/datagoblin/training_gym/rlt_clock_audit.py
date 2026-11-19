"""Audit phase, step size, and answer carry with frozen TRM-style checkpoints."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .rlt_pilot_data import jsonl
from .rlt_revision import score
from .rlt_trm import BATCH, DEPTHS, MODES, NODES, RULE_SPLITS, SEEDS, TRAIN_DEPTH, pending_rows, prepare, refine

POLICIES = ("native", "phase", "step", "phase_and_step")
CARRIES = ("native", "zero")


def clock_denominators(policy, depth, reference):
    phase = reference if policy in {"phase", "phase_and_step"} else depth
    step = reference if policy in {"step", "phase_and_step"} else depth
    return phase, step


def yield_cells(run, placement, pack, out, protocol, guard):
    for seed, mode in product(SEEDS, MODES):
        arm = run / str(seed) / mode
        fit = load_structured(arm / "train.json")
        for key, value in {
            "seed": seed, "mode": mode, "depth": TRAIN_DEPTH, "placement": placement,
            "training_manifest_sha256": protocol["rule_manifest_sha256"],
            "weights_sha256": sha256_file(arm / "weights.npz"),
        }.items():
            if fit[key] != value:
                raise ValueError(f"frozen checkpoint differs from its training receipt: {arm}/{key}")
        with np.load(arm / "weights.npz", allow_pickle=False) as state:
            weights = dict(state)
        for split, depth in product(RULE_SPLITS, DEPTHS[1:]):
            tasks = jsonl(pack / f"{split}.tasks.jsonl")
            policies = ("native",) if depth == TRAIN_DEPTH else POLICIES
            for policy in policies:
                options = {"placement": placement, "clock_depths": clock_denominators(policy, depth, TRAIN_DEPTH)}
                prefix = f"{placement}-{seed}-{mode}-{split}-{depth}-{policy}"
                paths = {carry: out / f"{prefix}-{carry}.jsonl" for carry in CARRIES}
                if not all(path.exists() for path in paths.values()):
                    with pending_rows(paths["native"]) as native, pending_rows(paths["zero"]) as zero:
                        for i in range(0, len(tasks), BATCH):
                            if guard is not None:
                                guard()
                            batch = tasks[i : i + BATCH]
                            start, hops, before, after = prepare(batch)
                            blank = np.zeros_like(start)
                            first, prior = refine(weights, before, start, hops, blank, mode, depth, **options)
                            carried, _ = refine(weights, after, start, hops, prior, mode, depth, **options)
                            reset, _ = refine(weights, after, start, hops, blank, mode, depth, **options)
                            for handle, output in ((native, carried), (zero, reset)):
                                for task, p1, p2 in zip(batch, first.argmax(1), output.argmax(1), strict=True):
                                    handle.write(canonical_json(
                                        {"id": task["id"], "before": NODES[p1], "after": NODES[p2]},
                                    ) + "\n")
                for carry, path in paths.items():
                    yield {
                        "seed": seed, "trained": mode, "placement": placement, "split": split, "depth": depth,
                        "clocks": policy, "carry": carry, "prediction_file": path.name,
                        "weights_sha256": fit["weights_sha256"],
                    }


def audit(pack, source, out, *, guard=None):
    manifest_path = pack / "manifest.json"
    manifest = load_structured(manifest_path)
    for split in RULE_SPLITS:
        if sha256_file(pack / f"{split}.tasks.jsonl") != manifest["splits"][split]["tasks_sha256"]:
            raise ValueError("frozen task hash mismatch")
    runs = {placement: source / leaf for placement, leaf in (("answer", "run"), ("reasoning", "reasoning"))}
    protocols = {placement: load_structured(run / "protocol.json") for placement, run in runs.items()}
    for placement, run in runs.items():
        if load_structured(run / "run.json")["status"] != "complete" or protocols[placement]["placement"] != placement:
            raise ValueError("frozen checkpoint run has not completed")
    if protocols["answer"]["rule_manifest_sha256"] != protocols["reasoning"]["rule_manifest_sha256"]:
        raise ValueError("frozen fits used different training packs")
    previous_pack = source / "rules"
    previous_manifest = previous_pack / "manifest.json"
    if sha256_file(previous_manifest) != protocols["answer"]["rule_manifest_sha256"]:
        raise ValueError("original training manifest changed")
    previous = set()
    for split, spec in load_structured(previous_manifest)["splits"].items():
        path = previous_pack / f"{split}.tasks.jsonl"
        if sha256_file(path) != spec["tasks_sha256"]:
            raise ValueError("original task hash mismatch")
        previous.update(task["lineage"] for task in jsonl(path))
    for split in RULE_SPLITS:
        if any(task["lineage"] in previous for task in jsonl(pack / f"{split}.tasks.jsonl")):
            raise ValueError("new lineage overlaps original fit or evaluation")
    source_files = (Path(__file__), Path(__file__).with_name("rlt_trm.py"))
    receipt = {
        "schema": "datagoblin.rlt-clock-audit.v1",
        "source_sha256": {path.name: sha256_file(path) for path in source_files},
        "rule_pack_sha256": sha256_file(manifest_path), "reference_depth": TRAIN_DEPTH,
        "training_rule_pack_sha256": protocols["answer"]["rule_manifest_sha256"],
        "training_protocol_sha256": {placement: sha256_file(run / "protocol.json") for placement, run in runs.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    audit_protocol = out / "protocol.json"
    if audit_protocol.exists():
        if load_structured(audit_protocol) != receipt:
            raise ValueError("cannot resume after the audit protocol changed")
    else:
        dump_structured(audit_protocol, receipt)
    predictions = out / "predictions"
    predictions.mkdir(exist_ok=True)
    cells = [
        cell for placement, run in runs.items()
        for cell in yield_cells(run, placement, pack, predictions, protocols[placement], guard)
    ]
    # Keep all target reads in the independent verification phase.
    for cell in cells:
        cell.update(score(pack, cell["split"], predictions / cell["prediction_file"]))
    dump_structured(out / "clock-evidence.json", {**receipt, "cells": cells})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.pack, args.source, args.out)


if __name__ == "__main__":
    main()
