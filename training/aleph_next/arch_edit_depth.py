"""Exploratory regrouping of the already-scored clean panel by exact logical correction size."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from arch_scoring import verified
from architecture_lab import MiniAleph, evaluation_rows, verdict
from gym import atomic_json, file_sha


@torch.inference_mode()
def regroup(out):
    record, custody = verified(out)
    scores = json.loads((out / "scores.json").read_text())
    if scores["training_sha256"] != custody["training_sha256"] or scores["seed_sha256"] != custody["seed_sha256"]:
        raise ValueError("different sealed panel")
    width = record["contract"]["recipe"]["width"]
    x, y, task, noise, families = evaluation_rows(int((out / "private.seed").read_text()), width, 16)
    repair = task < 2
    # A permitted elementary repair flips one truth-table row. Distinct rows
    # are independent, making this Hamming distance the exact minimum.
    distance = torch.where(repair, ((x[:, :, 4] > 0) != y.bool()).sum(1), y.sum(1).long())
    result = {"exploratory": True, "primary_scores_sha256": file_sha(out / "scores.json"), "cells": {}}
    family = np.asarray(families)
    for key, fit in record["fits"].items():
        if fit["arm"] not in ("direct3", "damp3", "rlt3"):
            continue
        m = MiniAleph(width, fit["arm"]).eval()
        m.load_state_dict(torch.load(out / f"{key}.pt", weights_only=True, map_location="cpu"))
        hits = []
        for start in range(0, len(x), 32):
            sl = slice(start, start + 32)
            level = torch.zeros(len(x[sl]))
            z = m(x[sl], level.bool(), level, noise[sl])[0][-1]
            hits.append(verdict(z, y[sl], task[sl]))
        hits = torch.cat(hits).numpy()
        original = np.mean([v["verified"][0] for v in scores["native"][key]["clean-0.0"].values()])
        if abs(hits.mean() - original) > 1e-7:
            raise ValueError("clean verdicts differ from frozen scoring")
        rows = {}
        for name, eligible in (("minimum_row_flips", repair), ("valid_witness_rows", ~repair)):
            entries = {}
            for k in range(5):
                mask = (eligible & (distance == k)).numpy()
                if not mask.any():
                    continue
                present = sorted(set(family[mask]))
                rates = [hits[mask & (family == g)].mean() for g in present]
                entries[str(k)] = {
                    "cases": int(mask.sum()),
                    "nonempty_task_compositions": len(present),
                    "equal_group_verified": float(np.mean(rates, dtype=np.float64)),
                }
            rows[name] = entries
        result["cells"][key] = rows
    atomic_json(out / "edit-depth.json", result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    regroup(args.out)
