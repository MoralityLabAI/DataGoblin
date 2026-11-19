"""Run the matched attention TRM checkpoint experiment on a capped CPU worker."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import psutil
from run_rlt_trm_windows import COMMIT_CAP, COMMIT_FLOOR, FREE_FLOOR, ResourcePause, memory

BLAS_THREADS = 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ramcap-package", type=Path, required=True)
    parser.add_argument("--torch-cpu", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    if sys.platform != "win32":
        raise SystemExit("requires a Windows Job Object")
    deadline = time.monotonic() + 180
    while True:
        free, commit = memory()
        if (root / "STOP").exists() or time.monotonic() >= deadline:
            raise SystemExit(75)
        if free >= FREE_FLOOR + COMMIT_CAP and commit >= COMMIT_FLOOR + COMMIT_CAP:
            break
        time.sleep(8)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(BLAS_THREADS)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "0"
    os.environ["DNNL_PRIMITIVE_CACHE_CAPACITY"] = "0"
    os.environ["MKL_DISABLE_FAST_MM"] = "1"
    sys.path[:0] = [
        str(args.torch_cpu.resolve()), str(args.ramcap_package.resolve()),
        str(Path(__file__).resolve().parents[1] / "src"),
    ]
    from lcgrt.ramcap import Job

    job = Job(COMMIT_CAP, priority="below_normal")
    job.assign_self()

    import numpy as np
    import torch
    import yaml

    from datagoblin.hashing import sha256_file
    from datagoblin.serialization import canonical_json, dump_structured, load_structured
    from datagoblin.training_gym import rlt_trm_sudoku as pilot

    if not torch.__version__.endswith("+cpu"):
        raise ValueError("expected a CPU-only Torch build")
    torch.set_num_threads(BLAS_THREADS)
    torch.set_num_interop_threads(1)

    resources = root / "resources.json"
    history = load_structured(resources)["segments"] if resources.exists() else []
    if history and history[-1]["status"] == "complete":
        return
    lock = root / "active.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    start = time.monotonic()
    prior = sum(s["wall_s"] for s in history)
    last = start - 2
    status, stage = "aborted", "load"
    process = psutil.Process()

    def guard():
        nonlocal last, free, commit
        now = time.monotonic()
        if (root / "STOP").exists() or prior + now - start >= 1800:
            raise RuntimeError("operator stop or active wall-time limit")
        if now - last >= 1:
            last = now
            if process.memory_info().peak_pagefile > COMMIT_CAP:
                raise RuntimeError("process private commit exceeds the configured cap")
            available, headroom = memory()
            free, commit = min(free, available), min(commit, headroom)
            if available < FREE_FLOOR or headroom < COMMIT_FLOOR:
                raise ResourcePause("host memory floor crossed")

    try:
        source = load_structured(root / "source.json")
        checkpoint = Path(source["checkpoint_path"])
        if sha256_file(checkpoint) != source["checkpoint_sha256"]:
            raise ValueError("cached checkpoint changed")
        for name, digest in source["upstream_sha256"].items():
            if sha256_file(root / "upstream" / name) != digest:
                raise ValueError("upstream reference changed")
        config_file = root / "upstream" / "config" / "arch" / "trm.yaml"
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        expected = {"H_cycles": pilot.H_CYCLES, "L_cycles": pilot.L_CYCLES, "halt_max_steps": pilot.SEGMENTS,
                    "hidden_size": pilot.HIDDEN, "num_heads": pilot.HEADS, "puzzle_emb_len": pilot.PUZZLE_PREFIX,
                    "L_layers": 2, "mlp_t": False, "pos_encodings": "rope", "forward_dtype": "bfloat16"}
        if any(config[key] != value for key, value in expected.items()):
            raise ValueError("expected the published TRM attention recipe")
        holdout = root / "holdout.json"
        items = load_structured(holdout)["puzzles"]
        if len(items) != 24 or len({item["id"] for item in items}) != 24:
            raise ValueError("unexpected holdout")
        ref_path = root / "reference.json"
        references = load_structured(ref_path)
        if (references["checkpoint_sha256"] != source["checkpoint_sha256"]
                or references["holdout_sha256"] != sha256_file(holdout)
                or references["torch"] != torch.__version__
                or references["peak_private_commit_bytes"] > COMMIT_CAP):
            raise ValueError("reference does not match the capped checkpoint run")
        protocol = {
            "checkpoint": {k: v for k, v in source.items() if k != "checkpoint_path"},
            "holdout_sha256": sha256_file(holdout), "reference_sha256": sha256_file(ref_path),
            "numpy": np.__version__, "torch": torch.__version__, "batch": 1, "blas_threads": BLAS_THREADS,
            "segments": pilot.SEGMENTS, "h_cycles": pilot.H_CYCLES, "l_cycles": pilot.L_CYCLES,
            "source_sha256": {"runner": sha256_file(Path(__file__)), "model": sha256_file(Path(pilot.__file__))},
            "coefficients": {
                arm: dict(zip(("answer", "reasoning"), pilot.feedback(arm), strict=True)) for arm in pilot.ARMS
            },
        }
        frozen = root / "protocol.json"
        if frozen.exists() and canonical_json(load_structured(frozen)) != canonical_json(protocol):
            raise ValueError("the fixed TRM protocol changed")
        if not frozen.exists():
            dump_structured(frozen, protocol)
        guard()
        with torch.inference_mode():
            model = pilot.SudokuTRM(pilot.checkpoint_weights(checkpoint, torch), torch)

        def trace(item, arm):
            with torch.inference_mode():
                return list(model.rollout(item["givens"], arm, guard))

        stage = "parity"
        parity = []
        baseline = {}
        for item, case in zip(items[:2], references["cases"], strict=True):
            rows = trace(item, "direct")
            if case["id"] != item["id"] or len(rows) != len(case["boards"]):
                raise ValueError("the reference sample or segment count changed")
            baseline[item["id"]] = rows
            boards = [r["prediction"] for r in rows]
            matches = [sum(a == b for a, b in zip(x, y, strict=True))
                       for x, y in zip(boards, case["boards"], strict=True)]
            parity.append({
                "id": case["id"], "cell_matches": sum(matches), "cells": 81 * len(rows),
                "board_matches": sum(n == 81 for n in matches), "final_agrees": boards[-1] == case["boards"][-1],
            })
        dump_structured(root / "parity.json", {"cases": parity, "minimum_cell_agreement": 1})
        print(canonical_json({"stage": "parity", "cases": parity}), flush=True)
        if any(row["cell_matches"] != row["cells"] for row in parity):
            raise ValueError("Torch trace differs from the pinned reference")
        stage = "infer"
        predictions = root / "predictions"
        predictions.mkdir(exist_ok=True)
        for item in items:
            for arm in pilot.ARMS:
                path = predictions / f"{item['id']}-{arm}.json"
                if path.exists():
                    continue
                steps = baseline.get(item["id"]) if arm == "direct" else None
                pending = path.with_suffix(".pending.json")
                dump_structured(pending, {"id": item["id"], "arm": arm, "steps": steps or trace(item, arm)})
                pending.replace(path)
            print(canonical_json({"id": item["id"], "status": "evaluated"}), flush=True)
        if process.memory_info().peak_pagefile > COMMIT_CAP:
            raise RuntimeError("process private commit exceeds the configured cap")
        status = "complete"
    except ResourcePause as exc:
        status = "paused"
        raise SystemExit(75) from exc
    finally:
        segment = {
            "status": status, "stage": stage, "wall_s": time.monotonic() - start, "cpu_workers": 1,
            "blas_threads": BLAS_THREADS, "hard_commit_cap_bytes": COMMIT_CAP,
            "peak_commit_bytes": process.memory_info().peak_pagefile,
            "job_peak_including_refused_bytes": job.peak_bytes(),
            "min_available_bytes": free, "min_headroom_bytes": commit,
        }
        dump_structured(resources, {"segments": [*history, segment]})
        lock.unlink()
        print(canonical_json(segment), flush=True)


if __name__ == "__main__":
    main()
