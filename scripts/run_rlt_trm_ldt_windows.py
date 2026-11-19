"""Run the live TRM-LDT rule-change evaluation with a hard Windows memory cap."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import psutil
from run_rlt_trm_windows import COMMIT_CAP, COMMIT_FLOOR, FREE_FLOOR, ResourcePause, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--previous-pack", type=Path, action="append", required=True)
    parser.add_argument("--upstream-ldt", type=Path, required=True)
    parser.add_argument("--ramcap-package", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("requires a Windows Job Object")
    root = args.out.resolve()
    deadline = time.monotonic() + 180
    while True:
        free, headroom = memory()
        if (root / "STOP").exists() or time.monotonic() >= deadline:
            raise SystemExit(75)
        if free >= FREE_FLOOR + COMMIT_CAP and headroom >= COMMIT_FLOOR + COMMIT_CAP:
            break
        time.sleep(8)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path[:0] = [
        str(args.ramcap_package.resolve()), str(args.upstream_ldt.resolve()),
        str(Path(__file__).resolve().parents[1] / "src"),
    ]
    from lcgrt.ramcap import Job

    job = Job(COMMIT_CAP, priority="below_normal")
    job.assign_self()

    from research_gym.core import hybrid

    from datagoblin.serialization import canonical_json, dump_structured, load_structured
    from datagoblin.training_gym.rlt_trm_ldt import run

    root.mkdir(parents=True, exist_ok=True)
    resources = root / "resources.json"
    history = load_structured(resources)["segments"] if resources.exists() else []
    if history and history[-1]["status"] == "complete":
        return
    lock = root / "active.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    started = time.monotonic()
    last_check = started - 1
    process = psutil.Process()
    status = "aborted"

    def guard():
        nonlocal last_check, free, headroom
        if (root / "STOP").exists():
            raise RuntimeError("operator stop")
        now = time.monotonic()
        if now - last_check < 1:
            return
        last_check = now
        available, commit_left = memory()
        free, headroom = min(free, available), min(headroom, commit_left)
        if available < FREE_FLOOR or commit_left < COMMIT_FLOOR:
            raise ResourcePause("host memory floor crossed")
        if process.memory_info().peak_pagefile > COMMIT_CAP:
            raise RuntimeError("process private commit exceeds cap")

    try:
        guard()
        run(root, args.checkpoints, args.previous_pack, hybrid, guard)
        if process.memory_info().peak_pagefile > COMMIT_CAP:
            raise RuntimeError("process private commit exceeds cap")
        status = "complete"
    except ResourcePause as exc:
        status = "paused"
        raise SystemExit(75) from exc
    finally:
        receipt = {
            "status": status, "wall_s": time.monotonic() - started, "cpu_workers": 1,
            "hard_commit_cap_bytes": COMMIT_CAP, "peak_commit_bytes": process.memory_info().peak_pagefile,
            "job_peak_commit_bytes": job.peak_bytes(), "min_available_bytes": free, "min_commit_headroom": headroom,
        }
        dump_structured(resources, {"segments": [*history, receipt]})
        lock.unlink()
        print(canonical_json(receipt), flush=True)


if __name__ == "__main__":
    main()
