"""Fit and analyze the TRM supervision study under the workstation memory cap."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from run_rlt_trm_windows import COMMIT_CAP, COMMIT_FLOOR, ResourcePause, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--ramcap-package", type=Path, required=True)
    parser.add_argument("--host-free-floor-mib", type=int, default=1536)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("This launcher requires a Windows Job Object")
    if args.host_free_floor_mib < 1:
        parser.error("the host reserve must be positive")
    root = args.out.resolve()
    floor = args.host_free_floor_mib * 2**20
    deadline = time.monotonic() + 180
    while True:
        available, headroom = memory()
        if (root / "STOP").exists() or time.monotonic() >= deadline:
            raise SystemExit(75)
        if available >= floor + COMMIT_CAP and headroom >= COMMIT_FLOOR + COMMIT_CAP:
            break
        time.sleep(8)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path[:0] = [str(args.ramcap_package.resolve()), str(Path(__file__).resolve().parents[1] / "src")]
    from lcgrt.ramcap import Job

    job = Job(COMMIT_CAP, priority="below_normal")
    job.assign_self()
    from datagoblin.serialization import canonical_json

    root.mkdir(parents=True, exist_ok=True)
    lock = root / "active.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    started = time.monotonic()
    status = "aborted"

    def guard():
        nonlocal available, headroom
        current, remaining = memory()
        available, headroom = min(available, current), min(headroom, remaining)
        if current < floor or remaining < COMMIT_FLOOR or (root / "STOP").exists():
            raise ResourcePause("host guard paused the experiment")

    try:
        guard()
        if args.phase == "fit":
            from datagoblin.training_gym.rlt_trm_supervision import fit_run

            fit_run(root, args.data_seed, guard)
        else:
            from datagoblin.training_gym.rlt_trm_svd import evaluate

            evaluate(root, args.data_seed, guard)
        status = "complete"
    except ResourcePause as exc:
        status = "paused"
        raise SystemExit(75) from exc
    finally:
        receipt = {"phase": args.phase, "status": status, "hard_cap_bytes": COMMIT_CAP,
                   "peak_job_commit_bytes": job.peak_bytes(), "host_reserve_bytes": floor,
                   "minimum_available_bytes": available, "minimum_commit_headroom_bytes": headroom,
                   "wall_s": time.monotonic() - started}
        with (root / "resource-history.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(receipt) + "\n")
        lock.unlink()
        print(canonical_json(receipt), flush=True)


if __name__ == "__main__":
    main()
