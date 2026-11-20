"""Run the Aleph proxy on CPU with the rollout campaign's 768 MiB job cap."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from run_rlt_trm_windows import COMMIT_FLOOR, memory

MIB = 2**20
CAP = 768 * MIB
RESERVE = 1536 * MIB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "out", "ramcap-package"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--cohort", type=Path, help="Reuse an already frozen development and held-out cohort")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("the resource wrapper requires Windows")
    if args.out.exists():
        parser.error("choose a fresh output directory")
    available, commit = memory()
    if available < RESERVE + CAP or commit < COMMIT_FLOOR + CAP:
        raise SystemExit("insufficient free memory for the capped proxy run")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path[:0] = [str(args.ramcap_package.resolve()), str(Path(__file__).resolve().parents[1] / "src")]
    from lcgrt.ramcap import Job

    job = Job(CAP, priority="below_normal")
    job.assign_self()
    from datagoblin.serialization import canonical_json, dump_structured
    from datagoblin.training_gym.rlt_gym.aleph_proxy import run

    started = time.monotonic()
    status = "aborted"

    def guard():
        nonlocal available, commit
        free, headroom = memory()
        available, commit = min(available, free), min(commit, headroom)
        if free < RESERVE or headroom < COMMIT_FLOOR or (args.out / "STOP").exists():
            raise SystemExit("stopped at the resource guard")

    try:
        result = run(args.campaign.resolve(), args.out.resolve(), guard, cohort=args.cohort)
        print(canonical_json(result["policies"]))
        status = "complete"
    finally:
        if args.out.is_dir():
            dump_structured(args.out / "resources.json", {
                "status": status, "device": "cpu", "threads": 1, "hard_cap_bytes": CAP,
                "peak_job_commit_bytes": job.peak_bytes(), "minimum_available_bytes": available,
                "minimum_commit_headroom_bytes": commit, "wall_seconds": time.monotonic() - started,
            })


if __name__ == "__main__":
    main()
