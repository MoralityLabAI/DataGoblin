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
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--cohort", type=Path, help="Reuse an already frozen development and held-out cohort")
    variant.add_argument("--previous-proxy", type=Path, help="Fit and test hard-pool repair using this completed proxy")
    variant.add_argument("--resume-hard", action="store_true", help="Resume a fitted hard-pool evaluation")
    parser.add_argument("--wait-seconds", type=int, default=0, help="Maximum wait for RAM headroom")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("the resource wrapper requires Windows")
    if args.out.exists() != args.resume_hard:
        parser.error("use an existing output with --resume-hard or a fresh output for training")
    allowance = CAP
    if args.resume_hard:
        from json import loads

        prior_run = loads((args.out / "resources.json").read_text(encoding="utf-8"))
        allowance = min(CAP, prior_run["peak_job_commit_bytes"] + 32 * MIB)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        available, commit = memory()
        if available >= RESERVE + allowance and commit >= COMMIT_FLOOR + allowance:
            break
        if time.monotonic() >= deadline:
            raise SystemExit("insufficient free memory for the capped proxy run")
        time.sleep(5)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path[:0] = [str(args.ramcap_package.resolve()), str(Path(__file__).resolve().parents[1] / "src")]
    from lcgrt.ramcap import Job

    job = Job(CAP, priority="below_normal")
    job.assign_self()
    from datagoblin.serialization import canonical_json, dump_structured

    started = time.monotonic()
    status = "aborted"

    def guard():
        nonlocal available, commit
        free, headroom = memory()
        available, commit = min(available, free), min(commit, headroom)
        if free < RESERVE or headroom < COMMIT_FLOOR or (args.out / "STOP").exists():
            raise SystemExit("stopped at the resource guard")

    try:
        if args.resume_hard:
            from datagoblin.training_gym.rlt_gym.hard_repair import resume

            result = resume(args.campaign.resolve(), args.out.resolve(), guard)
        elif args.previous_proxy is None:
            from datagoblin.training_gym.rlt_gym.aleph_proxy import run

            result = run(args.campaign.resolve(), args.out.resolve(), guard, cohort=args.cohort)
        else:
            from datagoblin.training_gym.rlt_gym.hard_repair import run as run_hard

            result = run_hard(args.campaign.resolve(), args.previous_proxy.resolve(), args.out.resolve(), guard)
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
