"""Run the live Lean stopping comparison under a 256 MiB process-tree cap."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from run_rlt_trm_windows import COMMIT_CAP, COMMIT_FLOOR, FREE_FLOOR, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "fit", "evaluate", "certify"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--torch-cpu", type=Path)
    parser.add_argument("--upstream-ldt", type=Path)
    parser.add_argument("--ramcap-package", type=Path, required=True)
    parser.add_argument("--host-free-floor-mib", type=int, default=FREE_FLOOR // 2**20)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--grounded", action="store_true")
    mode.add_argument("--trained", action="store_true")
    mode.add_argument("--composition", action="store_true")
    mode.add_argument("--pretrain", action="store_true")
    parser.add_argument("--input-spec", type=Path)
    parser.add_argument("--previous-fixture", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("requires a Windows Job Object")
    if args.phase == "prepare" and not args.pretrain and (args.checkpoints is None or args.torch_cpu is None):
        parser.error("prepare requires --checkpoints and --torch-cpu")
    if args.phase == "evaluate" and args.upstream_ldt is None:
        parser.error("evaluate requires --upstream-ldt")
    if args.phase == "collect" and (args.previous_fixture is None or args.upstream_ldt is None):
        parser.error("collect requires --previous-fixture and --upstream-ldt")
    if args.phase == "collect" and not args.trained:
        parser.error("collect requires --trained")
    if args.phase == "fit" and not (args.trained or args.pretrain):
        parser.error("fit requires --trained or --pretrain")
    if args.trained and args.phase == "prepare":
        parser.error("the trained ranker starts at collect")
    if args.composition and args.phase not in ("evaluate", "certify"):
        parser.error("composition supports evaluate and certify")
    if args.composition and args.phase == "evaluate" and args.previous_fixture is None:
        parser.error("composition evaluation requires --previous-fixture")
    if args.pretrain and args.phase not in ("prepare", "fit", "evaluate"):
        parser.error("pretrain supports prepare, fit and evaluate")
    if args.pretrain and args.phase == "prepare" and args.input_spec is None:
        parser.error("pretrain preparation requires --input-spec")
    if args.host_free_floor_mib <= 0:
        parser.error("host free-memory reserve must be positive")
    runtime_floor = args.host_free_floor_mib * 2**20
    root = args.out.resolve()
    deadline = time.monotonic() + 180
    while True:
        free, headroom = memory()
        if (root / "STOP").exists() or time.monotonic() >= deadline:
            raise SystemExit(75)
        if free >= runtime_floor + COMMIT_CAP and headroom >= COMMIT_FLOOR + COMMIT_CAP:
            break
        time.sleep(8)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "LEAN_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MKL_DISABLE_FAST_MM"] = "1"
    os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "0"
    paths = [args.ramcap_package.resolve(), Path(__file__).resolve().parents[1] / "src"]
    paths += [p.resolve() for p in (args.torch_cpu, args.upstream_ldt) if p is not None]
    sys.path[:0] = list(map(str, paths))
    from lcgrt.ramcap import Job

    job = Job(COMMIT_CAP, priority="below_normal")
    job.assign_self()
    root.mkdir(parents=True, exist_ok=True)

    from datagoblin.serialization import canonical_json, dump_structured

    started = time.monotonic()
    last_check = started - 1
    status = "aborted"

    def guard():
        nonlocal last_check, free, headroom
        now = time.monotonic()
        if (root / "STOP").exists():
            raise SystemExit(75)
        if now - last_check >= 1:
            last_check = now
            available, commit_left = memory()
            free, headroom = min(free, available), min(headroom, commit_left)
            if available < runtime_floor or commit_left < COMMIT_FLOOR:
                raise SystemExit(75)

    lock = root / "active.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        guard()
        if args.pretrain:
            if args.phase == "prepare":
                from datagoblin.training_gym.rlt_pretrain_data import prepare

                prepare(root, args.input_spec.resolve(), guard)
            elif args.phase == "fit":
                from datagoblin.training_gym.rlt_pretrain_fit import fit

                fit(root, guard)
            else:
                from datagoblin.training_gym.rlt_pretrain_eval import evaluate

                evaluate(root, guard)
        elif args.phase == "prepare":
            import torch

            from datagoblin.training_gym.rlt_lean_rank import export

            if not torch.__version__.endswith("+cpu"):
                raise ValueError("expected CPU-only Torch")
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            export(root, args.checkpoints.resolve(), torch, guard)
        elif args.phase == "collect":
            from datagoblin.training_gym.rlt_lean_labels import collect

            collect(root, args.previous_fixture.resolve(), guard)
        elif args.composition:
            from datagoblin.training_gym import rlt_lean_pool_audit

            if args.phase == "evaluate":
                rlt_lean_pool_audit.evaluate(root, args.previous_fixture.resolve(), guard)
            else:
                rlt_lean_pool_audit.certify(root, guard)
        elif args.trained:
            if args.phase == "fit":
                from datagoblin.training_gym.rlt_lean_train_rank import fit

                fit(root, guard)
            else:
                from datagoblin.training_gym import rlt_lean_rank_eval

                action = rlt_lean_rank_eval.evaluate if args.phase == "evaluate" else rlt_lean_rank_eval.certify
                action(root, guard)
        else:
            if args.grounded:
                from datagoblin.training_gym import rlt_lean_grounded as experiment
            else:
                from datagoblin.training_gym import rlt_lean_ldt as experiment

            action = experiment.evaluate if args.phase == "evaluate" else experiment.certify
            action(root, guard)
        guard()
        if job.peak_bytes() > COMMIT_CAP:
            raise RuntimeError("the job recorded a memory request above the cap")
        status = "complete"
    except SystemExit as exc:
        if exc.code == 75:
            status = "paused"
        raise
    finally:
        receipt = {
            "phase": args.phase, "status": status, "hard_commit_cap_bytes": COMMIT_CAP, "cpu_workers": 1,
            "wall_s": time.monotonic() - started, "job_peak_commit_bytes": job.peak_bytes(),
            "min_available_bytes": free, "min_commit_headroom_bytes": headroom,
            "host_free_floor_bytes": runtime_floor,
        }
        dump_structured(root / f"{args.phase}-resources.json", receipt)
        if args.trained or args.composition or args.pretrain:
            with (root / "resource-history.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(receipt) + "\n")
        lock.unlink()
        print(canonical_json(receipt), flush=True)


if __name__ == "__main__":
    main()
