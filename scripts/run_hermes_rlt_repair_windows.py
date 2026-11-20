"""Run the Hermes Campsite replay serially with host and GPU resource limits."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from run_rlt_trm_windows import COMMIT_FLOOR, ResourcePause, memory

MIB = 2**20
HOST_RESERVE = 1536 * MIB
GPU_BUDGET = {"rollout": 2304 * MIB, "fit": 896 * MIB}
GPU_RESERVE = 640 * MIB


def gpu_free():
    info = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True,
    )
    return int(info.strip()) * MIB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "rollout", "fit"))
    parser.add_argument("--cpu", "--cpu-fit", action="store_true", help="Use one CPU thread for fit or rollout")
    for name in ("out", "study", "hermes-lite", "ramcap-package"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--llama-server", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("This launcher uses a Windows Job Object")
    if args.phase == "rollout" and args.model is None:
        parser.error("rollout requires a local model")
    if args.cpu and args.phase == "prepare":
        parser.error("--cpu applies to training or inference")
    gguf = args.llama_server is not None
    if gguf and (args.phase != "rollout" or args.model.suffix.lower() != ".gguf"):
        parser.error("--llama-server requires rollout with a local GGUF model")
    root = args.out.resolve()
    gpu = args.phase != "prepare" and not args.cpu and not gguf
    gpu_cap = GPU_BUDGET[args.phase] if gpu else 0
    host_caps = {"prepare": 256, "fit": 1280, "rollout": 768 if gguf else (1536 if gpu else 3072)}
    cap = host_caps[args.phase] * MIB
    deadline = time.monotonic() + 180
    while True:
        free, commit = memory()
        if (root / "STOP").exists() or time.monotonic() >= deadline:
            raise SystemExit(75)
        if (free >= HOST_RESERVE + cap and commit >= COMMIT_FLOOR + cap
                and (not gpu or gpu_free() >= gpu_cap + GPU_RESERVE)):
            break
        time.sleep(8)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if gpu else ""
    sys.path[:0] = [str(args.ramcap_package.resolve()), str(Path(__file__).resolve().parents[1] / "src")]
    from lcgrt.ramcap import Job

    job = Job(cap, priority="below_normal")
    job.assign_self()
    from datagoblin.serialization import canonical_json
    from datagoblin.training_gym import rlt_hermes_repair as experiment

    root.mkdir(parents=True, exist_ok=True)
    lock = root / "active.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    started, last_gpu_poll = time.monotonic(), 0.0
    status, peak_gpu, minimum_gpu = "aborted", None, None

    def guard():
        nonlocal free, commit, last_gpu_poll, minimum_gpu
        available, headroom = memory()
        free, commit = min(free, available), min(commit, headroom)
        if available < HOST_RESERVE or headroom < COMMIT_FLOOR or (root / "STOP").exists():
            raise ResourcePause("host reserve paused the experiment")
        now = time.monotonic()
        if gpu and now - last_gpu_poll >= 10:
            remaining = gpu_free()
            last_gpu_poll = now
            minimum_gpu = remaining if minimum_gpu is None else min(minimum_gpu, remaining)
            if remaining < GPU_RESERVE:
                raise ResourcePause("GPU reserve paused the experiment")

    try:
        guard()
        if args.phase != "prepare" and not gguf:
            print(f"Initializing PyTorch for {args.phase}", flush=True)
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        if gpu:
            print(f"Applying CUDA memory limit for {args.phase}", flush=True)
            torch.cuda.set_per_process_memory_fraction(gpu_cap / torch.cuda.get_device_properties(0).total_memory, 0)
        if args.phase == "prepare":
            experiment.prepare(root, args.study.resolve(), args.hermes_lite.resolve(), guard)
        elif gguf:
            experiment.rollout_gguf(root, args.model.resolve(), args.llama_server.resolve(), guard)
        elif args.phase == "rollout":
            print("Loading the registered local Qwen checkpoint", flush=True)
            experiment.rollout(root, args.model.resolve(), "cuda:0" if gpu else "cpu", guard)
        else:
            from datagoblin.training_gym.rlt_hermes_fit import compare

            compare(root, args.study.resolve(), args.hermes_lite.resolve(), "cuda" if gpu else "cpu", guard)
        status = "complete"
    except ResourcePause as exc:
        status = "paused"
        raise SystemExit(75) from exc
    finally:
        if gpu and "torch" in locals() and torch.cuda.is_initialized():
            peak_gpu = torch.cuda.max_memory_reserved()
        receipt = {"phase": args.phase, "status": status, "hard_host_cap_bytes": cap,
                   "host_peak_commit_bytes": job.peak_bytes(), "minimum_available_bytes": free,
                   "minimum_commit_headroom_bytes": commit, "gpu_peak_reserved_bytes": peak_gpu,
                   "gpu_allocator_cap_bytes": gpu_cap or None, "minimum_gpu_free_bytes": minimum_gpu,
                   "wall_s": time.monotonic() - started}
        with (root / "resource-history.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(receipt) + "\n")
        lock.unlink()
        print(canonical_json(receipt), flush=True)


if __name__ == "__main__":
    main()
