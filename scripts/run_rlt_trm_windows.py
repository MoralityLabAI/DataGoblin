"""Run the matched RLT-TRM pilot serially under a Windows Job Object memory cap."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import os
import sys
import time
from pathlib import Path

import psutil

GIB = 2**30
COMMIT_CAP = 256 * 2**20
FREE_FLOOR = 3 * GIB
COMMIT_FLOOR = GIB
MAX_SECONDS = 20 * 60


class Performance(ctypes.Structure):
    _fields_ = [("size", wt.DWORD)] + [
        (key, ctypes.c_size_t) for key in (
            "commit_total", "commit_limit", "commit_peak", "physical_total", "physical_available",
            "system_cache", "kernel_total", "kernel_paged", "kernel_nonpaged", "page_size",
        )
    ] + [(key, wt.DWORD) for key in ("handles", "processes", "threads")]


class ResourcePause(RuntimeError):
    pass


def memory():
    info = Performance()
    info.size = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.size):
        raise OSError("GetPerformanceInfo failed")
    return info.physical_available * info.page_size, (info.commit_limit - info.commit_total) * info.page_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ramcap-package", type=Path, required=True, help="parent directory of lcgrt")
    args = parser.parse_args()
    pack, out = args.pack.resolve(), args.out.resolve()
    if sys.platform != "win32":
        raise SystemExit("This launcher requires a Windows Job Object")
    wait_started = time.monotonic()
    while True:
        free, headroom = memory()
        if (out / "STOP").exists() or time.monotonic() - wait_started >= 180:
            raise SystemExit(75)
        if free >= FREE_FLOOR + COMMIT_CAP and headroom >= COMMIT_FLOOR + COMMIT_CAP:
            break
        print("Waiting for physical and commit headroom", flush=True)
        time.sleep(10)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path[:0] = [str(args.ramcap_package.resolve()), str(Path(__file__).resolve().parents[1] / "src")]
    from lcgrt.ramcap import Job

    job = Job(COMMIT_CAP, priority="below_normal")
    job.assign_self()

    from datagoblin.hashing import sha256_file
    from datagoblin.serialization import canonical_json, dump_structured, load_structured
    from datagoblin.training_gym import rlt_pilot, rlt_revision
    from datagoblin.training_gym import rlt_trm as pilot

    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "run.json"
    previous = load_structured(state_path) if state_path.exists() else {}
    if previous.get("status") == "complete":
        return
    if previous and previous["status"] != "paused":
        raise SystemExit("Previous run did not pause cleanly")
    segments = previous.get("segments", [])
    prior_elapsed = sum(segment["elapsed_s"] for segment in segments)
    manifest = pack / "manifest.json"
    frozen = load_structured(manifest)
    if frozen["version"] != rlt_revision.VERSION:
        raise ValueError("unexpected rule pack version")
    protocol = {
        "seeds": pilot.SEEDS, "steps": pilot.STEPS, "modes": pilot.MODES, "batch": pilot.BATCH,
        "train_depth": pilot.TRAIN_DEPTH, "depths": pilot.DEPTHS, "crossover_depth": pilot.TRAIN_DEPTH,
        "lr": pilot.LR, "rule_manifest_sha256": sha256_file(manifest), "numpy": pilot.np.__version__,
        "source_sha256": {
            "runner": sha256_file(Path(__file__)), "model": sha256_file(Path(pilot.__file__)),
            "feedback": sha256_file(Path(rlt_pilot.__file__)), "scorer": sha256_file(Path(rlt_revision.__file__)),
        },
    }
    protocol_path = out / "protocol.json"
    lock = out / "active.lock"
    with lock.open("x", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))
    start = time.monotonic()
    last_sample = start - 2
    rss_peak = 0
    status, stage, current = "aborted", "freeze", None
    error = None
    process = psutil.Process()

    def guard():
        nonlocal last_sample, free, headroom, rss_peak
        now = time.monotonic()
        if (out / "STOP").exists() or prior_elapsed + now - start >= MAX_SECONDS:
            raise RuntimeError("operator stop or wall-time limit")
        if now - last_sample < 2:
            return
        last_sample = now
        available, commit_left = memory()
        rss = process.memory_info().rss
        free, headroom = min(free, available), min(headroom, commit_left)
        rss_peak = max(rss_peak, rss)
        with (out / "resources.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(canonical_json({
                "elapsed_s": prior_elapsed + now - start, "stage": stage, "arm": current,
                "available_bytes": available, "commit_headroom_bytes": commit_left,
                "rss_bytes": rss, "peak_job_commit_bytes": job.peak_bytes(),
            }) + "\n")
        if available < FREE_FLOOR or commit_left < COMMIT_FLOOR:
            raise ResourcePause("host headroom fell below the physical RAM or commit floor")

    def verify_split(split, kind):
        path = pack / f"{split}.{kind}.jsonl"
        if sha256_file(path) != frozen["splits"][split][f"{kind}_sha256"]:
            raise ValueError(f"frozen {split} {kind} changed")

    try:
        guard()
        if protocol_path.exists():
            if canonical_json(load_structured(protocol_path)) != canonical_json(protocol):
                raise ValueError("the frozen protocol changed")
        else:
            dump_structured(protocol_path, protocol)
        verify_split("train", "tasks")
        verify_split("train", "targets")
        stage = "train"
        training = pilot.load_training(pack)
        for seed in pilot.SEEDS:
            matched = set()
            for mode in pilot.MODES:
                current = [seed, mode]
                arm = out / str(seed) / mode
                arm.mkdir(parents=True, exist_ok=True)
                result = pilot.train(pack, arm, mode, seed, training, guard)
                matched.add((result["init_sha256"], result["schedule_sha256"], result["cell_calls"]))
            if len(matched) != 1:
                raise ValueError("unmatched initialization, batch plan, or compute")
        del training
        stage = "evaluate"
        for split in pilot.RULE_SPLITS:
            verify_split(split, "tasks")
        for seed in pilot.SEEDS:
            for mode in pilot.MODES:
                current = [seed, mode]
                pilot.evaluate(pack, out / str(seed) / mode, mode, guard)
        status, stage = "complete", "complete"
    except ResourcePause as exc:
        status, error = "paused", str(exc)
        raise SystemExit(75) from exc
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        segments.append({
            "status": status, "stage": stage, "arm": current, "error": error,
            "elapsed_s": time.monotonic() - start, "min_available_bytes": free,
            "min_commit_headroom_bytes": headroom, "peak_rss_bytes": rss_peak, "peak_commit_bytes": job.peak_bytes(),
        })
        receipt = {
            "status": status, "segments": segments, "cpu_workers": 1, "priority": "below_normal",
            "hard_job_commit_limit_bytes": COMMIT_CAP, "runtime_available_floor_bytes": FREE_FLOOR,
            "runtime_commit_floor_bytes": COMMIT_FLOOR, "max_wall_seconds": MAX_SECONDS,
            "peak_job_commit_bytes": max(s["peak_commit_bytes"] for s in segments),
            "peak_sampled_rss_bytes": max(s["peak_rss_bytes"] for s in segments),
            "min_host_available_bytes": min(s["min_available_bytes"] for s in segments),
        }
        dump_structured(state_path, receipt)
        lock.unlink()
        print(canonical_json(receipt), flush=True)


if __name__ == "__main__":
    main()
