"""Run a serial, RAM-capped CPU pilot using the existing Windows Job Object helper."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
from pathlib import Path

import psutil

GIB = 2**30
COMMIT_LIMIT = 256 * 2**20
RUN_FREE = 3 * GIB
ADMISSION_FREE = RUN_FREE + COMMIT_LIMIT
ADMISSION_WAIT_SECONDS = 180
MAX_SECONDS = 20 * 60
SEED = 131
STEPS = 512


class ResourcePause(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--ramcap-package", type=Path, required=True, help="parent directory of lcgrt")
    args = parser.parse_args()
    root = args.root.resolve()
    if sys.platform != "win32":
        raise SystemExit("This launcher requires the Windows Job Object cap")
    wait_started = time.monotonic()
    while psutil.virtual_memory().available < ADMISSION_FREE:
        if (root / "STOP").exists() or time.monotonic() - wait_started >= ADMISSION_WAIT_SECONDS:
            print("RAM admission canceled or timed out before loading the model", flush=True)
            raise SystemExit(75)
        print("Waiting for 256 MiB job headroom above the 3 GiB system reserve", flush=True)
        time.sleep(10)
    output = root / "run"
    output.mkdir(exist_ok=True)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path[:0] = [str(args.ramcap_package.resolve()), str(Path(__file__).resolve().parents[1] / "src")]
    from lcgrt import ramcap

    job = ramcap.Job(COMMIT_LIMIT, priority="below_normal")
    job.assign_self()

    from datagoblin.hashing import sha256_file
    from datagoblin.serialization import canonical_json, dump_structured, load_structured
    from datagoblin.training_gym import rlt_pilot_data

    started = time.monotonic()
    low_available = psutil.virtual_memory().available
    max_rss = 0
    last_sample = started - 5
    mode = None
    stage = "freeze"
    previous = load_structured(output / "run.json") if (output / "run.json").exists() else {}
    if previous.get("status") == "complete":
        return
    if previous and previous["status"] != "paused":
        raise SystemExit("Existing run is not a clean pause. Preserve it before launching another attempt.")
    lock_path = output / "active.lock"
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    receipt = {
        "status": "started",
        "device": "cpu",
        "hard_job_commit_limit_bytes": COMMIT_LIMIT,
        "admission_available_bytes": low_available,
        "runtime_available_floor_bytes": RUN_FREE,
        "threads": 1,
        "priority": "below_normal",
        "max_wall_seconds": MAX_SECONDS,
        "ramcap_source_sha256": sha256_file(Path(ramcap.__file__)),
    }
    dump_structured(output / "run.json", receipt)

    def guard() -> None:
        nonlocal last_sample, low_available, max_rss
        now = time.monotonic()
        if now - started >= MAX_SECONDS:
            raise RuntimeError("pilot reached its wall time limit")
        if now - last_sample < 2:
            return
        last_sample = now
        free = psutil.virtual_memory().available
        rss = psutil.Process().memory_info().rss
        low_available = min(low_available, free)
        max_rss = max(max_rss, rss)
        with (output / "resources.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(canonical_json({
                "elapsed_s": now - started, "stage": stage, "mode": mode,
                "host_available_bytes": free, "rss_bytes": rss, "job_peak_commit_bytes": job.peak_bytes(),
            }) + "\n")
        if free < RUN_FREE:
            raise ResourcePause("host available RAM dropped below the 3 GiB reserve")
        if (root / "STOP").exists():
            raise RuntimeError("operator requested stop")

    try:
        guard()
        manifest_path = root / "frozen" / "manifest.json"
        if not manifest_path.exists():
            rlt_pilot_data.freeze(root, secrets.randbits(128))
        frozen = load_structured(manifest_path)
        for filename, key in (("known.json", "known_sha256"), ("text_train.json", "train_text_sha256")):
            if sha256_file(root / "frozen" / filename) != frozen[key]:
                raise ValueError(f"frozen input changed: {filename}")
        rule_root = root / "rules"
        rule_manifest = rule_root / "manifest.json"
        if sha256_file(rule_manifest) != frozen["rules_manifest_sha256"]:
            raise ValueError("frozen rule manifest changed")
        rule_train = load_structured(rule_manifest)["splits"]["train"]
        for filetype in ("tasks", "targets"):
            if sha256_file(rule_root / f"train.{filetype}.jsonl") != rule_train[f"{filetype}_sha256"]:
                raise ValueError("frozen rule training input changed")
        stage = "backend_import"
        guard()
        import numpy as np

        from datagoblin.training_gym import rlt_pilot

        train_plan_sha = sha256_file(manifest_path)
        protocol = {
            "seed": SEED, "updates_per_arm": STEPS, "feedback_modes": rlt_pilot.MODES,
            "train_depth": rlt_pilot.TRAIN_DEPTH, "depths": rlt_pilot.DEPTHS,
            "width": rlt_pilot.WIDTH, "heads": rlt_pilot.HEADS,
            "training_examples_per_update": {"text": rlt_pilot.TEXT_BATCH, "rule_turns": rlt_pilot.RULE_BATCH},
            "optimizer": "AdamW lr 0.002 weight decay 0.01 gradient clip 1",
            "supervision": "mean per-visit text CE plus mean per-visit endpoint CE",
            "frozen_manifest_sha256": train_plan_sha, "backend": "NumPy/autograd", "numpy": np.__version__,
            "source_sha256": {
                "launcher": sha256_file(Path(__file__)),
                "trainer": sha256_file(Path(rlt_pilot.__file__)),
                "data": sha256_file(Path(rlt_pilot_data.__file__)),
            },
        }
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            if canonical_json(load_structured(protocol_path)) != canonical_json(protocol):
                raise ValueError("resume protocol changed")
        else:
            dump_structured(protocol_path, protocol)
        stage = "train"
        init_hashes = set()
        for mode in rlt_pilot.MODES:
            arm = output / mode
            arm.mkdir(exist_ok=True)
            training = rlt_pilot.train(root, arm, mode, SEED, STEPS, guard)
            init_hashes.add(training["init_sha256"])
        if len(init_hashes) != 1:
            raise RuntimeError("training initialization differs across feedback arms")
        stage = "evaluate"
        for mode in rlt_pilot.MODES:
            rlt_pilot.evaluate(root, output / mode, mode, guard)
        stage = "complete"
        receipt["status"] = "complete"
    except ResourcePause as exc:
        receipt["status"] = "paused"
        receipt["error"] = str(exc)
        raise SystemExit(75) from exc
    except BaseException as exc:
        receipt["status"] = "aborted"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        segment = {
            "status": receipt["status"], "stage": stage, "mode": mode, "elapsed_s": time.monotonic() - started,
            "minimum_available_bytes": low_available, "peak_rss_bytes": max_rss, "peak_commit_bytes": job.peak_bytes(),
        }
        receipt["segments"] = [*previous.get("segments", []), segment]
        receipt.update(stage=stage, mode=mode,
                       elapsed_s=sum(r["elapsed_s"] for r in receipt["segments"]),
                       min_host_available_bytes=min(r["minimum_available_bytes"] for r in receipt["segments"]),
                       sampled_peak_rss_bytes=max(r["peak_rss_bytes"] for r in receipt["segments"]),
                       peak_job_commit_bytes=max(r["peak_commit_bytes"] for r in receipt["segments"]))
        dump_structured(output / "run.json", receipt)
        lock_path.unlink()
        print(canonical_json(receipt), flush=True)


if __name__ == "__main__":
    main()
