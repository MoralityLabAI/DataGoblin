"""Run the binary carry comparison under the same Windows Job Object cap."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
from pathlib import Path

from run_rlt_trm_windows import COMMIT_CAP, COMMIT_FLOOR, FREE_FLOOR, ResourcePause, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ramcap-package", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if sys.platform != "win32":
        raise SystemExit("This launcher requires a Windows Job Object")
    limit = time.monotonic() + 180
    while True:
        free, commit = memory()
        if (out / "STOP").exists() or time.monotonic() >= limit:
            raise SystemExit(75)
        if free >= FREE_FLOOR + COMMIT_CAP and commit >= COMMIT_FLOOR + COMMIT_CAP:
            break
        time.sleep(8)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path[:0] = [str(args.ramcap_package.resolve()), str(Path(__file__).resolve().parents[1] / "src")]
    from lcgrt.ramcap import Job

    job = Job(COMMIT_CAP, priority="below_normal")
    job.assign_self()

    from datagoblin.hashing import sha256_file
    from datagoblin.serialization import canonical_json, dump_structured, load_structured
    from datagoblin.training_gym import rlt_hrm_carry as pilot
    from datagoblin.training_gym import rlt_pilot
    from datagoblin.training_gym.rlt_pilot_data import jsonl

    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "resources.json"
    history = load_structured(receipt)["segments"] if receipt.exists() else []
    if history and history[-1]["status"] == "complete":
        return
    if history and history[-1]["status"] != "paused":
        raise SystemExit("Previous run did not pause cleanly")
    lock = out / "active.lock"
    with lock.open("x", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))
    started = time.monotonic()
    prior = sum(segment["wall_s"] for segment in history)
    last = started - 2
    low_free, low_commit = free, commit
    stage, status = "freeze", "aborted"

    def guard():
        nonlocal last, low_free, low_commit
        now = time.monotonic()
        if (out / "STOP").exists() or prior + now - started >= 900:
            raise RuntimeError("operator stop or wall-time limit")
        if now - last >= 1:
            last = now
            free, commit = memory()
            low_free, low_commit = min(low_free, free), min(low_commit, commit)
            if free < FREE_FLOOR or commit < COMMIT_FLOOR:
                raise ResourcePause("host memory floor crossed")

    try:
        guard()
        training = {seed: pilot.batches(seed) for seed in pilot.SEEDS}
        tasks = out / "tasks.jsonl"
        if not tasks.exists():
            excluded = {tuple(pair) for plan in training.values() for batch in plan for pair in batch.T}
            with rlt_pilot.pending_rows(tasks) as fh:
                for task in pilot.cases(secrets.randbits(128), excluded):
                    fh.write(canonical_json(task) + "\n")
        protocol = {
            "tasks_sha256": sha256_file(tasks), "seeds": pilot.SEEDS, "arms": pilot.ARMS,
            "slow_depths": pilot.SLOW_DEPTHS, "train_slow": pilot.TRAIN_SLOW, "fast_per_slow": pilot.FAST_PER_SLOW,
            "updates": pilot.UPDATES, "batch": pilot.BATCH, "lr": pilot.LR,
            "source_sha256": {
                name: sha256_file(path) for name, path in (
                    ("runner", Path(__file__)), ("model", Path(pilot.__file__)), ("schedule", Path(rlt_pilot.__file__)),
                )
            },
        }
        protocol_path = out / "protocol.json"
        if protocol_path.exists():
            if canonical_json(load_structured(protocol_path)) != canonical_json(protocol):
                raise ValueError("the frozen carry protocol changed")
        else:
            dump_structured(protocol_path, protocol)
        stage = "fit"
        for seed in pilot.SEEDS:
            matched = set()
            for arm in pilot.ARMS:
                directory = out / str(seed) / arm
                directory.mkdir(parents=True, exist_ok=True)
                result = pilot.train(directory, seed, arm, training[seed], guard)
                matched.add((result["init_sha256"], result["batches_sha256"], result["params"]))
            if len(matched) != 1:
                raise ValueError("unmatched fitting conditions")
        del training
        stage = "evaluate"
        task_rows = jsonl(tasks)
        for seed in pilot.SEEDS:
            for arm in pilot.ARMS:
                pilot.evaluate(out / str(seed) / arm, arm, task_rows, guard)
        status = "complete"
    except ResourcePause as exc:
        status = "paused"
        raise SystemExit(75) from exc
    finally:
        segment = {
            "status": status, "stage": stage, "wall_s": time.monotonic() - started,
            "cpu_workers": 1, "hard_commit_cap_bytes": COMMIT_CAP, "peak_commit_bytes": job.peak_bytes(),
            "min_available_bytes": low_free, "min_headroom_bytes": low_commit,
        }
        dump_structured(receipt, {"segments": [*history, segment]})
        lock.unlink()
        print(canonical_json(segment), flush=True)


if __name__ == "__main__":
    main()
