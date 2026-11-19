"""Run a NumPy Sudoku HRM checkpoint pilot under the Windows Job Object cap."""

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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ramcap-package", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    if sys.platform != "win32":
        raise SystemExit("This runner requires a Windows Job Object")
    deadline = time.monotonic() + 180
    while True:
        free, commit = memory()
        if (root / "STOP").exists() or time.monotonic() >= deadline:
            raise SystemExit(75)
        if free >= FREE_FLOOR + COMMIT_CAP and commit >= COMMIT_FLOOR + COMMIT_CAP:
            break
        time.sleep(8)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path[:0] = [str(args.ramcap_package.resolve()), str(Path(__file__).resolve().parents[1] / "src")]
    from lcgrt.ramcap import Job

    job = Job(COMMIT_CAP, priority="below_normal")
    job.assign_self()

    import numpy as np
    import yaml

    from datagoblin.hashing import sha256_file
    from datagoblin.serialization import canonical_json, dump_structured, load_structured
    from datagoblin.training_gym import rlt_hrm_sudoku as pilot
    from datagoblin.training_gym import rlt_hrm_sudoku_cpu as cpu
    from datagoblin.training_gym import rlt_hrm_sudoku_weights as weights

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
        if (root / "STOP").exists() or prior + now - start >= 1200:
            raise RuntimeError("operator stop or active wall-time limit")
        if now - last >= 1:
            last = now
            if process.memory_info().peak_pagefile > COMMIT_CAP:
                raise RuntimeError("process private commit exceeded the configured cap")
            available, headroom = memory()
            free, commit = min(free, available), min(commit, headroom)
            if available < FREE_FLOOR or headroom < COMMIT_FLOOR:
                raise ResourcePause("host memory floor crossed")

    try:
        source = load_structured(root / "source.json")
        config_path = root / "all_config.yaml"
        if (sha256_file(root / "checkpoint") != source["checkpoint_sha256"]
                or sha256_file(config_path) != source["config_sha256"]):
            raise ValueError("release checkpoint or config changed")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["arch"]
        segments, h_cycles, l_cycles = (config[key] for key in ("halt_max_steps", "H_cycles", "L_cycles"))
        if (segments, h_cycles, l_cycles) != (16, 2, 2):
            raise ValueError("unexpected released recurrence")
        holdout = root / "holdout.json"
        items = load_structured(holdout)["puzzles"]
        if len(items) != 24 or len({item["id"] for item in items}) != 24:
            raise ValueError("unexpected fixed test sample")
        reference_root = root / "cpu-preflight-v1" / "predictions"
        references = [reference_root / f"{item['id']}-direct.json" for item in items[:2]]
        if not all(path.is_file() for path in references):
            raise ValueError("missing pinned upstream CPU preflight predictions")
        protocol = {
            "checkpoint": source, "holdout_sha256": sha256_file(holdout), "numpy": np.__version__,
            "forward_precision": "bfloat16 rounding at operation boundaries; fp32 NumPy accumulation",
            "segments": segments, "h_cycles": h_cycles, "l_cycles": l_cycles, "batch": 1,
            "reference_sha256": {path.name: sha256_file(path) for path in references},
            "source_sha256": {
                name: sha256_file(path) for name, path in (
                    ("runner", Path(__file__)), ("scorer", Path(pilot.__file__)),
                    ("cpu", Path(cpu.__file__)), ("weights", Path(weights.__file__)),
                )
            },
            "coefficients": {arm: pilot.slow_coefficients(arm, segments, h_cycles) for arm in pilot.ARMS},
        }
        frozen = root / "protocol.json"
        if frozen.exists() and canonical_json(load_structured(frozen)) != canonical_json(protocol):
            raise ValueError("the pinned run protocol changed")
        if not frozen.exists():
            dump_structured(frozen, protocol)
        guard()
        model = cpu.SudokuHRM(cpu.checkpoint_arrays(root / "checkpoint"), config)
        stage = "parity"
        parity = []
        for item, path in zip(items[:2], references, strict=True):
            reference = load_structured(path)["steps"]
            observed = list(model.rollout(item["givens"], protocol["coefficients"]["direct"], guard))
            pairs = [(a["prediction"], b["prediction"]) for a, b in zip(observed, reference, strict=True)]
            correct = sum(sum(x == y for x, y in zip(a, b, strict=True)) for a, b in pairs)
            parity.append({
                "id": item["id"], "cell_agreement": correct / (81 * segments),
                "board_agreements": sum(a == b for a, b in pairs), "final_board_agrees": pairs[-1][0] == pairs[-1][1],
            })
        dump_structured(root / "parity.json", {"cases": parity, "minimum_cell_agreement": 0.95})
        print(canonical_json({"stage": stage, "cases": parity, "peak_commit": process.memory_info().peak_pagefile}),
              flush=True)
        if any(row["cell_agreement"] < 0.95 or not row["final_board_agrees"] for row in parity):
            raise ValueError("NumPy predictions fail the upstream cross-backend acceptance check")
        stage = "infer"
        predictions = root / "predictions"
        predictions.mkdir(exist_ok=True)
        for item in items:
            for arm, coefficients in protocol["coefficients"].items():
                path = predictions / f"{item['id']}-{arm}.json"
                if path.exists():
                    continue
                rows = list(model.rollout(item["givens"], coefficients, guard))
                pending = path.with_suffix(".pending.json")
                dump_structured(pending, {"id": item["id"], "arm": arm, "steps": rows})
                pending.replace(path)
            print(canonical_json({"id": item["id"], "status": "evaluated"}), flush=True)
        if process.memory_info().peak_pagefile > COMMIT_CAP:
            raise RuntimeError("process private commit exceeded the configured cap")
        status = "complete"
    except ResourcePause as exc:
        status = "paused"
        raise SystemExit(75) from exc
    finally:
        segment = {
            "status": status, "stage": stage, "wall_s": time.monotonic() - start, "cpu_workers": 1,
            "hard_commit_cap_bytes": COMMIT_CAP, "peak_commit_bytes": process.memory_info().peak_pagefile,
            "job_peak_including_refused_bytes": job.peak_bytes(),
            "min_available_bytes": free, "min_headroom_bytes": commit,
        }
        dump_structured(resources, {"segments": [*history, segment]})
        lock.unlink()
        print(canonical_json(segment), flush=True)


if __name__ == "__main__":
    main()
