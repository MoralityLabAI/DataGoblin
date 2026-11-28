"""AFG-03 local campaign driver: BF16 cell, then an FP32 cell only after a BF16 parity failure.

Each cell runs afg03_checkpoint.py under afg03_watchdog.py in its own
fresh directory. The FP32 cell is a separately identified precision run,
started only when the BF16 cell stopped on an exact-parity assertion; a
watchdog kill, timeout or any other error ends the campaign.
`--cells fp32` runs only the FP32 cell (after a recorded BF16 parity
failure in an earlier campaign). Every invocation makes a new
`attempt-<UTC time>` directory under --root, so a supervisor restart
never reuses an output directory. A runner thermal pause (exit 75) is
returned as 75 so `vramorama run --pause-exit 75 --restart N` can retry
once the card is clear. Completed cells get the weight-free receipt
audit. Run it inside a VRAM lease:

    vramorama run --vram 11G --owner afg03 --guard-power 120 --pause-exit 75 --restart 3 -- python afg03_local.py ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARITY_STAGES = ("parity", "teacher", "calibration", "development")
PAUSE_EXIT = 75


def run_cell(a, root, precision, panel_sha):
    out, logs = root/precision, root/f"{precision}-watchdog"
    runner = [a.python, str(HERE/"afg03_checkpoint.py"), "--execute", "--checkpoint", str(a.checkpoint),
              "--data", str(a.panel), "--data-sha256", panel_sha, "--device", "cuda:0", "--precision", precision,
              "--allocator-gib", "10", "--steps", str(a.steps), "--max-new", "8", "--repeats", "3",
              "--threads", "2", "--wall-seconds", str(a.runner_wall), "--out", str(out)]
    cmd = [a.python, str(HERE/"afg03_watchdog.py"), "--log-dir", str(logs), "--wall-seconds", str(a.wall),
           "--ram-gib", "12", "--min-available-gib", "1.0", "--min-free-vram-gib", "10.5", "--", *runner]
    started = time.time()
    code = subprocess.run(cmd, check=False).returncode
    cell = {"precision": precision, "out": str(out), "watchdog_exit": code, "seconds": time.time()-started}
    summary = logs/"watchdog_summary.json"
    if summary.exists():
        w = json.loads(summary.read_text(encoding="utf-8"))
        cell["watchdog"] = {k: w.get(k) for k in ("status", "exit_code", "violations", "peaks", "elapsed_seconds",
                                                  "power_capped_samples")}
    progress = out/"progress.json"
    cell["progress"] = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else None
    if cell["progress"] and cell["progress"].get("status") == "complete":
        audit = subprocess.run([a.python, str(HERE/"afg03_checkpoint.py"), "--audit-report", str(out)],
                               capture_output=True, text=True, check=False)
        cell["audit"] = {"exit": audit.returncode, "stdout": audit.stdout.strip()[-2000:],
                         "stderr": audit.stderr.strip()[-2000:]}
    return cell


def parity_failure(cell):
    p = cell["progress"] or {}
    return (p.get("status") == "blocked" and p.get("error_type") == "AssertionError"
            and p.get("stage") in PARITY_STAGES and not (cell.get("watchdog") or {}).get("violations"))


def paused(cell):
    return (cell["progress"] or {}).get("error_type") == "ThermalPause" and cell["watchdog_exit"] == PAUSE_EXIT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True, help="local campaign directory; one attempt per invocation")
    p.add_argument("--cells", choices=("bf16-then-fp32", "fp32"), default="bf16-then-fp32")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--wall", type=int, default=1800, help="external watchdog wall cap per cell")
    p.add_argument("--runner-wall", type=int, default=1740, help="runner deadline, inside the watchdog cap")
    a = p.parse_args()
    root = a.root/time.strftime("attempt-%Y%m%dT%H%M%SZ", time.gmtime())
    root.mkdir(parents=True, exist_ok=False)
    panel_sha = hashlib.sha256(a.panel.read_bytes()).hexdigest()
    campaign = {"schema": "aleph.afg03.local_campaign.v1", "panel_sha256": panel_sha, "cells_mode": a.cells,
                "wall": a.wall, "runner_wall": a.runner_wall, "cells": []}
    if a.cells == "fp32":
        campaign["cells"].append(run_cell(a, root, "fp32", panel_sha))
    else:
        bf16 = run_cell(a, root, "bf16", panel_sha)
        campaign["cells"].append(bf16)
        if parity_failure(bf16):
            campaign["fp32_reason"] = "bf16 exact-parity failure at stage " + bf16["progress"]["stage"]
            campaign["cells"].append(run_cell(a, root, "fp32", panel_sha))
    (root/"campaign.json").write_text(json.dumps(campaign, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps([{k: c.get(k) for k in ("precision", "watchdog_exit")} | {
        "status": (c["progress"] or {}).get("status"), "stage": (c["progress"] or {}).get("stage")}
        for c in campaign["cells"]], sort_keys=True))
    if any((c["progress"] or {}).get("status") == "complete" for c in campaign["cells"]):
        return 0
    return PAUSE_EXIT if paused(campaign["cells"][-1]) else 1


if __name__ == "__main__":
    sys.exit(main())
