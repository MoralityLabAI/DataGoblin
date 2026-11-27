"""End-to-end Research Verifiers v2.1 evaluation runner with controller mode & telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from datagoblin.research_v2_1.controllers import CONTROLLER_MODES
from datagoblin.research_v2_1.runner import read_report, run_endpoint


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--binding-template", type=Path, required=True)
    parser.add_argument("--controller-mode", choices=CONTROLLER_MODES, default="autonomous")
    parser.add_argument("--max-episodes", type=int, default=32)
    parser.add_argument("--max-turns-per-episode", type=int, default=8)
    parser.add_argument("--run-wall-seconds", type=float, default=900.0)
    parser.add_argument("--max-requests-cap", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bridge-script", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("worker_cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    worker_cmd = args.worker_cmd
    if worker_cmd and worker_cmd[0] == "--":
        worker_cmd = worker_cmd[1:]
    if not worker_cmd:
        raise SystemExit("worker command required after --")
    if args.out.exists():
        raise SystemExit(f"output directory already exists: {args.out}")

    port = _free_port()
    work_dir = args.out.parent / f".meta-{args.out.name}"
    work_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = work_dir / "telemetry.json"
    bridge_log_path = work_dir / "bridge.stderr.log"

    binding = json.loads(args.binding_template.read_text(encoding="utf-8"))
    binding["base_url"] = f"http://127.0.0.1:{port}/v1"
    binding["resource_caps"]["max_requests"] = int(args.max_requests_cap)
    binding["resource_caps"]["wall_seconds"] = int(args.run_wall_seconds)
    cap_receipt = {
        "schema": "datagoblin-research-v2.1-cap-wrapper-1",
        "model_id": args.model_id,
        "controller_mode": args.controller_mode,
        "max_episodes": args.max_episodes,
        "max_turns_per_episode": args.max_turns_per_episode,
        "resource_caps": binding["resource_caps"],
        "operator_enforced": True,
    }
    cap_bytes = (_canon(cap_receipt) + "\n").encode("utf-8")
    (work_dir / "cap_wrapper_receipt.json").write_bytes(cap_bytes)
    binding["cap_wrapper_receipt_sha256"] = _sha256_bytes(cap_bytes)
    binding_path = work_dir / "binding.json"
    binding_path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    bridge_cmd = [
        sys.executable,
        str(args.bridge_script),
        "--port",
        str(port),
        "--telemetry-out",
        str(telemetry_path),
        "--",
        *worker_cmd,
    ]
    bridge_err = bridge_log_path.open("w", encoding="utf-8")
    bridge = subprocess.Popen(bridge_cmd, stdout=subprocess.DEVNULL, stderr=bridge_err)
    try:
        deadline = time.monotonic() + args.startup_timeout
        ready = False
        while time.monotonic() < deadline:
            if bridge.poll() is not None:
                bridge_err.flush()
                tail = bridge_log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                raise RuntimeError(f"bridge exited early ({bridge.returncode}):\n{tail}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.5)
        if not ready:
            raise RuntimeError("bridge startup timeout")

        run_endpoint(
            args.suite,
            args.out,
            model_id=args.model_id,
            binding_path=binding_path,
            registry_path=args.registry,
            max_episodes=args.max_episodes,
            controller_mode=args.controller_mode,
            max_turns_per_episode=args.max_turns_per_episode,
            run_wall_seconds=args.run_wall_seconds,
        )
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/telemetry", timeout=5) as r:
                final_tel = json.loads(r.read().decode("utf-8"))
                telemetry_path.write_text(json.dumps(final_tel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass
    finally:
        if bridge.poll() is None:
            bridge.send_signal(signal.SIGTERM)
            try:
                bridge.wait(timeout=15)
            except subprocess.TimeoutExpired:
                bridge.kill()
        bridge_err.close()

    verified_report, _rows = read_report(args.out)
    meta_out = {
        "candidate_id": verified_report["candidate_id"],
        "controller_mode": verified_report["controller_mode"],
        "report_sha256": verified_report["report_sha256"],
        "suite_sha256": verified_report["suite_sha256"],
        "macro_score": verified_report["summary"]["macro_score"],
        "macro_capability_score": verified_report["summary"]["macro_capability_score"],
        "success_rate": verified_report["summary"]["success_rate"],
        "checkpoint_rank_eligible": verified_report["checkpoint_rank_eligible"],
        "request_failures": verified_report["request_failures"],
        "controller_accounting": verified_report["controller_accounting"],
        "usage": verified_report["usage"],
        "wall_seconds": verified_report["wall_seconds"],
        "telemetry": json.loads(telemetry_path.read_text(encoding="utf-8")) if telemetry_path.exists() else {},
    }
    (work_dir / "summary_with_telemetry.json").write_text(
        json.dumps(meta_out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta_out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
