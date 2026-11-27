"""End-to-end Research Verifiers v2 model evaluation runner with telemetry & binding receipts.

Runs inside Linux (WSL) so datagoblin.research_v2 enforces Linux RLIMIT worker
caps natively, while bridging loopback HTTP 127.0.0.1:<port> to local_worker.py
(CPU or vramorama-leased CUDA).
"""

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

from datagoblin.research_v2.runner import read_report, run_endpoint


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
    parser.add_argument("--max-episodes", type=int, default=32)
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

    # Build immutable operator resource-cap wrapper receipt
    binding = json.loads(args.binding_template.read_text(encoding="utf-8"))
    binding["base_url"] = f"http://127.0.0.1:{port}/v1"
    cap_receipt = {
        "schema": "datagoblin-research-v2-cap-wrapper-1",
        "model_id": args.model_id,
        "suite": str(args.suite),
        "max_episodes": args.max_episodes,
        "resource_caps": binding["resource_caps"],
        "worker_command_basename": [Path(worker_cmd[0]).name, *[x for x in worker_cmd[1:] if x.startswith("--")]],
        "operator_enforced": True,
    }
    cap_bytes = (_canon(cap_receipt) + "\n").encode("utf-8")
    cap_path = work_dir / "cap_wrapper_receipt.json"
    cap_path.write_bytes(cap_bytes)
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
        )
        # Fetch final telemetry before shutdown
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

    # Verify report replay integrity
    verified_report, _verified_rows = read_report(args.out)
    # Persist non-sensitive telemetry & cap receipt alongside run metadata (outside report.json digest)
    meta_out = {
        "candidate_id": verified_report["candidate_id"],
        "report_sha256": verified_report["report_sha256"],
        "suite_sha256": verified_report["suite_sha256"],
        "macro_score": verified_report["summary"]["macro_score"],
        "success_rate": verified_report["summary"]["success_rate"],
        "checkpoint_rank_eligible": verified_report["checkpoint_rank_eligible"],
        "request_failures": verified_report["request_failures"],
        "usage": verified_report["usage"],
        "wall_seconds": verified_report["wall_seconds"],
        "telemetry": json.loads(telemetry_path.read_text(encoding="utf-8")) if telemetry_path.exists() else {},
    }
    meta_text = json.dumps(meta_out, indent=2, sort_keys=True) + "\n"
    (work_dir / "summary_with_telemetry.json").write_text(meta_text, encoding="utf-8")
    print(json.dumps(meta_out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
