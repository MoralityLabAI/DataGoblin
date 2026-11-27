"""WSL loopback HTTP adapter bridging Research Verifiers v2 to local_worker.py.

Listens strictly on 127.0.0.1:<port> inside Linux so datagoblin.research_v2's
transport validation sees a standard loopback HTTP endpoint with exact
`/tokenize`, `/v1/completions`, and `/v1/chat/completions` routes, while
delegating inference to the persistent local_worker.py process and persisting
an immutable telemetry JSON receipt.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class WorkerBridge:
    def __init__(self, cmd: list[str], telemetry_out: Path | None = None) -> None:
        self.cmd = cmd
        self.telemetry_out = telemetry_out
        self.lock = threading.Lock()
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdout is not None and self.proc.stdin is not None
        ready_line = self.proc.stdout.readline()
        if not ready_line:
            code = self.proc.wait()
            raise RuntimeError(f"local_worker exited before ready (rc={code})")
        info = json.loads(ready_line)
        if not info.get("ready"):
            raise RuntimeError(f"local_worker unexpected startup message: {info}")
        self.latest_telemetry: dict[str, Any] = info.get("telemetry", {})
        self._save_telemetry()

    def call(self, op: str, payload: dict[str, Any] | None = None, timeout_seconds: float = 29.0) -> dict[str, Any]:
        with self.lock:
            assert self.proc.stdin is not None and self.proc.stdout is not None
            msg = {"op": op, "payload": payload or {}, "timeout_seconds": timeout_seconds}
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("local_worker pipe closed")
            resp = json.loads(line)
            if not resp.get("ok"):
                raise RuntimeError(resp.get("error", "worker_error"))
            return resp["result"]

    def refresh_telemetry(self) -> dict[str, Any]:
        self.latest_telemetry = self.call("telemetry")
        self._save_telemetry()
        return self.latest_telemetry

    def _save_telemetry(self) -> None:
        if self.telemetry_out is None:
            return
        self.telemetry_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.telemetry_out.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.latest_telemetry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.telemetry_out)

    def close(self) -> None:
        try:
            self.latest_telemetry = self.call("shutdown")
            self._save_telemetry()
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def make_handler(bridge: WorkerBridge):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write(f"[bridge] {format % args}\n")

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            elif self.path == "/telemetry":
                try:
                    self._send_json(200, bridge.refresh_telemetry())
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
            else:
                self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                if self.path == "/tokenize":
                    res = bridge.call("tokenize", payload, timeout_seconds=10.0)
                    self._send_json(200, res)
                elif self.path in ("/v1/completions", "/v1/chat/completions"):
                    res = bridge.call("generate", payload, timeout_seconds=29.2)
                    bridge.refresh_telemetry()
                    self._send_json(200, res)
                else:
                    self._send_json(404, {"error": "not_found"})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)[:300]})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--telemetry-out", type=Path)
    parser.add_argument("worker_cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.worker_cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("worker command required after --")

    bridge = WorkerBridge(cmd, telemetry_out=args.telemetry_out)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(bridge))

    def _handle_signal(signum: int, frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    sys.stderr.write(f"[bridge] listening on http://127.0.0.1:{args.port} (started {time.time():.0f})\n")
    sys.stderr.flush()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
