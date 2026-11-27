"""MCP stdio server for the DataGoblin hill-climb tools (newline-delimited JSON-RPC 2.0, MCP 2025-06-18).

    python -m datagoblin.hillclimb.server [--root DIR] [--state-dir DIR] [--lanes FILE]
        [--max-tier offline|loopback|gpu] [--read-only]

Stdout carries protocol frames only; recipes run with their output redirected to files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO, Any

from .lanes import TIERS, HillclimbError
from .service import Service

PROTOCOL = "2025-06-18"
VERSION = "0.1.0"
INSTRUCTIONS = (
    "DataGoblin hill-climb tools. A move is: propose_move (lane, hypothesis, plan) -> call each planned recipe's "
    "own tool (named like the recipe, e.g. rlt_run) with move_id and that recipe's parameters -> record_result on "
    "the run that produces the lane metric -> decide. Pass an earlier run's output as run:<run_id>. The service "
    "reads metric values from artifacts and applies the lane gate; never report a number or verdict a tool did not "
    "return. Only recipes within this session's tier are listed as tools."
)


def _result(payload: dict[str, Any], error: bool = False) -> dict[str, Any]:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text}], "structuredContent": payload, "isError": error}


def handle(service: Service, message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    if "id" not in message:
        return None
    reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
    params = message.get("params") or {}
    if method == "initialize":
        reply["result"] = {
            "protocolVersion": PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "datagoblin-hillclimb", "version": VERSION},
            "instructions": INSTRUCTIONS,
        }
    elif method == "ping":
        reply["result"] = {}
    elif method == "tools/list":
        reply["result"] = {"tools": service.tools()}
    elif method == "tools/call":
        name = params.get("name")
        try:
            reply["result"] = _result(service.call(str(name), params.get("arguments")))
        except HillclimbError as exc:
            reply["result"] = _result({"error": exc.code, "detail": exc.detail}, error=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reply["result"] = _result({"error": "internal_error", "detail": f"{type(exc).__name__}: {exc}"}, True)
    else:
        reply["error"] = {"code": -32601, "message": f"method not found: {method}"}
    return reply


def serve(service: Service, reader: IO[bytes], writer: IO[bytes]) -> None:
    for raw in reader:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("not an object")
        except ValueError:
            response: dict[str, Any] | None = {"jsonrpc": "2.0", "id": None,
                                               "error": {"code": -32700, "message": "parse error"}}
        else:
            response = handle(service, message)
        if response is not None:
            writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            writer.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DataGoblin hill-climb MCP server")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--lanes", type=Path)
    parser.add_argument("--max-tier", choices=TIERS, default="offline")
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args(argv)
    service = Service(root=args.root, state_dir=args.state_dir, lanes=args.lanes, max_tier=args.max_tier,
                      read_only=args.read_only)
    serve(service, sys.stdin.buffer, sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
