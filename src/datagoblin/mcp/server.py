"""MCP stdio server for DataGoblin (newline-delimited JSON-RPC 2.0, MCP 2025-06-18).

    python -m datagoblin.mcp [--root DIR]

Stdout carries protocol frames only. Tools read from examples/, runs/ and configs/ under the root and write
only under runs/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO, Any

from datagoblin.config import repository_root

from .errors import McpToolError, Sandbox
from .tools import BY_NAME, tool_list

PROTOCOL = "2025-06-18"
VERSION = "0.1.0"
INSTRUCTIONS = (
    "DataGoblin research orchestrator. Typical flow: validate_recipe on a recipe, propose_plan from a learner "
    "snapshot, emit_tickets from the plan, run_demo for the offline toy cycle, then ledger_summary / ledger_get / "
    "report to read the run back, and export_episodes to turn the research episode into SFT or preference rows. "
    "Inputs come from examples/, runs/ or configs/; outputs go only under runs/. Never report a metric, decision "
    "or rule id a tool did not return."
)


def _result(payload: dict[str, Any], error: bool = False) -> dict[str, Any]:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text}], "structuredContent": payload, "isError": error}


def call_tool(box: Sandbox, name: str, arguments: Any) -> dict[str, Any]:
    tool = BY_NAME.get(name)
    if tool is None:
        raise McpToolError("unknown_tool", f"no tool named {name}")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise McpToolError("invalid_arguments", "arguments must be an object")
    allowed = set(tool["inputSchema"]["properties"])
    extra = sorted(set(arguments) - allowed)
    if extra:
        raise McpToolError("invalid_arguments", f"unexpected arguments: {', '.join(extra)}")
    missing = [key for key in tool["inputSchema"]["required"] if key not in arguments]
    if missing:
        raise McpToolError("invalid_arguments", f"missing arguments: {', '.join(missing)}")
    try:
        return tool["handler"](box, arguments)  # type: ignore[no-any-return]
    except McpToolError:
        raise
    except FileNotFoundError as exc:
        raise McpToolError("not_found", str(exc)) from exc
    except (KeyError, ValueError, TypeError) as exc:
        raise McpToolError("invalid_input", f"{type(exc).__name__}: {exc}") from exc
    except RuntimeError as exc:
        raise McpToolError("run_failed", str(exc)) from exc
    except OSError as exc:
        raise McpToolError("io_error", f"{type(exc).__name__}: {exc}") from exc


def handle(box: Sandbox, message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    if "id" not in message:
        return None
    reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
    params = message.get("params") or {}
    if method == "initialize":
        reply["result"] = {
            "protocolVersion": PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "datagoblin", "version": VERSION},
            "instructions": INSTRUCTIONS,
        }
    elif method == "ping":
        reply["result"] = {}
    elif method == "tools/list":
        reply["result"] = {"tools": tool_list()}
    elif method == "tools/call":
        try:
            reply["result"] = _result(call_tool(box, str(params.get("name")), params.get("arguments")))
        except McpToolError as exc:
            reply["result"] = _result({"error": exc.code, "detail": exc.detail}, error=True)
    else:
        reply["error"] = {"code": -32601, "message": f"method not found: {method}"}
    return reply


def serve(box: Sandbox, reader: IO[bytes], writer: IO[bytes]) -> None:
    for raw in reader:
        line = raw.decode("utf-8-sig", errors="replace").strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("not an object")
        except ValueError:
            response: dict[str, Any] | None = {
                "jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}
            }
        else:
            response = handle(box, message)
        if response is not None:
            writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            writer.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DataGoblin MCP server")
    parser.add_argument("--root", type=Path, default=repository_root(), help="repository root (default: this checkout)")
    args = parser.parse_args(argv)
    serve(Sandbox(args.root), sys.stdin.buffer, sys.stdout.buffer)
    return 0
