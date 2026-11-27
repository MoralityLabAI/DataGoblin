"""End-to-end check of the data-goblin binary against a scripted OpenAI-compatible endpoint.

    python -I gdk/ci_smoke.py PATH/TO/data-goblin [--python PY]

Run 1 (native tool calls, --approve ledger) drives one real looped_modules move through the GDK app:
the /datagoblin-hillclimb slash command, the lane card via load_skill, suite_status, propose_move,
rlt_make_suite, the same bad rlt_run call three times (two server refusals naming the parameter, then
the app's repeated-failure guard), rlt_run done right (a real CPU fit of recurrent controls), a loopback
recipe the app must deny at this approval level, record_result and decide. Run 2 (--tool-mode text, --approve read-only)
checks the text protocol and that write tools are hidden. The script checks the receipts, the
transcript and the hash-chained ledger.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEEN: list[dict[str, Any]] = []


def tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found = []
    for message in messages:
        if message.get("role") == "tool":
            try:
                found.append(json.loads(message.get("content") or ""))
            except ValueError:
                found.append({"text": message.get("content")})
    return found


def last_with(messages: list[dict[str, Any]], key: str) -> Any:
    for result in reversed(tool_results(messages)):
        if isinstance(result, dict) and key in result:
            return result[key]
    raise AssertionError(f"no tool result with {key}")


def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"tool": name, "arguments": arguments}


def native_script(messages: list[dict[str, Any]]) -> dict[str, Any] | str:
    """The next scripted reply, from how many tool results the conversation already holds."""
    done = len(tool_results(messages))
    if done == 0:
        return call("load_skill", {"name": "datagoblin-hillclimb/lanes/looped_modules.md"})
    if done == 1:
        return call("suite_status", {})
    if done == 2:
        return call("propose_move", {"lane": "looped_modules", "plan": ["rlt_make_suite", "rlt_run"],
                                     "expected": 0.2,
                                     "hypothesis": "Depth-1 and depth-2 RLT controls on a one-task-per-family suite "
                                                   "set the lane baseline."})
    move = last_with(messages, "move_id")
    if done == 3:
        return call("rlt_make_suite", {"move_id": move, "per_family": 1})
    if done in (4, 5, 6):
        # The same bad call three times: two server refusals, then the app's repeated-failure guard.
        return call("rlt_run", {"move_id": move})
    if done == 7:
        suite = next(r["run_id"] for r in tool_results(messages) if isinstance(r, dict) and r.get("run_id"))
        return call("rlt_run", {"move_id": move, "suite": f"run:{suite}", "seeds": [7], "depths": [1, 2],
                                "max_wall_seconds": 120})
    if done == 8:
        return call("agonist_send", {"move_id": move, "requests": "Agonist/fixtures/tastes.jsonl",
                                     "base_url": "http://127.0.0.1:9/v1", "model": "x"})
    if done == 9:
        fits = [r for r in tool_results(messages) if isinstance(r, dict) and r.get("produces_lane_metric")]
        return call("record_result", {"move_id": move, "run_id": fits[-1]["run_id"]})
    if done == 10:
        return call("decide", {"move_id": move})
    verdict = last_with(messages, "verdict")
    return f"Lane looped_modules, move {move}: rlt_make_suite and rlt_run ran; decide returned {verdict}."


def text_script(messages: list[dict[str, Any]]) -> str:
    if not any(m.get("role") != "system" and "<tool_response>" in str(m.get("content")) for m in messages):
        return '<tool_call>\n{"name": "suite_status", "arguments": {}}\n</tool_call>'
    return "Read the suite: six lanes, read-only session."


class Endpoint(BaseHTTPRequestHandler):
    mode = "native"

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send({"object": "list", "data": [{"id": "scripted-driver", "object": "model"}]})

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(request)
        messages = request["messages"]
        reply = native_script(messages) if type(self).mode == "native" else text_script(messages)
        message: dict[str, Any] = {"role": "assistant", "content": None}
        finish = "stop"
        if isinstance(reply, dict):
            message["tool_calls"] = [{"id": f"call_{len(SEEN)}", "type": "function", "function": {
                "name": reply["tool"], "arguments": json.dumps(reply["arguments"])}}]
            finish = "tool_calls"
        else:
            message["content"] = reply
        self._send({"id": f"cmpl-{len(SEEN)}", "object": "chat.completion", "model": request.get("model"),
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})


def run(binary: str, python: str, port: int, state: Path, *extra: str) -> dict[str, Any]:
    command = [binary, "run", "--root", str(ROOT), "--python", python, "--state-dir", str(state),
               "--base-url", f"http://127.0.0.1:{port}/v1", "--model", "scripted-driver",
               "--timeout-secs", "120", *extra]
    done = subprocess.run(command, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    assert done.returncode == 0, f"exit {done.returncode}\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}"
    return json.loads(done.stdout.strip().splitlines()[-1])


def main() -> int:
    binary = sys.argv[1]
    python = sys.argv[sys.argv.index("--python") + 1] if "--python" in sys.argv else sys.executable
    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp) / "state"
        receipt = run(binary, python, port, state, "--skill", "datagoblin-hillclimb", "--args", "looped_modules",
                      "--approve", "ledger")
        first = json.dumps(SEEN[0]["messages"])
        assert "# Loaded Skill: datagoblin-hillclimb" in first, "slash command did not load the skill"
        assert "/datagoblin-hillclimb looped_modules" in first
        offered = {t["function"]["name"]: t["function"] for t in SEEN[0]["tools"]}
        assert {"load_skill", "propose_move", "rlt_make_suite", "rlt_run", "curate"} <= set(offered), sorted(offered)
        assert "run_recipe" not in offered and "agonist_send" not in offered, "listed tools must respect the tier"
        rlt_params = offered["rlt_run"]["parameters"]
        assert rlt_params["required"] == ["move_id", "suite"], rlt_params
        assert rlt_params["properties"]["depths"]["items"]["enum"] == [1, 2, 4, 8, 16], rlt_params
        calls = receipt["toolCalls"]
        assert [c["name"] for c in calls] == ["load_skill", "suite_status", "propose_move", "rlt_make_suite",
                                              *["rlt_run"] * 4, "agonist_send", "record_result", "decide"], calls
        refused = calls[4]["result"] or ""
        assert "missing_param" in refused and "parameter 'suite'" in refused and "run:r0001" in refused, refused
        assert calls[5]["ok"] is False and "missing_param" in (calls[5]["result"] or "")
        assert "already failed 2 times" in (calls[6]["result"] or ""), calls[6]
        assert "loopback recipe" in (calls[8]["result"] or ""), calls[8]
        failed = {i for i, c in enumerate(calls) if not c["ok"]}
        assert failed == {4, 5, 6, 8}, [calls[i] for i in sorted(failed)]
        assert receipt["decisions"][0]["verdict"] == "baseline", receipt["decisions"]
        assert receipt["moves"] == ["m0001"] and receipt["skillsLoaded"] == [
            "datagoblin-hillclimb/lanes/looped_modules.md"]
        assert "decide returned baseline" in receipt["answer"]
        roles = [m["role"] for m in receipt["transcript"]]
        assert roles[0] == "system" and roles.count("tool") == 11, roles
        ledger = [json.loads(line) for line in (state / "ledger.jsonl").read_text().splitlines()]
        assert [r["kind"] for r in ledger] == ["proposal", "run", "run", "result", "decision"], ledger
        assert ledger[-1]["body"]["verdict"] == "baseline"
        assert 0.0 <= ledger[3]["body"]["value"] <= 1.0 and ledger[3]["body"]["split"] == "calibration"
        receipts = (state / "receipts.jsonl").read_text().splitlines()
        assert len(receipts) == 1

        Endpoint.mode = "text"
        SEEN.clear()
        text = run(binary, python, port, state, "--prompt", "What can this session climb?", "--tool-mode", "text",
                   "--approve", "read-only")
        assert "tools" not in SEEN[0] or not SEEN[0]["tools"], "text mode must not send native tools"
        system = SEEN[0]["messages"][0]["content"]
        assert "<tool_call>" in system and "\n- suite_status:" in system and "\n- propose_move:" not in system
        assert [c["name"] for c in text["toolCalls"]] == ["suite_status"] and text["toolCalls"][0]["ok"], (
            json.dumps({k: text[k] for k in ("toolCalls", "answer", "error", "inferenceCalls")})[:3000])
        assert text["approve"] == "read-only" and text["maxTier"] == "offline"
        print(json.dumps({"ok": True, "move": receipt["moves"], "verdict": receipt["decisions"][0]["verdict"],
                          "value": ledger[3]["body"]["value"], "ledger_records": len(ledger),
                          "native_seconds": round(receipt["seconds"], 1), "text_calls": len(text["toolCalls"])}))
    server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
