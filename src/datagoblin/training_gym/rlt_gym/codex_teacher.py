"""Collect public gym packets through the authenticated Codex CLI."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from datagoblin.hashing import sha256_file, sha256_text
from datagoblin.serialization import canonical_json, dump_structured, load_structured

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answers"],
    "properties": {"answers": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["index", "candidate_json", "summary"],
        "properties": {"index": {"type": "integer"}, "candidate_json": {"type": "string"},
                       "summary": {"type": "string"}},
    }}},
}
INSTRUCTIONS = (
    "Solve each synthetic task independently from the public packet. Use no tools, files, or internet. "
    "Return a full candidate in the task's declared JSON grammar as the candidate_json string. "
    "Copy index from the task. Give a short observable revision summary, especially after checker feedback. "
    "Treat prior attempts as evidence and submit your best candidate for each index."
)


def replies(final, requests):
    for answer in final["answers"]:
        index = answer.get("index")
        if type(index) is not int or not 0 <= index < len(requests):
            continue
        request = requests[index]
        try:
            candidate = json.loads(answer["candidate_json"])
            canonical_json(candidate)
        except (KeyError, TypeError, ValueError, RecursionError):
            candidate = None
        yield {"request_id": request["request_id"], "task_digest": request["task_digest"],
               "candidate": candidate, "revision_summary": answer["summary"][:2000]}


def collect(request_file, out, *, model, batch_size, max_calls, timeout):
    manifest = load_structured(request_file.parent / "manifest.json")
    if sha256_file(request_file) != manifest["requests_sha256"]:
        raise ValueError("public packets differ from the prepared batch")
    requests = [json.loads(line) for line in request_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not requests or any(r["provider"] != "codex" or r["model"] != model for r in requests):
        raise ValueError("packet provider and model must match the authenticated invocation")
    cli = shutil.which("codex")
    if cli is None:
        raise RuntimeError("Codex CLI is unavailable")
    auth = subprocess.run([cli, "login", "status"], capture_output=True, check=True, timeout=30)
    if b"Logged in using ChatGPT" not in auth.stdout + auth.stderr:
        raise RuntimeError("the campaign requires existing ChatGPT authentication")
    out.mkdir(parents=True, exist_ok=False)
    schema_file = out / "response-schema.json"
    dump_structured(schema_file, SCHEMA)
    version = subprocess.check_output([cli, "--version"], timeout=30, text=True).strip()
    options = ["exec", "--json", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
               "--sandbox", "read-only", "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
               "-c", 'model_reasoning_effort="low"', "--model", model, "--output-schema",
               str(schema_file.resolve()), "-"]
    log = out / "invocations.jsonl"
    response_path = out / "responses.jsonl"
    with (
        tempfile.TemporaryDirectory(prefix="codex-packets-", dir=out) as isolated,
        log.open("x", encoding="utf-8", newline="\n") as invocations,
        response_path.open("x", encoding="utf-8", newline="\n") as responses,
    ):
        for offset in range(0, min(len(requests), batch_size * max_calls), batch_size):
            chunk = requests[offset:offset + batch_size]
            payload = [{"index": i, **r["context"], "response_schema": r["response_schema"]}
                       for i, r in enumerate(chunk)]
            prompt = INSTRUCTIONS + "\n" + canonical_json(payload)
            start = time.monotonic()
            environment = {k: v for k, v in os.environ.items() if k not in {"OPENAI_API_KEY", "CODEX_API_KEY"}}
            try:
                result = subprocess.run(
                    [cli, *options], input=prompt.encode("utf-8"), capture_output=True,
                    cwd=isolated, timeout=timeout, env=environment, check=False,
                )
                stdout, stderr, code = result.stdout, result.stderr, result.returncode
            except subprocess.TimeoutExpired as error:
                stdout, stderr, code = error.stdout or b"", error.stderr or b"", None
            raw_path = out / f"events-{offset:05d}.jsonl"
            raw_path.write_bytes(stdout)
            (out / f"stderr-{offset:05d}.log").write_bytes(stderr)
            events = [json.loads(line) for line in stdout.decode("utf-8", errors="replace").splitlines()
                      if line.startswith("{")]
            finishes = [e for e in events if e.get("type") == "turn.completed"]
            messages = [e["item"]["text"] for e in events if e.get("type") == "item.completed"
                        and e["item"].get("type") == "agent_message"]
            item_kinds = {e["item"].get("type") for e in events if e.get("type") == "item.completed"}
            disallowed = item_kinds - {"agent_message", "reasoning", "todo_list"}
            status = "complete" if code == 0 and len(finishes) == 1 and messages and not disallowed else "incomplete"
            parsed = []
            if status == "complete":
                try:
                    parsed = list(replies(json.loads(messages[-1]), chunk))
                except (TypeError, ValueError, KeyError):
                    status = "unparseable"
            for row in parsed:
                responses.write(canonical_json(row) + "\n")
            responses.flush()
            receipt = {
                "provider": "openai_codex_cli", "authentication": "chatgpt", "requested_model": model,
                "cli_version": version, "reasoning_effort": "low", "sandbox": "read-only",
                "request_ids": [r["request_id"] for r in chunk], "request_file_sha256": manifest["requests_sha256"],
                "prompt_sha256": sha256_text(prompt), "event_file": raw_path.name,
                "event_sha256": sha256_file(raw_path),
                "thread_id": next((e["thread_id"] for e in events if e.get("type") == "thread.started"), None),
                "usage": finishes[0].get("usage") if len(finishes) == 1 else None,
                "cost_usd": None, "billing": "ChatGPT quota; no per-invocation USD receipt",
                "temperature": None, "max_output_tokens": None, "returncode": code,
                "status": status, "emitted_replies": len(parsed), "wall_seconds": time.monotonic() - start,
            }
            invocations.write(canonical_json(receipt) + "\n")
            invocations.flush()
            print(f"Codex batch {offset // batch_size + 1} {status}, {len(parsed)}/{len(chunk)} replies", flush=True)
            if status != "complete":
                break
    dump_structured(out / "manifest.json", {
        "schema_version": "rlt_gym.codex_capture.v1", "provider": "openai_codex_cli",
        "request_file_sha256": manifest["requests_sha256"], "invocations_sha256": sha256_file(log),
        "responses_sha256": sha256_file(response_path),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 80 or not 1 <= args.max_calls <= 64 or not 10 <= args.timeout <= 900:
        parser.error("collection exceeds the per-call or invocation bounds")
    collect(args.requests, args.out, model=args.model, batch_size=args.batch_size,
            max_calls=args.max_calls, timeout=args.timeout)


if __name__ == "__main__":
    main()
