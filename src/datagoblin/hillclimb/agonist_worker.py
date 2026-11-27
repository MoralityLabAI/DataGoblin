"""Agonist worker: send staged AGN-01 teacher requests to an already running bigger local model.

This is the permissioned provider worker `Agonist/SKILL.md` leaves to the operator. It talks only to
a loopback OpenAI-compatible server (llama.cpp, squid5090, vLLM), never starts or stops one, and
writes three files: `teacher_replies.jsonl` (one reply object per attempted request, for
`python -m Agonist.gym audit`), `provider_receipts.jsonl` (settings, token counts, latency and
failure class per request) and `manifest.json`.

Before each request it asks the server's `/tokenize` endpoint for the prompt's token count and
skips the request when prompt plus output budget exceeds the request's context cap. A reply keeps
only the AGN-01 reply keys; request_id and teacher_model are set from the staged request and the
model identity the server returned. A request that yields no JSON object still gets a reply line
carrying only those two keys, so the audit counts it as a rejected reply.

    python -m datagoblin.hillclimb.agonist_worker --requests runs/q/teacher_requests.jsonl \
        --base-url http://127.0.0.1:18090/v1 --model inkling-2-mini --out runs/agonist/replies
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .lanes import is_loopback_url

SCHEMA = "datagoblin.agonist01.v1"
REPLY_KEYS = ("request_id", "teacher_model", "world_a", "world_b", "intervention", "design_note")
MAX_RESPONSE_BYTES = 1_048_576
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def first_json_object(text: str) -> dict[str, Any] | None:
    """The first balanced top-level JSON object in a reply (thinking blocks and fences ignored)."""
    text = _THINK.sub("", text)
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    return value if isinstance(value, dict) else None
        start = text.find("{", start + 1)
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=canonical(body).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    with _OPENER.open(request, timeout=timeout) as response:
        data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("oversized response")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    return value


def count_tokens(base_url: str, text: str, timeout: float = 30.0) -> int | None:
    """Prompt token count from the server's /tokenize endpoint (llama.cpp layout), or None."""
    parts = urlsplit(base_url)
    url = urlunsplit((parts.scheme, parts.netloc, "/tokenize", "", ""))
    try:
        data = _post(url, {"content": text, "add_special": True}, timeout)
    except (OSError, ValueError):
        return None
    tokens = data.get("tokens")
    return len(tokens) if isinstance(tokens, list) else None


def send(
    requests_path: Path,
    base_url: str,
    model: str,
    out: Path,
    *,
    max_requests: int = 8,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    timeout_seconds: float = 600.0,
    max_wall_seconds: float = 7000.0,
) -> dict[str, Any]:
    if not is_loopback_url(base_url):
        raise ValueError("base URL must be http on 127.0.0.1, localhost or ::1 with path /v1")
    requests = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for request in requests:
        if request.get("schema") != SCHEMA or not isinstance(request.get("prompt"), str):
            raise ValueError("requests file is not an AGN-01 teacher queue")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    out.mkdir(parents=True, exist_ok=False)
    replies: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    started = time.monotonic()
    for request in requests[:max_requests]:
        receipt: dict[str, Any] = {"request_id": request["request_id"], "model_requested": model,
                                   "max_tokens": max_tokens, "temperature": temperature}
        if time.monotonic() - started > max_wall_seconds:
            receipt["status"] = "wall_budget_exhausted"
            receipts.append(receipt)
            continue
        cap = int(request.get("max_context_tokens", 12_288))
        prompt_tokens = count_tokens(base_url, request["prompt"])
        receipt["prompt_tokens_preflight"] = prompt_tokens
        if prompt_tokens is not None and prompt_tokens + max_tokens > min(cap, 16_384):
            receipt["status"] = "context_cap_exceeded"
            receipts.append(receipt)
            continue
        body = {"model": model, "messages": [{"role": "user", "content": request["prompt"]}],
                "temperature": temperature, "max_tokens": max_tokens}
        began = time.monotonic()
        returned_model = model
        parsed: dict[str, Any] | None = None
        try:
            data = _post(endpoint, body, timeout_seconds)
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            returned_model = str(data.get("model") or model)
            usage = data.get("usage") or {}
            receipt.update(
                returned_model=returned_model,
                finish_reason=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
            parsed = first_json_object(content)
            receipt["status"] = "ok" if parsed is not None else "no_json_object"
        except urllib.error.HTTPError as exc:
            receipt.update(status="http_error", http_status=exc.code)
        except OSError:
            receipt["status"] = "connection_error"
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            receipt["status"] = "bad_response"
        receipt["latency_seconds"] = round(time.monotonic() - began, 3)
        reply: dict[str, Any] = {}
        if parsed is not None:
            reply = {key: parsed[key] for key in REPLY_KEYS[2:] if key in parsed}
            receipt["dropped_keys"] = sorted(set(parsed) - set(REPLY_KEYS))
            receipt["missing_keys"] = sorted(set(REPLY_KEYS[2:]) - set(parsed))
        reply = {"request_id": request["request_id"], "teacher_model": returned_model, **reply}
        replies.append(reply)
        receipts.append(receipt)

    for name, rows in (("teacher_replies.jsonl", replies), ("provider_receipts.jsonl", receipts)):
        with (out / name).open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
    statuses = Counter(receipt.get("status", "unknown") for receipt in receipts)
    manifest = {
        "schema": "datagoblin.hillclimb.agonist_worker.v1",
        "requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
        "endpoint": endpoint,
        "model_requested": model,
        "models_returned": sorted({r["returned_model"] for r in receipts if "returned_model" in r}),
        "settings": {"max_requests": max_requests, "max_tokens": max_tokens, "temperature": temperature,
                     "timeout_seconds": timeout_seconds, "max_wall_seconds": max_wall_seconds},
        "requests_staged": len(requests),
        "requests_attempted": len(replies),
        "statuses": dict(sorted(statuses.items())),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in receipts),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in receipts),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "outputs": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                    for name in ("teacher_replies.jsonl", "provider_receipts.jsonl")},
        "teacher_execution": "LOCAL_LOOPBACK_SERVER",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send staged AGN-01 requests to a local model server.")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-wall-seconds", type=float, default=7000.0)
    args = parser.parse_args(argv)
    try:
        manifest = send(args.requests, args.base_url, args.model, args.out, max_requests=args.max_requests,
                        max_tokens=args.max_tokens, temperature=args.temperature,
                        timeout_seconds=args.timeout_seconds, max_wall_seconds=args.max_wall_seconds)
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": "agonist_send_error", "detail": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"attempted": manifest["requests_attempted"], "statuses": manifest["statuses"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
