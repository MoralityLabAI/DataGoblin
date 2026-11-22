"""Run BSAR-02 public Step7 prompts against a locally hosted GGUF chat model.

Uses a loopback-only OpenAI-compatible llama-server. No remote endpoints.
Requires an already-running server; this script does not download model weights.
Labels are never placed in user prompts, and model responses stay separate from
the gold label catalogue. Pauses/resumes are explicit.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .evaluation import ABSTAIN, LABELS, load_holdout, public_tasks, read_predictions, score

SYSTEM = (
    "You review the scope of a Lean theorem statement. "
    "Respond with EXACTLY ONE uppercase label, with no explanation or punctuation. "
    "Allowed labels: " + ", ".join(LABELS) + ". "
    "Classify only what the displayed theorem signature directly establishes. "
    "A conditional mathematical lemma does not automatically prove a global PDE claim."
)


def validated_endpoint(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Only local HTTP endpoints on loopback are permitted")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint contains unsupported credentials or query parameters")
    if not parsed.path.endswith("/v1/chat/completions"):
        raise ValueError("Expected an OpenAI-compatible /v1/chat/completions endpoint")
    return url


def label_from_response(content: str) -> str:
    stripped = content.strip().strip("`").strip()
    if re.fullmatch(r"[A-Z_]+", stripped) and stripped in LABELS:
        return stripped
    return ABSTAIN


def chat_once(
    endpoint: str, model: str, prompt: str, timeout: float, seed: int
) -> tuple[str, str, int | None]:
    validated_endpoint(endpoint)
    payload = {
        "model": model,
        "temperature": 0,
        "stream": False,
        "max_tokens": 48,
        "seed": seed,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as handle:
        response = json.loads(handle.read().decode("utf-8"))
    content = response["choices"][0]["message"]["content"]
    tokens = response.get("usage", {}).get("total_tokens")
    if not isinstance(content, str):
        raise ValueError("Model did not produce text content")
    return label_from_response(content), content, tokens if isinstance(tokens, int) else None


def run(
    output: Path,
    endpoint: str,
    model: str,
    *,
    limit: int = 164,
    timeout: float = 120,
    seed: int = 20261122,
) -> dict:
    endpoint = validated_endpoint(endpoint)
    if limit < 1 or timeout <= 0:
        raise ValueError("Limit and timeout must be positive")
    heldout = load_holdout()
    tasks = public_tasks(heldout)
    expected = set(heldout)
    previous = read_predictions(output, expected) if output.exists() else {}
    for item in tasks:
        if item["id"] not in previous:
            continue
        # Stop users from silently pooling responses from different models.
        # The simple label parser ensures each recorded response stays valid.
        if previous[item["id"]] not in LABELS + (ABSTAIN,):
            raise ValueError("Invalid existing prediction")
    remaining = [item for item in tasks if item["id"] not in previous][:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    for item in remaining:
        start = time.monotonic()
        label, raw, tokens = chat_once(endpoint, model, item["prompt"], timeout, seed)
        row = {
            "id": item["id"],
            "label": label,
            "model": model,
            "raw_text": raw[:512],
            "tokens": tokens,
            "elapsed_ms": round(1000 * (time.monotonic() - start), 2),
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    predictions = read_predictions(output, expected)
    metrics = score(heldout, predictions)
    metrics["producer"] = "LOCAL_GGUF_CHAT_SERVER"
    metrics["model_identifier"] = model
    metrics["new_requests"] = len(remaining)
    metrics["prediction_file"] = str(output)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=164)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args()
    results = run(args.predictions, args.endpoint, args.model,
                  limit=args.limit, timeout=args.timeout_seconds, seed=args.seed)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
