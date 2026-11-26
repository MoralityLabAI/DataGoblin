"""MOLT-03: bounded public memory panels, with optional Research v2 transport."""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datagoblin.training_gym.molt_episode import VARIANTS, Episode, World, make_world
from datagoblin.training_gym.molt_episode_policy import PublicPolicy
from datagoblin.training_gym.molt_goblin import digest

MODES = ("full", "recent2", "current", "bank1", "bank2")
MAX_REPLY_BYTES = 16384
MAX_OUTPUT_TOKENS = 512
CONTEXT_TOKENS = 2048
SYSTEM = (
    "Select the maximum-utility admissible array under capacity. Each active source receipt "
    "excludes its named candidate. Break ties by lexicographically smallest sorted ID list. "
    "After start and distractor, advance. On source_change, request updated evidence if the "
    "source belongs to the task's sources; metadata changes require no request. A request "
    "costs one verifier call. Submit using current frame and exactly the issued current "
    "receipt IDs for all task sources. If information is missing, abstain. Output one JSON "
    "object only: {\"action\":\"advance\"}, {\"action\":\"request\",\"source\":\"s0\"}, "
    "{\"action\":\"submit\",\"selected\":[\"c0\"],\"receipts\":[\"receipt_id\"],"
    "\"frame_id\":\"current_frame_id\"}, or {\"action\":\"abstain\"}."
)
ActionClient = Callable[[str, float], dict[str, Any]]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def reject_constant(_: str) -> None:
    raise ValueError("nonfinite JSON")


def parse_action(reply: str) -> dict[str, Any]:
    if type(reply) is not str or len(reply.encode("utf-8")) > MAX_REPLY_BYTES:
        raise ValueError("reply byte cap")
    obj = json.loads(reply, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    if type(obj) is not dict:
        raise ValueError("action object required")
    return obj


class ObservationView:
    """Only serialized public observations enter the memory adapter."""

    def __init__(self, mode: str):
        if mode not in MODES:
            raise ValueError("unknown history mode")
        self.mode = mode
        self.static: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.bank: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.limit = int(mode[-1]) if mode.startswith("bank") else 0
        self.writes = 0
        self.invalidations = 0
        self.evictions = 0
        self.bank_read_bytes = 0
        self.peak_bank_bytes = 0

    def _put(self, receipt: dict[str, Any]) -> None:
        if not self.limit:
            return
        source = receipt["source"]
        self.bank[source] = {k: receipt[k] for k in ("source", "version", "excluded", "receipt_id")}
        self.bank.move_to_end(source)
        self.writes += 1
        if len(self.bank) > self.limit:
            self.bank.popitem(last=False)
            self.evictions += 1
        self.peak_bank_bytes = max(self.peak_bank_bytes, len(canonical(list(self.bank.values())).encode()))

    def observe(self, observation: dict[str, Any]) -> str:
        obs = json.loads(canonical(observation))
        if obs["event"] == "start":
            self.static = {
                "task_id": obs["task_id"], "candidates": obs["candidates"],
                "capacity": obs["capacity"], "sources": sorted(r["source"] for r in obs["receipts"]),
                "verifier_call_cap": obs["verifier_calls_remaining"],
                "turn_cap": obs["turns_remaining"],
            }
            for receipt in obs["receipts"]:
                self._put(receipt)
        if obs["event"] == "source_change" and obs["source"] in self.bank:
            del self.bank[obs["source"]]
            self.invalidations += 1
        if obs["event"] == "receipt":
            self._put(obs)
        # History variants expose observed events only. The fixed task is available
        # in every arm; receipt contents are excluded from this static task record.
        if self.mode == "full":
            self.history.append(obs)
        elif self.mode == "recent2":
            self.history = [*self.history[-1:], obs]
        else:
            self.history = [obs]
        packet: dict[str, Any] = {"task": self.static, "observations": self.history}
        if self.limit:
            packet["retained_receipts"] = list(self.bank.values())
            self.bank_read_bytes += len(canonical(packet["retained_receipts"]).encode())
        return SYSTEM + "\nOBSERVATION_PACKET\n" + canonical(packet) + "\nACTION\n"

    def receipt(self) -> dict[str, int]:
        return {"bank_slots": self.limit, "bank_writes": self.writes, "bank_evictions": self.evictions,
                "bank_invalidations": self.invalidations, "bank_read_bytes": self.bank_read_bytes,
                "peak_serialized_bank_bytes": self.peak_bank_bytes}


def exact_public_client(prompt: str, _timeout: float) -> dict[str, Any]:
    """Stateless exact policy, reconstructed from the rendered view at each call."""
    packet = json.loads(prompt.split("\nOBSERVATION_PACKET\n", 1)[1].rsplit("\nACTION\n", 1)[0])
    history = packet["observations"]
    last = history[-1]
    if last["event"] in {"start", "distractor"}:
        action: dict[str, Any] = {"action": "advance"}
    elif last["event"] == "source_change" and last["source"] in packet["task"]["sources"]:
        action = {"action": "request", "source": last["source"]}
    else:
        receipts = {r["source"]: r for r in packet.get("retained_receipts", [])}
        for obs in history:
            if obs["event"] == "start":
                receipts.update({r["source"]: r for r in obs["receipts"]})
            elif obs["event"] == "source_change":
                receipts.pop(obs["source"], None)
            elif obs["event"] == "receipt":
                receipts[obs["source"]] = obs
        if set(receipts) != set(packet["task"]["sources"]):
            action = {"action": "abstain"}
        else:
            policy = PublicPolicy()
            policy.candidates = packet["task"]["candidates"]
            policy.capacity = packet["task"]["capacity"]
            policy.receipts = receipts
            policy.frame = last["frame_id"]
            action = policy._submit()
    return {"text": canonical(action), "http_requests": 0, "inference_requests": 0}


def roster(seed: int, roots: int, split: str) -> list[World]:
    # No release of protected confirmation roots through this development runner.
    if split not in {"train", "calibration"} or not 1 <= roots <= 128:
        raise ValueError("development split and 1..128 roots required")
    return [make_world(split, seed, i, v) for i in range(roots) for v in VARIANTS]


def usage_record(response: dict[str, Any]) -> tuple[int, int, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, int(response.get("inference_requests", 0) > 0)
    p, c = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if type(p) is not int or type(c) is not int or p < 0 or c < 0:
        raise ValueError("invalid usage")
    if p + c > CONTEXT_TOKENS or c > MAX_OUTPUT_TOKENS:
        raise ValueError("token budget exceeded")
    return p, c, 0


def run_panel(worlds: list[World], mode: str, client: ActionClient, *,
              max_actions: int, wall_seconds: float, identity: dict[str, Any]) -> dict[str, Any]:
    if not worlds or len({w.task_id for w in worlds}) != len(worlds):
        raise ValueError("nonempty unique task roster required")
    if not 1 <= max_actions <= 8192 or not 0 < wall_seconds <= 900:
        raise ValueError("invalid resource caps")
    if any(w.split not in {"train", "calibration"} for w in worlds):
        raise ValueError("development-only panel")
    started = time.monotonic()
    attempts = 0
    rows = []
    for world in worlds:
        ep = Episode(world)
        view = ObservationView(mode)
        obs = ep.start()
        calls = []
        stop_reason = "terminal"
        while not ep.closed:
            remaining = wall_seconds - (time.monotonic() - started)
            if attempts >= max_actions or remaining <= 0:
                stop_reason = "run_budget_exhausted"
                break
            prompt = view.observe(obs)
            attempts += 1
            record = {"prompt_digest": digest(prompt), "prompt_bytes": len(prompt.encode())}
            try:
                response = client(prompt, min(30.0, remaining))
                record.update(http_requests=response.get("http_requests", 0),
                              inference_requests=response.get("inference_requests", 0))
                p, c, missing = usage_record(response)
                record.update(prompt_tokens=p, completion_tokens=c, unknown_usage_calls=missing)
                action = parse_action(response["text"])
                record["action"] = action
                if time.monotonic() - started > wall_seconds:
                    stop_reason = "run_deadline_exceeded"
                    calls.append(record)
                    break
                obs = ep.step(action)
                record["observation"] = obs
            except (ValueError, TypeError, KeyError, OSError, RuntimeError, TimeoutError) as error:
                stop_reason = "invalid_or_failed_action"
                record["error_type"] = type(error).__name__
                # The Research v2 transport exposes sanitized cost receipts on failures.
                receipt = getattr(error, "receipt", {})
                if isinstance(receipt, dict):
                    record["transport_error"] = receipt.get("error_code")
                    for k in ("http_requests", "inference_requests"):
                        record[k] = receipt.get(k, record.get(k, 0))
                    try:
                        p, c, missing = usage_record(receipt)
                        if receipt:
                            record.update(prompt_tokens=p, completion_tokens=c, unknown_usage_calls=missing)
                    except ValueError:
                        record["unknown_usage_calls"] = int(record.get("inference_requests", 0) > 0)
                calls.append(record)
                break
            calls.append(record)
        terminal = ep.result or {"reason": stop_reason, "verified": False, "optimal": False,
                                 "utility": 0, "turns": ep.turns, "verifier_calls": ep.calls}
        rows.append({"root_id": world.root_id, "task_id": world.task_id, "split": world.split,
                     "variant": world.variant, "mode": mode, "terminal": terminal,
                     "stop_reason": stop_reason, "memory": view.receipt(), "calls": calls})
    summary = {
        "episodes": len(rows), "roots": len({r["root_id"] for r in rows}),
        "optimal": sum(r["terminal"]["optimal"] for r in rows),
        "verified": sum(r["terminal"]["verified"] for r in rows),
        "verifier_calls": sum(r["terminal"]["verifier_calls"] for r in rows),
        "action_attempts": attempts,
        "http_requests": sum(c.get("http_requests", 0) for r in rows for c in r["calls"]),
        "inference_requests": sum(c.get("inference_requests", 0) for r in rows for c in r["calls"]),
        "prompt_tokens_reported": sum(c.get("prompt_tokens", 0) for r in rows for c in r["calls"]),
        "completion_tokens_reported": sum(c.get("completion_tokens", 0) for r in rows for c in r["calls"]),
        "unknown_usage_calls": sum(c.get("unknown_usage_calls", 0) for r in rows for c in r["calls"]),
        "prompt_bytes": sum(c["prompt_bytes"] for r in rows for c in r["calls"]),
        "wall_seconds": time.monotonic() - started,
    }
    by_root = {}
    for row in rows:
        by_root.setdefault(row["root_id"], []).append(int(row["terminal"]["optimal"]))
    summary["equal_root_macro_optimal"] = sum(sum(v) / len(v) for v in by_root.values()) / len(by_root)
    return {"schema": "molt-memory-panel-v3", "status": "development_only", "mode": mode,
            "identity": identity, "roster_digest": digest([asdict(w) for w in worlds]),
            "limits": {"max_actions": max_actions, "wall_seconds": wall_seconds,
                       "context_tokens": CONTEXT_TOKENS, "max_output_tokens": MAX_OUTPUT_TOKENS},
            "summary": summary, "rows": rows, "promotion_authorized": False}


def endpoint_client(model_id: str, registry_path: Path, binding_path: Path):
    # This dependency is intentionally optional until Research Verifiers v2 is integrated.
    try:
        from datagoblin.research_v2.runner import registry
        from datagoblin.research_v2.transport import request_action, validate_binding
    except ImportError as error:
        raise RuntimeError("Research Verifiers v2 transport required; integrate PR #81 first") from error
    entry = registry(registry_path)[model_id]
    binding = validate_binding(entry, json.loads(binding_path.read_text()))
    identity = {"role": "checkpoint", "registry_id": model_id, "identity_digest": digest([entry, binding]),
                "weights_sha256": binding["weights_sha256"], "identity_authority": "operator_attested"}

    def client(prompt: str, timeout: float) -> dict[str, Any]:
        return request_action(entry, binding, prompt, max_tokens=MAX_OUTPUT_TOKENS, timeout_seconds=timeout)

    caps = binding["resource_caps"]
    return client, identity, caps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--roots", type=int, default=4)
    parser.add_argument("--split", choices=("train", "calibration"), default="calibration")
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--wall-seconds", type=float, default=900)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--binding", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refuse to overwrite report")
    client = exact_public_client
    identity = {"role": "exact_public_control", "model_executed": False}
    max_actions, wall = args.max_actions, args.wall_seconds
    if args.model_id or args.registry or args.binding:
        if not (args.model_id and args.registry and args.binding):
            parser.error("model-id, registry and binding are jointly required")
        client, identity, caps = endpoint_client(args.model_id, args.registry, args.binding)
        max_actions = min(max_actions, caps["max_requests"] // 2)
        wall = min(wall, caps["wall_seconds"])
    result = run_panel(roster(args.seed, args.roots, args.split), args.mode, client,
                       max_actions=max_actions, wall_seconds=wall, identity=identity)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(canonical(result) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
