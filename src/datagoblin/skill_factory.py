"""GB-SF-09 public-development bridge to the pinned HybridTRMLDT evaluator.

Uses Research Verifiers v2's exact rendered-token preflight and registered
checkpoint binding. The evaluator runs as a local subprocess; model sees only
the public task and current compact observation. No runtime is provisioned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any

from .research_v2.runner import DEFAULT_REGISTRY, registry
from .research_v2.transport import TransportError, request_action, validate_binding

SCHEMA = "datagoblin-goblinbench-sf09-run-v1"
HYBRID_REVISION = "3d2db3856c3cade1d32cda335d464e666f7a25a2"
GYMS = tuple(f"GSF-{i:02}" for i in range(1, 9))
MAX_TURNS = 4
MAX_LINE = 1048576
SESSION_SECONDS = 150
RUN_SECONDS = 1200
MAX_NEW_TOKENS = 512
MODULE = "research_gym.benchmarks.goblin_skill_factory.goblinbench"


def digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key")
        result[key] = value
    return result


def parse_action(raw: str) -> dict[str, Any] | None:
    try:
        action = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
        digest(action)
        if type(action) is not dict:
            return None
        return action
    except (ValueError, TypeError, RecursionError, OverflowError):
        return None


def checked_checkout(path: Path) -> Path:
    path = path.resolve()
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], timeout=15, text=True
        ).strip()
    if git("rev-parse", "HEAD") != HYBRID_REVISION:
        raise ValueError("HybridTRMLDT must be checked out at the pinned revision")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("HybridTRMLDT checkout must be clean")
    return path


def prompt(public: dict[str, Any], observation: dict[str, Any]) -> str:
    return (
        "You are a researcher in a bounded skill training gym. "
        "Run the required control first. Use the observed calibration metrics to "
        "choose experiments and submit the best eligible artifact. "
        "Return exactly one JSON object, no prose or markdown. "
        'Actions: {"action":"run_experiment","choice":"<allowed choice>"} or '
        '{"action":"submit","artifact_id":"a1","claim":"improved|equivalent|regressed"}. '
        "The claim compares quality and unsafe count to the calibration control. "
        "Final results are evaluator-only.\n"
        + json.dumps({"task": public, "observation": observation}, sort_keys=True,
                     separators=(",", ":"), allow_nan=False)
    )


def remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("session deadline")
    return seconds


def read_stream(pipe, messages: queue.Queue) -> None:
    try:
        while True:
            line = pipe.readline(MAX_LINE + 1)
            if not line or len(line) > MAX_LINE or not line.endswith(b"\n"):
                messages.put(None)
                return
            messages.put(line)
    except OSError:
        messages.put(None)


def receive(messages: queue.Queue, deadline: float) -> dict[str, Any]:
    try:
        raw = messages.get(timeout=remaining(deadline))
    except queue.Empty as error:
        raise TimeoutError("evaluator response deadline") from error
    if raw is None:
        raise ValueError("evaluator closed or oversized response")
    value = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
    if type(value) is not dict:
        raise ValueError("evaluator response must be an object")
    return value


def verify_episode(path: Path, gym: str, role: str, index: int) -> dict[str, Any]:
    row = json.loads((path / "episode.json").read_text(encoding="utf-8"))
    seal = row.pop("receipt_hash")
    if (row.get("schema") != "goblinbench.skill_factory.v1.episode"
            or row.get("gym") != gym or row.get("role") != role or row.get("index") != index
            or row.get("training_ready") is not False or digest(row) != seal):
        raise ValueError("evaluator receipt binding failed")
    result = row["result"]
    artifact = result.get("artifact")
    if artifact is not None:
        if artifact["file"] != "selected.weights":
            raise ValueError("unexpected artifact path")
        actual = hashlib.sha256((path / "selected.weights").read_bytes()).hexdigest()
        if actual != artifact["sha256"]:
            raise ValueError("selected artifact digest failed")
    if type(result["score"]) not in (int, float) or not 0 <= result["score"] <= 1:
        raise ValueError("invalid evaluator score")
    return {**row, "receipt_hash": seal}


def episode(hybrid: Path, entry: dict[str, Any], binding: dict[str, Any],
            gym: str, role: str, index: int, output: Path, run_deadline: float) -> dict[str, Any]:
    deadline = min(run_deadline, time.monotonic() + SESSION_SECONDS)
    row: dict[str, Any] = {"gym": gym, "role": role, "index": index, "score": 0.0,
                           "status": "failed", "turns": [], "training_ready": False}
    process = None
    try:
        remaining(deadline)
        # The trusted evaluator does not need the model server credential.
        env = dict(os.environ)
        credential_env = binding.get("api_key_env")
        if credential_env:
            env.pop(credential_env, None)
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
        messages: queue.Queue = queue.Queue()
        with output.with_suffix(".stderr.log").open("xb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-m", MODULE, "--session", gym, role, str(index),
                 "--out", str(output.resolve())],
                cwd=hybrid, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            )
            assert process.stdout is not None and process.stdin is not None
            reader = threading.Thread(target=read_stream, args=(process.stdout, messages), daemon=True)
            reader.start()
            initial = receive(messages, deadline)
            public, observation = initial["public"], initial["observation"]
            if (public.get("gym") != gym or public.get("role") != role
                    or public.get("schema") != "goblinbench.skill_factory.v1"):
                raise ValueError("unexpected evaluator task")
            row["root"] = public["root"]
            for _ in range(MAX_TURNS):
                if observation["done"]:
                    break
                text = prompt(public, observation)
                turn: dict[str, Any] = {"public_prompt": text}
                row["turns"].append(turn)
                try:
                    response = request_action(
                        entry, binding, text, max_tokens=MAX_NEW_TOKENS,
                        timeout_seconds=min(30.0, remaining(deadline)),
                    )
                except TransportError as error:
                    turn["transport_error"] = error.receipt
                    raise
                turn["transport"] = response
                action = parse_action(response["text"])
                turn["action"] = action
                process.stdin.write((json.dumps(action, allow_nan=False) + "\n").encode())
                process.stdin.flush()
                tool = receive(messages, deadline)
                turn["tool"] = tool
                observation = tool["observation"]
            if not observation["done"]:
                raise ValueError("evaluator did not close within turn budget")
            process.stdin.close()
            if process.wait(timeout=remaining(deadline)) != 0:
                raise ValueError("evaluator exit failure")
        receipt = verify_episode(output, gym, role, index)
        if receipt["root"] != row["root"] or [t["action"] for t in receipt["turns"]] != [
            t["action"] for t in row["turns"]
        ]:
            raise ValueError("transport/action receipt mismatch")
        row.update(status="completed", score=receipt["result"]["score"], evaluator=receipt)
    except (TransportError, TimeoutError, subprocess.TimeoutExpired, ValueError, KeyError,
            TypeError, OSError, RecursionError) as error:
        row["failure"] = error.code if isinstance(error, TransportError) else type(error).__name__
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if process is not None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.stdout and not process.stdout.closed:
                process.stdout.close()
    row["receipt_hash"] = digest(row)
    return row


def run(hybrid: Path, entry: dict[str, Any], binding: dict[str, Any], role: str,
        per_gym: int, out: Path) -> dict[str, Any]:
    if role not in ("development", "trace_train") or not 1 <= per_gym <= 4:
        raise ValueError("unsupported public suite")
    validate_binding(entry, binding)
    hybrid = checked_checkout(hybrid)
    out = out.resolve()
    if out == hybrid or out.is_relative_to(hybrid):
        raise ValueError("evaluator output must be outside the checkout")
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    deadline = time.monotonic() + RUN_SECONDS
    rows = []
    with (out / "episodes.jsonl").open("x", encoding="utf-8") as f:
        for index in range(per_gym):
            for gym in GYMS:
                row = episode(hybrid, entry, binding, gym, role, index,
                              out / f"{gym}-{index:02}", deadline)
                rows.append(row)
                f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                f.flush()
    families = {
        gym: sum(r["score"] for r in rows if r["gym"] == gym) / per_gym
        for gym in GYMS
    }
    report = {"schema": SCHEMA, "status": "public_development_only",
              "hybrid_revision": HYBRID_REVISION, "model_id": entry["id"],
              "binding_sha256": digest(binding), "registry_entry_sha256": digest(entry),
              "identity_verification": "operator_attested", "role": role,
              "cases": len(rows), "completed": sum(r["status"] == "completed" for r in rows),
              "per_gym": families, "macro_score": sum(families.values()) / len(GYMS),
              "context_tokens": 2048, "max_new_tokens": MAX_NEW_TOKENS,
              "max_turns": MAX_TURNS, "ranking_eligible": False,
              "training_ready": False, "episode_sha256": hashlib.sha256(
                  (out / "episodes.jsonl").read_bytes()).hexdigest()}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hybrid", type=Path, required=True)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--model-id", required=True)
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--role", choices=("development", "trace_train"), default="development")
    p.add_argument("--per-gym", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    entry = registry(a.registry)[a.model_id]
    binding = json.loads(a.binding.read_text(encoding="utf-8"))
    report = run(a.hybrid, entry, binding, a.role, a.per_gym, a.out)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
