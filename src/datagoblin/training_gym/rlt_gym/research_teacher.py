"""Collect real model responses through a chat endpoint and replay their scored actions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .hypergame_taste import _digest, _file_sha256, _json
from .research_campaign import _write_json, best, case, records, source_hashes
from .research_world import DEPTHS, Game

SYSTEM = (
    "Choose a legal research action to maximize true utility. Three causal modes are possible. The native "
    "hypotheses omit one mode, which can be targeted by a legal intervention. Experiment p16 lists the "
    "probability of seeing 1 out of 16 for modes 0, 1, 2. Each experiment costs credits and one turn. "
    "Publication is irreversible. Correct and wrong rewards change after goal_change turns. "
    "The posterior summarizes the visible actor endorsements and experiments. "
    "Return one JSON object with an action string. Use the experiment history when available."
)


def messages(task, state, game):
    public = task["public"]
    masses = game.masses(state)
    compact = {
        "family": public["family"], "K": public["K"], "T": public["T"],
        "credits": state.credits, "turns_left": state.turns, "goal_change": public["goal_change"],
        "native_hypotheses": public["native_hypotheses"], "legal_actions": game.legal(state),
        "experiment_history": state.history, "experiments": game.experiments,
        "posterior": [round(mass / sum(masses), 5) for mass in masses],
        "early_rewards": public["early_rewards"], "late_rewards": public["late_rewards"],
    }
    return [{"role": "system", "content": SYSTEM}, {
        "role": "user", "content": _json(compact),
    }]


def ask(endpoint, model, prompt, legal, seed, timeout):
    payload = {
        "model": model, "messages": prompt, "temperature": 0, "seed": seed, "max_tokens": 40,
        "response_format": {"type": "json_schema", "json_schema": {"name": "research_action", "schema": {
            "type": "object", "properties": {"action": {"type": "string", "enum": legal}},
            "required": ["action"], "additionalProperties": False,
        }}},
    }
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("DATAGOBLIN_TEACHER_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(endpoint, data=_json(payload).encode(), headers=headers)
    started = perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.load(response)
        choice = body["choices"][0]
        result = {
            "status": "received", "text": choice["message"]["content"], "usage": body.get("usage"),
            "model": body.get("model"), "response_id": body.get("id"), "finish_reason": choice.get("finish_reason"),
        }
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, ValueError) as error:
        result = {"status": "backend_error", "error": str(error)}
    return {**result, "latency_seconds": round(perf_counter() - started, 6), "request_sha256": _digest(payload)}


def ask_binary(binary, weights, model, prompt, legal, seed, timeout):
    """Run a local CPU-only GGUF model directly where loopback sockets are unavailable."""
    chat = "".join(f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n" for item in prompt)
    chat += "<|im_start|>assistant\n"
    choices = " | ".join(json.dumps(json.dumps(action)) for action in legal)
    grammar = 'root ::= "{" ws "\\"action\\"" ws ":" ws (' + choices + ') ws "}"\nws ::= [ \\t\\n]*\n'
    started = perf_counter()
    result = {}
    with TemporaryDirectory(prefix="research-teacher-") as directory:
        prompt_path = Path(directory) / "input.txt"
        grammar_path = Path(directory) / "action.gbnf"
        prompt_path.write_text(chat, encoding="utf-8")
        grammar_path.write_text(grammar, encoding="utf-8")
        try:
            process = subprocess.run([
                str(binary), "-m", str(weights), "--device", "none", "-ngl", "0", "--fit", "off",
                "-t", "6", "-c", "4096", "--no-conversation", "--no-warmup", "--no-display-prompt", "--temp", "0",
                "--seed", str(seed), "-n", "40", "-f", str(prompt_path), "--grammar-file", str(grammar_path),
            ], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=timeout)
            if process.returncode:
                result = {"status": "backend_error", "error": process.stderr[-2000:], "returncode": process.returncode}
            else:
                usage = {}
                for name, pattern in (
                    ("prompt_tokens", r"prompt eval time[^\n]*/\s*(\d+) tokens"),
                    ("completion_tokens", r"(?m)^.*?\beval time[^\n]*/\s*(\d+) runs"),
                ):
                    match = re.search(pattern, process.stderr)
                    if match:
                        usage[name] = int(match[1])
                raw = process.stdout.strip()
                text = raw.removesuffix(" [end of text]")
                result = {"status": "received", "text": text, "raw_stdout": raw, "usage": usage, "model": model,
                          "runtime_receipt": process.stderr[-3000:]}
        except (OSError, subprocess.TimeoutExpired) as error:
            result = {"status": "backend_error", "error": str(error)}
    return {**result, "latency_seconds": round(perf_counter() - started, 6),
            "request_sha256": _digest([prompt, legal, seed, model])}


def score_reply(game, state, sealed, reply):
    status, action = reply["status"], "abstain"
    if status == "received":
        try:
            response = json.loads(reply["text"])
            if set(response) != {"action"} or response["action"] not in game.legal(state):
                raise ValueError("malformed or illegal action")
            action, status = response["action"], "valid"
        except (TypeError, ValueError):
            status = "parse_error"
    future, reward = game.outcome(state, action, sealed)
    values = game.values(state)
    return future, {
        "status": status, "action": action, "reward": reward,
        "observation": future.history[-1][1] if future is not None else None,
        "regret": round(values[best(values)] - values[action], 10),
    }


def collect(out, key, runner, model, per_depth, timeout, weights_sha256):
    if not key or not 1 <= per_depth <= 54 or timeout <= 0:
        raise ValueError("teacher requires a private key and 1..54 roots per depth")
    out.mkdir(parents=True)
    start = perf_counter()
    path = out / "model_rollouts.jsonl"
    with path.open("xb") as stream:
        for depth in DEPTHS:
            for offset in range(per_depth):
                ordinal = (offset * 17 + depth) % 54
                task, sealed = case("train", depth, ordinal, 0, key)
                game = Game(task["public"])
                state, events = game.initial, []
                while state is not None:
                    prompt = messages(task, state, game)
                    reply = runner(prompt, game.legal(state), ordinal, timeout)
                    state, verdict = score_reply(game, state, sealed, reply)
                    events.append({"prompt_sha256": _digest(prompt), "reply": reply, "verdict": verdict})
                row = {
                    "root": task["root"], "K": depth, "split": "train", "ordinal": ordinal,
                    "requested_model": model, "events": events, "evidence_class": "actual_model_on_synthetic_world",
                    "utility": sum(event["verdict"]["reward"] for event in events),
                }
                stream.write((_json(row) + "\n").encode())
                stream.flush()
                game._solve.cache_clear()
                last_status = events[-1]["verdict"]["status"]
                print(f"model K={depth} ordinal={ordinal} visits={len(events)} status={last_status}", flush=True)
    _write_json(out / "manifest.json", {
        "model": model, "per_depth": per_depth, "weights_sha256": weights_sha256,
        "max_training_presentations_per_root": 4, "rollouts_sha256": _file_sha256(path),
        "sources": {**source_hashes(), Path(__file__).name: _file_sha256(Path(__file__))},
        "private_seed_commitment": hashlib.sha256(key).hexdigest(), "collection_wall_seconds": perf_counter() - start,
    })


def replay(folder, key):
    manifest = json.loads((folder / "manifest.json").read_bytes())
    path = folder / "model_rollouts.jsonl"
    if (manifest["private_seed_commitment"] != hashlib.sha256(key).hexdigest()
            or manifest["rollouts_sha256"] != _file_sha256(path)
            or manifest["sources"] != {**source_hashes(), Path(__file__).name: _file_sha256(Path(__file__))}):
        raise ValueError("model archive identity differs")
    totals = dict.fromkeys((
        "roots", "visits", "verifier_calls", "valid", "parse_error", "backend_error", "prompt_tokens",
        "completion_tokens", "token_receipts", "utility", "non_native_actions", "correct_publications",
    ), 0)
    seen = set()
    models = set()
    total_latency = 0
    for row in records(path):
        task, sealed = case("train", row["K"], row["ordinal"], 0, key)
        if row["root"] != task["root"] or task["root"] in seen or not 1 <= len(row["events"]) <= 4:
            raise ValueError("model root or custody differs")
        seen.add(task["root"])
        totals["roots"] += 1
        game = Game(task["public"])
        state = game.initial
        utility = 0
        for event in row["events"]:
            if state is None or _digest(messages(task, state, game)) != event["prompt_sha256"]:
                raise ValueError("model prompt differs")
            state, expected = score_reply(game, state, sealed, event["reply"])
            if expected != event["verdict"]:
                raise ValueError("model verifier receipt differs")
            utility += expected["reward"]
            totals[expected["status"]] += 1
            totals["visits"] += 1
            totals["verifier_calls"] += 1
            totals["non_native_actions"] += int(expected["action"] not in game.native)
            usage = event["reply"].get("usage")
            if usage:
                totals["token_receipts"] += 1
                for field in ("prompt_tokens", "completion_tokens"):
                    totals[field] += usage.get(field, 0)
            identity = event["reply"].get("model")
            if identity:
                models.add(identity)
            total_latency += event["reply"]["latency_seconds"]
        if state is not None or utility != row["utility"]:
            raise ValueError("model episode did not end")
        totals["utility"] += utility
        totals["correct_publications"] += int(row["events"][-1]["verdict"]["action"] == f"publish:{sealed['truth']}")
        game._solve.cache_clear()
    if len(seen) != len(DEPTHS) * manifest["per_depth"]:
        raise ValueError("missing model rollout roots")
    return {**totals, "models_returned": sorted(models), "request_wall_seconds": round(total_latency, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    dest = parser.add_mutually_exclusive_group(required=True)
    dest.add_argument("--out", type=Path)
    dest.add_argument("--replay", type=Path)
    parser.add_argument("--private-seed-file", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--weights-sha256")
    parser.add_argument("--per-depth", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=75)
    args = parser.parse_args()
    key = hashlib.sha256(args.private_seed_file.read_bytes() + b"/model").digest()
    if args.out:
        if bool(args.endpoint) == bool(args.binary) or not args.model or (args.binary and not args.weights):
            parser.error("collection requires a model and either an endpoint or a binary with weights")
        runner = (partial(ask_binary, args.binary, args.weights, args.model) if args.binary
                  else partial(ask, args.endpoint, args.model))
        digest = _file_sha256(args.weights) if args.weights else args.weights_sha256
        collect(args.out, key, runner, args.model, args.per_depth, args.timeout, digest)
    else:
        result = replay(args.replay, key)
        _write_json(args.replay / "replay.json", result)
        print(_json(result))


if __name__ == "__main__":
    main()
