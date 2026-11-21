"""Replay the Lean-CGRT tactic receipts with an automation-first model harness."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, dump_structured

from ..rlt_lean_ldt import AXIOMS

AUTO_PREFIX = 4
CHECK_BUDGET = 8
MAX_MENU = 64
FORBIDDEN = re.compile(r"sorry|admit|native_decide|run_tac|run_cmd|run_meta|by_elab|set_option|#|unsafe|"
                       r"implemented_by|extern|IO\.|ofReduceBool|trustCompiler")


def verified(result: Any) -> bool:
    if (not isinstance(result, dict) or result.get("ok") is not True
            or type(result.get("goals_after")) is not int or result["goals_after"] != 0):
        return False
    cert = result.get("cert")
    if not isinstance(cert, dict):
        return False
    axioms = cert.get("axioms")
    return (cert.get("kernel") is True and isinstance(axioms, list)
            and all(isinstance(axiom, str) and axiom in AXIOMS for axiom in axioms))


@dataclass(frozen=True)
class Case:
    identity: str
    split: str
    goals: list[str]
    auto: tuple[str, ...]
    menu: dict[int, str]
    results: dict[str, Any]

    def packet(self, limit: int, failed: tuple[int, ...] = ()) -> dict[str, Any]:
        return {
            "goals": self.goals, "failed_choices": [{"id": i, "tactic": self.menu[i]} for i in failed],
            "max_choices": limit,
            "menu": [{"id": i, "tactic": tactic} for i, tactic in self.menu.items() if i not in failed],
        }


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def position(row: dict[str, Any]) -> tuple[str, int, int]:
    return row["file"], row["line"], row["col"]


def context_index(states: Path, directory: Path, encoder_sha256: str) -> dict[tuple[str, int, int], list[str]]:
    rows = list(jsonl(states))
    wanted = {}
    for row in rows:
        split = row["split"]
        if not re.fullmatch(r"[a-z][a-z_0-9]*", split):
            raise ValueError("invalid Lean-CGRT split name")
        wanted.setdefault(split, {})[position(row)] = row
    found: dict[tuple[str, int, int], list[str]] = {}
    for split, cohort in wanted.items():
        for trace in jsonl(directory / f"{split}.jsonl.gz"):
            key = position(trace)
            row = cohort.get(key)
            if not row or key in found or trace.get("checker") != "certified":
                continue
            goals = trace["goals_before"]
            terminal = trace.get("terminal", len(trace["goals_after"]) < len(goals))
            tactic, kind = trace["tactic"], trace["kind"]
            atom = kind.startswith("«") and kind.endswith("»") and kind[1:-1] == tactic.strip()
            if not (goals and terminal and kind != "Lean.cdot" and not atom
                    and 0 < len(re.sub(r"\s+", " ", tactic).strip().encode()) < 96):
                continue
            first, *rest = trace["tactic"].split("\n")
            source = "\n".join([first] + [line[min(key[2], len(line) - len(line.lstrip(" "))):] for line in rest])
            if trace.get("decl") != row["decl"] or source != row["policies"]["human"][0]:
                raise ValueError(f"archived Lean-CGRT source tactic mismatch at {key}")
            if not isinstance(goals, list) or not all(isinstance(goal, str) for goal in goals):
                raise ValueError("invalid printed Lean goals")
            found[key] = goals
    # Exact c4 encoder digest, in archived row order, before censoring any row.
    digest = hashlib.sha256()
    for row in rows:
        key = position(row)
        if key not in found:
            raise ValueError(f"missing printed Lean goal at {key}")
        digest.update(json.dumps(list(key)).encode())
        digest.update(found[key][0].encode()[-384:])
        digest.update(b"\0")
    if digest.hexdigest() != encoder_sha256:
        raise ValueError("joined goals differ from the archived Lean-CGRT encoder digest")
    return found


def read_cases(states: Path, contexts: dict[tuple[str, int, int], list[str]]) -> Iterator[Case]:
    seen = set()
    with gzip.open(states, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not (row["reached"] and row.get("main_prop", True) and row.get("main_closed", True)):
                continue
            policies = row["policies"]
            auto = tuple(policies["auto"][:CHECK_BUDGET])
            if len(auto) != CHECK_BUDGET or len(set(auto)) != CHECK_BUDGET:
                raise ValueError("Lean-CGRT receipt is missing its eight distinct automation tactics")
            results = row["results"]
            if not isinstance(results, dict) or any(tactic not in results for tactic in auto):
                raise ValueError("Lean-CGRT receipt is missing automation checks")
            identity = sha256_json(position(row))
            if identity in seen:
                raise ValueError("duplicate Lean-CGRT source position")
            seen.add(identity)
            excluded = set(auto[:AUTO_PREFIX]) | (set(policies["human"]) - set(auto))
            bank = list(auto[AUTO_PREFIX:])
            for name, tactics in sorted(policies.items()):
                if name not in {"auto", "human"} and "+" not in name:
                    bank.extend(tactics)
            bank = [t for t in dict.fromkeys(bank) if t not in excluded and not FORBIDDEN.search(t)]
            yield Case(identity, row["split"], contexts.get(position(row), []), auto,
                       dict(enumerate(bank[:MAX_MENU], 1)), results)


def rank_ids(content: Any, offered: set[int], limit: int) -> list[int]:
    if not isinstance(content, str):
        return []
    content = re.sub(r"^\s*<think>\s*</think>\s*", "", content)
    try:
        choices = json.loads(content)
    except ValueError:
        return []
    if not isinstance(choices, list) or len(choices) > limit:
        return []
    if any(type(choice) is not int or choice not in offered for choice in choices):
        return []
    return choices if len(set(choices)) == len(choices) else []


class LocalChat:
    def __init__(self, port: int, model: str, timeout: float) -> None:
        if not 1 <= port <= 65535 or not model.strip() or not 1 <= timeout <= 120:
            raise ValueError("use a valid loopback port, model alias, and 1-120 second timeout")
        self.port, self.model, self.timeout = port, model, timeout
        models = self.request("/v1/models").get("data", [])
        if not isinstance(models, list) or not any(isinstance(entry, dict) and entry.get("id") == model
                                                   for entry in models):
            raise ValueError(f"model {model!r} is absent from loopback port {port}")

    def request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            with closing(HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)) as connection:
                connection.request(
                    "GET" if body is None else "POST", path,
                    body=canonical_json(body).encode("utf-8") if body is not None else None,
                    headers={"Content-Type": "application/json"},
                )
                with connection.getresponse() as response:
                    if response.status != 200:
                        raise OSError(f"model port {self.port} returned HTTP {response.status}")
                    data = response.read(65537)
        except HTTPException as exc:
            raise OSError(f"model port {self.port}: {exc}") from exc
        if len(data) > 65536:
            raise ValueError("model response exceeded 64 KiB")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError("expected one JSON object from local model")
        return result

    def __call__(self, packet: dict[str, Any]) -> dict[str, Any]:
        prompt = ("Rank distinct candidate tactics for closing this Lean goal. Return a JSON array of at most "
                  "max_choices integer IDs, best first, with no other text. Exclude tried tactics.\n"
                  f"{canonical_json(packet)}\n/no_think")
        request = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": 96, "stream": False}
        started = time.perf_counter()
        response = self.request("/v1/chat/completions", request)
        choices = response.get("choices")
        if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
                or not isinstance(choices[0].get("message"), dict)):
            raise ValueError("expected one choice from local model")
        return {"content": choices[0]["message"].get("content"), "usage": response.get("usage"),
                "wall_s": time.perf_counter() - started, "request_sha256": sha256_json(request),
                "response_sha256": sha256_json(response)}


Ranker = Callable[[dict[str, Any]], dict[str, Any]]


def rescue(case: Case, steps: tuple[tuple[str, int], ...], models: dict[str, Ranker]) -> dict[str, Any]:
    failed: tuple[int, ...] = ()
    calls = []
    for model, budget in steps:
        packet = case.packet(budget, failed)
        try:
            reply = models[model](packet)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            calls.append({"model": model, "error": str(exc)})
            break
        ids = rank_ids(reply.get("content"), set(case.menu) - set(failed), budget)
        calls.append({"model": model, "ids": ids, **{k: v for k, v in reply.items() if k != "content"}})
        for i in ids:
            if verified(case.results.get(case.menu[i])):
                return {"closed": True, "winning_choice": i, "checks": len(failed) + 1, "calls": calls}
            failed += (i,)
    return {"closed": False, "checks": len(failed), "calls": calls}


def run(states: Path, out: Path, *, expected_sha256: str, max_goals: int,
        models: dict[str, Ranker], model_ids: dict[str, str],
        context_dir: Path | None = None, encoder_sha256: str | None = None) -> dict[str, Any]:
    if not 1 <= max_goals <= 2000 or set(models) != set(model_ids) or set(models) - {"aleph", "bonsai"}:
        raise ValueError("use 1-2000 goals and named Aleph/Bonsai model receipts")
    if sha256_file(states) != expected_sha256:
        raise ValueError("Lean-CGRT states receipt hash mismatch")
    if (context_dir is None) != (encoder_sha256 is None) or (models and context_dir is None):
        raise ValueError("model runs require both the original context directory and its encoder digest")
    contexts = context_index(states, context_dir, encoder_sha256) if context_dir and encoder_sha256 else {}
    arms = {name: ((name, 4),) for name in models}
    if set(models) == {"aleph", "bonsai"}:
        arms["aleph_then_bonsai"] = (("aleph", 2), ("bonsai", 2))
    out.mkdir(parents=True)
    counts: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = {}
    with (out / "rollouts.jsonl").open("w", encoding="utf-8") as traces:
        for case in read_cases(states, contexts):
            counts["goals"] += 1
            split = by_split.setdefault(case.split, Counter())
            split["goals"] += 1
            prefix = any(verified(case.results[t]) for t in case.auto[:AUTO_PREFIX])
            baseline = any(verified(case.results[t]) for t in case.auto)
            has_witness = any(verified(case.results.get(t)) for t in case.menu.values())
            metrics = {"automation4": prefix, "automation8": baseline, "prefix_open": not prefix,
                       "open_with_menu_witness": not prefix and has_witness,
                       "automation8_miss_with_menu_witness": not baseline and has_witness}
            counts.update({key: int(value) for key, value in metrics.items()})
            split.update({key: int(value) for key, value in metrics.items()})
            rows = {} if prefix else {name: rescue(case, steps, models) for name, steps in arms.items()}
            for name in arms:
                closed = prefix or rows[name]["closed"]
                counts[f"{name}_close8"] += closed
                split[f"{name}_close8"] += closed
                if not prefix:
                    calls = rows[name]["calls"]
                    counts[f"{name}_rescues"] += closed
                    counts[f"{name}_checks"] += rows[name]["checks"]
                    counts[f"{name}_model_calls"] += len(calls)
                    counts[f"{name}_inconclusive"] += any("error" in call for call in calls)
                    for call in calls:
                        usage = call.get("usage")
                        tokens = ("prompt_tokens", "completion_tokens")
                        if isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0
                                                          for key in tokens):
                            for key in tokens:
                                counts[f"{name}_{key}"] += usage[key]
                        elif "error" not in call:
                            counts[f"{name}_unknown_usage"] += 1
            traces.write(canonical_json({"id": case.identity, "split": case.split, **metrics, "arms": rows}) + "\n")
            if counts["goals"] == max_goals:
                break
    if not counts["goals"]:
        raise ValueError("Lean-CGRT archive contains no certifiable goals")
    contexts_sha256 = sha256_json([[list(key), value] for key, value in sorted(contexts.items())]) if contexts else None
    summary = {"schema": "datagoblin.lean-cgrt-harness-replay.v1", "states_sha256": expected_sha256,
               "encoder_sha256": encoder_sha256, "contexts_sha256": contexts_sha256,
               "evidence": "historical_kernel_receipt_replay", "training_eligible": False,
               "automation_prefix": AUTO_PREFIX, "check_budget": CHECK_BUDGET, "model_aliases": model_ids,
               "counts": dict(counts), "by_split": {key: dict(val) for key, val in by_split.items()},
               "rollouts_sha256": sha256_file(out / "rollouts.jsonl")}
    dump_structured(out / "summary.json", summary)
    return summary
