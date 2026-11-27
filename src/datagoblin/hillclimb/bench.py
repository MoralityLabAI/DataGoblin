"""Hill-climb decision bench: exact tasks for evaluating and training the driver model itself.

Three task types, each answered by the function the MCP service itself uses:

- gate_verdict: apply the promotion gate (`ledger.gate_verdict`) to an incumbent and results;
- proposal_legality: the first failing check for a call to a real recipe (`lanes.call_reason`);
- next_lane: name a lane that can take a new proposal (`ledger.lane_eligible`).

A generated view belongs to the split its SHA-256 maps to (digest mod 3), so an identical task can
never sit in two splits. `make` writes tasks.jsonl (what a model sees) and gold.jsonl separately.

    python -m datagoblin.hillclimb.bench make --split dev --seed 0 --count 120 --out runs/bench-dev
    python -m datagoblin.hillclimb.bench run --tasks runs/bench-dev --base-url http://127.0.0.1:18090/v1 \
        --model inkling-2-mini --out runs/bench-dev-inkling
    python -m datagoblin.hillclimb.bench export-sft --tasks runs/bench-train --out runs/driver-sft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
from collections import Counter
from pathlib import Path
from typing import Any

from .agonist_worker import _post, first_json_object
from .lanes import TIERS, Param, Recipe, Registry, call_reason, is_loopback_url
from .ledger import canonical, gate_verdict, lane_eligible

SCHEMA = "datagoblin.hillclimb.bench.v1"
SPLITS = ("train", "dev", "test")
TYPES = ("gate_verdict", "proposal_legality", "next_lane")
TYPE_WEIGHTS = (4, 4, 2)

RULES = {
    "gate_verdict": (
        "You are the DataGoblin hill-climb gate. Apply these rules exactly to the JSON you are given.\n"
        "1. If results is empty, or any result's split differs from lane_gate.selection_split, the verdict is "
        "invalid.\n"
        "2. Otherwise, if incumbent is null, the verdict is baseline.\n"
        "3. Otherwise let value be the mean of the result values, and delta = value - incumbent when direction is "
        "up, or incumbent - value when direction is down.\n"
        "4. If delta >= min_delta, the verdict is promote.\n"
        "5. Otherwise, if delta >= min_delta / 2 and there is exactly one result, the verdict is replicate.\n"
        "6. Otherwise the verdict is reject.\n"
        "lane_stopped is true only when the verdict is reject or invalid, patience > 0 and plateau + 1 >= patience; "
        "otherwise it is false.\n"
        'Reply with JSON only: {"verdict": "...", "lane_stopped": true or false}'
    ),
    "proposal_legality": (
        "You check one DataGoblin hill-climb recipe call. Tiers are ordered offline < loopback < gpu. Apply the "
        "checks in this order and report the first that fails:\n"
        "1. lane.status is not open: lane_not_open.\n"
        "2. call.recipe is not in lane.recipes: recipe_not_in_lane.\n"
        "3. the recipe's tier is above max_tier: tier_not_allowed.\n"
        "4. call.params names a param the recipe does not declare: unknown_param.\n"
        "5. a declared param that has no default and is not optional is absent: missing_param.\n"
        "6. go through the given params in the order the recipe declares them; at the first bad value report "
        "non_loopback_url if it is a loopback_url param holding a string that is not an http URL with host "
        "127.0.0.1, localhost or ::1, an explicit port, path /v1 and no user, query or fragment; otherwise report "
        "bad_param. A value is bad when its type is wrong, an int or float is outside [min, max], an enum value or "
        "list item is not in choices, a list is empty, longer than max_items or repeats an item, or a str does not "
        "fully match its pattern. path values only need to be non-empty strings.\n"
        "7. otherwise: ok.\n"
        'legal is true exactly when the reason is ok. Reply with JSON only: {"legal": true or false, "reason": "..."}'
    ),
    "next_lane": (
        "Pick a DataGoblin hill-climb lane that can take a new proposal: its status is open, open_move is null "
        "and moves_used < max_moves. Any such lane is correct. If no lane qualifies, answer none.\n"
        'Reply with JSON only: {"lane": "<lane id or none>"}'
    ),
}

_MODEL_IDS = ("inkling-2-mini", "bonsai-2.5-27b", "bonsai-2-9b", "qwen3.5-4b")
_STRINGS = (*_MODEL_IDS, "plans/bdb_compare.json", "plans/driver_adapter.json", "eval-bonsai", "train-bonsai,suite")
_URLS = ("http://127.0.0.1:18090/v1", "http://localhost:8080/v1", "http://127.0.0.1:18298/v1")
_BAD_URLS = ("https://127.0.0.1:18090/v1", "http://10.0.0.5:18090/v1", "http://api.example.com/v1",
             "http://127.0.0.1:18090/v2", "http://127.0.0.1/v1")
_PATHS = ("run:r0003", "run:r0012/out", "Agonist/fixtures/tastes.jsonl", "runs/hillclimb/receipts.jsonl")
_UNKNOWN = ("learning_rate", "shell", "gpu", "max_steps", "out")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_of(view: dict[str, Any]) -> str:
    return SPLITS[int(hashlib.sha256(canonical(view).encode()).hexdigest()[:8], 16) % 3]


def _valid_value(rng: random.Random, param: Param) -> Any:
    kind = param.type
    if kind == "int":
        if param.choices:
            return rng.choice(param.choices)
        low = int(param.min if param.min is not None else 0)
        return rng.randint(low, int(param.max if param.max is not None else low + 99))
    if kind == "float":
        low, high = float(param.min or 0.0), float(param.max if param.max is not None else 1.0)
        return round(rng.uniform(low, high), 2)
    if kind == "enum":
        return rng.choice(param.choices or ("x",))
    if kind == "str":
        matching = [s for s in _STRINGS if param.check(s) == "ok"]
        return rng.choice(matching) if matching else "x"
    if kind == "loopback_url":
        return rng.choice(_URLS)
    if kind == "path":
        return rng.choice(_PATHS)
    limit = param.max_items or 3
    if kind == "path_list":
        return rng.sample(_PATHS, rng.randint(1, min(limit, len(_PATHS))))
    if kind == "enum_list":
        pool = list(param.choices or ("x",))
        return rng.sample(pool, rng.randint(1, min(limit, len(pool))))
    pool = list(param.choices) if param.choices else list(range(int(param.min or 0), int(param.min or 0) + 50))
    return rng.sample(pool, rng.randint(1, min(limit, len(pool))))


def _bad_value(rng: random.Random, param: Param) -> Any:
    kind = param.type
    if kind in ("int", "float"):
        high = param.max if param.max is not None else 100
        return rng.choice([int(high) + rng.randint(1, 50), str(param.min if param.min is not None else 1)])
    if kind == "enum":
        return "other"
    if kind == "str":
        return "bad value!"
    if kind == "loopback_url":
        return 18090
    if kind == "path":
        return ""
    if kind in ("int_list", "enum_list", "path_list"):
        good = _valid_value(rng, param)
        return rng.choice([[], [*good, good[0]], ["bogus"] if kind != "path_list" else [""]])
    return None


def _gate_task(rng: random.Random, registry: Registry) -> tuple[dict[str, Any], dict[str, Any]]:
    lane = rng.choice([lane for lane in registry.lanes.values() if lane.metric is not None])
    assert lane.metric is not None
    direction = lane.metric.direction if rng.random() < 0.7 else rng.choice(("up", "down"))
    min_delta = lane.metric.min_delta if rng.random() < 0.6 else rng.choice((0.01, 0.02, 0.03, 0.05, 0.1, 1.0, 2.0))
    patience = lane.patience if rng.random() < 0.7 else rng.randint(1, 5)
    plateau = rng.randint(0, max(0, patience - 1))
    counts = min_delta >= 1
    incumbent = None
    if rng.random() >= 0.15:
        incumbent = float(rng.randint(0, 200)) if counts else round(rng.uniform(0.05, 0.95), 3)
    region = rng.choice(("above", "near", "small", "below"))
    span = {"above": (1.0, 3.0), "near": (0.5, 1.0), "small": (0.0, 0.5), "below": (-2.0, 0.0)}[region]
    sign = 1 if direction == "up" else -1
    base = incumbent if incumbent is not None else (50.0 if counts else 0.5)
    mean = base + sign * rng.uniform(*span) * min_delta
    n = 1 if rng.random() < 0.8 else 2
    values = [mean] if n == 1 else [mean - min_delta * 0.3, mean + min_delta * 0.3]
    values = [round(v) if counts else min(1.0, max(0.0, round(v, 3))) for v in values]
    splits = [lane.metric.split] * n
    if rng.random() < 0.1:
        splits[0] = rng.choice([s for s in ("train", "dev", "calibration", "test") if s != lane.metric.split])
    results = [{"value": v, "split": s} for v, s in zip(values, splits, strict=True)]
    if rng.random() < 0.03:
        results = []
    view = {
        "lane_gate": {"metric": lane.metric.name, "direction": direction, "min_delta": min_delta,
                      "selection_split": lane.metric.split, "patience": patience},
        "incumbent": incumbent,
        "results": results,
        "plateau": plateau,
    }
    verdict = gate_verdict(direction=direction, min_delta=min_delta, selection_split=lane.metric.split,
                           patience=patience, plateau=plateau, incumbent=incumbent, results=results)
    return view, {"verdict": verdict["verdict"], "lane_stopped": verdict["lane_stopped"]}


def _recipe_view(recipe: Recipe) -> dict[str, Any]:
    view = recipe.public()
    view.pop("description", None)
    view.pop("produces", None)
    view.pop("requires", None)
    return view


def _legality_task(rng: random.Random, registry: Registry) -> tuple[dict[str, Any], dict[str, Any]]:
    lane = rng.choice(list(registry.lanes.values()))
    outside = [r for r in registry.recipes.values() if r.name not in lane.recipes]
    pick_outside = bool(outside) and rng.random() < 0.15
    inside = [registry.recipes[name] for name in lane.recipes]
    weights = [1 if r.tier == "offline" else 3 for r in inside]
    recipe = rng.choice(outside) if pick_outside else rng.choices(inside, weights=weights)[0]
    status = "open" if rng.random() < 0.85 else rng.choice(("plateaued", "exhausted"))
    max_tier = rng.choices(TIERS, weights=(2, 1, 1))[0]
    params = {name: _valid_value(rng, p) for name, p in recipe.params.items() if p.required or rng.random() < 0.5}
    corruption = rng.choice(("none", "none", "unknown", "missing", "value", "url"))
    if corruption == "unknown":
        params[rng.choice([u for u in _UNKNOWN if u not in recipe.params])] = 1
    elif corruption == "missing":
        required = [name for name, p in recipe.params.items() if p.required]
        if required:
            params.pop(rng.choice(required), None)
    elif corruption == "value" and recipe.params:
        name = rng.choice(list(recipe.params))
        params[name] = _bad_value(rng, recipe.params[name])
    elif corruption == "url":
        urls = [name for name, p in recipe.params.items() if p.type == "loopback_url"]
        if urls:
            params[urls[0]] = rng.choice(_BAD_URLS)
    reason = call_reason(lane_status=status, lane_recipes=lane.recipes, recipe=recipe, max_tier=max_tier,
                         params=params)
    view = {
        "lane": {"id": lane.id, "status": status, "recipes": list(lane.recipes)},
        "max_tier": max_tier,
        "recipe": _recipe_view(recipe),
        "call": {"recipe": recipe.name, "params": params},
    }
    return view, {"legal": reason == "ok", "reason": reason}


def _next_lane_task(rng: random.Random, registry: Registry) -> tuple[dict[str, Any], dict[str, Any]]:
    ids = rng.sample(sorted(registry.lanes), rng.randint(3, len(registry.lanes)))
    lanes = []
    for lane_id in ids:
        max_moves = registry.lanes[lane_id].max_moves
        status = rng.choices(("open", "plateaued", "exhausted"), weights=(6, 2, 2))[0]
        moves_used = max_moves if status == "exhausted" else rng.randint(0, max_moves)
        open_move = f"m{rng.randint(1, 99):04d}" if rng.random() < 0.3 else None
        lanes.append({"id": lane_id, "status": status, "open_move": open_move, "moves_used": moves_used,
                      "max_moves": max_moves})
    eligible = sorted(lane["id"] for lane in lanes
                      if lane_eligible(lane["status"], lane["open_move"], lane["moves_used"], lane["max_moves"]))
    return {"lanes": lanes}, {"eligible": eligible}


_MAKERS = {"gate_verdict": _gate_task, "proposal_legality": _legality_task, "next_lane": _next_lane_task}


def make_tasks(
    split: str, seed: int, count: int, registry: Registry
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    rng = random.Random(int(hashlib.sha256(f"{SCHEMA}/{split}/{seed}".encode()).hexdigest()[:16], 16))
    tasks: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(tasks) < count:
        attempts += 1
        if attempts > count * 50:
            raise ValueError("could not generate enough distinct tasks for this split")
        kind = rng.choices(TYPES, weights=TYPE_WEIGHTS)[0]
        view, answer = _MAKERS[kind](rng, registry)
        view_digest = hashlib.sha256(canonical(view).encode()).hexdigest()
        if _split_of(view) != split or view_digest in seen:
            continue
        seen.add(view_digest)
        task_id = f"hcb-{split}-{seed}-{len(tasks):05d}"
        tasks.append({"task_id": task_id, "split": split, "type": kind, "view": view, "view_sha256": view_digest})
        gold.append({"task_id": task_id, "type": kind, "gold": answer})
    return tasks, gold


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make(split: str, seed: int, count: int, out: Path, registry: Registry) -> dict[str, Any]:
    tasks, gold = make_tasks(split, seed, count, registry)
    out.mkdir(parents=True, exist_ok=False)
    _write_jsonl(out / "tasks.jsonl", tasks)
    _write_jsonl(out / "gold.jsonl", gold)
    manifest = {
        "schema": SCHEMA, "split": split, "seed": seed, "count": count,
        "by_type": dict(sorted(Counter(task["type"] for task in tasks).items())),
        "tasks_sha256": sha256_file(out / "tasks.jsonl"), "gold_sha256": sha256_file(out / "gold.jsonl"),
        "lanes_sha256": registry.sha256,
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def messages_for(task: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": RULES[task["type"]]}, {"role": "user", "content": canonical(task["view"])}]


def score_answer(kind: str, gold: dict[str, Any], answer: Any) -> bool:
    if not isinstance(answer, dict):
        return False
    if kind == "gate_verdict":
        return answer.get("verdict") == gold["verdict"] and answer.get("lane_stopped") is gold["lane_stopped"]
    if kind == "proposal_legality":
        return answer.get("legal") is gold["legal"] and answer.get("reason") == gold["reason"]
    lane = answer.get("lane")
    return lane in gold["eligible"] if gold["eligible"] else lane == "none"


def target_for(kind: str, gold: dict[str, Any]) -> dict[str, Any]:
    if kind == "gate_verdict":
        return {"verdict": gold["verdict"], "lane_stopped": gold["lane_stopped"]}
    if kind == "proposal_legality":
        return {"legal": gold["legal"], "reason": gold["reason"]}
    return {"lane": gold["eligible"][0] if gold["eligible"] else "none"}


def _load_bench(tasks_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = json.loads((tasks_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("not a hill-climb bench task directory")
    if sha256_file(tasks_dir / "tasks.jsonl") != manifest["tasks_sha256"]:
        raise ValueError("tasks.jsonl does not match its manifest")
    gold = {row["task_id"]: row for row in _read_jsonl(tasks_dir / "gold.jsonl")}
    return manifest, _read_jsonl(tasks_dir / "tasks.jsonl"), gold


def run(tasks_dir: Path, base_url: str, model: str, out: Path, *, limit: int = 120, max_tokens: int = 1024,
        timeout_seconds: float = 300.0) -> dict[str, Any]:
    if not is_loopback_url(base_url):
        raise ValueError("base URL must be http on 127.0.0.1, localhost or ::1 with path /v1")
    manifest, tasks, gold = _load_bench(tasks_dir)
    tasks = tasks[:limit]
    endpoint = base_url.rstrip("/") + "/chat/completions"
    out.mkdir(parents=True, exist_ok=False)
    answers: list[dict[str, Any]] = []
    started = time.monotonic()
    for task in tasks:
        row: dict[str, Any] = {"task_id": task["task_id"], "type": task["type"]}
        began = time.monotonic()
        try:
            data = _post(endpoint, {"model": model, "messages": messages_for(task), "temperature": 0,
                                    "max_tokens": max_tokens}, timeout_seconds)
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            answer = first_json_object(content)
            row.update(answer=answer, content=content[:4000], finish_reason=choice.get("finish_reason"),
                       returned_model=data.get("model"), usage=data.get("usage"))
        except urllib.error.HTTPError as exc:
            row.update(answer=None, error=f"http_error:{exc.code}")
        except OSError:
            row.update(answer=None, error="connection_error")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            row.update(answer=None, error="bad_response")
        row["latency_seconds"] = round(time.monotonic() - began, 3)
        row["correct"] = score_answer(task["type"], gold[task["task_id"]]["gold"], row["answer"])
        answers.append(row)
    _write_jsonl(out / "answers.jsonl", answers)
    by_type: dict[str, dict[str, Any]] = {}
    for kind in TYPES:
        rows = [row for row in answers if row["type"] == kind]
        if rows:
            hits = sum(row["correct"] for row in rows)
            by_type[kind] = {"n": len(rows), "correct": hits, "accuracy": hits / len(rows)}
    correct = sum(row["correct"] for row in answers)
    score = {
        "schema": "datagoblin.hillclimb.bench_score.v1",
        "split": manifest["split"],
        "model_requested": model,
        "models_returned": sorted({str(r["returned_model"]) for r in answers if r.get("returned_model")}),
        "base_url": base_url,
        "n": len(answers),
        "correct": correct,
        "accuracy": correct / len(answers) if answers else 0.0,
        "by_type": by_type,
        "parse_failures": sum(row["answer"] is None for row in answers),
        "errors": dict(sorted(Counter(row["error"] for row in answers if "error" in row).items())),
        "tasks_sha256": manifest["tasks_sha256"],
        "evaluated_sha256": hashlib.sha256(canonical([t["task_id"] for t in tasks]).encode()).hexdigest(),
        "temperature": 0,
        "max_tokens": max_tokens,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _write_json(out / "bench_score.json", score)
    return score


def export_sft(tasks_dir: Path, out: Path, *, answers_dir: Path | None = None,
               receipts: Path | None = None) -> dict[str, Any]:
    """Train-split SFT rows for the driver adapter; every row has training_ready false until curated.

    Rows carry `id` and `messages` ending in one assistant turn, the format bonsai-gym's trainer
    (adapter-bits `experiments/bonsai-gym`, `load_examples`) reads. Trace rows keep OpenAI tool calls
    and are for trainers that render them.
    """
    manifest, tasks, gold = _load_bench(tasks_dir)
    if manifest["split"] != "train":
        raise ValueError("export-sft takes train-split bench tasks only")
    rows: list[dict[str, Any]] = []
    for task in tasks:
        target = target_for(task["type"], gold[task["task_id"]]["gold"])
        rows.append({
            "id": f"{task['task_id']}/exact", "root": task["task_id"],
            "family_id": f"hillclimb-bench/{task['type']}", "split": "train",
            "source": "datagoblin.hillclimb.bench", "evidence_class": "synthetic_exact_oracle",
            "messages": [*messages_for(task), {"role": "assistant", "content": canonical(target)}],
            "training_ready": False,
        })
    sources: dict[str, str] = {"tasks.jsonl": manifest["tasks_sha256"]}
    if answers_dir is not None:
        score = json.loads((answers_dir / "bench_score.json").read_text(encoding="utf-8"))
        if score.get("tasks_sha256") != manifest["tasks_sha256"]:
            raise ValueError("answers were produced on different bench tasks")
        by_id = {task["task_id"]: task for task in tasks}
        for answer in _read_jsonl(answers_dir / "answers.jsonl"):
            if not answer.get("correct"):
                continue
            task = by_id[answer["task_id"]]
            rows.append({
                "id": f"{task['task_id']}/teacher", "root": task["task_id"],
                "family_id": f"hillclimb-bench/{task['type']}", "split": "train",
                "source": str(answer.get("returned_model") or score.get("model_requested")),
                "evidence_class": "verified_teacher",
                "messages": [*messages_for(task), {"role": "assistant", "content": canonical(answer["answer"])}],
                "training_ready": False,
            })
        sources["answers.jsonl"] = sha256_file(answers_dir / "answers.jsonl")
    traces = 0
    if receipts is not None:
        for line in receipts.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            receipt = json.loads(line)
            calls = receipt.get("toolCalls") or []
            transcript = receipt.get("transcript") or []
            decided = any(call.get("name") == "decide" and call.get("ok") for call in calls)
            if receipt.get("error") or not calls or not decided or not all(call.get("ok") for call in calls):
                continue
            root = "trace-" + hashlib.sha256(canonical(transcript).encode()).hexdigest()[:20]
            rows.append({
                "id": root, "root": root,
                "family_id": "hillclimb-trace", "split": "train", "source": str(receipt.get("servedModel")),
                "evidence_class": "driver_trace_tools_ok", "messages": transcript,
                "moves": receipt.get("moves", []), "training_ready": False,
            })
            traces += 1
        sources["receipts.jsonl"] = sha256_file(receipts)
    out.mkdir(parents=True, exist_ok=False)
    _write_jsonl(out / "sft.jsonl", rows)
    result = {
        "schema": "datagoblin.hillclimb.driver_sft.v1",
        "rows": len(rows),
        "by_evidence_class": dict(sorted(Counter(row["evidence_class"] for row in rows).items())),
        "traces": traces,
        "sources": sources,
        "sft_sha256": sha256_file(out / "sft.jsonl"),
        "training_ready": False,
    }
    _write_json(out / "manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DataGoblin hill-climb decision bench")
    commands = parser.add_subparsers(dest="command", required=True)
    m = commands.add_parser("make")
    m.add_argument("--split", choices=SPLITS, required=True)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--count", type=int, default=120)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--lanes", type=Path)
    r = commands.add_parser("run")
    r.add_argument("--tasks", type=Path, required=True)
    r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--limit", type=int, default=120)
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--timeout-seconds", type=float, default=300.0)
    e = commands.add_parser("export-sft")
    e.add_argument("--tasks", type=Path, required=True)
    e.add_argument("--answers", type=Path)
    e.add_argument("--receipts", type=Path)
    e.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "make":
            result = make(args.split, args.seed, args.count, args.out, Registry.load(args.lanes))
        elif args.command == "run":
            result = run(args.tasks, args.base_url, args.model, args.out, limit=args.limit,
                         max_tokens=args.max_tokens, timeout_seconds=args.timeout_seconds)
        else:
            result = export_sft(args.tasks, args.out, answers_dir=args.answers, receipts=args.receipts)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"error": f"bench_{args.command}_error", "detail": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
