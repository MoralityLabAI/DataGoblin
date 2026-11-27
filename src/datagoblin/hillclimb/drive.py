"""Scripted self_adapter evaluation move: the service calls the driver model makes, made by a script.

A gym that serves the driver (the base model, or the base with an adapter loaded) runs the app first.
When the model driving the app ends without a decision, this records the evaluation through the same
service calls, so the lane's ledger still holds the result and the gate's verdict:

  propose_move (unless the lane has an open move) -> the dev bench (an earlier bench_make run whose task
  list matches the lane's eval contract, else bench_make in this move) -> driver_bench -> record_result
  -> decide; on replicate, one more driver_bench, record_result and decide.

    python -m datagoblin.hillclimb.drive --state-dir runs/hillclimb --base-url http://127.0.0.1:18296/v1 \
        --model bonsai-2.5-27b --seed 0 --count 120 --limit 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .lanes import HillclimbError
from .ledger import State, canonical
from .service import Service

LANE = "self_adapter"
ROOT = Path(__file__).resolve().parents[3]


def _effective(service: Service, run: dict[str, Any]) -> dict[str, Any]:
    recipe = service.registry.recipes[run["recipe"]]
    given = run.get("params") or {}
    return {name: given.get(name, param.default) for name, param in recipe.params.items()}


def _task_ids(service: Service, run: dict[str, Any]) -> list[str]:
    tasks = Path(service._abs(run["out_dir"])) / "tasks.jsonl"
    return [json.loads(line)["task_id"] for line in tasks.read_text(encoding="utf-8").splitlines() if line.strip()]


def dev_run(service: Service, state: State, *, contract: str | None, seed: int, count: int,
            limit: int) -> str | None:
    """The first finished dev bench_make run that is the lane's eval set.

    With an eval contract, that is the run whose first `limit` task ids hash to it (driver_bench's
    evaluated_sha256); without one, the run made with this seed and count.
    """
    for run_id, run in state.runs.items():
        if run["recipe"] != "bench_make" or run["status"] != "ok":
            continue
        params = _effective(service, run)
        if params.get("split") != "dev":
            continue
        if contract is None:
            if params.get("seed") == seed and params.get("count") == count:
                return run_id
            continue
        ids = _task_ids(service, run)[:limit]
        if hashlib.sha256(canonical(ids).encode()).hexdigest() == contract:
            return run_id
    return None


def _bench(service: Service, move_id: str, dev: str, base_url: str, model: str, limit: int,
           max_tokens: int, calls: list[dict[str, Any]]) -> None:
    run = service.run_recipe(move_id, "driver_bench", {"tasks": f"run:{dev}", "base_url": base_url, "model": model,
                                                       "limit": limit, "max_tokens": max_tokens})
    calls.append({"tool": "driver_bench", "run_id": run["run_id"], "status": run["status"]})
    if run["status"] != "ok":
        raise HillclimbError("run_failed", f"driver_bench {run['run_id']} ended {run['status']}: "
                                           f"{run['stderr_tail'][-400:]}")
    result = service.record_result(move_id, run["run_id"])
    calls.append({"tool": "record_result", "run_id": run["run_id"], "value": result["value"]})


def evaluate(service: Service, *, base_url: str, model: str, seed: int = 0, count: int = 120, limit: int = 120,
             max_tokens: int = 1024, hypothesis: str | None = None, expected: float | None = None,
             note: str = "") -> dict[str, Any]:
    state = service.ledger.state(service.registry)
    lane = state.lanes[LANE]
    dev = dev_run(service, state, contract=lane.contract, seed=seed, count=count, limit=limit)
    if dev is None and lane.contract is not None:  # checked before anything is written
        raise HillclimbError("no_eval_set", f"no bench_make run in this ledger matches the {LANE} eval contract "
                                            f"{lane.contract} at limit {limit}")
    calls: list[dict[str, Any]] = []
    move_id = lane.open_move
    if move_id is None:
        text = hypothesis or (f"Scripted evaluation of {model} on the self_adapter dev bench (split dev, seed {seed}, "
                              f"{count} tasks, limit {limit}); the gate decides against the incumbent.")
        move_id = service.propose_move(LANE, text, ["bench_make", "driver_bench"], expected)["move_id"]
        calls.append({"tool": "propose_move", "move_id": move_id})
        state = service.ledger.state(service.registry)
    if dev is None:
        made = service.run_recipe(move_id, "bench_make", {"split": "dev", "seed": seed, "count": count})
        calls.append({"tool": "bench_make", "run_id": made["run_id"], "status": made["status"]})
        if made["status"] != "ok":
            raise HillclimbError("run_failed", f"bench_make {made['run_id']} ended {made['status']}")
        dev = made["run_id"]
    move = state.moves[move_id]
    last = move.decisions[-1] if move.decisions else None
    if not move.results or (last and last["verdict"] == "replicate" and last["results"] >= len(move.results)):
        _bench(service, move_id, dev, base_url, model, limit, max_tokens, calls)
    decision = service.decide(move_id, note)
    calls.append({"tool": "decide", "verdict": decision["verdict"]})
    if decision["verdict"] == "replicate":
        _bench(service, move_id, dev, base_url, model, limit, max_tokens, calls)
        decision = service.decide(move_id, note)
        calls.append({"tool": "decide", "verdict": decision["verdict"]})
    return {"lane": LANE, "move_id": move_id, "dev_run": dev, "calls": calls, **decision}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--hypothesis")
    parser.add_argument("--expected", type=float)
    parser.add_argument("--note", default="scripted evaluation (datagoblin.hillclimb.drive)")
    args = parser.parse_args(argv)
    try:
        service = Service(root=args.root, state_dir=args.state_dir, max_tier="loopback")
        result = evaluate(service, base_url=args.base_url, model=args.model, seed=args.seed, count=args.count,
                          limit=args.limit, max_tokens=args.max_tokens, hypothesis=args.hypothesis,
                          expected=args.expected, note=args.note)
    except HillclimbError as exc:
        print(json.dumps({"error": exc.code, "detail": exc.detail}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
