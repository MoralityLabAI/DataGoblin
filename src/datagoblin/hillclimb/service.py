"""Tools behind the DataGoblin hill-climb MCP server.

The model proposes a move in a lane, runs allowlisted recipes for it, asks the service to read the
lane metric from a run's artifact, and asks for a decision. The service, not the model, extracts
metric values, applies the frozen gate and keeps the ledger. Tool errors carry stable codes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .lanes import (
    TIERS,
    HillclimbError,
    Recipe,
    Registry,
    call_reason,
    is_loopback_url,
    render_argv,
    render_text,
    tier_rank,
)
from .ledger import Ledger, MoveState, State, gate_verdict, lane_eligible

WRITE_TOOLS = frozenset({"propose_move", "run_recipe", "record_result", "decide", "abandon_move"})
MAX_HYPOTHESIS = 1200
TAIL = 1500


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "suite_status",
        "description": "Start here. Lanes (what each climbs, gate, incumbent, plateau, budget, open move), the "
                       "highest recipe tier this session may run, local model endpoints and GPU holders.",
        "inputSchema": _obj({}, []),
        "readOnly": True,
    },
    {
        "name": "ledger_status",
        "description": "Hash-chain check, lane states, open moves with their runs and results, and the last records.",
        "inputSchema": _obj({"lane": {"type": "string"}, "last": {"type": "integer", "minimum": 1, "maximum": 50}}, []),
        "readOnly": True,
    },
    {
        "name": "list_recipes",
        "description": "Recipes a lane may run: tier, parameters (type, range, choices, default) and the metric each "
                       "produces. Parameters left out take their defaults.",
        "inputSchema": _obj({"lane": {"type": "string"}}, []),
        "readOnly": True,
    },
    {
        "name": "propose_move",
        "description": "Open one move in a lane: a falsifiable hypothesis about what will raise the lane metric, "
                       "the recipes you plan to run, and your expected metric value. One open move per lane.",
        "inputSchema": _obj({
            "lane": {"type": "string"},
            "hypothesis": {"type": "string"},
            "plan": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 12},
            "expected": {"type": "number"},
            "parent": {"type": "string", "description": "An earlier move this one revises."},
        }, ["lane", "hypothesis", "plan"]),
        "readOnly": False,
    },
    {
        "name": "run_recipe",
        "description": "Run one recipe for an open move. Path parameters take run:<run_id> (that run's output "
                       "directory, optionally run:<run_id>/sub/path) or a path inside the DataGoblin checkout.",
        "inputSchema": _obj({
            "move_id": {"type": "string"},
            "recipe": {"type": "string"},
            "params": {"type": "object"},
        }, ["move_id", "recipe"]),
        "readOnly": False,
    },
    {
        "name": "record_result",
        "description": "Read the lane metric from a finished run's artifact and record it for the move. The service "
                       "extracts the value; it also previews the gate verdict.",
        "inputSchema": _obj({"move_id": {"type": "string"}, "run_id": {"type": "string"}}, ["move_id", "run_id"]),
        "readOnly": False,
    },
    {
        "name": "decide",
        "description": "Close a move with the lane gate: baseline, promote, replicate (record one more result, then "
                       "decide again), reject or invalid; lanes without a metric are closed as explored.",
        "inputSchema": _obj({"move_id": {"type": "string"}, "note": {"type": "string"}}, ["move_id"]),
        "readOnly": False,
    },
    {
        "name": "abandon_move",
        "description": "Close an open move that cannot produce a valid result (counts as invalid toward the plateau).",
        "inputSchema": _obj({"move_id": {"type": "string"}, "reason": {"type": "string"}}, ["move_id", "reason"]),
        "readOnly": False,
    },
]


def _pointer(document: Any, pointer: str) -> Any:
    if pointer in ("", "/"):
        return document
    current = document
    for raw in pointer.lstrip("/").split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not key.isdigit() or int(key) >= len(current):
                raise HillclimbError("metric_missing", f"{pointer}: no index {key}")
            current = current[int(key)]
        elif isinstance(current, dict) and key in current:
            current = current[key]
        else:
            raise HillclimbError("metric_missing", f"{pointer}: no key {key}")
    return current


def _number(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise HillclimbError("metric_not_numeric", where)
    return float(value)


def extract_metric(spec: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    """Read one metric value (plus split and eval contract) from a run's artifact."""
    artifact = out_dir / spec["artifact"]
    if not artifact.is_file():
        raise HillclimbError("metric_missing", f"artifact {spec['artifact']} was not written")
    document = json.loads(artifact.read_text(encoding="utf-8"))
    forbid = spec.get("forbid")
    if forbid is not None:
        rows_path = out_dir / forbid["artifact"]
        lines = rows_path.read_text(encoding="utf-8").splitlines() if rows_path.is_file() else []
        for line in lines:
            if line.strip() and json.loads(line).get(forbid["field"]) in forbid["values"]:
                raise HillclimbError("not_evidence", f"{forbid['artifact']} has {forbid['field']} "
                                                     f"in {forbid['values']}: wiring fixtures never become results")
    if "aggregate" in spec:
        agg = spec["aggregate"]
        items = [item for item in _pointer(document, agg["list"])
                 if isinstance(item, dict) and all(item.get(k) == v for k, v in agg.get("where", {}).items())]
        numerator = sum(_number(item.get(agg["num"]), agg["num"]) for item in items)
        denominator = sum(_number(item.get(agg["den"]), agg["den"]) for item in items)
        if denominator == 0:
            raise HillclimbError("empty_aggregate", f"no {agg['den']} in rows matching {agg.get('where')}")
        value = numerator / denominator
    else:
        value = _number(_pointer(document, spec["pointer"]), spec["pointer"])
        if "denominator" in spec:
            denominator = _number(_pointer(document, spec["denominator"]), spec["denominator"])
        elif "denominator_sum" in spec:
            denominator = sum(_number(_pointer(document, p), p) for p in spec["denominator_sum"])
        else:
            denominator = 1.0
        if denominator == 0:
            raise HillclimbError("empty_aggregate", "denominator is zero")
        value = value / denominator
    split = spec.get("split")
    if split is None and "split_pointer" in spec:
        split = _pointer(document, spec["split_pointer"])
    contract = None
    if "contract" in spec:
        contract_doc = json.loads((out_dir / spec["contract"]["artifact"]).read_text(encoding="utf-8"))
        contract = str(_pointer(contract_doc, spec["contract"]["pointer"]))
    return {"value": value, "split": split, "contract": contract,
            "artifact": spec["artifact"], "artifact_sha256": _sha256(artifact)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tail(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-TAIL:]


class Service:
    def __init__(
        self,
        *,
        root: Path,
        state_dir: Path | None = None,
        lanes: Path | None = None,
        max_tier: str = "offline",
        read_only: bool = False,
        python: str | None = None,
    ) -> None:
        tier_rank(max_tier)
        self.root = Path(root).resolve()
        self.state_dir = Path(state_dir).resolve() if state_dir else self.root / "runs" / "hillclimb"
        self.registry = Registry.load(lanes)
        self.ledger = Ledger(self.state_dir)
        self.max_tier = max_tier
        self.read_only = read_only
        self.python = python or sys.executable
        self.adaptergym_python = os.environ.get("DATAGOBLIN_ADAPTERGYM_PYTHON", "python")
        self.roots = self.registry.resolve_roots(self.root)
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "suite_status": self.suite_status,
            "ledger_status": self.ledger_status,
            "list_recipes": self.list_recipes,
            "propose_move": self.propose_move,
            "run_recipe": self.run_recipe,
            "record_result": self.record_result,
            "decide": self.decide,
            "abandon_move": self.abandon_move,
        }

    # ---- MCP surface ----

    def tools(self) -> list[dict[str, Any]]:
        listed = []
        for spec in TOOL_SPECS:
            if self.read_only and not spec["readOnly"]:
                continue
            listed.append({
                "name": spec["name"],
                "description": spec["description"],
                "inputSchema": spec["inputSchema"],
                "annotations": {"readOnlyHint": spec["readOnly"], "destructiveHint": False,
                                "openWorldHint": spec["name"] == "run_recipe"},
            })
        return listed

    def call(self, name: str, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            raise HillclimbError("unknown_tool", name)
        if name in WRITE_TOOLS and self.read_only:
            raise HillclimbError("read_only", f"{name} needs a session started without --read-only")
        arguments = dict(arguments or {})
        spec = next(s for s in TOOL_SPECS if s["name"] == name)["inputSchema"]
        unknown = sorted(set(arguments) - set(spec["properties"]))
        missing = [key for key in spec["required"] if key not in arguments]
        if unknown or missing:
            raise HillclimbError("bad_arguments", f"unknown {unknown}, missing {missing}")
        return handler(**arguments)

    # ---- helpers ----

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _abs(self, recorded: str) -> Path:
        path = Path(recorded)
        return (path if path.is_absolute() else self.root / path).resolve()

    def _state(self) -> State:
        return self.ledger.state(self.registry)

    def _lane_view(self, lane_id: str, state: State) -> dict[str, Any]:
        lane = self.registry.lanes[lane_id]
        ls = state.lanes[lane_id]
        return {
            "id": lane_id, "title": lane.title, "climbs": lane.climbs,
            "metric": lane.metric.public() if lane.metric else None,
            "patience": lane.patience, "max_moves": lane.max_moves, **ls.public(),
            "eligible_for_new_move": lane_eligible(ls.status, ls.open_move, ls.moves_used, lane.max_moves),
        }

    def _move(self, state: State, move_id: Any) -> MoveState:
        if not isinstance(move_id, str) or move_id not in state.moves:
            raise HillclimbError("unknown_move", str(move_id))
        return state.moves[move_id]

    def _probe(self, base: str) -> dict[str, Any]:
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(base.rstrip("/") + "/models", timeout=3) as response:
                data = json.loads(response.read(1_000_000))
            models = [str(m.get("id")) for m in data.get("data", []) if isinstance(m, dict)]
            return {"url": base, "ok": True, "models": models}
        except (OSError, ValueError, AttributeError):
            return {"url": base, "ok": False, "models": []}

    def _endpoints(self) -> list[dict[str, Any]]:
        urls = [u.strip() for u in os.environ.get("DATAGOBLIN_ENDPOINTS", "").split(",") if u.strip()]
        urls = [u for u in dict.fromkeys(urls) if is_loopback_url(u)]
        if not urls:
            return []
        with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
            return list(pool.map(self._probe, urls))

    def _gpu(self) -> dict[str, Any] | None:
        exe = shutil.which("vramorama")
        if exe is None:
            return None
        try:
            done = subprocess.run([exe, "ps", "--json"], capture_output=True, text=True, timeout=20,
                                  stdin=subprocess.DEVNULL, check=False)
            data = json.loads(done.stdout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return {"available": False}
        adapters = []
        for adapter in data.get("adapters", []):
            procs = sorted(adapter.get("procs", []), key=lambda p: -int(p.get("mib") or 0))[:8]
            adapters.append({
                "name": adapter.get("name"), "total_mib": adapter.get("total_mib"),
                "used_mib": adapter.get("used_mib"),
                "free_mib": (adapter.get("total_mib") or 0) - (adapter.get("used_mib") or 0),
                "holders": [{"mib": p.get("mib"), "label": p.get("label") or p.get("owner") or p.get("name")}
                            for p in procs],
            })
        return {"available": True, "adapters": adapters,
                "rule": "One queued GPU job per session; launch GPU work only through vramorama leases."}

    def _resolve_path(self, value: str, kind: str | None, state: State) -> Path:
        if value.startswith("run:"):
            run_id, _, sub = value[4:].partition("/")
            run = state.runs.get(run_id)
            if run is None:
                raise HillclimbError("bad_path", f"unknown run {run_id}")
            path = self._abs(run["out_dir"])
            if sub:
                path = (path / sub).resolve()
        else:
            path = self._abs(value)
        inside = any(path == base or base in path.parents for base in (self.root, self.state_dir.resolve()))
        if not inside:
            raise HillclimbError("bad_path", f"{value} is outside the DataGoblin checkout and state directory")
        if kind == "dir" and not path.is_dir():
            raise HillclimbError("bad_path", f"{value} is not a directory")
        if kind == "file" and not path.is_file():
            raise HillclimbError("bad_path", f"{value} is not a file")
        if kind is None and not path.exists():
            raise HillclimbError("bad_path", f"{value} does not exist")
        return path

    def _recipe_available(self, recipe: Recipe) -> list[str]:
        return [name for name in recipe.requires if self.roots.get(name) is None]

    # ---- tools ----

    def suite_status(self) -> dict[str, Any]:
        state = self._state()
        return {
            "root": self.root.as_posix(),
            "state_dir": self._rel(self.state_dir),
            "max_tier": self.max_tier,
            "tiers": list(TIERS),
            "read_only": self.read_only,
            "lanes_sha256": self.registry.sha256,
            "lanes": [self._lane_view(lane_id, state) for lane_id in self.registry.lanes],
            "external_roots": {name: (p.as_posix() if p else None) for name, p in self.roots.items()},
            "endpoints": self._endpoints(),
            "gpu": self._gpu(),
        }

    def ledger_status(self, lane: str | None = None, last: int = 10) -> dict[str, Any]:
        records = self.ledger.read()
        state = self.ledger.state(self.registry, records)
        lanes = [lane] if lane else list(self.registry.lanes)
        for lane_id in lanes:
            if lane_id not in self.registry.lanes:
                raise HillclimbError("unknown_lane", lane_id)
        open_moves = []
        for move in state.moves.values():
            if move.status == "open" and move.lane in lanes:
                open_moves.append({
                    "move_id": move.move_id, "lane": move.lane, "hypothesis": move.proposal["hypothesis"],
                    "plan": move.proposal["plan"],
                    "runs": [{k: state.runs[r][k] for k in ("run_id", "recipe", "status", "out_dir")}
                             for r in move.runs],
                    "results": [{k: res[k] for k in ("run_id", "value", "split")} for res in move.results],
                })
        recent = []
        for record in records[-max(1, min(int(last), 50)):]:
            body = record["body"]
            summary = {"seq": record["seq"], "kind": record["kind"], "move_id": body.get("move_id")}
            for key in ("lane", "recipe", "run_id", "status", "value", "verdict", "reason"):
                if key in body:
                    summary[key] = body[key]
            recent.append(summary)
        return {"chain_ok": True, "records": len(records),
                "lanes": [self._lane_view(lane_id, state) for lane_id in lanes],
                "open_moves": open_moves, "recent": recent}

    def list_recipes(self, lane: str | None = None) -> dict[str, Any]:
        if lane is not None and lane not in self.registry.lanes:
            raise HillclimbError("unknown_lane", lane)
        names = self.registry.lanes[lane].recipes if lane else tuple(self.registry.recipes)
        recipes = []
        for name in names:
            recipe = self.registry.recipes[name]
            missing = self._recipe_available(recipe)
            recipes.append({
                **recipe.public(),
                "lanes": self.registry.lanes_using(name),
                "allowed": tier_rank(recipe.tier) <= tier_rank(self.max_tier) and not self.read_only,
                "available": not missing,
                **({"missing": missing} if missing else {}),
            })
        return {"max_tier": self.max_tier, "recipes": recipes}

    def propose_move(self, lane: str, hypothesis: str, plan: list[str], expected: float | None = None,
                     parent: str | None = None) -> dict[str, Any]:
        state = self._state()
        if lane not in self.registry.lanes:
            raise HillclimbError("unknown_lane", lane)
        spec = self.registry.lanes[lane]
        ls = state.lanes[lane]
        if ls.open_move is not None:
            raise HillclimbError("move_already_open", f"{lane} has open move {ls.open_move}: decide or abandon it")
        if not lane_eligible(ls.status, ls.open_move, ls.moves_used, spec.max_moves):
            raise HillclimbError("lane_not_open", f"{lane} is {ls.status}")
        if not isinstance(hypothesis, str) or not 10 <= len(hypothesis.strip()) <= MAX_HYPOTHESIS:
            raise HillclimbError("bad_arguments", f"hypothesis must be 10..{MAX_HYPOTHESIS} characters")
        if not isinstance(plan, list) or not plan or not all(isinstance(p, str) for p in plan):
            raise HillclimbError("bad_arguments", "plan must be a non-empty list of recipe names")
        outside = [name for name in plan if name not in spec.recipes]
        if outside:
            raise HillclimbError("recipe_not_in_lane", f"{outside} not in {lane}")
        if expected is not None and (type(expected) not in (int, float) or not math.isfinite(expected)):
            raise HillclimbError("bad_arguments", "expected must be a finite number")
        if parent is not None and parent not in state.moves:
            raise HillclimbError("unknown_move", parent)
        move_id = state.next_move_id()
        body = {"move_id": move_id, "lane": lane, "hypothesis": hypothesis.strip(), "plan": plan,
                "expected": expected, "parent": parent, "lanes_sha256": self.registry.sha256,
                "gate": spec.metric.public() if spec.metric else None, "incumbent": ls.incumbent}
        self.ledger.append("proposal", body)
        return {"move_id": move_id, "lane": lane, "gate": body["gate"], "incumbent": ls.incumbent,
                "plateau": ls.plateau, "patience": spec.patience,
                "moves_left_after_this": spec.max_moves - ls.moves_used - 1}

    def run_recipe(self, move_id: str, recipe: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        state = self._state()
        move = self._move(state, move_id)
        if move.status != "open":
            raise HillclimbError("move_not_open", move_id)
        if recipe not in self.registry.recipes:
            raise HillclimbError("unknown_recipe", recipe)
        spec = self.registry.recipes[recipe]
        lane = self.registry.lanes[move.lane]
        params = dict(params or {})
        reason = call_reason(lane_status=state.lanes[move.lane].status, lane_recipes=lane.recipes, recipe=spec,
                             max_tier=self.max_tier, params=params)
        if reason != "ok":
            raise HillclimbError(reason, f"{recipe} in {move.lane}")
        missing = self._recipe_available(spec)
        if missing:
            raise HillclimbError("recipe_unavailable", f"{recipe} needs {missing}; set the root's environment variable")
        values: dict[str, Any] = {}
        for name, param in spec.params.items():
            value = params.get(name, param.default)
            if value is None:
                continue
            if param.type == "path":
                value = str(self._resolve_path(value, param.kind, state))
            elif param.type == "path_list":
                value = [str(self._resolve_path(item, param.kind, state)) for item in value]
            values[name] = value
        run_id = state.next_run_id()
        run_dir = self.state_dir / "moves" / move_id / f"{run_id}-{recipe}"
        run_dir.mkdir(parents=True, exist_ok=False)
        out = run_dir / "out"
        if spec.mkdir_out:
            out.mkdir()
        context = {"python": self.python, "out": str(out), "root": str(self.root),
                   "adaptergym_python": self.adaptergym_python}
        context.update({name: str(path) for name, path in self.roots.items() if path is not None})
        argv = render_argv(spec, values, context)
        cwd = Path(render_text(spec.cwd, {}, context)) if spec.cwd else self.root
        env = dict(os.environ)
        sep = os.pathsep
        env["PYTHONPATH"] = sep.join(p for p in (str(self.root / "src"), str(self.root),
                                                 env.get("PYTHONPATH", "")) if p)
        env.update(spec.env)
        started = time.monotonic()
        status, exit_code = "ok", None
        with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
            try:
                done = subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=stdout,
                                      stderr=stderr, timeout=spec.timeout_seconds, check=False)
                exit_code = done.returncode
                status = "ok" if exit_code == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
            except OSError as exc:
                status = "failed"
                stderr.write(f"spawn failed: {exc}\n".encode())
        seconds = round(time.monotonic() - started, 3)
        artifacts = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())[:60] \
            if out.exists() else []
        body = {
            "run_id": run_id, "move_id": move_id, "recipe": recipe, "tier": spec.tier,
            "params": params, "in_plan": recipe in move.proposal["plan"], "argv": argv, "cwd": self._rel(cwd),
            "status": status, "exit_code": exit_code, "seconds": seconds, "out_dir": self._rel(out),
            "stdout_sha256": _sha256(run_dir / "stdout.log"), "artifacts": artifacts,
        }
        self.ledger.append("run", body)
        metric = lane.metric.name if lane.metric else None
        return {
            "run_id": run_id, "status": status, "exit_code": exit_code, "seconds": seconds,
            "out": f"run:{run_id}", "out_dir": body["out_dir"], "artifacts": artifacts[:30],
            "produces_lane_metric": bool(metric and metric in spec.produces),
            "stdout_tail": _tail(run_dir / "stdout.log"), "stderr_tail": _tail(run_dir / "stderr.log"),
        }

    def record_result(self, move_id: str, run_id: str) -> dict[str, Any]:
        state = self._state()
        move = self._move(state, move_id)
        if move.status != "open":
            raise HillclimbError("move_not_open", move_id)
        run = state.runs.get(run_id)
        if run is None or run["move_id"] != move_id:
            raise HillclimbError("unknown_run", f"{run_id} is not a run of {move_id}")
        if run["status"] != "ok":
            raise HillclimbError("run_failed", f"{run_id} ended {run['status']}")
        lane = self.registry.lanes[move.lane]
        if lane.metric is None:
            raise HillclimbError("no_metric", f"{move.lane} has no metric; decide closes the move as explored")
        if any(res["run_id"] == run_id for res in move.results):
            raise HillclimbError("duplicate_result", run_id)
        produces = self.registry.recipes[run["recipe"]].produces.get(lane.metric.name)
        if produces is None:
            raise HillclimbError("no_metric", f"{run['recipe']} does not produce {lane.metric.name}")
        extracted = extract_metric(produces, self._abs(run["out_dir"]))
        body = {"move_id": move_id, "run_id": run_id, "metric": lane.metric.name, **extracted}
        self.ledger.append("result", body)
        ls = state.lanes[move.lane]
        preview = gate_verdict(direction=lane.metric.direction, min_delta=lane.metric.min_delta,
                               selection_split=lane.metric.split, patience=lane.patience, plateau=ls.plateau,
                               incumbent=ls.incumbent["value"] if ls.incumbent else None,
                               results=[*move.results, body])
        return {"move_id": move_id, "metric": lane.metric.name, "value": extracted["value"],
                "split": extracted["split"], "incumbent": ls.incumbent, "gate_preview": preview}

    def decide(self, move_id: str, note: str | None = None) -> dict[str, Any]:
        state = self._state()
        move = self._move(state, move_id)
        if move.status != "open":
            raise HillclimbError("move_not_open", move_id)
        lane = self.registry.lanes[move.lane]
        ls = state.lanes[move.lane]
        incumbent = ls.incumbent["value"] if ls.incumbent else None
        contracts = {res.get("contract") for res in move.results if res.get("contract")}
        contract = next(iter(contracts)) if len(contracts) == 1 else None
        if lane.metric is None:
            verdict = {"verdict": "explored", "lane_stopped": False, "value": None, "delta": None,
                       "reason": "lane_has_no_metric"}
        elif len(contracts) > 1 or (ls.contract and contract and contract != ls.contract):
            stopped = lane.patience > 0 and ls.plateau + 1 >= lane.patience
            verdict = {"verdict": "invalid", "lane_stopped": stopped, "value": None, "delta": None,
                       "reason": "eval_contract_changed"}
        else:
            verdict = gate_verdict(direction=lane.metric.direction, min_delta=lane.metric.min_delta,
                                   selection_split=lane.metric.split, patience=lane.patience, plateau=ls.plateau,
                                   incumbent=incumbent, results=move.results)
        body = {"move_id": move_id, "lane": move.lane, **verdict, "incumbent_before": ls.incumbent,
                "contract": contract, "note": (note or "")[:600], "results": len(move.results)}
        self.ledger.append("decision", body)
        after = self._state().lanes[move.lane]
        return {**verdict, "move_id": move_id, "lane": move.lane, "incumbent_before": ls.incumbent,
                "incumbent_after": after.incumbent, "lane_status": after.status, "plateau": after.plateau,
                "move_open": verdict["verdict"] == "replicate"}

    def abandon_move(self, move_id: str, reason: str) -> dict[str, Any]:
        state = self._state()
        move = self._move(state, move_id)
        if move.status != "open":
            raise HillclimbError("move_not_open", move_id)
        lane = self.registry.lanes[move.lane]
        ls = state.lanes[move.lane]
        stopped = lane.metric is not None and lane.patience > 0 and ls.plateau + 1 >= lane.patience
        verdict = "invalid" if lane.metric is not None else "explored"
        body = {"move_id": move_id, "lane": move.lane, "verdict": verdict, "lane_stopped": stopped,
                "value": None, "delta": None, "reason": "abandoned", "note": str(reason)[:600],
                "incumbent_before": ls.incumbent, "contract": None, "results": len(move.results)}
        self.ledger.append("decision", body)
        after = self._state().lanes[move.lane]
        return {"move_id": move_id, "verdict": verdict, "lane_status": after.status, "plateau": after.plateau}
