"""Append-only, hash-chained hill-climb ledger and the exact promotion gate.

Every record carries the previous record's SHA-256, so an edited or reordered ledger fails to load.
Lane state (incumbent, plateau, budget, open move) is derived from the records, never stored.
`gate_verdict` is the one decision rule; the decision bench asks models to apply the same function.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .lanes import HillclimbError, Registry

GENESIS = "0" * 64
KINDS = frozenset({"proposal", "run", "result", "decision"})
VERDICTS = ("baseline", "promote", "replicate", "reject", "invalid", "explored")
_EPS = 1e-12


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def gate_verdict(
    *,
    direction: str,
    min_delta: float,
    selection_split: str,
    patience: int,
    plateau: int,
    incumbent: float | None,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Decide one move from its recorded results against the lane incumbent.

    invalid: no result, a non-finite value, or a result off the selection split.
    baseline: no incumbent yet. promote: delta >= min_delta. replicate: min_delta/2 <= delta < min_delta
    with exactly one result. reject: anything else. lane_stopped: reject or invalid with
    plateau + 1 >= patience (patience > 0).
    """

    def failed(verdict: str, reason: str, value: float | None = None, delta: float | None = None) -> dict[str, Any]:
        stopped = patience > 0 and plateau + 1 >= patience
        return {"verdict": verdict, "lane_stopped": stopped, "value": value, "delta": delta, "reason": reason}

    if not results:
        return failed("invalid", "no_result")
    values: list[float] = []
    for result in results:
        value = result.get("value")
        if type(value) not in (int, float) or not math.isfinite(value):
            return failed("invalid", "non_finite_value")
        if result.get("split") != selection_split:
            return failed("invalid", "wrong_split")
        values.append(float(value))
    value = sum(values) / len(values)
    if incumbent is None:
        return {"verdict": "baseline", "lane_stopped": False, "value": value, "delta": None, "reason": "first_result"}
    delta = (value - incumbent) if direction == "up" else (incumbent - value)
    if delta + _EPS >= min_delta:
        return {"verdict": "promote", "lane_stopped": False, "value": value, "delta": delta, "reason": "gate_passed"}
    if delta + _EPS >= min_delta / 2 and len(values) == 1:
        return {"verdict": "replicate", "lane_stopped": False, "value": value, "delta": delta, "reason": "near_gate"}
    return failed("reject", "below_gate", value, delta)


def lane_eligible(status: str, open_move: str | None, moves_used: int, max_moves: int) -> bool:
    """Whether a lane can take a new proposal (the decision bench's next-lane rule)."""
    return status == "open" and open_move is None and moves_used < max_moves


@dataclass
class MoveState:
    move_id: str
    lane: str
    proposal: dict[str, Any]
    status: str = "open"
    runs: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LaneState:
    lane: str
    status: str = "open"
    incumbent: dict[str, Any] | None = None
    plateau: int = 0
    moves_used: int = 0
    open_move: str | None = None
    contract: str | None = None
    promotions: int = 0

    def public(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "incumbent": self.incumbent,
            "plateau": self.plateau,
            "moves_used": self.moves_used,
            "open_move": self.open_move,
            "eval_contract": self.contract,
            "promotions": self.promotions,
        }


@dataclass
class State:
    lanes: dict[str, LaneState]
    moves: dict[str, MoveState]
    runs: dict[str, dict[str, Any]]
    records: int

    def next_move_id(self) -> str:
        return f"m{len(self.moves) + 1:04d}"

    def next_run_id(self) -> str:
        return f"r{len(self.runs) + 1:04d}"


class Ledger:
    """`<state_dir>/ledger.jsonl`, one JSON record per line, with a lock file for appends."""

    def __init__(self, state_dir: Path) -> None:
        self.dir = Path(state_dir)
        self.path = self.dir / "ledger.jsonl"
        self.lock_path = self.dir / "ledger.lock"

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        prev = GENESIS
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HillclimbError("ledger_corrupt", f"line {number} is not JSON") from exc
            claimed = record.pop("sha256", None)
            if record.get("prev") != prev or hashlib.sha256(canonical(record).encode()).hexdigest() != claimed:
                raise HillclimbError("ledger_corrupt", f"hash chain breaks at line {number}")
            if record.get("seq") != len(records) + 1 or record.get("kind") not in KINDS:
                raise HillclimbError("ledger_corrupt", f"bad seq or kind at line {number}")
            record["sha256"] = claimed
            records.append(record)
            prev = claimed
        return records

    def _lock(self) -> int:
        self.dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 30
        while True:
            try:
                return os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    if time.time() - self.lock_path.stat().st_mtime > 120:
                        self.lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise HillclimbError("ledger_locked", str(self.lock_path)) from None
                time.sleep(0.05)

    def append(self, kind: str, body: Mapping[str, Any]) -> dict[str, Any]:
        if kind not in KINDS:
            raise HillclimbError("ledger_corrupt", f"unknown record kind {kind}")
        handle = self._lock()
        try:
            records = self.read()
            record: dict[str, Any] = {
                "seq": len(records) + 1,
                "ts": utc_now(),
                "kind": kind,
                "body": dict(body),
                "prev": records[-1]["sha256"] if records else GENESIS,
            }
            record["sha256"] = hashlib.sha256(canonical(record).encode()).hexdigest()
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return record
        finally:
            os.close(handle)
            self.lock_path.unlink(missing_ok=True)

    def state(self, registry: Registry, records: list[dict[str, Any]] | None = None) -> State:
        records = self.read() if records is None else records
        lanes = {lane_id: LaneState(lane_id) for lane_id in registry.lanes}
        moves: dict[str, MoveState] = {}
        runs: dict[str, dict[str, Any]] = {}
        for record in records:
            body = record["body"]
            kind = record["kind"]
            if kind == "proposal":
                lane = lanes.setdefault(body["lane"], LaneState(body["lane"]))
                moves[body["move_id"]] = MoveState(body["move_id"], body["lane"], body)
                lane.moves_used += 1
                lane.open_move = body["move_id"]
            elif kind == "run":
                runs[body["run_id"]] = body
                moves[body["move_id"]].runs.append(body["run_id"])
            elif kind == "result":
                moves[body["move_id"]].results.append(body)
            elif kind == "decision":
                move = moves[body["move_id"]]
                move.decisions.append(body)
                lane = lanes[move.lane]
                verdict = body["verdict"]
                if verdict == "replicate":
                    continue
                move.status = "decided"
                lane.open_move = None
                if verdict in ("baseline", "promote"):
                    lane.incumbent = {"value": body["value"], "move_id": move.move_id, "verdict": verdict}
                    lane.plateau = 0
                    lane.promotions += verdict == "promote"
                    lane.contract = body.get("contract") or lane.contract
                elif verdict in ("reject", "invalid"):
                    lane.plateau += 1
        for lane_id, lane in lanes.items():
            spec = registry.lanes.get(lane_id)
            if spec is None:
                lane.status = "retired"
            elif spec.metric is not None and spec.patience > 0 and lane.plateau >= spec.patience:
                lane.status = "plateaued"
            elif lane.moves_used >= spec.max_moves and lane.open_move is None:
                lane.status = "exhausted"
            else:
                lane.status = "open"
        return State(lanes=lanes, moves=moves, runs=runs, records=len(records))
