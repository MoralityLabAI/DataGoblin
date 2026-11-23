"""Matched call-ceiling policies for courier proposals and certified STOP repair."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

from .native import STOP, Session, State, World, successors
from .oracle import repair_routes


@dataclass(frozen=True)
class Observation:
    world: World
    state: State
    legal: tuple[int, ...]
    moves_left: int
    remaining_calls: int


@dataclass(frozen=True)
class Proposal:
    action: int
    stop_confidence: float = 0.0
    raw_action: int | None = None
    visits: int = 0


@dataclass(frozen=True)
class Repair:
    mode: str = "standard"
    depth: int = 4
    attempt_limit: int = 2
    proposer: Callable[[Observation, int, int], tuple[tuple[int, ...], ...]] | None = None
    gate: Callable[[Proposal], bool] | None = None


def _repair(session: Session, obs: Observation, policy: Repair, attempts: list[dict], trigger: str) -> bool:
    depth = min(obs.moves_left, policy.depth)
    allowance = min(session.remaining - 1, policy.attempt_limit - len(attempts))
    if depth <= 0 or allowance <= 0:
        return False
    if policy.proposer is None:
        candidates = repair_routes(obs.world, obs.state, depth, allowance)
    else:
        candidates = policy.proposer(obs, depth, allowance)
    for route in candidates[:allowance]:
        if not route or len(route) > depth or any(type(a) is not int or not 0 <= a < STOP for a in route):
            continue
        receipt = session.certify(tuple(route), macro=True)
        attempts.append(
            {
                "trigger": trigger,
                "actions": route,
                "accepted": receipt.accepted,
                "completed": receipt.accepted and session.state.complete,
            }
        )
        if receipt.accepted:
            return True
    return False


def run_episode(session: Session, policy: Callable[[Observation], Proposal], repair: Repair = Repair()) -> dict:
    if repair.mode not in {"standard", "reserve", "early", "gate"}:
        raise ValueError("Unknown courier repair controller")
    if repair.mode == "gate" and repair.gate is None:
        raise ValueError("Learned gate controller requires a fitted gate")
    visits = invalid = raw_invalid = premature = decisions = 0
    seen = {session.state}
    attempts = []
    early_considered = False
    exit_reason = "call_ceiling"
    while session.remaining:
        choices = successors(session.world, session.state)
        legal = tuple(sorted(a for a, after in choices.items() if after not in seen))
        if not session.moves_left:
            legal = ()
        obs = Observation(session.world, session.state, (*legal, STOP), session.moves_left, session.remaining)
        proposal = policy(obs)
        decisions += 1
        visits += proposal.visits
        raw_invalid += int(proposal.raw_action is not None and proposal.raw_action not in obs.legal)
        if proposal.action == STOP:
            premature += int(not session.state.complete)
            use_repair = repair.mode == "reserve" or (repair.mode == "gate" and repair.gate(proposal))
            if not session.state.complete and use_repair and _repair(session, obs, repair, attempts, "stop"):
                seen.add(session.state)
                continue
            stop_receipt = session.stop()
            exit_reason = "completed" if stop_receipt.accepted else "premature_stop"
            break
        if repair.mode == "early" and not early_considered and len(session.actions) >= 2:
            early_considered = True
            if _repair(session, obs, repair, attempts, "early"):
                seen.add(session.state)
                continue
        if session.remaining <= 1:
            exit_reason = "only_stop_call_remaining"
            break
        if proposal.action not in obs.legal:
            invalid += 1
            exit_reason = "invalid_proposal"
            break
        receipt = session.certify((proposal.action,))
        if not receipt.accepted:
            invalid += 1
            exit_reason = "rejected_primitive"
            break
        seen.add(session.state)
    result = {
        "verified_completion": session.stopped,
        "moves": len(session.actions),
        "actions": session.actions,
        "calls": session.calls,
        "unused_calls": session.remaining,
        "recurrent_visits": visits,
        "decisions": decisions,
        "invalid_proposals": invalid,
        "raw_illegal_top": raw_invalid,
        "premature_stops": premature,
        "macro_attempts": len(attempts),
        "macro_commits": sum(record["accepted"] for record in attempts),
        "completion_macro": session.stopped and any(record["completed"] for record in attempts),
        "stop_repair_completion": session.stopped
        and any(record["trigger"] == "stop" and record["completed"] for record in attempts),
        "exit_reason": exit_reason,
        "macro_trace": attempts,
        "receipts": [asdict(receipt) for receipt in session.receipts],
    }
    return result
