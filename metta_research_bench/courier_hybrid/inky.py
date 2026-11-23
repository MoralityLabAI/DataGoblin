"""Render the published Inky curriculum and verify bounded offline replies."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

from MeTTa.inky_skills import skill_names, system_prompt

from .controller import Observation
from .native import ACTION_NAMES, STOP, Receipt, digest, successors
from .oracle import repair_routes

TASKS = {
    "metta-compare-policies": "choose_action",
    "metta-curate-sources": "explain_rejection",
    "metta-evidence-dockets": "explain_rejection",
    "metta-repair-actions": "propose_routes",
    "metta-storyworld-routes": "propose_routes",
    "metta-verify-rules": "explain_rejection",
}
FORMATS = {
    "choose_action": 'Return {"action":"<one allowed action>"} as JSON.',
    "explain_rejection": (
        'Return {"reason":"<one code>"} as JSON. Codes: permit_required, pickup_required, '
        "missing_independent_evidence, already_attested, no_route, or undelivered_stop."
    ),
    "propose_routes": 'Return {"routes":[["<typed action>", "..."], ["..."]]} as JSON in candidate order.',
}


@dataclass(frozen=True)
class Reply:
    model: str
    request_sha256: str
    content: str
    inference_seconds: float
    input_tokens: int
    output_tokens: int


def request_for(
    obs: Observation, skill: str, model: str, *, depth: int = 0, limit: int = 0, rejection: Receipt | None = None
) -> dict:
    if set(TASKS) != set(skill_names()) or skill not in TASKS or not model.strip():
        raise ValueError("Use the six published curriculum skills and an explicit model identity")
    task = TASKS[skill]
    if (task == "explain_rejection") != (rejection is not None) or (rejection is not None and rejection.accepted):
        raise ValueError("Rejection tasks require a failed native receipt")
    task_state = {
        "task": task,
        "world": asdict(obs.world),
        "current_state": asdict(obs.state),
        "legal_first_actions": [ACTION_NAMES[a] for a in obs.legal],
        "moves_left": obs.moves_left,
        "verifier_calls_left": obs.remaining_calls,
        "proof_obligations": "Deliver only while carrying cargo at the desk with attestations from both identities. "
        "First may have two outlets. A second outlet for First cannot attest Second.",
    }
    if task == "propose_routes":
        task_state.update(max_route_length=min(depth, obs.moves_left), max_routes=max(0, limit))
    if rejection is not None:
        task_state["native_rejection"] = {**asdict(rejection), "actions": [ACTION_NAMES[a] for a in rejection.actions]}
    request = {
        "model": model,
        "temperature": 0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": system_prompt(skill)},
            {"role": "user", "content": FORMATS[task] + "\n" + json.dumps(task_state, sort_keys=True)},
        ],
    }
    return {"skill": skill, "request_sha256": digest(request), "request": request}


def parse_reply(request: dict, reply: Reply, *, depth: int = 0, limit: int = 0):
    if (
        reply.model != request["request"]["model"]
        or reply.request_sha256 != request["request_sha256"]
        or type(reply.input_tokens) is not int
        or type(reply.output_tokens) is not int
        or min(reply.input_tokens, reply.output_tokens) < 0
        or reply.output_tokens > 256
        or not math.isfinite(reply.inference_seconds)
        or reply.inference_seconds < 0
    ):
        raise ValueError("Missing or mismatched Inky inference provenance")
    if len(reply.content.encode()) > 4096:
        return None
    try:
        payload = json.loads(reply.content)
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict):
        return None
    task = TASKS[request["skill"]]
    if task == "choose_action":
        value = payload.get("action")
        return ACTION_NAMES.index(value) if isinstance(value, str) and value in ACTION_NAMES else None
    if task == "explain_rejection":
        value = payload.get("reason")
        return value if isinstance(value, str) else None
    entries = payload.get("routes")
    if not isinstance(entries, list) or len(entries) > limit:
        return ()
    checked = []
    for route in entries:
        if (
            isinstance(route, list)
            and 0 < len(route) <= depth
            and all(isinstance(item, str) and item in ACTION_NAMES[:STOP] for item in route)
        ):
            checked.append(tuple(ACTION_NAMES.index(item) for item in route))
    return tuple(dict.fromkeys(checked))


class InkyAdapter:
    """A bounded repair proposer. Model claims can never authorize state changes."""

    def __init__(self, model: str, respond: Callable[[dict], Reply], skill: str = "metta-repair-actions"):
        if TASKS.get(skill) != "propose_routes":
            raise ValueError("An Inky repair provider requires a route skill")
        self.model, self.respond, self.skill = model, respond, skill
        self.inferences: list[dict] = []

    def __call__(self, obs: Observation, depth: int, limit: int) -> tuple[tuple[int, ...], ...]:
        request = request_for(obs, self.skill, self.model, depth=depth, limit=limit)
        reply = self.respond(request)
        routes = parse_reply(request, reply, depth=depth, limit=limit)
        self.inferences.append(
            {
                "skill": self.skill,
                "model": reply.model,
                "request_sha256": reply.request_sha256,
                "response_sha256": digest(reply.content),
                "inference_seconds": reply.inference_seconds,
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "typed_candidates": len(routes or ()),
            }
        )
        return routes or ()


def score_answer(
    request: dict, reply: Reply, *, optimal_actions: tuple[int, ...] = (), rejection_reason: str | None = None
) -> bool:
    answer = parse_reply(request, reply)
    task = TASKS[request["skill"]]
    if task == "choose_action" and optimal_actions:
        return answer in optimal_actions
    if task == "explain_rejection" and rejection_reason:
        return answer == rejection_reason
    raise ValueError("Advice scoring requires independent labels from the audit fixture")


def looped_routes(model, obs: Observation, depth: int, limit: int) -> tuple[tuple[int, ...], ...]:
    """Use the frozen pointer to propose one bounded continuation without native authority."""
    if not limit:
        return ()
    world, state = obs.world, obs.state
    visited = {state}
    actions = []
    for step in range(min(depth, obs.moves_left)):
        choices = {a: s for a, s in successors(world, state).items() if s not in visited}
        local = Observation(world, state, (*sorted(choices), STOP), obs.moves_left - step, obs.remaining_calls)
        action = model.propose(local).action
        if action == STOP or action not in choices:
            break
        actions.append(action)
        state = choices[action]
        visited.add(state)
        if state.complete:
            return (tuple(actions),)
    return (tuple(actions),) if actions else ()


def public_routes(obs: Observation, depth: int, limit: int) -> tuple[tuple[int, ...], ...]:
    return repair_routes(obs.world, obs.state, min(depth, obs.moves_left), limit)
