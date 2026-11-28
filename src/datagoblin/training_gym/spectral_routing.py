"""Independent integer grader for offline SG-03 source-bound routing tasks."""
from __future__ import annotations

from itertools import combinations
from typing import Any


class TaskError(ValueError):
    """Malformed task, invalid action, or exhausted verification budget."""


def _integer(value: Any, lo: int, hi: int) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise TaskError("integer outside registered bounds")
    return value


def _vector(value: Any, size: int, lo: int, hi: int) -> list[int]:
    if not isinstance(value, list) or len(value) != size:
        raise TaskError("invalid vector shape")
    return [_integer(v, lo, hi) for v in value]


def validate(task: dict[str, Any]) -> tuple[int, int, list[int], list[int], list[int], list[list[int]], int]:
    """Accept the public SG-03 task schema, with explicit verifier resource limits."""
    try:
        d, t = _integer(task["d"], 1, 10), _integer(task["T"], 1, 10)
        bits = _vector(task["bits"], d, 1, 100)
        x = _vector(task["x"], t + 1, -1000, 1000)
        k = _vector(task["k"], t, 0, d)
        lam = _integer(task["lambda"], 0, 100)
        u = task["u"]
        if not isinstance(u, list) or len(u) != t or any(a > b for a, b in zip(x, x[1:], strict=False)):
            raise TaskError("invalid utility shape or unsorted coordinates")
        return d, t, bits, x, k, [_vector(row, d, -10000, 10000) for row in u], lam
    except (KeyError, TypeError) as exc:
        raise TaskError("malformed task") from exc


def score(task: dict[str, Any], schedule: list[int]) -> dict[str, int]:
    """Score a hard read schedule directly by last consumer, not by the DP recurrence."""
    d, t, bits, x, k, u, lam = validate(task)
    masks = _vector(schedule, t, 0, (1 << d) - 1)
    if any(mask.bit_count() != k[i] for i, mask in enumerate(masks)):
        raise TaskError("per-visit read budget violated")
    last = [x[0]] * d
    utility = 0
    for i, mask in enumerate(masks):
        for j in range(d):
            if mask & (1 << j):
                last[j] = x[i + 1]
                utility += u[i][j]
    cost = sum(b * (v - x[0]) for b, v in zip(bits, last, strict=True))
    return {"utility": utility, "cost": cost, "objective": utility - lam * cost}


def oracle(task: dict[str, Any], *, transition_cap: int = 1_000_000) -> dict[str, Any]:
    """Exact gap-union DP, independently formulated from the upstream last-use DP."""
    d, t, bits, x, k, u, lam = validate(task)
    _integer(transition_cap, 1, 10_000_000)
    weights = [sum(bits[j] for j in range(d) if mask & (1 << j)) for mask in range(1 << d)]
    states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    transitions = 0
    for i in range(t - 1, -1, -1):
        choices = [sum(1 << j for j in a) for a in combinations(range(d), k[i])]
        utilities = {mask: sum(u[i][j] for j in range(d) if mask & (1 << j)) for mask in choices}
        next_states: dict[int, tuple[int, tuple[int, ...]]] = {}
        for suffix, (value, tail) in states.items():
            for mask in choices:
                transitions += 1
                if transitions > transition_cap:
                    raise TaskError("verification transition cap exceeded")
                union = suffix | mask
                proposed = (
                    value + utilities[mask] - lam * (x[i + 1] - x[i]) * weights[union],
                    (mask, *tail),
                )
                old = next_states.get(union)
                if old is None or proposed[0] > old[0] or (proposed[0] == old[0] and proposed[1] < old[1]):
                    next_states[union] = proposed
        states = next_states
    value, schedule = min(states.values(), key=lambda pair: (-pair[0], pair[1]))
    result = score(task, list(schedule))
    if result["objective"] != value:
        raise AssertionError("independent objective accounting failed")
    return {**result, "schedule": list(schedule), "transitions": transitions}


def grade(task: dict[str, Any], schedule: list[int], *, transition_cap: int = 1_000_000) -> dict[str, Any]:
    """Optimality requires independent exact verification; ties are accepted."""
    candidate = score(task, schedule)
    optimum = oracle(task, transition_cap=transition_cap)
    regret = optimum["objective"] - candidate["objective"]
    if regret < 0:
        raise AssertionError("candidate exceeds exact upper bound")
    return {**candidate, "optimal_objective": optimum["objective"], "regret": regret, "optimal": regret == 0}


def training_example(task: dict[str, Any]) -> dict[str, Any]:
    """Create an offline teacher record from an explicitly public development task."""
    validate(task)
    public = {name: task[name] for name in ("d", "T", "bits", "x", "k", "u", "lambda")}
    teacher = oracle(public)
    return {"schema": "sg03-routing/v1", "task": public, "teacher": teacher, "evidence": "exact finite objective"}
