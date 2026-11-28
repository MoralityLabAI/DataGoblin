"""Transparent CPU reference for ZK-DG-01; never a cryptographic proof."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

Vector = tuple[int, ...]
BACKEND = "transparent-replay-v1"


@dataclass(frozen=True)
class Job:
    name: str
    parents: tuple[str, ...]
    source: Vector
    weights: tuple[Vector, ...]
    bias: Vector
    steps: int


@dataclass(frozen=True)
class Plan:
    tenant: str
    run: str
    policy: str
    jobs: tuple[Job, ...]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(domain: str, value: Any) -> str:
    return hashlib.sha256(b"zk-dg-01/v1/" + domain.encode("ascii") + b"\0" + canonical(value)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identifier(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is not None


def bounded_vector(value: Vector, width: int) -> bool:
    return len(value) == width and all(type(v) is int and -127 <= v <= 127 for v in value)


def validate(plan: Plan) -> int:
    require(all(identifier(x) for x in (plan.tenant, plan.run, plan.policy)), "invalid context")
    require(1 <= len(plan.jobs) <= 128, "invalid graph size")
    width = len(plan.jobs[0].source)
    require(1 <= width <= 16, "invalid width")
    seen: set[str] = set()
    consumed: set[str] = set()
    for job in plan.jobs:
        require(identifier(job.name) and job.name not in seen, "duplicate or invalid job")
        require(len(set(job.parents)) == len(job.parents), "duplicate edge")
        require(set(job.parents) <= seen, "missing or forward parent")
        require(type(job.steps) is int and 1 <= job.steps <= 64, "invalid depth")
        require(bounded_vector(job.source, width) and bounded_vector(job.bias, width), "invalid vector")
        require(len(job.weights) == width and all(bounded_vector(r, width) for r in job.weights), "invalid matrix")
        seen.add(job.name)
        consumed.update(job.parents)
    require(seen - consumed == {plan.jobs[-1].name}, "graph must have one terminal")
    return width


def clamp(x: int) -> int:
    return max(-127, min(127, x))


def step(job: Job, x: Vector, h: Vector) -> Vector:
    return tuple(
        clamp(sum(w * v for w, v in zip(row, h, strict=True)) + b + z)
        for row, b, z in zip(job.weights, job.bias, x, strict=True)
    )


def execute(plan: Plan) -> list[dict[str, Any]]:
    width = validate(plan)
    context = digest("plan", asdict(plan))
    prior: dict[str, tuple[Vector, str]] = {}
    receipts: list[dict[str, Any]] = []
    for job in plan.jobs:
        parents = [prior[p] for p in job.parents]
        x = tuple(clamp(job.source[i] + sum(v[i] for v, _ in parents)) for i in range(width))
        trace = [x]
        for _ in range(job.steps):
            trace.append(step(job, x, trace[-1]))
        statement = {
            "backend": BACKEND,
            "context": context,
            "job": job.name,
            "parents": [r for _, r in parents],
            "model": digest("model", [job.weights, job.bias]),
            "input": digest("vector", x),
            "output": digest("vector", trace[-1]),
            "steps": job.steps,
        }
        receipt = {"statement": statement, "trace": trace}
        receipts.append(receipt)
        prior[job.name] = (trace[-1], digest("receipt", receipt))
    return receipts


def verify(plan: Plan, receipts: list[dict[str, Any]]) -> str:
    """Replay an independently authorized plan, then return its terminal receipt hash."""
    expected = execute(plan)
    require(len(receipts) == len(expected), "incomplete or extended graph")
    for want, got in zip(expected, receipts, strict=True):
        require(canonical(got) == canonical(want), "receipt mismatch")
    return digest("receipt", expected[-1])
