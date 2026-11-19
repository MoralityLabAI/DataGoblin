"""Reuse an environment-verified graph route across a typed patch."""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .rlt_revision import follow


@dataclass(frozen=True)
class RouteWitness:
    nodes: tuple[str, ...]
    first_exit: Mapping[str, int]


def observe_route(rules: Mapping[str, str], start: str, steps: int) -> RouteWitness:
    """Record the exact old route in the environment, including repeated visits."""
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    nodes = [start]
    first_exit = {}
    for visit in range(steps):
        source = nodes[-1]
        first_exit.setdefault(source, visit)
        nodes.append(rules[source])
    return RouteWitness(tuple(nodes), MappingProxyType(first_exit))


def patched_endpoint(rules: Mapping[str, str], witness: RouteWitness, edits: Mapping[str, str]) -> tuple[str, int]:
    """Report the new endpoint and traversals after the earliest affected edge.

    The verifier owns the old rules and witness. Edits represent the final patch
    relative to those rules, with all updates to the same source resolved.
    """
    steps = len(witness.nodes) - 1
    first = min(
        (witness.first_exit.get(source, steps) for source, target in edits.items() if rules.get(source) != target),
        default=steps,
    )
    remaining = steps - first
    if not remaining:
        return witness.nodes[-1], 0
    return follow(ChainMap(edits, rules), witness.nodes[first], remaining), remaining
