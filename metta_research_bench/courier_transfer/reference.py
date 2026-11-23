"""Finite deterministic CPU oracle for DG-METTA-12 adapter wiring.

This independently implemented Python reference is not the native Storyworld
MeTTa kernel. Its outputs cannot count as native domain-transfer results.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

STOP = "STOP"


@dataclass(frozen=True)
class CourierWorld:
    start: str
    # Undirected edges: (node_a, node_b, required_permit), "" if unrestricted.
    edges: tuple[tuple[str, str, str], ...]
    # (permit_id, office_node)
    offices: tuple[tuple[str, str], ...]
    # (parcel_id, pickup_node, destination_node, required_delivery_permit)
    parcels: tuple[tuple[str, str, str, str], ...]
    # (witness_alias, witness_node, independent_source_id)
    witnesses: tuple[tuple[str, str, str], ...] = ()
    # (permit_id, tuple of required independent source ids)
    permit_sources: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not self.start:
            raise ValueError("missing starting node")
        for ids in (
            [p for p, _ in self.offices],
            [p for p, *_ in self.parcels],
            [w for w, *_ in self.witnesses],
            [p for p, _ in self.permit_sources],
        ):
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate typed IDs")
        if any(a == b for a, b, _ in self.edges):
            raise ValueError("self-loop edges are unsupported by the reference")


@dataclass(frozen=True)
class CourierState:
    at: str
    permits: frozenset[str] = frozenset()
    observed_sources: frozenset[str] = frozenset()
    carrying: frozenset[str] = frozenset()
    delivered: frozenset[str] = frozenset()


def initial(world: CourierWorld) -> CourierState:
    return CourierState(at=world.start)


def legal_actions(world: CourierWorld, state: CourierState) -> tuple[str, ...]:
    actions = {STOP}
    for a, b, permit in world.edges:
        if not permit or permit in state.permits:
            if state.at == a:
                actions.add(f"move:{b}")
            elif state.at == b:
                actions.add(f"move:{a}")
    requirements = dict(world.permit_sources)
    for permit, office in world.offices:
        if (office == state.at and permit not in state.permits
                and set(requirements.get(permit, ())).issubset(state.observed_sources)):
            actions.add(f"permit:{permit}")
    for alias, node, source in world.witnesses:
        if node == state.at and source not in state.observed_sources:
            actions.add(f"inspect:{alias}")
    for parcel, pickup, destination, required in world.parcels:
        if (parcel not in state.carrying and parcel not in state.delivered
                and state.at == pickup):
            actions.add(f"pickup:{parcel}")
        if (parcel in state.carrying and state.at == destination
                and (not required or required in state.permits)):
            actions.add(f"deliver:{parcel}")
    return tuple(sorted(actions))


def transition(world: CourierWorld, state: CourierState, action: str) -> CourierState:
    if action == STOP or action not in legal_actions(world, state):
        raise ValueError(f"invalid state-changing action: {action}")
    kind, identifier = action.split(":", 1)
    if kind == "move":
        return replace(state, at=identifier)
    if kind == "permit":
        return replace(state, permits=state.permits | {identifier})
    if kind == "inspect":
        source = next(src for alias, _, src in world.witnesses if alias == identifier)
        return replace(state, observed_sources=state.observed_sources | {source})
    if kind == "pickup":
        return replace(state, carrying=state.carrying | {identifier})
    if kind == "deliver":
        return replace(state, carrying=state.carrying - {identifier},
                       delivered=state.delivered | {identifier})
    raise AssertionError(action)


def success(world: CourierWorld, state: CourierState, stopped: bool) -> bool:
    return stopped and state.delivered == frozenset(p for p, *_ in world.parcels)


class FiniteOracle:
    """Lexicographic exact optimum: (delivered count, negative action count).

    STOP scores the current state. Every accepted action costs one step.
    Different shortest successful continuations all receive supervision.
    Unsolvable states are excluded from the successful-action training pool.
    """

    def __init__(self, world: CourierWorld) -> None:
        self.world = world
        self._memo: dict[tuple[CourierState, int], tuple[int, int]] = {}

    def value(self, state: CourierState, remaining: int) -> tuple[int, int]:
        if remaining < 0:
            raise ValueError("remaining must be nonnegative")
        key = (state, remaining)
        if key in self._memo:
            return self._memo[key]
        best = (len(state.delivered), 0)
        if remaining:
            for action in legal_actions(self.world, state):
                if action != STOP:
                    nxt = self.value(transition(self.world, state, action), remaining - 1)
                    best = max(best, (nxt[0], nxt[1] - 1))
        self._memo[key] = best
        return best

    def action_set(self, state: CourierState, remaining: int) -> tuple[str, ...]:
        best = self.value(state, remaining)
        choices = []
        for action in legal_actions(self.world, state):
            if action == STOP:
                score = (len(state.delivered), 0)
            elif remaining:
                nxt = self.value(transition(self.world, state, action), remaining - 1)
                score = (nxt[0], nxt[1] - 1)
            else:
                continue
            if score == best:
                choices.append(action)
        return tuple(choices)

    def can_finish(self, state: CourierState, remaining: int) -> bool:
        return self.value(state, remaining)[0] == len(self.world.parcels)


def serialize_state(state: CourierState) -> str:
    """Canonical across Python hash seeds and frozenset iteration orders."""
    return json.dumps({
        "at": state.at,
        "permits": sorted(state.permits),
        "observed_sources": sorted(state.observed_sources),
        "carrying": sorted(state.carrying),
        "delivered": sorted(state.delivered),
    }, sort_keys=True, separators=(",", ":"))


def hashed_singleton(
    root_id: str, state: CourierState, actions: tuple[str, ...], seed: int
) -> str:
    """Stable action-tie control independent of Python hash randomization."""
    if not actions or len(set(actions)) != len(actions):
        raise ValueError("expected nonempty unique action set")
    ordered = sorted(actions)
    key = "|".join((root_id, str(seed), serialize_state(state), *ordered)).encode()
    j = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(ordered)
    return ordered[j]


def set_nll(logits, allowed_indices: tuple[int, ...]):
    """PyTorch action-set NLL with full legal-action normalization."""
    import torch
    if not allowed_indices:
        raise ValueError("empty optimal action set")
    indices = torch.tensor(sorted(set(allowed_indices)), device=logits.device)
    if (logits.ndim != 1 or indices.min().item() < 0
            or indices.max().item() >= len(logits)):
        raise ValueError("indices must select valid logits")
    return torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[indices], dim=0)
