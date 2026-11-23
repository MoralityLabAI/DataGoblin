"""Finite-horizon continuation labels and public bounded route search."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass

from .native import STOP, State, World, digest, successors

Graph = dict[State, dict[int, State]]
OBJECTIVES = ("canonical", "hashed", "shortest", "budget", "soft")


def state_graph(
    world: World, transition: Callable[[World, State], dict[int, State]] = successors
) -> tuple[Graph, dict[State, int]]:
    graph = {}
    parents = defaultdict(set)
    queue = deque([world.initial])
    seen = {world.initial}
    while queue:
        state = queue.popleft()
        graph[state] = transition(world, state)
        for after in graph[state].values():
            parents[after].add(state)
            if after not in seen:
                seen.add(after)
                queue.append(after)
    distances = {state: 0 for state in graph if state.complete}
    queue.extend(distances)
    while queue:
        state = queue.popleft()
        for before in parents[state]:
            if before not in distances:
                distances[before] = distances[state] + 1
                queue.append(before)
    return graph, distances


@dataclass(frozen=True)
class Outcome:
    moves: int
    calls: int


def continuations(
    state: State, graph: Graph, distances: dict[State, int], moves: int, calls: int
) -> dict[int, Outcome]:
    """Best successful primitive continuation per first action, including a charged STOP."""
    if state.complete:
        return {STOP: Outcome(0, 1)} if calls else {}
    return {
        action: Outcome(distances[after] + 1, distances[after] + 2)
        for action, after in sorted(graph[state].items())
        if after in distances and distances[after] + 1 <= moves and distances[after] + 2 <= calls
    }


def objective_weights(world: World, state: State, outcomes: dict[int, Outcome]) -> dict[str, dict[int, float]]:
    if not outcomes:
        raise ValueError("Training presentations require a verified feasible continuation")
    best = min(outcome.moves for outcome in outcomes.values())
    shortest = tuple(sorted(a for a, outcome in outcomes.items() if outcome.moves == best))
    seed_hash = int(digest([world.sha256, state.place, state.permit, state.cargo, state.evidence]), 16)
    optimal_units = 2 * best + 1
    return {
        "canonical": {shortest[0]: 1.0},
        "hashed": {shortest[seed_hash % len(shortest)]: 1.0},
        "shortest": dict.fromkeys(shortest, 1.0),
        "budget": dict.fromkeys(outcomes, 1.0),
        "soft": {a: 2.0 ** (-(cost.moves + cost.calls - optimal_units) / 2) for a, cost in outcomes.items()},
    }


def repair_routes(world: World, start: State, horizon: int, limit: int) -> tuple[tuple[int, ...], ...]:
    """Breadth-first public search, keeping up to one shortest route per first action."""
    if start.complete or horizon <= 0 or limit <= 0:
        return ()
    queue = deque([(start, ())])
    seen = set()
    results = {}
    while queue and len(results) < limit:
        state, route = queue.popleft()
        if route and route[0] in results:
            continue
        if state.complete:
            results[route[0]] = route
        elif len(route) < horizon:
            for action, after in sorted(successors(world, state).items()):
                first = route[0] if route else action
                key = first, after
                if key not in seen:
                    seen.add(key)
                    queue.append((after, (*route, action)))
    return tuple(results.values())
