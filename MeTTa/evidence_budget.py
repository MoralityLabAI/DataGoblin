"""Public evidence-check requests under a host-owned verification budget."""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from math import prod

from .ram_hillclimb import digest

BUDGET = 4
WIDTH = 7
ROUTES = {
    "fork": ((0, 1), (0, 2), (3, 4), (5, 6)),
    "chain": ((0, 1, 2), (0, 1, 3), (3, 4), (5, 6)),
    "bottleneck": ((0, 1, 2), (0, 3, 4), (0, 5, 6)),
    "overlap_transfer": ((0, 1, 2), (0, 3), (2, 4, 5), (1, 4, 6)),
}
UNCHECKED, CURRENT, REJECTED = 0, 1, -1
State = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Catalog:
    routes: tuple[tuple[int, ...], ...]
    priors: tuple[int, ...]

    @property
    def source_root(self) -> str:
        return digest((self.routes, self.priors))

    def available(self, state: State, remaining: int) -> tuple[tuple[int, ...], ...]:
        return tuple(route for route in self.routes if all(state[i] != REJECTED for i in route)
                     and sum(state[i] == UNCHECKED for i in route) <= remaining)


@dataclass(frozen=True, slots=True)
class PublicObservation:
    catalog: Catalog
    state: State
    remaining: int
    legal: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Episode:
    catalog: Catalog
    current: tuple[bool, ...]


def make_episode(family: str, split: str, seed: int) -> Episode:
    rng = random.Random(f"metta-check-budget-v1/{family}/{split}/{seed}")
    permutation = rng.sample(range(WIDTH), WIDTH)
    routes = tuple(sorted(tuple(sorted(permutation[i] for i in route)) for route in ROUTES[family]))
    priors = tuple(rng.randrange(18, 93) for _ in range(WIDTH))
    current = tuple(rng.randrange(100) < prior for prior in priors)
    return Episode(Catalog(routes, priors), current)


def written(state: State, slot: int, observation: int) -> State:
    return (*state[:slot], observation, *state[slot + 1:])


def witness(catalog: Catalog, state: State) -> tuple[int, ...] | None:
    return next((route for route in catalog.routes if all(state[i] == CURRENT for i in route)), None)


def observe(catalog: Catalog, state: State, remaining: int) -> PublicObservation:
    viable = catalog.available(state, remaining)
    legal = tuple(sorted({i for route in viable for i in route if state[i] == UNCHECKED}))
    return PublicObservation(catalog, state, remaining, legal)


def check_run(episode: Episode, policy: Callable[[PublicObservation], int], budget: int = BUDGET) -> dict:
    if type(budget) is not int or not 0 <= budget <= BUDGET:
        raise ValueError("Verification budget outside the registered bounds")
    catalog = episode.catalog
    state = (UNCHECKED,) * len(catalog.priors)
    receipts = []
    for remaining in range(budget, 0, -1):
        public = observe(catalog, state, remaining)
        if witness(catalog, state) is not None or not public.legal:
            break
        slot = policy(public)
        if type(slot) is not int or slot not in public.legal:
            raise ValueError("Policy proposed an unavailable verification request")
        outcome = CURRENT if episode.current[slot] else REJECTED
        receipts.append({"slot": slot, "outcome": outcome})
        state = written(state, slot, outcome)
    proof = witness(catalog, state)
    return {"source_root_sha256": catalog.source_root, "checks": len(receipts), "receipts": receipts,
            "final_state": state, "accepted_route": proof, "completed": proof is not None}


def greedy_check(public: PublicObservation) -> int:
    viable = public.catalog.available(public.state, public.remaining)
    route = max(viable, key=lambda path: prod(public.catalog.priors[i] / 100
                                             for i in path if public.state[i] == UNCHECKED))
    return min((i for i in route if public.state[i] == UNCHECKED), key=lambda i: (public.catalog.priors[i], i))


def bayes_check(catalog: Catalog, *, minimize_work: bool = False) -> Callable[[PublicObservation], int]:
    """Exact finite-horizon controller using public independent priors only."""
    def score(expectation: tuple[float, float]) -> tuple[float, float]:
        success, calls = expectation
        return (round(success, 12), -round(calls, 12)) if minimize_work else (success, 0.0)

    @cache
    def value(state: State, remaining: int) -> tuple[float, float]:
        if witness(catalog, state) is not None:
            return 1.0, 0.0
        if not remaining:
            return 0.0, 0.0
        choices = observe(catalog, state, remaining).legal
        return max((expected(state, remaining, slot) for slot in choices), key=score, default=(0.0, 0.0))

    def expected(state: State, remaining: int, slot: int) -> tuple[float, float]:
        p = catalog.priors[slot] / 100
        yes, yes_calls = value(written(state, slot, CURRENT), remaining - 1)
        no, no_calls = value(written(state, slot, REJECTED), remaining - 1)
        return p * yes + (1 - p) * no, 1 + p * yes_calls + (1 - p) * no_calls

    def choose(public: PublicObservation) -> int:
        def priority(slot: int) -> tuple[float, float, int]:
            success, calls = score(expected(public.state, public.remaining, slot))
            return round(success, 12), calls, -slot

        return max(public.legal, key=priority)

    return choose
