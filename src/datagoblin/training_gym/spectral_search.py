"""Replayable bounded SG-04 search; student state choices are never proof bounds."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from datagoblin.training_gym.spectral_routing import TaskError, score, validate


@dataclass(frozen=True)
class Node:
    value: int
    tail: tuple[int, ...]
    upper: int


class SearchSession:
    """Exact integer certificate, with complete expansions and no frontier eviction."""

    def __init__(self, task: dict[str, Any], initial: list[int], *, transition_cap: int = 4096):
        self.d, self.t, self.bits, self.x, self.k, self.u, self.lam = validate(task)
        if type(transition_cap) is not int or not 1 <= transition_cap <= 1_000_000:
            raise TaskError("invalid transition cap")
        self.task = {"d": self.d, "T": self.t, "bits": self.bits, "x": self.x,
                     "k": self.k, "u": self.u, "lambda": self.lam}
        self.incumbent = score(self.task, initial)["objective"]
        self.schedule = list(initial)
        self.cap = transition_cap
        self.transitions = 0
        self.trace: list[list[int]] = []
        self.actions = [
            [(sum(1 << j for j in a), sum(row[j] for j in a)) for a in combinations(range(self.d), k)]
            for row, k in zip(self.u, self.k, strict=True)
        ]
        self.weights = [sum(b for j, b in enumerate(self.bits) if mask & (1 << j))
                        for mask in range(1 << self.d)]
        local = [sorted(row, reverse=True) for row in self.u]
        cutoff = [row[k - 1] if k else row[0] + 1 for row, k in zip(local, self.k, strict=True)]
        self.etas = [[0] * self.t, *[[v + shift for v in cutoff] for shift in (-6, -3, 0, 3, 6)]]
        self.weak = [0]
        for row, k in zip(local, self.k, strict=True):
            self.weak.append(self.weak[-1] + sum(row[:k]))
        self.cache: dict[tuple[int, int], int] = {}
        self.live = {(self.t, 0): Node(0, (), self.remaining_bound(self.t, 0))}
        self.seen = {(self.t, 0): 0}
        self._prune()

    def remaining_bound(self, t: int, suffix: int) -> int:
        """Min of six cardinality relaxations and the cost-free top-k bound."""
        if type(t) is not int or type(suffix) is not int or not 0 <= t <= self.t or not 0 <= suffix < 1 << self.d:
            raise TaskError("invalid bound state")
        key = (t, suffix)
        if key in self.cache:
            return self.cache[key]
        bound = self.weak[t]
        for eta in self.etas:
            value = sum(eta[i] * self.k[i] for i in range(t))
            for j, bits in enumerate(self.bits):
                adjusted = [self.u[i][j] - eta[i] for i in range(t)]
                if suffix & (1 << j):
                    value += sum(max(0, v) for v in adjusted)
                else:
                    value += max([0, *[
                        v + sum(max(0, earlier) for earlier in adjusted[:i])
                        - self.lam * bits * (self.x[i + 1] - self.x[0])
                        for i, v in enumerate(adjusted)
                    ]])
            bound = min(bound, value)
        self.cache[key] = bound
        return bound

    def _prune(self) -> None:
        self.live = {key: n for key, n in self.live.items()
                     if n.upper > self.incumbent and n.value == self.seen[key]}

    def frontier(self) -> list[tuple[int, int, int]]:
        """Public controller choices: remaining visits, suffix mask, upper bound."""
        return [(t, s, n.upper) for (t, s), n in sorted(self.live.items())]

    def advance(self, t: int, suffix: int) -> None:
        """Expand all legal actions for a live state, or reject without mutation."""
        if type(t) is not int or type(suffix) is not int or (t, suffix) not in self.live:
            raise TaskError("state is not live")
        actions = self.actions[t - 1]
        if self.transitions + len(actions) > self.cap:
            raise TaskError("transition cap would be exceeded")
        node = self.live.pop((t, suffix))
        self.trace.append([t, suffix])
        self.transitions += len(actions)
        for mask, utility in actions:
            union = suffix | mask
            value = node.value + utility - self.lam * (self.x[t] - self.x[0]) * self.weights[mask & ~suffix]
            tail = (mask, *node.tail)
            if t == 1:
                if value > self.incumbent:
                    if score(self.task, list(tail))["objective"] != value:
                        raise AssertionError("incumbent accounting failed")
                    self.incumbent, self.schedule = value, list(tail)
                continue
            key = (t - 1, union)
            upper = value + self.remaining_bound(*key)
            if upper <= self.incumbent or (key in self.seen and value <= self.seen[key]):
                continue
            self.seen[key] = value
            self.live[key] = Node(value, tail, upper)
        self._prune()

    def receipt(self) -> dict[str, Any]:
        upper = max([self.incumbent, *[n.upper for n in self.live.values()]])
        return {**score(self.task, self.schedule), "schedule": list(self.schedule), "upper": upper,
                "gap": upper - self.incumbent, "transitions": self.transitions,
                "live_states": len(self.live), "trace": [list(a) for a in self.trace]}


def verify_trace(task: dict[str, Any], initial: list[int], trace: list[list[int]], *,
                 transition_cap: int = 4096) -> dict[str, Any]:
    """Rebuild every expanded state and recompute the certificate; ignore student claims."""
    session = SearchSession(task, initial, transition_cap=transition_cap)
    if not isinstance(trace, list) or len(trace) > transition_cap:
        raise TaskError("invalid trace length")
    for choice in trace:
        if not isinstance(choice, list) or len(choice) != 2:
            raise TaskError("invalid trace state")
        session.advance(*choice)
    return session.receipt()


def best_bound(task: dict[str, Any], initial: list[int], *, transition_cap: int = 4096) -> dict[str, Any]:
    """Deterministic bounded baseline; the trace can be independently replayed."""
    session = SearchSession(task, initial, transition_cap=transition_cap)
    while session.live:
        t, s, _ = min(session.frontier(), key=lambda v: (-v[2], v[0], v[1]))
        if session.transitions + len(session.actions[t - 1]) > transition_cap:
            break
        session.advance(t, s)
    return session.receipt()
