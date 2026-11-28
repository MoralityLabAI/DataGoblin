"""Exact integer multiplier proposals for the replayable spectral routing verifier."""
from __future__ import annotations

from typing import Any

from datagoblin.training_gym.spectral_routing import TaskError, validate
from datagoblin.training_gym.spectral_search import Node, SearchSession

MAX_MULTIPLIER = 100_000_000


def _checked(task: dict[str, Any], t: int, suffix: int, eta: list[int]) -> None:
    d, visits, *_ = validate(task)
    if type(t) is not int or type(suffix) is not int or not 0 <= t <= visits or not 0 <= suffix < 1 << d:
        raise TaskError("invalid dual state")
    if not isinstance(eta, list) or len(eta) != t:
        raise TaskError("invalid multiplier shape")
    if any(type(v) is not int or abs(v) > MAX_MULTIPLIER for v in eta):
        raise TaskError("invalid integer multiplier")


def dual_bound(task: dict[str, Any], t: int, suffix: int, eta: list[int]) -> int:
    """Admissible remaining objective bound, evaluated independently of the student."""
    _checked(task, t, suffix, eta)
    d, _, bits, x, k, u, lam = validate(task)
    value = sum(eta[i] * k[i] for i in range(t))
    for j in range(d):
        prefix = best = 0
        for i in range(t):
            adjusted = u[i][j] - eta[i]
            best = max(best, prefix + adjusted - lam * bits[j] * (x[i + 1] - x[0]))
            prefix += max(0, adjusted)
        value += prefix if suffix & (1 << j) else best
    return value


def coordinate_minimizer(task: dict[str, Any], t: int, suffix: int, eta: list[int], i: int) -> int:
    """Exact one-coordinate minimizer over integers, keeping all other coordinates fixed."""
    _checked(task, t, suffix, eta)
    if type(i) is not int or not 0 <= i < t:
        raise TaskError("invalid coordinate")
    d, _, bits, x, k, u, lam = validate(task)
    thresholds = []
    for j in range(d):
        if suffix & (1 << j):
            thresholds.append(u[i][j])
            continue
        prefix, exclude = 0, 0
        include = u[i][j] + sum(max(0, u[l][j] - eta[l]) for l in range(i))
        include -= lam * bits[j] * (x[i + 1] - x[0])
        for l in range(t):
            if l == i:
                continue
            adjusted = u[l][j] - eta[l]
            base = prefix + adjusted - lam * bits[j] * (x[l + 1] - x[0])
            exclude = max(exclude, base)
            if l > i:
                include = max(include, base + u[i][j])
            prefix += max(0, adjusted)
        thresholds.append(include - exclude)
    thresholds.sort(reverse=True)
    if k[i] == 0:
        return thresholds[0]
    if k[i] == d:
        return thresholds[-1]
    return (thresholds[k[i] - 1] + thresholds[k[i]]) // 2


class DualSearchSession(SearchSession):
    """A student can propose multipliers for live states; only exact bounds are used."""

    def __init__(self, task: dict[str, Any], initial: list[int], *,
                 transition_cap: int = 4096, proposal_cap: int = 4096):
        if type(proposal_cap) is not int or not 0 <= proposal_cap <= 100_000:
            raise TaskError("invalid proposal cap")
        self.proposal_cap = proposal_cap
        self.proposals = 0
        self.events: list[dict[str, Any]] = []
        super().__init__(task, initial, transition_cap=transition_cap)

    def advance(self, t: int, suffix: int) -> None:
        super().advance(t, suffix)
        self.events.append({"kind": "expand", "state": [t, suffix]})

    def propose(self, t: int, suffix: int, eta: list[int]) -> None:
        """Validate before mutation; a valid but weak proposal consumes one slot."""
        if type(t) is not int or type(suffix) is not int or (t, suffix) not in self.live:
            raise TaskError("state is not live")
        if self.proposals >= self.proposal_cap:
            raise TaskError("proposal cap exceeded")
        value = dual_bound(self.task, t, suffix, eta)
        key = (t, suffix)
        n = self.live[key]
        bound = min(self.remaining_bound(t, suffix), value)
        self.cache[key] = bound
        self.live[key] = Node(n.value, n.tail, n.value + bound)
        self.proposals += 1
        self.events.append({"kind": "dual", "state": [t, suffix], "eta": list(eta)})
        self._prune()

    def receipt(self) -> dict[str, Any]:
        events = [{**e, "state": list(e["state"]), **({"eta": list(e["eta"])} if "eta" in e else {})}
                  for e in self.events]
        return {**super().receipt(), "proposals": self.proposals, "events": events}


def verify_dual_trace(task: dict[str, Any], initial: list[int], events: list[dict[str, Any]], *,
                      transition_cap: int = 4096, proposal_cap: int = 4096) -> dict[str, Any]:
    """Replay all proposals and complete expansions; never trust claimed gap or values."""
    session = DualSearchSession(task, initial, transition_cap=transition_cap, proposal_cap=proposal_cap)
    if not isinstance(events, list) or len(events) > transition_cap + proposal_cap:
        raise TaskError("invalid event count")
    for e in events:
        if not isinstance(e, dict) or not isinstance(e.get("state"), list) or len(e["state"]) != 2:
            raise TaskError("invalid event state")
        if e.get("kind") == "expand" and set(e) == {"kind", "state"}:
            session.advance(*e["state"])
        elif e.get("kind") == "dual" and set(e) == {"kind", "state", "eta"}:
            session.propose(*e["state"], e["eta"])
        else:
            raise TaskError("invalid event")
    return session.receipt()
