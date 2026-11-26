"""Public-development efficiency frontiers with conservative interval dominance."""
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Bound:
    low: float
    high: float

    def __post_init__(self):
        if not isfinite(self.low) or not isfinite(self.high) or self.low > self.high:
            raise ValueError("invalid bounds")


@dataclass(frozen=True)
class Receipt:
    key: str
    context: tuple[str, ...]  # task, split, initial checkpoint, hardware, estimand, objective schema
    status: str  # measured or estimated
    benefits: tuple[Bound, ...]
    costs: tuple[Bound, ...]  # wall seconds, RAM bytes, VRAM bytes, verifier calls
    competent: bool

    def __post_init__(self):
        if not self.key or len(self.context) != 6 or not all(self.context):
            raise ValueError("incomplete comparison identity")
        if self.status not in ("measured", "estimated"):
            raise ValueError("incomplete resource evidence")
        if not self.benefits or len(self.costs) != 4 or any(c.low < 0 for c in self.costs):
            raise ValueError("invalid resource vector")


def dominates(a: Receipt, b: Receipt) -> bool:
    if (a.context, a.status, len(a.benefits)) != (b.context, b.status, len(b.benefits)):
        raise ValueError("incomparable receipts")
    margins = [x.low - y.high for x, y in zip(a.benefits, b.benefits, strict=True)]
    margins += [y.low - x.high for x, y in zip(a.costs, b.costs, strict=True)]
    return all(m >= 0 for m in margins) and any(m > 0 for m in margins)


def frontier(receipts: Iterable[Receipt], caps: tuple[float, ...]) -> tuple[str, ...]:
    rows = tuple(receipts)
    if len(caps) != 4 or any(not isfinite(c) or c < 0 for c in caps):
        raise ValueError("invalid caps")
    if len({r.key for r in rows}) != len(rows):
        raise ValueError("duplicate receipt identity")
    if len({(r.context, r.status, len(r.benefits)) for r in rows}) > 1:
        raise ValueError("mixed comparison cohort")
    feasible = [r for r in rows if r.competent and all(c.high <= b for c, b in zip(r.costs, caps, strict=True))]
    return tuple(sorted(r.key for r in feasible if not any(dominates(s, r) for s in feasible if s.key != r.key)))


def optimistic_index(gain: Bound, cost: Bound, exploration_weight: float) -> float:
    """Planning heuristic only; cost denominator is a conservative upper bound."""
    if cost.low < 0 or cost.high <= 0 or not isfinite(exploration_weight) or exploration_weight < 0:
        raise ValueError("invalid acquisition inputs")
    return ((gain.low + gain.high) / 2 + exploration_weight * (gain.high - gain.low) / 2) / cost.high
