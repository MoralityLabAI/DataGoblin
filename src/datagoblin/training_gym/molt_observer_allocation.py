"""Development-only MMS observer-map offer allocation.

All forecasts are host-supplied assumptions. The planner never grants a
certificate, runs a model, changes a lattice, or enforces process-tree RAM.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

SCALE = 10**9
MAX_DIMENSION = 64
MAX_OFFERS = 32
MAX_BUDGET = 65536
MODES = ("linear", "conditional_nonlinear")


def nonnegative(value: int) -> bool:
    return type(value) is int and 0 <= value <= 10**18


@dataclass(frozen=True)
class MapOffer:
    binding_digest: str
    dimension: int
    tail_upper_nano: tuple[int, ...]
    radius_upper_nano: int
    hessian_upper_nano: int | None
    rounding_allowance_nano: int
    error_budget_nano: int
    forecast_requests: int
    cold_build_us: int
    hit_us: tuple[int, ...]
    encode_us: tuple[int, ...]

    def validate(self) -> None:
        if not isinstance(self.binding_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", self.binding_digest):
            raise ValueError("immutable binding digest required")
        d = self.dimension
        if type(d) is not int or not 1 <= d <= MAX_DIMENSION:
            raise ValueError("dimension outside development bounds")
        if (len(self.tail_upper_nano) != d + 1 or len(self.hit_us) != d
                or len(self.encode_us) != d):
            raise ValueError("dimension mismatch")
        values = (*self.tail_upper_nano, self.radius_upper_nano, self.rounding_allowance_nano,
                  self.error_budget_nano, self.forecast_requests, self.cold_build_us,
                  *self.hit_us, *self.encode_us)
        if not all(nonnegative(v) for v in values):
            raise ValueError("nonnegative integer forecasts required")
        if self.hessian_upper_nano is not None and not nonnegative(self.hessian_upper_nano):
            raise ValueError("invalid conditional curvature bound")
        if self.tail_upper_nano[-1] != 0 or any(
            a < b for a, b in zip(self.tail_upper_nano, self.tail_upper_nano[1:], strict=False)
        ):
            raise ValueError("tail upper bounds must be nonincreasing and end in zero")
        if not 1 <= self.forecast_requests <= 100000:
            raise ValueError("forecast request count outside bounds")


def map_bytes(d: int, k: int) -> int:
    """MOLT-07 serialization: digest key, header, float64 basis and spectrum."""
    if type(d) is not int or type(k) is not int or not 1 <= k <= d <= MAX_DIMENSION:
        raise ValueError("invalid rank")
    return 32 + 12 + 8 * d * (k + 1)


def amortization(d: int, k: int) -> dict:
    size = map_bytes(d, k)
    savings = 8 * (d - k)
    return {
        "map_bytes": size,
        "coefficient_bytes": 8 * k,
        "ambient_bytes": 8 * d,
        "break_even_vectors": (size + savings - 1) // savings if savings else None,
        "scope": "serialized payload only; shared binding and rank required",
    }


def error_upper_nano(offer: MapOffer, rank: int, mode: str) -> int | None:
    """Conservative integer ceiling conditional on supplied upper bounds.

    Full rank has no mathematical projection error. The caller's rounding
    allowance is still included. Approximate nonlinear use requires a
    uniform Hessian bound on the entire declared intervention ball.
    """
    if mode not in MODES or not 1 <= rank <= offer.dimension:
        raise ValueError("invalid mode or rank")
    if rank == offer.dimension:
        return offer.rounding_allowance_nano
    curvature = 0
    if mode == "conditional_nonlinear":
        if offer.hessian_upper_nano is None:
            return None
        curvature = offer.hessian_upper_nano * offer.radius_upper_nano**2
    linear = offer.tail_upper_nano[rank] * offer.radius_upper_nano * SCALE
    den = SCALE**2
    return (linear + curvature + den - 1) // den + offer.rounding_allowance_nano


@dataclass(frozen=True)
class Choice:
    binding_digest: str
    rank: int
    bytes: int
    error_upper_nano: int
    forecast_saved_us: int
    forecast_hits: int


def options(offer: MapOffer, mode: str) -> tuple[Choice, ...]:
    offer.validate()
    if mode not in MODES:
        raise ValueError("unknown allocation mode")
    choices = []
    for k in range(1, offer.dimension + 1):
        bound = error_upper_nano(offer, k, mode)
        if bound is None or bound > offer.error_budget_nano:
            continue
        hits = offer.forecast_requests - 1
        savings = hits * (offer.cold_build_us - offer.hit_us[k - 1]) - offer.encode_us[k - 1]
        if savings > 0:
            choices.append(Choice(offer.binding_digest, k, map_bytes(offer.dimension, k),
                                  bound, savings, hits))
    return tuple(choices)


def allocate(offers: tuple[MapOffer, ...], cap_bytes: int, mode: str) -> dict:
    """Exact multiple-choice knapsack over positive-value rank alternatives.

    Each chosen map is assumed resident after one cold construction. No
    eviction, stochastic reuse prediction, nonlinear certificate, or real
    workload speedup is inferred from this planning model.
    """
    if type(cap_bytes) is not int or not 0 <= cap_bytes <= MAX_BUDGET or mode not in MODES:
        raise ValueError("invalid allocation budget or mode")
    if not 1 <= len(offers) <= MAX_OFFERS:
        raise ValueError("offer count outside development bounds")
    if len({o.binding_digest for o in offers}) != len(offers):
        raise ValueError("duplicate binding")
    ordered = sorted(offers, key=lambda o: o.binding_digest)
    # Cost -> (forecast utility, selected ranks). Keep lexicographic ties
    # deterministic, then eliminate strictly dominated cost/utility points.
    frontier: dict[int, tuple[int, tuple[Choice, ...]]] = {0: (0, ())}

    def identity(seq: tuple[Choice, ...]) -> tuple[tuple[str, int], ...]:
        return tuple((c.binding_digest, c.rank) for c in seq)

    available = 0
    for offer in ordered:
        candidates = options(offer, mode)
        available += len(candidates)
        next_frontier = dict(frontier)
        for cost, (value, selected) in frontier.items():
            for c in candidates:
                new_cost = cost + c.bytes
                if new_cost > cap_bytes:
                    continue
                entry = (value + c.forecast_saved_us, (*selected, c))
                old = next_frontier.get(new_cost)
                if old is None or entry[0] > old[0] or (
                    entry[0] == old[0] and identity(entry[1]) < identity(old[1])
                ):
                    next_frontier[new_cost] = entry
        best = -1
        frontier = {}
        for cost, entry in sorted(next_frontier.items()):
            if entry[0] > best:
                frontier[cost] = entry
                best = entry[0]
    cost, (value, chosen) = min(
        frontier.items(), key=lambda item: (-item[1][0], item[0], identity(item[1][1]))
    )
    return {
        "schema": "molt07-map-allocation-v1",
        "status": "forecast_only",
        "mode": mode,
        "cap_bytes": cap_bytes,
        "retained_map_bytes": cost,
        "forecast_saved_us": value,
        "forecast_hits": sum(c.forecast_hits for c in chosen),
        "available_rank_options": available,
        "offers": len(ordered),
        "selected": [asdict(c) for c in chosen],
        "offer_ids": [o.binding_digest for o in ordered],
        "conditional_on_host_bounds": True,
        "hard_process_tree_cap_verified": False,
        "certificate_authorized": False,
        "promotion_authorized": False,
    }


def from_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"schema", "offers", "cap_bytes", "mode"}:
        raise ValueError("invalid allocation input")
    if data["schema"] != "molt07-map-offers-v1" or not isinstance(data["offers"], list):
        raise ValueError("invalid schema")
    offers = []
    for raw in data["offers"]:
        if not isinstance(raw, dict):
            raise ValueError("invalid offer")
        row = dict(raw)
        for field in ("tail_upper_nano", "hit_us", "encode_us"):
            if not isinstance(row.get(field), list):
                raise ValueError("numeric array required")
            row[field] = tuple(row[field])
        offers.append(MapOffer(**row))
    return allocate(tuple(offers), data["cap_bytes"], data["mode"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = from_json(args.offers)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "selected"}, sort_keys=True))


if __name__ == "__main__":
    main()
