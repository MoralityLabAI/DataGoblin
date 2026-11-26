from dataclasses import replace
from itertools import product

import pytest

from datagoblin.training_gym.molt_observer_allocation import (
    SCALE,
    MapOffer,
    allocate,
    amortization,
    error_upper_nano,
    map_bytes,
    options,
)


def offer(i: int, requests: int = 10) -> MapOffer:
    return MapOffer(f"{i:064x}", 4, (4 * SCALE, 2 * SCALE, SCALE, 0, 0),
                    SCALE // 2, None, 1, SCALE // 2 + 1, requests, 100,
                    (2, 3, 4, 5), (10, 10, 10, 10))


def test_payload_and_amortization():
    assert map_bytes(16, 5) == 812
    assert amortization(16, 5)["break_even_vectors"] == 10
    assert amortization(16, 16)["break_even_vectors"] is None


def test_conditional_gate_and_integer_bound():
    o = offer(1)
    assert error_upper_nano(o, 2, "linear") == SCALE // 2 + 1
    assert error_upper_nano(o, 2, "conditional_nonlinear") is None
    assert [c.rank for c in options(o, "conditional_nonlinear")] == [4]
    with_curvature = replace(o, hessian_upper_nano=SCALE)
    assert error_upper_nano(with_curvature, 3, "conditional_nonlinear") == SCALE // 4 + 1
    assert [c.rank for c in options(with_curvature, "conditional_nonlinear")] == [3, 4]
    assert not allocate((o,), 1024, "linear")["promotion_authorized"]


def test_multiple_choice_against_exhaustive_search():
    offers = tuple(offer(i, requests=i + 3) for i in range(1, 6))
    for cap in (0, 139, 140, 200, 500, 1000):
        result = allocate(offers, cap, "linear")
        candidates = []
        for combination in product(*[(None, *options(o, "linear")) for o in offers]):
            chosen = tuple(c for c in combination if c is not None)
            cost = sum(c.bytes for c in chosen)
            if cost <= cap:
                candidates.append((-sum(c.forecast_saved_us for c in chosen), cost,
                                   tuple((c.binding_digest, c.rank) for c in chosen)))
        expected = min(candidates)
        actual = (-result["forecast_saved_us"], result["retained_map_bytes"],
                  tuple((c["binding_digest"], c["rank"]) for c in result["selected"]))
        assert actual == expected
        assert len({c["binding_digest"] for c in result["selected"]}) == len(result["selected"])


def test_reject_invalid_and_duplicate():
    o = offer(1)
    with pytest.raises(ValueError):
        allocate((o, o), 1024, "linear")
    with pytest.raises(ValueError):
        options(replace(o, tail_upper_nano=(1, 2, 3, 4, 0)), "linear")
    with pytest.raises(ValueError):
        allocate((o,), 65537, "linear")
    assert not options(replace(o, forecast_requests=1), "linear")
