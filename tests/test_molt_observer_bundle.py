import itertools

import pytest

from datagoblin.training_gym.molt_observer_allocation import map_bytes
from datagoblin.training_gym.molt_observer_bundle import Record, plan_bundle, report_bundles


def test_scalar_and_signed_break_even():
    for metadata, threshold in ((8, 11), (80, 102)):
        for n in (threshold - 1, threshold):
            records = tuple(Record(str(i), (5,), metadata) for i in range(n))
            result = plan_bundle(16, records)
            assert result["map_rank"] == (5 if n == threshold else 0)
            assert result["planned_bytes"] == min(128 * n, 812 + (40 + metadata) * n)
            assert not result["certificate_authorized"]


def test_mixed_rank_exhaustive():
    records = (
        Record("a", (1, 3), 8), Record("b", (2,), 0),
        Record("c", (), 8), Record("d", (1, 2), 0),
    )
    # Increase reuse without changing the eligible rank pattern.
    records = tuple(Record(f"{i}-{r.id}", r.eligible_ranks, r.bound_bytes)
                    for i in range(8) for r in records)
    result = plan_bundle(8, records)
    alternatives = []
    for cap in range(0, 9):
        size = map_bytes(8, cap) if cap else 0
        for r in records:
            size += min([64, *(8 * k + r.bound_bytes for k in r.eligible_ranks if k <= cap)])
        alternatives.append((size, cap))
    assert (result["planned_bytes"], result["map_rank"]) == min(alternatives)
    # A genuinely exhaustive per-record choice enumeration on a tiny case.
    tiny = (Record("a", (1, 2), 0), Record("b", (2, 3), 0))
    brute = []
    for a, b in itertools.product((None, 1, 2), (None, 2, 3)):
        ranks = [k for k in (a, b) if k is not None]
        cost = sum(8 * (k if k is not None else 16) for k in (a, b))
        if ranks:
            cost += map_bytes(16, max(ranks))
        brute.append(cost)
    assert plan_bundle(16, tiny)["planned_bytes"] == min(brute)


def test_report_ingest_exact_roster():
    report = {"schema": "molt08-observer-error-v1", "status": "development_only",
              "conditional_real_arithmetic_only": True, "formal_certificate_authorized": False,
              "model_seeds": [137], "world_roots": 2, "rows": []}
    for length, dose, rank in itertools.product((8, 12), (0.25, 0.5, 1.0), range(1, 6)):
        report["rows"].append({"seed": 137, "root": f"molt08-{length}-0", "dose": dose, "rank": rank,
                               "accepts": {"interval16_scalar": rank == 5},
                               "sign_parity": True, "approx": [1, 2, 3, 4, 5]})
    result = report_bundles(report, "interval16_scalar")
    assert result["groups"] == 2
    assert result["records"] == 6
    assert result["dense_bytes"] == result["planned_bytes"] == 768
    assert result["map_groups"] == 0
    with pytest.raises(ValueError):
        report_bundles(report, "linear_only")
    with pytest.raises(ValueError):
        report_bundles({**report, "rows": report["rows"][:15]}, "interval16_scalar")
    report["rows"][-1]["sign_parity"] = False
    with pytest.raises(ValueError):
        report_bundles(report, "interval16_scalar")
