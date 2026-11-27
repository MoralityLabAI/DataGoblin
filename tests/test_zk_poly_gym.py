import os
from pathlib import Path

from datagoblin.zk_ideal_gym import BN254_FR, file_sha
from datagoblin.zk_poly_gym import DEVELOPMENT, HOLDOUT, campaign


def test_degree_families_and_finite_relations():
    assert set(DEVELOPMENT).isdisjoint(HOLDOUT)
    for p in (7, 11):
        for split in ("development", "holdout"):
            rows = list(campaign(27, split, 3, p))
            for row in rows:
                assert row["finite_receipt"]["equivalent"] == (row["construction_expected_degree"] is not None)
                assert not row["tier_receipts"]
                assert not row["lean_replayed"]
                assert not row["production_accepted"]


def test_bn254_without_enumeration():
    rows = list(campaign(27, "holdout", 3, BN254_FR))
    assert len(rows) == 12
    assert all(r["finite_receipt"]["status"] == "NOT_ENUMERATED" for r in rows)


def test_external_checker_when_pinned_checkout_provided():
    value = os.environ.get("ZKL12_CHECKER_ROOT")
    if not value:
        return
    root = Path(value)
    pins = {name: file_sha(root / name) for name in ("poly_certificate.py", "ideal_certificate.py", "audit.py")}
    for p in (7, 11, BN254_FR):
        for split in ("development", "holdout"):
            rows = list(campaign(29, split, 2, p, root / "poly_certificate.py", pins))
            for row in rows:
                expected = row["construction_expected_degree"]
                found = [r["degree_budget"] for r in row["tier_receipts"] if r["status"] == "EXACT_IDENTITY_CHECKED"]
                assert found == ([] if expected is None else list(range(expected, 3)))
