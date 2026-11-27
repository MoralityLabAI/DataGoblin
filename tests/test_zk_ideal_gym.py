from datagoblin.zk_ideal_gym import BN254_FR, DEVELOPMENT, HOLDOUT, campaign


def test_topology_and_small_field_relations():
    assert set(DEVELOPMENT).isdisjoint(HOLDOUT)
    for p in (7, 11):
        for split, seed in (("development", 7), ("holdout", 19)):
            records = list(campaign(seed, split, 3, p))
            assert len(records) == 9
            assert all(r["finite_receipt"]["status"] == "EXHAUSTIVE_FINITE_DOMAIN" for r in records)
            assert all(r["identity_receipt"]["status"] == "NOT_RUN" for r in records)
            for r in records:
                assert r["finite_receipt"]["equivalent"] == r["construction_expected_equivalent"]
                assert not r["lean_replayed"]
                assert not r["production_accepted"]
                if r["family"] == "multiplicative_implication":
                    assert r["finite_receipt"]["equivalent"]
                    assert not r["construction_expected_span"]
                elif not r["construction_expected_equivalent"]:
                    assert r["finite_receipt"]["counterexample"] is not None


def test_large_field_does_not_enumerate():
    records = list(campaign(17, "holdout", 2, BN254_FR))
    assert len(records) == 6
    assert all(r["finite_receipt"]["status"] == "NOT_ENUMERATED" for r in records)
    assert all(r["finite_receipt"]["equivalent"] is None for r in records)


def test_reproducible_and_unpromoted():
    a = list(campaign(7, "development", 2, 7))
    b = list(campaign(7, "development", 2, 7))
    assert a == b
    assert all(not r["production_accepted"] for r in a)
