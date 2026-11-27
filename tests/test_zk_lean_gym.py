import random

from datagoblin.zk_lean_gym import (
    DEVELOPMENT,
    HOLDOUT,
    campaign,
    evaluate,
    optimize,
    shape,
    variant,
)


def test_clean_relations_and_duplicate_optimization():
    for family in DEVELOPMENT + HOLDOUT:
        for p in (7, 11):
            original = shape(p, family, 3)
            result = evaluate(original, family, 3)
            assert result["sound"] and result["complete"]
            assert not result["lean_replayed"] and not result["production_accepted"]
            duplicate = variant(original, "duplicate", random.Random(0))
            candidate, edits = optimize(duplicate, family, 3)
            assert len(candidate["rows"]) < len(duplicate["rows"])
            assert edits
            result = evaluate(candidate, family, 3)
            assert result["sound"] and result["complete"]


def test_deleted_bit_has_counterexample():
    clean = shape(7, "bit", 1)
    broken = variant(clean, "drop", random.Random(1))
    result = evaluate(broken, "bit", 1)
    assert not result["sound"]
    assert result["complete"]
    assert result["counterexample"]["public"] == (2,)
    assert result["counterexample"]["assignment"] == (1, 2)
    candidate, edits = optimize(broken, "bit", 1)
    assert candidate == broken and not edits


def test_split_and_reproducibility():
    a = list(campaign(7, "development", 1))
    b = list(campaign(7, "development", 1))
    h = list(campaign(19, "holdout", 1))
    assert a == b
    assert len(a) == len(h) == 12
    assert {r["family"] for r in a}.isdisjoint({r["family"] for r in h})
