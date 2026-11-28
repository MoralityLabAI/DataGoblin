import random
from itertools import combinations

import pytest

from datagoblin.training_gym.molt_episode_policy import PublicPolicy
from datagoblin.training_gym.molt_packed_policy import (
    PackedPublicPolicy,
    campaign,
    native_row,
    packed_select,
    prepare,
    public_row,
)


def exhaustive(candidates, cap, forbidden):
    eligible = [c for c in candidates if c["id"] not in forbidden]
    return min(
        (-sum(c["utility"] for c in group), tuple(sorted(c["id"] for c in group)))
        for n in range(len(eligible) + 1)
        for group in combinations(eligible, n)
        if sum(c["cost"] for c in group) <= cap
    )[1]


def test_packed_ties_and_random_exactness():
    rng = random.Random(910)
    for n in range(1, 8):
        for _ in range(10):
            candidates = [
                {"id": f"c{i:02}", "cost": rng.randint(1, 5), "utility": rng.randint(1, 5)}
                for i in range(n)
            ]
            rng.shuffle(candidates)
            for cap in range(12):
                p = prepare(candidates, cap)
                for forbidden in (set(), {c["id"] for c in candidates[::2]}):
                    expected = exhaustive(candidates, cap, forbidden)
                    assert packed_select(p, forbidden, dense=False) == expected
                    assert packed_select(p, forbidden, dense=True) == expected
    tied = [{"id": "a", "cost": 1, "utility": 1},
            {"id": "b", "cost": 2, "utility": 2},
            {"id": "c", "cost": 1, "utility": 1}]
    assert packed_select(prepare(tied, 2), set(), dense=True) == ("a", "c")


def test_high_bit_width_and_source_rebinding():
    candidates = [{"id": f"c{i:03}", "cost": 1, "utility": 1} for i in range(128)]
    prepared = prepare(candidates, 3)
    for dense in (False, True):
        assert packed_select(prepared, {"c000"}, dense=dense) == ("c001", "c002", "c003")
        p = PackedPublicPolicy(dense=dense)
        p.action({"event": "start", "candidates": candidates, "capacity": 3,
                  "frame_id": "old", "receipts": [{"source": "s0", "excluded": "c000", "receipt_id": "r1"}]})
        action = p.action({"event": "source_change", "source": "s0", "frame_id": "new"})
        assert action == {"action": "request", "source": "s0"}
        action = p.action({"event": "receipt", "source": "s0", "excluded": "c004", "receipt_id": "r2"})
        assert action["selected"] == ["c000", "c001", "c002"]
        assert action["frame_id"] == "new" and action["receipts"] == ["r2"]
    with pytest.raises(ValueError):
        packed_select(prepared, {"absent"}, dense=False)


def test_reference_and_host_parity():
    for width in (5, 8, 12, 16):
        row = native_row("calibration", 11, 0, width)
        assert all(a["optimal"] == a["verified"] == 2 for a in row["arms"].values())
    for width in (16, 32, 64, 128):
        row = public_row(width, "medium", 0)
        assert all(a["queries"] == 4 for a in row["arms"].values())
    with pytest.raises(ValueError):
        native_row("ordinary_test", 11, 0, 5)


def test_preparation_scope():
    p = PublicPolicy()
    assert p._submit()["selected"] == []
    with pytest.raises(ValueError):
        prepare([{"id": "a", "cost": 0, "utility": 1}], 3)
    with pytest.raises(ValueError):
        prepare([{"id": "a", "cost": 1, "utility": 0}], 3)
    with pytest.raises(ValueError):
        PackedPublicPolicy(dense=True)._submit()


def test_campaign_smoke():
    r = campaign(1, 1)
    assert not r["promotion_authorized"] and not r["training_admission"]
    assert r["provider_calls"] == 0
    assert len(r["native_rows"]) == len(r["public_rows"]) == 8
