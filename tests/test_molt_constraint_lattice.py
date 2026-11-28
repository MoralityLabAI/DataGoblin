import random
from itertools import combinations

import pytest

from datagoblin.training_gym.molt_constraint_lattice import (
    ARMS,
    MODES,
    LatticePlanner,
    LatticePolicy,
    campaign,
    factory,
    metrics,
    public_row,
)
from datagoblin.training_gym.molt_exclusion_atlas import native_row
from datagoblin.training_gym.molt_packed_policy import packed_select, prepare


def brute(rows, cap, forbidden):
    eligible = [c for c in rows if c["id"] not in forbidden]
    return min(
        (-sum(c["utility"] for c in group), tuple(sorted(c["id"] for c in group)))
        for k in range(len(eligible) + 1)
        for group in combinations(eligible, k)
        if sum(c["cost"] for c in group) <= cap
    )[1]


def test_all_exclusion_sets_against_enumeration():
    rng = random.Random(22)
    rows = [{"id": f"c{i}", "cost": rng.randint(1, 5), "utility": rng.randint(1, 5)} for i in range(7)]
    sets = [{c["id"] for i, c in enumerate(rows) if mask >> i & 1} for mask in range(128)]
    rng.shuffle(sets)
    for cap in range(10):
        p = prepare(rows, cap)
        for mode in MODES:
            planner = LatticePlanner(p, mode, limit=8)
            for forbidden in sets + sets[::9]:
                assert planner.choose(forbidden) == brute(rows, cap, forbidden)
                assert len(planner.cache) <= 8 and len(planner.recent) <= 8
                for key, entry in planner.cache.items():
                    assert not key & entry.membership
            assert planner.counts.queries == (
                planner.counts.direct_solves + planner.counts.exact_hits
                + planner.counts.subset_hits + planner.counts.relaxed_hits
            )


def test_relaxed_witness_crosses_source_scopes():
    rows = [{"id": name, "cost": 1, "utility": utility}
            for name, utility in (("a", 20), ("b", 10), ("c", 2), ("d", 1))]
    p = prepare(rows, 1)
    m = LatticePlanner(p, "lattice_memo")
    assert m.choose(set()) == ("a",)
    assert m.choose({"b", "c"}) == ("a",)
    assert m.choose({"d"}) == ("a",)
    assert m.counts.direct_solves == 1 and m.counts.subset_hits == 2
    assert m.choose({"a"}) == ("b",)
    assert m.choose({"a", "d"}) == ("b",)
    assert m.choose({"a", "b"}) == ("c",)
    assert m.counts.direct_solves == 3

    m = LatticePlanner(p, "lattice_relax")
    assert m.choose({"b", "c"}) == ("a",)
    assert m.choose({"b", "d"}) == ("a",)
    assert m.counts.relaxed_solves == 1 and m.counts.relaxed_hits == 1
    assert m.choose({"b"}) == ("a",)
    assert m.counts.exact_hits == 1
    assert m.choose({"a", "b"}) == ("c",)
    assert m.counts.direct_solves == 2


def test_wide_masks_and_inert_exclusions():
    p = prepare([{"id": f"c{i:03}", "cost": 1, "utility": 1} for i in range(127)]
                + [{"id": "unaffordable", "cost": 4, "utility": 99}], 3)
    m = LatticePlanner(p, "lattice_guarded")
    for forbidden in (set(), {"unaffordable"}, {"c000"}, {"c125"}, set(p.ids) - {"c126"}):
        assert m.choose(forbidden) == packed_select(p, forbidden, dense=True)
    assert m.counts.exact_hits == 1
    with pytest.raises(ValueError):
        m.choose({"unknown"})


@pytest.mark.parametrize("mode", MODES)
def test_refresh_and_duplicate_constraints(mode):
    p = LatticePolicy(mode)
    p.action({
        "event": "start", "candidates": [{"id": "a", "cost": 1, "utility": 3},
                                         {"id": "b", "cost": 1, "utility": 2}],
        "capacity": 1, "frame_id": "old",
        "receipts": [{"source": "s0", "excluded": "b", "receipt_id": "r0"},
                     {"source": "s1", "excluded": "b", "receipt_id": "r1"}],
    })
    assert p.mode == "selective_refresh"
    assert p.action({"event": "source_change", "source": "s1", "frame_id": "new"}) == {
        "action": "request", "source": "s1"
    }
    a = p.action({"event": "receipt", "source": "s1", "excluded": "b", "receipt_id": "r2"})
    assert a == {"action": "submit", "selected": ["a"], "receipts": ["r0", "r2"], "frame_id": "new"}
    p.frame = "newer"
    a = p.action({"event": "receipt", "source": "s0", "excluded": "b", "receipt_id": "r3"})
    assert a["receipts"] == ["r2", "r3"] and a["frame_id"] == "newer"
    assert p.planner is not None and p.planner.counts.exact_hits == 1


def test_native_and_rotating_public_parity():
    row = native_row("calibration", 22, 0, 5, arm_names=ARMS, factory=factory, metrics=metrics)
    assert all(a["verified"] == a["optimal"] == 2 for a in row["arms"].values())
    row = public_row(32, "tight", 0, "rotate4", 32)
    assert all(a["requests"] == 32 for a in row["arms"].values())


def test_campaign_smoke():
    r = campaign(1, 1)
    assert not r["promotion_authorized"] and not r["training_admission"] and r["provider_calls"] == 0
    assert len(r["native_rows"]) == 6 and len(r["public_rows"]) == 30
    assert all(
        a["peak_entries"] <= r["memo_limit"]
        for row in r["public_rows"] for name, a in row["arms"].items() if name in MODES
    )
