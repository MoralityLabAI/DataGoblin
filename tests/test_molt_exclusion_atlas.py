import random
from itertools import combinations

import pytest

from datagoblin.training_gym.molt_exclusion_atlas import (
    AtlasPolicy,
    RevisionPlanner,
    build_atlas,
    campaign,
    decode,
    native_row,
    public_row,
)
from datagoblin.training_gym.molt_packed_policy import packed_select, prepare


def test_all_single_exclusions_against_enumeration():
    rng = random.Random(2020)
    for n in range(1, 8):
        rows = [{"id": f"c{i}", "cost": rng.randint(1, 4), "utility": rng.randint(1, 5)} for i in range(n)]
        rng.shuffle(rows)
        for cap in range(10):
            p = prepare(rows, cap)
            for fixed in (frozenset(), frozenset(c["id"] for c in rows[::3])):
                atlas = build_atlas(p, fixed)
                for variable in (None, *sorted(p.ids)):
                    forbidden = set(fixed)
                    if variable is not None:
                        forbidden.add(variable)
                    eligible = [c for c in rows if c["id"] not in forbidden]
                    expected = min(
                        (-sum(c["utility"] for c in group), tuple(sorted(c["id"] for c in group)))
                        for k in range(len(eligible) + 1)
                        for group in combinations(eligible, k)
                        if sum(c["cost"] for c in group) <= cap
                    )[1]
                    assert decode(p, atlas.score(variable)) == expected


def test_delay_invalidation_and_budget_fallback():
    p = prepare([{"id": f"c{i}", "cost": i + 1, "utility": i + 2} for i in range(8)], 12)
    planner = RevisionPlanner(p, delay=4)
    fixed = frozenset({"c7"})
    for i in range(4):
        variable = f"c{i}"
        assert planner.choose(fixed, variable) == packed_select(p, set(fixed) | {variable}, dense=True)
    assert planner.counts.fallbacks == 3
    assert planner.counts.builds == planner.counts.atlas_uses == 1
    assert planner.choose(frozenset({"c6"}), "c0") == packed_select(p, {"c6", "c0"}, dense=True)
    assert planner.counts.invalidations == 1
    assert planner.atlas is None
    blocked = RevisionPlanner(p, delay=1, limit=1)
    for i in range(2):
        assert blocked.choose(fixed, f"c{i}") == packed_select(p, set(fixed) | {f"c{i}"}, dense=True)
    assert blocked.counts.cell_rejections == 1 and blocked.counts.fallbacks == 2
    with pytest.raises(ValueError):
        build_atlas(p, fixed, 1)
    with pytest.raises(ValueError):
        planner.choose(fixed, "unknown")


def test_current_receipts_and_new_objective():
    p = AtlasPolicy(delay=1)
    rows = [{"id": f"c{i}", "cost": 1, "utility": i + 1} for i in range(5)]
    initial = {"event": "start", "candidates": rows, "capacity": 2, "frame_id": "start",
               "receipts": [{"source": "s0", "excluded": "c0", "receipt_id": "r0"},
                            {"source": "s1", "excluded": "c4", "receipt_id": "r1"}]}
    p.action(initial)
    for j in (1, 2):
        p.frame = f"f{j}"
        action = p.action({"event": "receipt", "source": "s0", "excluded": "c0", "receipt_id": f"r{j+1}"})
        assert action["selected"] == ["c2", "c3"]
        assert action["frame_id"] == f"f{j}"
        assert action["receipts"] == sorted([f"r{j+1}", "r1"])
    assert p.planner is not None and p.planner.counts.builds == 1
    p.action({**initial, "capacity": 1})
    assert p.planner is not None and p.planner.atlas is None


def test_host_and_churn_parity():
    for width in (5, 12, 16):
        row = native_row("calibration", 11, 0, width)
        assert all(x["verified"] == x["optimal"] == 2 for x in row["arms"].values())
    row = public_row(16, "tight", 0, "churn", 16)
    assert row["arms"]["atlas_eager"]["builds"] == 5
    assert row["arms"]["atlas_eager"]["invalidations"] == 4
    assert row["arms"]["atlas_eager"]["atlas_uses"] == 16


def test_campaign_smoke():
    r = campaign(1, 1)
    assert r["provider_calls"] == 0 and not r["promotion_authorized"] and not r["training_admission"]
    assert len(r["native_rows"]) == 6 and len(r["public_rows"]) == 40
    assert all(
        arm["peak_cells"] <= r["max_atlas_cells"]
        for row in r["public_rows"] for arm in row["arms"].values()
    )
