import random
from itertools import combinations

import pytest

from datagoblin.training_gym.molt_critical_witness import (
    ARMS,
    MODES,
    WitnessPlanner,
    WitnessPolicy,
    build_critical_atlas,
    campaign,
    decode,
    factory,
    metrics,
)
from datagoblin.training_gym.molt_exclusion_atlas import native_row, public_row
from datagoblin.training_gym.molt_packed_policy import packed_select, prepare


def test_witness_and_critical_omissions_exhaustive():
    rng = random.Random(21)
    for n in range(1, 9):
        rows = [{"id": f"c{i}", "cost": rng.randint(1, 5), "utility": rng.randint(1, 5)} for i in range(n)]
        for cap in range(10):
            p = prepare(rows, cap)
            for fixed in (frozenset(), frozenset(c["id"] for c in rows[::3])):
                witness = packed_select(p, set(fixed), dense=True)
                omitted, cells = build_critical_atlas(p, fixed, frozenset(witness))
                assert set(omitted) == set(witness)
                assert cells <= (len(witness) + 2) * (cap + 1)
                for variable in (None, *sorted(p.ids)):
                    eligible = [c for c in rows if c["id"] not in fixed and c["id"] != variable]
                    expected = min(
                        (-sum(c["utility"] for c in group), tuple(sorted(c["id"] for c in group)))
                        for k in range(len(eligible) + 1)
                        for group in combinations(eligible, k)
                        if sum(c["cost"] for c in group) <= cap
                    )[1]
                    actual = decode(p, omitted[variable]) if variable in omitted else witness
                    assert actual == expected


def test_lazy_scope_and_critical_admission():
    rows = [{"id": f"c{i:02}", "cost": 1, "utility": 100 - i} for i in range(12)]
    p = prepare(rows, 8)
    fixed = frozenset({"c11"})
    for mode in MODES:
        planner = WitnessPlanner(p, mode, memo_limit=2)
        # First direct query; second novel query builds a witness only in
        # witness arms; four novel critical misses then trigger the atlas.
        sequence = ("c10", "c09", "c00", "c01", "c02", "c03", "c04", "c10", "c00")
        for variable in sequence:
            assert planner.choose(fixed, variable) == packed_select(p, set(fixed) | {variable}, dense=True)
        assert planner.counts.peak_entries <= 2
        if mode == "critical_atlas":
            assert planner.counts.builds == 1
            assert planner.counts.atlas_hits >= 2
            assert planner.counts.witness_hits >= 2
            assert planner.counts.peak_cells == (8 + 2) * 9
        assert planner.choose(frozenset({"c00"}), "c01") == packed_select(p, {"c00", "c01"}, dense=True)
        assert planner.counts.invalidations == 1 and planner.atlas is None
    planner = WitnessPlanner(p, "critical_atlas", cell_limit=1)
    for variable in sequence:
        assert planner.choose(fixed, variable) == packed_select(p, set(fixed) | {variable}, dense=True)
    assert planner.counts.cell_rejections == 1 and planner.counts.builds == 0
    with pytest.raises(ValueError):
        planner.choose(fixed, "missing")


def test_repeated_key_does_not_trigger_full_solve():
    p = prepare([{"id": "a", "cost": 1, "utility": 1}, {"id": "b", "cost": 1, "utility": 2}], 1)
    planner = WitnessPlanner(p, "witness")
    for _ in range(5):
        assert planner.choose(frozenset(), "a") == ("b",)
    assert planner.counts.direct_solves == 1
    assert planner.counts.memo_hits == 4 and planner.counts.full_solves == 0


def test_receipt_binding_and_native_replay():
    p = WitnessPolicy("witness")
    p.action({
        "event": "start", "candidates": [{"id": "a", "cost": 1, "utility": 1},
                                         {"id": "b", "cost": 1, "utility": 2}],
        "capacity": 1, "frame_id": "old",
        "receipts": [{"source": "s0", "excluded": "a", "receipt_id": "old-r"}],
    })
    for j in range(3):
        p.frame = f"f{j}"
        a = p.action({"event": "receipt", "source": "s0", "excluded": "a", "receipt_id": f"r{j}"})
        assert a["selected"] == ["b"] and a["receipts"] == [f"r{j}"] and a["frame_id"] == f"f{j}"
    assert p.planner is not None and p.planner.counts.memo_hits == 2
    row = native_row("calibration", 21, 0, 5, arm_names=ARMS, factory=factory, metrics=metrics)
    assert all(a["verified"] == a["optimal"] == 2 for a in row["arms"].values())
    row = public_row(16, "medium", 1000, "churn", 16, arm_names=ARMS, factory=factory, metrics=metrics)
    assert row["arms"]["witness"]["invalidations"] == 4


@pytest.mark.parametrize("mode", MODES)
def test_witness_mode_preserves_required_refresh(mode):
    p = WitnessPolicy(mode)
    p.action({
        "event": "start", "candidates": [{"id": "a", "cost": 1, "utility": 1},
                                         {"id": "b", "cost": 1, "utility": 2}],
        "capacity": 1, "frame_id": "old",
        "receipts": [{"source": "s0", "excluded": "a", "receipt_id": "old-r"}],
    })
    assert p.mode == "selective_refresh" and p.witness_mode == mode
    assert p.planner is not None and p.planner.mode == mode
    a = p.action({"event": "source_change", "source": "s0", "frame_id": "new"})
    assert a == {"action": "request", "source": "s0"}
    a = p.action({"event": "receipt", "source": "s0", "excluded": "b", "receipt_id": "new-r"})
    assert a == {"action": "submit", "selected": ["a"], "receipts": ["new-r"], "frame_id": "new"}


def test_campaign_smoke():
    r = campaign(1, 1)
    assert not r["training_admission"] and not r["promotion_authorized"] and r["provider_calls"] == 0
    assert len(r["native_rows"]) == 6 and len(r["public_rows"]) == 40
    assert all(
        a["peak_cells"] <= r["max_atlas_cells"] and a["peak_entries"] <= r["memo_limit"]
        for row in r["public_rows"] for name, a in row["arms"].items() if name in MODES
    )
