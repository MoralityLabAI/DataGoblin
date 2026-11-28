from dataclasses import replace
from itertools import combinations

import pytest

from datagoblin.training_gym.molt_compact_policy import (
    CompiledPublicPolicy,
    Counts,
    SelectionMemory,
    campaign,
    episode,
    normalize,
    solve,
)
from datagoblin.training_gym.molt_episode import Episode, make_world
from datagoblin.training_gym.molt_episode_policy import PublicPolicy


def test_exhaustive_small_selection_parity():
    candidates = [
        {"id": "c0", "cost": 1, "utility": 3},
        {"id": "c1", "cost": 2, "utility": 3},
        {"id": "c2", "cost": 1, "utility": 2},
        {"id": "c3", "cost": 3, "utility": 5},
        {"id": "c4", "cost": 4, "utility": 8},
    ]
    for mask in range(32):
        forbidden = {c["id"] for i, c in enumerate(candidates) if mask >> i & 1}
        receipts = {name: {"excluded": name, "receipt_id": name} for name in forbidden}
        for cap in range(14):
            p = PublicPolicy()
            p.candidates = candidates
            p.receipts = receipts
            p.capacity = cap
            expected = tuple(p._submit()["selected"])
            eligible = [c for c in candidates if c["id"] not in forbidden]
            choices = [
                (-sum(c["utility"] for c in group), tuple(sorted(c["id"] for c in group)))
                for n in range(len(eligible) + 1)
                for group in combinations(eligible, n)
                if sum(c["cost"] for c in group) <= cap
            ]
            assert expected == min(choices)[1]
            assert solve(normalize(candidates, cap, receipts), Counts()) == expected


def test_quotient_and_current_authority():
    stats = Counts()
    memory = SelectionMemory(2, stats)
    p = CompiledPublicPolicy(memory)
    p.candidates = [
        {"id": "a", "cost": 1, "utility": 2},
        {"id": "b", "cost": 2, "utility": 4},
        {"id": "x", "cost": 9, "utility": 30},
    ]
    p.capacity = 4
    p.frame = "frame-old"
    p.receipts = {"s0": {"excluded": "x", "receipt_id": "r-old"}}
    first = p._submit()
    p.frame = "frame-new"
    p.receipts = {"s0": {"excluded": "x", "receipt_id": "r-new"}}
    p.candidates[2]["utility"] = 300
    p.capacity = 8
    second = p._submit()
    assert first["selected"] == second["selected"] == ["a", "b"]
    assert second["frame_id"] == "frame-new"
    assert second["receipts"] == ["r-new"]
    assert stats.hits == 1 and stats.writes == 1
    assert len(memory.entries) == 1
    assert all(isinstance(value, tuple) for value in memory.entries.values())
    for utility in (5, 6):
        p.candidates[1]["utility"] = utility
        p._submit()
    assert stats.evictions == 1 and len(memory.entries) == 2


def test_native_revision_and_stale_receipt_rejection():
    world = make_world("calibration", 7, 0, "relevant")
    cold = episode(world, PublicPolicy())
    stats = Counts()
    memory = SelectionMemory(2, stats)
    for _ in range(2):
        assert episode(world, CompiledPublicPolicy(memory)).trace == cold.trace
    assert stats.hits == 1
    other = replace(world, variant="unrelated")
    assert episode(other, CompiledPublicPolicy(memory)).trace == episode(other, PublicPolicy()).trace
    host = Episode(world)
    start = host.start()
    for _ in world.distractors:
        host.step({"action": "advance"})
    change = host.step({"action": "advance"})
    selected = cold.trace[-1]["action"]["selected"]
    result = host.step({
        "action": "submit", "frame_id": change["frame_id"], "selected": selected,
        "receipts": sorted(r["receipt_id"] for r in start["receipts"]),
    })
    assert not result["verified"] and result["reason"] == "stale_or_missing_evidence"


def test_scope_and_campaign():
    with pytest.raises(ValueError):
        normalize([{"id": "a", "cost": 1, "utility": 0}], 3, {})
    with pytest.raises(ValueError):
        normalize([{"id": "a", "cost": 1, "utility": 2}], 3, {"s0": {"excluded": "b"}})
    r = campaign(2, (1, 3))
    assert r["provider_calls"] == 0 and not r["promotion_authorized"] and not r["training_admission"]
    for summary in r["summary"].values():
        assert all(x["verified"] == x["optimal"] == x["episodes"] for x in summary.values())
        assert summary["compiled"]["dp_visits"] <= summary["reference"]["dp_visits"]
        assert summary["cache2"]["peak_entries"] <= 2
