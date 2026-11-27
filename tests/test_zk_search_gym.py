import os

import pytest


def test_zkl13_paired_search_pinned_checkout():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires scoped ZKL-13 checker checkout")
    from datagoblin.zk_search_gym import ARMS, BUDGETS, TEST, TRAIN, campaign, summarize

    assert set(TRAIN).isdisjoint(TEST)
    for start in (13101, 13201):
        result = campaign(test_start=start)
        assert len(result["tasks"]) == 32
        assert not result["lean_replayed"]
        for task in result["tasks"]:
            essential = task["family"] == "essential"
            assert task["full"]["exact_identity_checked"] == (not essential)
            for budget in BUDGETS:
                for arm in ARMS:
                    rec = task["arms"][str(budget)][arm]
                    assert rec["columns"] <= budget
                    assert not rec["production_accepted"]
                    if essential:
                        assert not rec["exact_identity_checked"]
            assert task["arms"]["24"]["peel_local"]["exact_identity_checked"] == (not essential)
        stats = {(r["budget"], r["arm"]): r for r in summarize(result)}
        assert stats[24, "peel_local"]["certified"] == 24
        assert stats[24, "learned"]["certified"] == 16
