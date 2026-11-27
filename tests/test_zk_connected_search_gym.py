import os

import pytest


def test_connected_nonlinear_search_pinned():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned SatsHedgeZero checker")
    from datagoblin.zk_connected_search_gym import ARMS, BUDGETS, TEST, TRAIN, campaign

    assert set(TRAIN).isdisjoint(TEST)
    result = campaign(roots=2)
    assert len(result["tasks"]) == 8
    assert not result["lean_replayed"]
    assert not result["production_accepted"]
    for task in result["tasks"]:
        for budget in BUDGETS:
            for arm in ARMS:
                rec = task["arms"][str(budget)][arm]
                assert rec["columns"] <= budget
                assert rec["prepared_columns"] == rec["columns"]
                if task["family"] == "essential":
                    assert not rec["exact_identity_checked"]
                assert not rec["production_accepted"]
