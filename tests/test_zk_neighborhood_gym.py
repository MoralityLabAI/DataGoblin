import os

import pytest


def test_zkl16_fresh_neighborhood_policies():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned SatsHedgeZero checker")
    from datagoblin.zk_neighborhood_gym import ARMS, PROTOCOLS, TEST, TRAIN, campaign

    assert set(TRAIN).isdisjoint(TEST)
    result = campaign(roots=2)
    assert len(result["tasks"]) == 8
    assert not result["production_accepted"]
    assert result["fit"]["examples"] > 0
    for task in result["tasks"]:
        for protocol, (budget, limit) in PROTOCOLS.items():
            for arm in ARMS:
                rec = task["arms"][protocol][arm]
                assert rec["columns"] <= budget
                assert rec["prepared_columns"] == rec["columns"]
                assert not rec["lean_replayed"]
                assert not rec["production_accepted"]
                if task["family"] == "essential_chain":
                    assert not rec["exact_identity_checked"]
                if rec["credited"] and limit is not None:
                    assert rec["cpu_ns"] <= limit
                if limit is None:
                    assert rec["credited"] == rec["exact_identity_checked"]
