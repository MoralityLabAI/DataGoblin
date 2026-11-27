import os

import pytest


def test_zkl17_exploration_gate():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned checker")
    from datagoblin.zk_exploration_gym import ARMS, HIDDEN, PROTOCOLS, campaign

    result = campaign(roots=2, train_roots=2)
    assert len(result["tasks"]) == 8
    assert result["training"]["updates"] > 0
    assert len(result["model"]["wh"]) == HIDDEN
    assert not result["production_accepted"]
    for task in result["tasks"]:
        for protocol, (budget, limit) in PROTOCOLS.items():
            for arm in ARMS:
                rec = task["arms"][protocol][arm]
                assert rec["columns"] <= budget
                assert rec["prepared_columns"] == rec["columns"]
                assert rec["explore_actions"] <= rec["columns"]
                assert not rec["production_accepted"]
                if task["family"] == "essential_chain":
                    assert not rec["exact_identity_checked"]
                if limit is not None and rec["credited"]:
                    assert rec["cpu_ns"] <= limit


def test_hybrid_index_remains_exact_after_exploration():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned checker")
    from datagoblin.zk_exploration_gym import Hybrid, fixture

    problem = fixture(11, "product_chain_zero", 17101)
    policy = Hybrid(problem, 17101)
    for t in range(16):
        if policy.state.solved:
            break
        expected = max(policy.index.active, key=lambda k: (policy.value(k), policy.tie[k]))
        assert policy.exploit() == expected
        action = bool(t % 2)
        key = policy.explore() if action else expected
        policy.advance(key, action)
        for k in policy.index.active:
            assert policy.index.rc[k] == len(policy.supports[k] & policy.index.remaining)
            assert policy.index.fc[k] == len(policy.supports[k] & policy.index.frontier)
