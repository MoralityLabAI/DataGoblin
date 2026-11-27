import os

import pytest


def test_new_topologies_exact_reference():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned checker")
    from datagoblin.zk_advantage_gym import SHIFT
    from datagoblin.zk_neighborhood_gym import full_reference

    for p in (11, 21888242871839275222246405745257275088548364400416034343698204186575808495617):
        for family in SHIFT:
            cert, columns = full_reference(p, family, 19101)
            assert (cert is None) == family.startswith("essential_")
            assert columns > 0


def test_signed_advantage_and_matched_memory_controls():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned checker")
    from datagoblin.zk_advantage_gym import ARMS, PROTOCOLS, campaign

    result = campaign(roots=1, train_roots=4)
    assert len(result["tasks"]) == 8
    tr = result["training"]
    assert tr["observed_advantages"] == tr["positive_advantages"]+tr["negative_advantages"]+tr["observed_ties"]
    assert tr["positive_advantages"] == tr["explore_labels"]
    assert tr["negative_advantages"] == tr["exploit_labels"]
    assert tr["sign_updates"] > 0 and tr["value_updates"] > 0
    for task in result["tasks"]:
        for protocol, (budget, limit) in PROTOCOLS.items():
            for arm in ARMS:
                rec = task["arms"][protocol][arm]
                assert rec["columns"] <= budget
                assert not rec["lean_replayed"] and not rec["production_accepted"]
                if task["essential"]:
                    assert not rec["exact_identity_checked"]
                if limit is not None and rec["credited"]:
                    assert rec["cpu_ns"] <= limit
