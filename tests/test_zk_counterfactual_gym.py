import copy
import os

import pytest


def test_counterfactual_fork_isolation_and_replay():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned checker")
    from datagoblin.zk_counterfactual_gym import counterfactual, fork
    from datagoblin.zk_exploration_gym import Hybrid
    from datagoblin.zk_neighborhood_gym import fixture, verified_certificate

    p = Hybrid(fixture(11, "product_chain_zero", 18001), 18001)
    a, b = p.exploit(), p.explore()
    before = (p.state.counts(), dict(p.state.remaining), set(p.index.active),
              dict(p.index.rc), dict(p.index.fc), list(p.index.heap),
              dict(p.state.problem.columns.cache), p.rng.getstate(), copy.deepcopy(p.pending))
    x = fork(p)
    x.advance(a, False)
    assert x.state.problem is not p.state.problem
    assert x.state.problem.columns.cache is not p.state.problem.columns.cache
    assert a in p.index.active and a not in x.index.active
    if x.state.solved:
        assert verified_certificate(x.state.problem, x.state)[0] is not None
    first = counterfactual(p, a, b, 12, 91)
    second = counterfactual(p, a, b, 12, 91)
    assert first[0] == second[0]
    for left, right in zip(first[1], second[1], strict=True):
        for arm in ("exploit", "explore"):
            assert {k: v for k, v in left[arm].items() if k != "verify_ns"} == {
                k: v for k, v in right[arm].items() if k != "verify_ns"}
    after = (p.state.counts(), dict(p.state.remaining), set(p.index.active),
             dict(p.index.rc), dict(p.index.fc), list(p.index.heap),
             dict(p.state.problem.columns.cache), p.rng.getstate(), p.pending)
    assert before == after


def test_counterfactual_labels_and_deadline():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned checker")
    from datagoblin.zk_counterfactual_gym import ARMS, PROTOCOLS, campaign

    result = campaign(roots=2, train_roots=4)
    tr = result["training"]
    assert tr["exploit_labels"] > 0 and tr["explore_labels"] > 0
    assert tr["trajectory_steps"] == tr["exploit_labels"]+tr["explore_labels"]+tr["masked"]
    assert tr["fork_columns"] <= 48*tr["fork_pairs"]
    assert tr["fork_verifier_calls"] <= 2*tr["fork_pairs"]
    assert len(result["tasks"]) == 8
    for t in result["tasks"]:
        for protocol, (budget, limit) in PROTOCOLS.items():
            for arm in ARMS:
                rec = t["arms"][protocol][arm]
                assert rec["columns"] <= budget
                assert not rec["production_accepted"] and not rec["lean_replayed"]
                if t["family"] == "essential_chain":
                    assert not rec["exact_identity_checked"]
                if limit is not None and rec["credited"]:
                    assert rec["cpu_ns"] <= limit
