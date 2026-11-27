import os

import pytest


def test_row_heap_matches_scan_with_arbitrary_actions():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned checker")
    from datagoblin.zk_exploration_gym import Hybrid
    from datagoblin.zk_neighborhood_gym import fixture
    from datagoblin.zk_row_heap_gym import HeapHybrid

    a = Hybrid(fixture(11, "zero_pair_transport", 20101), 20101)
    b = HeapHybrid(fixture(11, "zero_pair_transport", 20101), 20101)
    for t in range(96):
        if a.state.solved:
            assert b.state.solved
            break
        x, y = a.exploit(), b.exploit()
        assert x == y
        x2, y2 = a.explore(), b.explore()
        assert x2 == y2
        action = bool(t % 3)
        a.advance(x2 if action else x, action)
        b.advance(y2 if action else y, action)
        assert a.state.counts() == b.state.counts()
        assert a.state.remaining == b.state.remaining
        assert a.state.certificate() == b.state.certificate()
    assert b.local_scores == 0


def test_row_heap_paired_campaign():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned checker")
    from datagoblin.zk_row_heap_gym import campaign

    result = campaign(roots=1, train_roots=4)
    assert len(result["tasks"]) == 8
    assert result["training"]["updates"] > 0
    for task in result["tasks"]:
        for protocol in ("c24", "c96"):
            records = task["arms"][protocol]
            for variant in ("recurrent", "reset"):
                scan, heap = records[f"scan_{variant}"], records[f"heap_{variant}"]
                assert scan["trace"] == heap["trace"]
                assert scan["certificate"] == heap["certificate"]
                assert heap["column_scores"] <= scan["column_scores"]
