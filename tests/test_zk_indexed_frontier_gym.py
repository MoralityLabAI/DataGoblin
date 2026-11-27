import os

import pytest


def test_indexed_frontier_matches_original():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned SatsHedgeZero checker")
    from datagoblin.zk_connected_search_gym import run as original
    from datagoblin.zk_indexed_frontier_gym import BUDGETS, ROOT_BLOCKS, campaign, summarize

    for root in ROOT_BLOCKS:
        result = campaign(roots=2, test_start=root)
        assert len(result["tasks"]) == 8
        for task in result["tasks"]:
            for budget in BUDGETS:
                pair = task["pairs"][str(budget)]
                reference = original(int(task["modulus"]), task["family"], task["seed"], "frontier", budget)
                for k in ("certificate", "columns", "coefficient_work", "prepared_terms"):
                    assert pair["scan"][k] == reference[k]
                    assert pair["indexed"][k] == reference[k]
                assert pair["scan"]["trace"] == pair["indexed"]["trace"]
        stats = {(r["budget"], r["arm"]): r for r in summarize(result)}
        for budget in BUDGETS:
            assert stats[budget, "scan"]["certified"] == stats[budget, "indexed"]["certified"] == 6
            assert stats[budget, "indexed"]["score_evals"] < stats[budget, "scan"]["score_evals"]


def test_index_updates_match_brute_force():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned SatsHedgeZero checker")
    from datagoblin.zk_indexed_frontier_gym import IndexedFrontier, score

    keys = [(i, ()) for i in range(8)]
    supports = {key: frozenset({(i+1,), (i % 3+1,), (i % 2+1, i % 2+1)})
                for i, key in enumerate(keys)}
    base = {key: (i % 3)*0.17 for i, key in enumerate(keys)}
    tie = {key: i/9 for i, key in enumerate(keys)}
    remaining = {(1,), (2,)}
    frontier = set(remaining)
    index = IndexedFrontier(keys, supports, base, tie, remaining)
    pending = set(keys)
    for step in range(len(keys)):
        key = max(pending, key=lambda k: (score(base[k], len(supports[k] & remaining),
                                                len(supports[k] & frontier), len(supports[k])), tie[k]))
        assert index.take() == key
        pending.remove(key)
        remaining = {(step % 3+1,), (step % 2+1, step % 2+1)}
        frontier.update(supports[key])
        index.update(remaining, supports[key])
        for k in pending:
            assert index.rc[k] == len(supports[k] & remaining)
            assert index.fc[k] == len(supports[k] & frontier)
