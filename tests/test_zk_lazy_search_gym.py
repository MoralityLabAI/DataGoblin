import os

import pytest


def test_zkl14a_lazy_columns_paired():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires scoped checker checkout")
    from datagoblin.zk_lazy_search_gym import campaign, summarize

    result = campaign()
    assert len(result["tasks"]) == 32
    stats = summarize(result)
    assert stats["eager"]["certified"] == stats["lazy"]["certified"] == 24
    assert stats["lazy"]["prepared_columns"] == stats["lazy"]["columns"]
    assert stats["lazy"]["prepared_columns"] < stats["eager"]["prepared_columns"]
    assert stats["lazy"]["pool_terms"] < stats["eager"]["pool_terms"]
    assert not result["lean_replayed"]
    assert not result["production_accepted"]
