import os

import pytest


def test_cached_symbolic_preparation_matches_all_features_used():
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned checker")
    from neighborhood_search import BN254_FR

    from datagoblin.zk_cached_prep_gym import FAMILIES, prepare_cached
    from datagoblin.zk_neighborhood_gym import fixture, prepare

    for family in FAMILIES:
        for p in (11, BN254_FR):
            problem = fixture(p, family, 21101)
            a, b = prepare(problem, 21101), prepare_cached(problem, 21101)
            assert a[:3] == b[:3]
            assert a[3].getstate() == b[3].getstate()
            assert problem.prepared_columns == 0


def test_cached_prep_campaign_and_replay_source(tmp_path):
    if not os.environ.get("ZKL13_CHECKER_ROOT"):
        pytest.skip("requires pinned checker")
    from datagoblin.zk_cached_prep_gym import campaign, emit_replay

    result = campaign(roots=1, train_roots=4)
    assert len(result["tasks"]) == 8
    for task in result["tasks"]:
        for protocol in ("c24", "c96"):
            records = task["arms"][protocol]
            for variant in ("recurrent", "reset"):
                a, b = records[f"heap_{variant}"], records[f"cached_{variant}"]
                assert a["trace"] == b["trace"]
                assert a["certificate"] == b["certificate"]
    cases = emit_replay([result], tmp_path / "replay")
    assert len(cases) == 12
    assert any(c["status"] == "generated" for c in cases)
    for case in cases:
        if case["status"] == "generated":
            source = (tmp_path / "replay" / case["file"]).read_text(encoding="utf-8")
            assert "theorem exact_poly_delete_closed" in source
            assert "theorem original_well_formed" in source
