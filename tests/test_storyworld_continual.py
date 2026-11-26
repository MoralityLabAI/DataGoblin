from pathlib import Path

import pytest

from Agonist.storyworld_continual import FILES, check_public, digest, make_views, source_digest


def packet():
    return {"schema": "SW-PROV/session/v1", "title": "fixture", "remaining": 1, "turn": 0,
            "terminal": False, "text": "public", "ledger": [], "actions": [], "swmd_card": "public"}


def test_source_digest_binds_all_files(tmp_path: Path):
    for p in FILES:
        (tmp_path / p).write_text(p)
    before, hashes = source_digest(tmp_path)
    assert len(hashes) == 4 and before == digest(hashes)
    (tmp_path / FILES[1]).write_text("changed")
    assert source_digest(tmp_path)[0] != before


def test_private_keys_rejected():
    assert check_public(packet()) == packet()
    with pytest.raises(ValueError):
        check_public({**packet(), "truth": 1})
    with pytest.raises(ValueError):
        check_public({**packet(), "ledger": [{"query": "x", "outcome": "1", "text": "x", "truth": 1}]})


def test_terminal_actions_rejected():
    with pytest.raises(ValueError):
        check_public({**packet(), "terminal": True, "actions": [{"id": "x", "label": "x", "cost": 1}]})


def test_candidate_custody():
    steps = [{"observation": packet(), "action": a, "optimal_actions": [a], "next_observation": packet()}
             for a in ("apply", "registry")]
    views = make_views([{"social": {"steps": steps}}], "digest")
    assert len(views) == 4
    assert all(x["split"] == "dev" and not x["training_ready"] for x in views)
    assert len({x["root_id"] for x in views}) == 1
    assert views[-1]["kernel_checked"] is False
