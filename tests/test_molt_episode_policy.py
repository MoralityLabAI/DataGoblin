import json

import pytest

from datagoblin.training_gym.molt_episode import make_world
from datagoblin.training_gym.molt_episode_policy import export, report, run_public, train_messages


def test_public_policy_controls():
    r = report(71, 64)
    assert r["roots"] == 64 and len(r["rows"]) == 512
    assert r["summaries"]["selective_refresh"] == {
        "episodes": 128, "optimal": 128, "verified": 128, "verifier_calls": 64}
    assert r["summaries"]["always_refresh"] == {
        "episodes": 128, "optimal": 128, "verified": 128, "verifier_calls": 128}
    assert r["summaries"]["stale"] == {
        "episodes": 128, "optimal": 64, "verified": 64, "verifier_calls": 0}
    assert r["summaries"]["forget_receipts"] == {
        "episodes": 128, "optimal": 0, "verified": 0, "verifier_calls": 128}
    assert not r["model_executed"] and not r["promotion_authorized"]


def test_ood_public_policy():
    for i in range(32):
        for variant in ("relevant", "unrelated"):
            ep = run_public(make_world("ood_test", 17, i, variant), "selective_refresh")
            assert ep.result["optimal"]


def test_chat_export_train_only(tmp_path):
    with pytest.raises(ValueError):
        train_messages(make_world("ordinary_test", 7, 0, "relevant"))
    p = tmp_path / "chat.jsonl"
    export(7, 2, p)
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert len(rows) == 4
    for row in rows:
        roles = [m["role"] for m in row["messages"]]
        assert roles[0] == "system" and roles[1:] == ["user", "assistant"] * ((len(roles) - 1) // 2)
        assert json.loads(row["messages"][-1]["content"])["action"] == "submit"
        assert row["history_policy"] == "full_transcript"
