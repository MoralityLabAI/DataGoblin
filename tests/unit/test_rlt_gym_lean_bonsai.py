from __future__ import annotations

import json
from subprocess import CompletedProcess

import pytest

from datagoblin.training_gym.rlt_gym import lean_bonsai as bonsai


@pytest.mark.parametrize(("content", "expected"), [
    ("2", 2), ("<think>\n\n</think>\n2", 2), ("sorry", None), ("2; sorry", None), ("1", None),
])
def test_bonsai_accepts_only_available_choice_ids(monkeypatch, content, expected):
    engine = bonsai.OxiBonsai(port=8080, timeout=5)

    def reply(path, payload):
        assert path == "/v1/chat/completions"
        assert payload["temperature"] == 0
        assert payload["max_tokens"] == bonsai.MAX_TOKENS
        prompt = payload["messages"][0]["content"]
        assert "Previous choices failed verification: 1 (simp)." in prompt
        assert "2. rfl" in prompt
        return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 12, "completion_tokens": 1}}

    monkeypatch.setattr(engine, "request", reply)
    assert engine.choose("1 = 1", {2: "rfl"}, {1: "simp"})["choice"] == expected


@pytest.mark.parametrize(("split", "exported"), [("train", True), ("held_templates", False)])
def test_bonsai_stops_on_verification_and_exports_only_train(tmp_path, monkeypatch, split, exported):
    model = tmp_path / "Bonsai-8B.gguf"
    model.write_bytes(b"model fixture")
    lean = tmp_path / "lean"
    lean.touch()
    monkeypatch.setattr(bonsai, "CHECKPOINT_SHA256", bonsai.sha256_file(model))
    monkeypatch.setattr(bonsai.subprocess, "run", lambda *args, **kwargs: CompletedProcess(
        args, 0, "Lean (version 4.35.0, test)\n", "",
    ))
    monkeypatch.setattr(bonsai, "probe_library", lambda _lean: ([], "test"))
    monkeypatch.setattr(bonsai.OxiBonsai, "model_id", lambda _self: "fixture")

    def choose(_self, _statement, menu, failed):
        choice = next(iter(menu)) if not failed else next(i for i, tactic in menu.items() if tactic == "rfl")
        return {"choice": choice, "usage": {"prompt_tokens": 4, "completion_tokens": 1}}

    monkeypatch.setattr(bonsai.OxiBonsai, "choose", choose)
    monkeypatch.setattr(bonsai, "check_tactic", lambda _lean, _statement, tactic, _cwd: {
        "status": "verified" if tactic == "rfl" else "rejected", "proof": tactic,
    })
    out = tmp_path / "run"
    result = bonsai.run_lean_bonsai(
        out, lean=lean, checkpoint=model, port=8080, split=split, fixtures=None,
        seed=7, max_goals=1, attempts=4, request_timeout=10,
    )
    assert result["arms"]["oxibonsai"]["verified"] == 1
    assert result["arms"]["oxibonsai"]["completion_tokens"] == 2
    assert result["arms"]["lexical"]["checks"] == 4
    trace = json.loads((out / "rollouts.jsonl").read_text(encoding="utf-8"))
    assert [r["status"] for r in trace["arms"]["oxibonsai"]] == ["rejected", "verified"]
    assert bool((out / "sft.jsonl").read_text(encoding="utf-8")) is exported


def test_fixtures_reject_noninteger_salts(tmp_path):
    goal = bonsai.fixture_goals(None, "train", 7)[0]
    goal["salt"] = str(goal["salt"])
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps({
        "schema": "datagoblin.lean-grounded-split.v1",
        "splits": {"train": [goal], "same_templates": [], "held_templates": []},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="integer"):
        bonsai.fixture_goals(path, "train", 7)
