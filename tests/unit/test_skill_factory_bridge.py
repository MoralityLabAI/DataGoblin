"""Protocol checks without contacting a model endpoint."""
from __future__ import annotations

import hashlib
import json

import pytest

from datagoblin.skill_factory import digest, parse_action, prompt, verify_episode


def test_actions_strict_json():
    action = {"action": "run_experiment", "choice": "direct"}
    assert parse_action(json.dumps(action)) == action
    for bad in (
        "[]", "null", "not json", '{"a":1,"a":2}', '{"a":NaN}',
        '{"a":1e999}', '```json\n{}\n```',
    ):
        assert parse_action(bad) is None


def test_prompt_only_public_inputs():
    text = prompt({"gym": "GSF-01"}, {"turns_left": 4})
    assert "GSF-01" in text
    assert "turns_left" in text
    assert "Final results are evaluator-only" in text


def test_episode_receipt_and_artifact(tmp_path):
    weights = b"fixture bytes"
    (tmp_path / "selected.weights").write_bytes(weights)
    row = {
        "schema": "goblinbench.skill_factory.v1.episode",
        "gym": "GSF-01",
        "role": "development",
        "index": 0,
        "training_ready": False,
        "result": {
            "score": 0.5,
            "artifact": {
                "file": "selected.weights",
                "sha256": hashlib.sha256(weights).hexdigest(),
            },
        },
    }
    row["receipt_hash"] = digest(row)
    (tmp_path / "episode.json").write_text(json.dumps(row))
    assert verify_episode(tmp_path, "GSF-01", "development", 0) == row
    with pytest.raises(ValueError):
        verify_episode(tmp_path, "GSF-01", "trace_train", 0)
    (tmp_path / "selected.weights").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        verify_episode(tmp_path, "GSF-01", "development", 0)
