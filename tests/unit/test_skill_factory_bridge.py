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


def test_floodgates_setup_and_selected_descriptor(tmp_path):
    target_hash = "a" * 64
    adapter_hashes = {"first": "b" * 64, "deep": "c" * 64}
    descriptor = {
        "schema": "hybrid.gsf.floodgates.fg09.v1.policy",
        "choice": "fixed_first4",
        "target_hash": target_hash,
        "adapter_hashes": adapter_hashes,
        "controller_state": None,
    }
    selected_bytes = json.dumps(descriptor).encode()
    (tmp_path / "selected.weights").write_bytes(selected_bytes)

    def entry(name):
        payload = name.encode()
        (tmp_path / name).write_bytes(payload)
        return {"artifact": name, "artifact_sha256": hashlib.sha256(payload).hexdigest()}

    row = {
        "schema": "goblinbench.skill_factory.v1.episode",
        "fg09_schema": "hybrid.gsf.floodgates.fg09.v1",
        "gym": "GSF-10",
        "role": "trace_train",
        "index": 0,
        "training_ready": False,
        "setup": {
            "target": {"weights_hash": target_hash},
            "adapters": {mode: {"weights_hash": value} for mode, value in adapter_hashes.items()},
        },
        "result": {
            "score": 0.65,
            "artifact": {
                "file": "selected.weights",
                "sha256": hashlib.sha256(selected_bytes).hexdigest(),
                "weights_hash": digest(descriptor),
                "choice": "fixed_first4",
            },
            "setup_artifacts": {
                "target": entry("target.pt"),
                "adapters": {mode: entry(f"{mode}-adapter.pt") for mode in adapter_hashes},
            },
        },
    }
    row["receipt_hash"] = digest(row)
    (tmp_path / "episode.json").write_text(json.dumps(row))
    assert verify_episode(tmp_path, "GSF-10", "trace_train", 0) == row
    (tmp_path / "deep-adapter.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="setup artifact digest"):
        verify_episode(tmp_path, "GSF-10", "trace_train", 0)
