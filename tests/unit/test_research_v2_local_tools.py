"""Focused unit tests for Research Verifiers v2 local evaluation and persona tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "research_v2"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from aggregate_benchmark_results import _summarize_reply_safely  # noqa: E402
from export_sanitized_artifacts import _assert_clean_text  # noqa: E402
from jinn_beast_persona_exp import (  # noqa: E402
    AdditiveLDTHead,
    TRMRoutedRLTHead,
    _evaluate_frame_head,
    _reconstruct_store,
)


def test_reconstruct_store_enforces_owner_scope_temporal_and_revocation() -> None:
    events = [
        {"actor": "u1", "scope": "research", "revision": 1, "op": "set", "key": "tone", "value": "warm"},
        {"actor": "u1", "scope": "research", "revision": 2, "op": "set", "key": "verbosity", "value": "brief"},
        {"actor": "u2", "scope": "research", "revision": 5, "op": "set", "key": "tone", "value": "hostile"},
        {"actor": "u1", "scope": "fiction", "revision": 5, "op": "set", "key": "tone", "value": "theatrical"},
        {"actor": "u1", "scope": "research", "revision": 4, "op": "set", "key": "tone", "value": "formal"},
        {"actor": "u1", "scope": "research", "revision": 5, "op": "revoke", "key": "verbosity"},
        {"actor": "u1", "scope": "research", "revision": 6, "op": "set", "key": "tone", "value": "future"},
    ]
    # Reversed delivery order tests explicit temporal sorting
    rev_events = list(reversed(events))
    bound = _reconstruct_store(rev_events, owner="u1", scope="research", at_revision=5)
    assert bound == {"tone": "formal"}

    no_owner = _reconstruct_store(rev_events, owner="u1", scope="research", at_revision=5, enforce_owner=False)
    assert no_owner["tone"] == "hostile"

    no_temporal = _reconstruct_store(rev_events, owner="u1", scope="research", at_revision=5, enforce_temporal=False)
    assert no_temporal["tone"] != "formal" or "verbosity" in no_temporal


def test_trm_routed_rlt_head_guarantees_exact_opposite_frame_invariance_and_zero_veto_fa() -> None:
    rng = np.random.default_rng(7)
    n, d = 120, 12
    x = (rng.random((n, d)) < 0.35).astype(np.float64)
    own_idx = [0, 1, 2, 3, 4, 5]
    crit_idx = [0]
    y = np.where(x[:, 0] == 1, 2, np.where(x[:, 1] == 1, 0, 1)).astype(np.int64)

    head = TRMRoutedRLTHead(d, own_indices=own_idx, critical_indices=crit_idx, seed=7)
    info = head.fit(x, y, seed=7)
    assert info["rlt_parameters"] > 0

    x_own = x.copy()
    x_own[:, 6:] = 0.0
    p_u = head.predict_probs(x)
    p_o = head.predict_probs(x_own)
    metrics = _evaluate_frame_head(p_u, p_o, y, x[:, 0] == 1, apply_veto=True)
    assert metrics["opposite_frame_top_lane_invariance"] == 1.0
    assert metrics["guarded_critical_false_accepts"] == 0
    assert metrics["guarded_accuracy"] >= 0.9


def test_additive_ldt_head_learns_and_reduces_loss() -> None:
    rng = np.random.default_rng(11)
    x = (rng.random((96, 8)) < 0.5).astype(np.float64)
    y = np.where(x[:, 0] == 1, 2, np.where(x[:, 1] == 1, 0, 1)).astype(np.int64)
    head = AdditiveLDTHead(8, seed=11)
    info = head.fit(x, y, epochs=50, seed=11)
    assert info["parameters"] == 27
    assert info["learning_curve"][-1]["train_loss"] < info["learning_curve"][0]["train_loss"]


def test_mesh_registry_schema_and_distinct_ids() -> None:
    reg_path = Path(__file__).resolve().parents[2] / "configs" / "research_v2" / "models_mesh_20261126.json"
    data = json.loads(reg_path.read_text(encoding="utf-8"))
    assert data["schema"] == "datagoblin-research-v2-model-registry-1"
    ids = [m["id"] for m in data["models"]]
    assert len(ids) == len(set(ids))
    for req in (
        "baby-goblin-10m-v2-native",
        "baby-goblin-10m-v2-q4km",
        "dory-4b-bf16-loop2",
        "nanbeige42-3b-loop2",
        "qwen3-4b",
        "hrm-text-1b-h2-l3",
        "inky-mini-varloop-research-pass3",
    ):
        assert req in ids


def test_reply_redaction_omits_answer_values() -> None:
    raw = json.dumps({"action": "submit", "answer": {"checkpoint": "c1", "next_position": 5242880}})
    summary = _summarize_reply_safely(raw)
    assert summary["valid_json"] is True
    assert summary["action"] == "submit"
    assert summary["answer_keys"] == ["checkpoint", "next_position"]
    assert "5242880" not in json.dumps(summary)


def test_sanitizer_rejects_local_paths_and_tokens() -> None:
    import pytest

    _assert_clean_text('{"status":"ok","base_url":"http://127.0.0.1:18000/v1"}', "clean")
    for bad in (r"C:\Users\alice\secret", "/home/tester/private", "hf_" + "A" * 24):
        with pytest.raises(ValueError):
            _assert_clean_text(bad, "bad")

