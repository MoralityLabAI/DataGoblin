from __future__ import annotations

import json
from pathlib import Path

import pytest

from datagoblin.adapter_verifier.evidence_replay import (
    _bind_they_sing_record,
    _find_centauri_dataset,
    _project_diplomacy_outcomes,
    _they_sing_turn_commitment,
    _validate_challenge_binding,
    _validate_method_specs,
)
from datagoblin.data_etiquette.evidence_receipts import EvidenceVerifierRegistry


def test_game_replay_registry_commits_the_checked_in_method_specs() -> None:
    root = Path(__file__).resolve().parents[2]
    registry = EvidenceVerifierRegistry(root / "configs/game_evidence_verifiers.yaml")
    _validate_method_specs(registry, root / "configs/evidence_methods")


def test_challenge_binding_rejects_subject_substitution() -> None:
    challenge = {
        "item_id": "safe-1",
        "subject_sha256": "a" * 64,
        "game": "they_sing",
        "asserted_by": "model-a",
        "required_claim_types": ["OUTCOME_VALIDITY"],
        "required_verifier_id": "they-sing-environment-replay-v1",
        "required_method_id": "they-sing-state-transition-replay-v1",
    }
    map_row = {
        "item_id": "safe-1",
        "record_hash": "b" * 64,
        "game": "they_sing",
        "model_family": "model-a",
    }
    ledger = {"item_id": "safe-1", "game": "they_sing"}
    with pytest.raises(ValueError, match="subject"):
        _validate_challenge_binding(challenge, map_row, ledger)


def test_they_sing_binding_requires_both_exact_order_lists() -> None:
    rows = [
        {"type": "session_created", "sessionId": "episode-1", "data": {}},
        {
            "type": "phase_reasoning_diary",
            "sessionId": "episode-1",
            "turn": 1,
            "phase": "ALLOCATION",
            "data": {"factionId": "CANTOR", "requestedOrders": [{"type": "RESEARCH"}]},
        },
        {
            "type": "phase_reasoning_diary",
            "sessionId": "episode-1",
            "turn": 1,
            "phase": "ACTION_DECLARATION",
            "data": {"factionId": "CANTOR", "requestedOrders": [{"type": "AUDIT"}]},
        },
    ]
    record = {
        "episode_id": "episode-1",
        "turn": 1,
        "seat": "CANTOR",
        "target": {
            "allocation": {"orders": [{"type": "RESEARCH"}]},
            "action": {"orders": [{"type": "AUDIT"}]},
        },
    }
    assert _bind_they_sing_record(rows, record)
    record["target"]["action"]["orders"] = [{"type": "MOVE"}]
    assert not _bind_they_sing_record(rows, record)


def test_centauri_pilot_lookup_binds_the_episode_not_the_directory_name(tmp_path: Path) -> None:
    preferred = tmp_path / "pilot-centauri" / "native-centauri.jsonl"
    preferred.parent.mkdir()
    preferred.write_text(
        json.dumps({"episode_id": "older", "event": "reset"}) + "\n",
        encoding="utf-8",
    )
    recovered = tmp_path / "pilot-2026-01-01" / "centauri" / "native-centauri.jsonl"
    recovered.parent.mkdir(parents=True)
    recovered.write_text(
        json.dumps({"episode_id": "selected", "event": "reset"}) + "\n",
        encoding="utf-8",
    )
    assert _find_centauri_dataset(tmp_path, preferred, "selected") == recovered


def test_diplomacy_projection_uses_only_public_orders_and_results() -> None:
    saved = {
        "phases": [
            {
                "name": "S1915M",
                "orders": {"AUSTRIA": ["A VIE H"]},
                "order_results": {
                    "AUSTRIA": {
                        "hold": [{"order": "A VIE H", "result": "success"}],
                    }
                },
                "messages": [{"content": "excluded"}],
            }
        ]
    }
    assert _project_diplomacy_outcomes(saved) == [
        {
            "phase": "S1915M",
            "power": "AUSTRIA",
            "orders": ["A VIE H"],
            "results": [{"order": "A VIE H", "result": "success", "order_type": "hold"}],
            "visibility": "public",
        }
    ]


def test_they_sing_replay_commitment_ignores_nondeterministic_log_metadata(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    stable_turn = {
        "type": "turn_completed",
        "turn": 1,
        "trace": {"post_state_hash": "a" * 64},
    }
    left.write_text(
        "\n".join(
            [
                json.dumps({"type": "session_created", "timestamp": "first"}),
                json.dumps(stable_turn),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    right.write_text(
        "\n".join(
            [
                json.dumps({"type": "session_created", "timestamp": "second"}),
                json.dumps(stable_turn),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert _they_sing_turn_commitment(left) == _they_sing_turn_commitment(right)
