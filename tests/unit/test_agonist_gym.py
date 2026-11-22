"""Deterministic AGN-01 checks: generation, independent labels, leakage and replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Agonist.gym import (
    audit,
    fixture_reply,
    make_requests,
    oracle_action,
    read_jsonl,
    replay,
    stage,
    validate_tastes,
    verify_pair,
    write_jsonl,
)


def _taste(name: str = "revision", split: str = "train", rights: str = "project_original") -> dict:
    return {
        "taste_id": name,
        "family_id": f"family-{name}",
        "idea": "An old source can reverse a justified decision after a rule changes.",
        "source_kind": "fixture",
        "source_ref": "original unittest fixture",
        "rights": rights,
        "split": split,
    }


def _prepare(tmp_path: Path, seeds: list[dict] | None = None) -> tuple[Path, Path]:
    tastes_path = tmp_path / "tastes.jsonl"
    write_jsonl(tastes_path, seeds or [_taste(), _taste("second", split="dev")])
    stage(tastes_path, tmp_path / "queue")
    requests = tmp_path / "queue" / "teacher_requests.jsonl"
    replies = tmp_path / "teacher_replies.jsonl"
    write_jsonl(replies, [fixture_reply(row) for row in read_jsonl(requests)])
    return requests, replies


def test_verified_causal_flip_and_fixed_cue() -> None:
    requests = make_requests([_taste()])
    reply = fixture_reply(requests[0])
    labels = verify_pair(reply["world_a"], reply["world_b"], reply["intervention"])
    assert set(labels) == {"deploy", "hold"}
    assert oracle_action(reply["world_a"]) == labels[0]
    assert reply["world_a"]["surface_cue"] == reply["world_b"]["surface_cue"]
    assert reply["world_a"]["scenario"] == reply["world_b"]["scenario"]


def test_stage_import_and_exact_replay(tmp_path: Path) -> None:
    requests, replies = _prepare(tmp_path)
    report = audit(requests, replies, tmp_path / "archive")
    assert report["accepted_pairs"] == 2
    assert report["rejected_replies"] == 0
    assert report["evaluation"] == {
        "oracle_correct": 4,
        "surface_cue_correct": 2,
        "decisions": 4,
        "learner_uplift_measured": False,
    }
    assert replay(tmp_path / "archive") == {"verified_pairs": 2, "train_candidates": 2}
    views = read_jsonl(tmp_path / "archive" / "train_candidates.jsonl")
    assert all(row["training_ready"] is False for row in views)
    assert all(row["split"] == "train" for row in views)
    public = read_jsonl(tmp_path / "archive" / "public_worlds.jsonl")
    assert all("actions" not in row and "gold_action" not in row for row in public)
    assert all(row["world_digest"] for row in public)
    with pytest.raises(FileExistsError):
        audit(requests, replies, tmp_path / "archive")


def test_queue_is_inert_and_context_bounded(tmp_path: Path) -> None:
    requests, _ = _prepare(tmp_path)
    rows = read_jsonl(requests)
    assert all(row["model_role"] == "local_inkling_2_mini_world_builder" for row in rows)
    assert all(row["hard_context_cap_tokens"] == 16_384 for row in rows)
    assert not (tmp_path / "queue" / "verifier_receipts.jsonl").exists()
    assert replay(tmp_path / "queue") == {"verified_pairs": 0, "queue_rows": 2}
    with pytest.raises(ValueError, match="context cap"):
        make_requests([_taste()], 16_385)


def test_story_change_is_rejected(tmp_path: Path) -> None:
    requests, replies = _prepare(tmp_path, [_taste()])
    [candidate] = read_jsonl(replies)
    candidate["world_b"]["scenario"] = "An entirely different scene about an unrelated world."
    write_jsonl(replies, [candidate])
    report = audit(requests, replies, tmp_path / "invalid")
    assert report["accepted_pairs"] == 0
    assert report["rejected_replies"] == 1
    assert replay(tmp_path / "invalid")["verified_pairs"] == 0
    assert "story surface" in read_jsonl(tmp_path / "invalid" / "rejected.jsonl")[0]["reason"]


def test_causal_change_without_action_flip_is_rejected(tmp_path: Path) -> None:
    requests, replies = _prepare(tmp_path, [_taste()])
    [candidate] = read_jsonl(replies)
    candidate["world_b"]["state"] = dict(candidate["world_a"]["state"])
    candidate["world_b"]["state"]["available"] = 5
    candidate["intervention"] = {"field": "available", "from": 4, "to": 5}
    write_jsonl(replies, [candidate])
    report = audit(requests, replies, tmp_path / "invalid")
    assert report["accepted_pairs"] == 0
    assert "did not reverse" in read_jsonl(tmp_path / "invalid" / "rejected.jsonl")[0]["reason"]


def test_extra_answer_field_and_bool_as_integer_are_rejected() -> None:
    reply = fixture_reply(make_requests([_taste()])[0])
    reply["world_a"]["gold_action"] = "deploy"
    with pytest.raises(ValueError, match="exactly"):
        verify_pair(reply["world_a"], reply["world_b"], reply["intervention"])
    reply = fixture_reply(make_requests([_taste()])[0])
    reply["world_a"]["state"]["available"] = True
    with pytest.raises(ValueError, match="integer"):
        verify_pair(reply["world_a"], reply["world_b"], reply["intervention"])


def test_training_rights_and_dev_custody(tmp_path: Path) -> None:
    requests, replies = _prepare(tmp_path, [_taste(rights="reference_only"), _taste("dev", split="dev")])
    report = audit(requests, replies, tmp_path / "archive")
    assert report["accepted_pairs"] == 2
    assert read_jsonl(tmp_path / "archive" / "train_candidates.jsonl") == []


def test_root_and_family_split_boundaries() -> None:
    with pytest.raises(ValueError, match="duplicate taste"):
        validate_tastes([_taste(), _taste()])
    dev = _taste("other", split="dev")
    dev["family_id"] = "family-revision"
    with pytest.raises(ValueError, match="family crosses"):
        validate_tastes([_taste(), dev])
    with pytest.raises(ValueError, match="train and dev"):
        validate_tastes([_taste(split="sealed")])


def test_replay_detects_corrupted_artifact(tmp_path: Path) -> None:
    requests, replies = _prepare(tmp_path, [_taste()])
    audit(requests, replies, tmp_path / "archive")
    artifact = tmp_path / "archive" / "verifier_receipts.jsonl"
    artifact.write_text(artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        replay(tmp_path / "archive")


def test_unknown_request_preserved_as_rejection(tmp_path: Path) -> None:
    requests, replies = _prepare(tmp_path, [_taste()])
    [candidate] = read_jsonl(replies)
    candidate["request_id"] = "not-in-queue"
    write_jsonl(replies, [candidate])
    report = audit(requests, replies, tmp_path / "archive")
    assert report["rejected_replies"] == 1
    rejected = read_jsonl(tmp_path / "archive" / "rejected.jsonl")
    assert rejected[0]["request_id"] == "not-in-queue"
    assert len(rejected[0]["reply_sha256"]) == 64


def test_fixture_tastes_are_valid_jsonl() -> None:
    path = Path("Agonist/fixtures/tastes.jsonl")
    rows = read_jsonl(path)
    validate_tastes(rows)
    assert len(rows) == 3
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["taste_id"] == "stale_evidence"
