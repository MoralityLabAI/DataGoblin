"""Replay/tamper tests for AIRIS paired synthetic world receipts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from MeTTa.airis_counterfactual import canonical, run_experiment
from MeTTa.airis_receipts import audit_bundle


def _fixture(directory: Path):
    report, rows = run_experiment(roots_per_family=48, case_limit=8,
                                  min_support=2, top_k=6, seed=73)
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    (directory / "internal_confirmation_only.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in rows)
    )
    return report, rows


def test_full_replay_covers_each_root_and_preserves_source_identity(tmp_path):
    report, rows = _fixture(tmp_path)
    result = audit_bundle(tmp_path, full_replay=True)
    assert result["valid"] is True
    assert result["oracle_replayed"] is True
    assert result["paired_cases"] == len(rows) == report["actual_paired_cases"] == 8
    assert result["new_sft_presentations"] == 0
    assert result["cross_arm_root_ids_matched"] is True
    assert result["training_ready"] is False
    assert result["receipt_sha256"] == report["source_root_receipts_sha256"]


def test_tampered_target_label_detected_without_oracle(tmp_path):
    _fixture(tmp_path)
    file = tmp_path / "internal_confirmation_only.jsonl"
    records = [json.loads(line) for line in file.read_text().splitlines()]
    records[0]["guided"]["target_violation_changed"] = not records[0]["guided"]["target_violation_changed"]
    file.write_text("".join(canonical(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="digest mismatch"):
        audit_bundle(tmp_path, full_replay=False)


def test_forged_hash_with_tampered_case_detected_by_full_replay(tmp_path):
    report, _ = _fixture(tmp_path)
    file = tmp_path / "internal_confirmation_only.jsonl"
    records = [json.loads(line) for line in file.read_text().splitlines()]
    records[0]["guided"]["world_sha256"] = "0" * 64
    file.write_text("".join(canonical(r) + "\n" for r in records))
    from MeTTa.collect import sha256_bytes

    report["source_root_receipts_sha256"] = sha256_bytes(
        "\n".join(canonical(r) for r in records).encode()
    )
    (tmp_path / "summary.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="source-root replay differs"):
        audit_bundle(tmp_path, full_replay=True)


def test_no_training_permission_can_be_smuggled_into_case(tmp_path):
    report, _ = _fixture(tmp_path)
    file = tmp_path / "internal_confirmation_only.jsonl"
    records = [json.loads(line) for line in file.read_text().splitlines()]
    records[0]["training_ready"] = True
    file.write_text("".join(canonical(r) + "\n" for r in records))
    from MeTTa.collect import sha256_bytes

    report["source_root_receipts_sha256"] = sha256_bytes(
        "\n".join(canonical(r) for r in records).encode()
    )
    (tmp_path / "summary.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="case-level admission"):
        audit_bundle(tmp_path, full_replay=False)


def test_unmatched_or_reserved_root_fails(tmp_path):
    report, _ = _fixture(tmp_path)
    file = tmp_path / "internal_confirmation_only.jsonl"
    records = [json.loads(line) for line in file.read_text().splitlines()]
    records[0]["root"] = "metta-sft-v1:conflict:000000"
    file.write_text("".join(canonical(r) + "\n" for r in records))
    from MeTTa.collect import sha256_bytes

    report["source_root_receipts_sha256"] = sha256_bytes(
        "\n".join(canonical(r) for r in records).encode()
    )
    (tmp_path / "summary.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="crosses internal confirmation"):
        audit_bundle(tmp_path, full_replay=False)
