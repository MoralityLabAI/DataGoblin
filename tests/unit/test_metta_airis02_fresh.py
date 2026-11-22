"""Fresh AIRIS-02 source split and oracle-replay custody tests."""
from __future__ import annotations

import json

import pytest

from MeTTa.airis02_audit import audit_bundle
from MeTTa.airis02_fresh import (
    ARMS,
    DEV_RANGE,
    FROZEN_MOSES_PROGRAM,
    PUBLIC_TEST_RANGE,
    STEWARD_RANGE,
    TRAIN_RANGE,
    _frozen_motifs,
    build_bundle,
    build_row,
    canonical,
    partition,
    root_id,
    steward_root_manifest,
)
from MeTTa.collect import sha256_bytes
from MeTTa.sft_oracle import FAMILIES, OOD_FAMILY


def test_partition_registered_ranges_preserve_old_sft_and_ood():
    for family in FAMILIES:
        if family == OOD_FAMILY:
            with pytest.raises(ValueError, match="reserved"):
                partition(family, 512)
            continue
        for seed, expected in (
            (TRAIN_RANGE.start, "train_candidate"),
            (DEV_RANGE.start, "development"),
            (PUBLIC_TEST_RANGE.start, "public_test_reserved"),
            (STEWARD_RANGE.start, "steward_reserved"),
        ):
            assert partition(family, seed) == expected
            assert root_id(family, seed).startswith("metta-airis02-v1:")
        with pytest.raises(ValueError, match="seed outside"):
            partition(family, 42)


def test_steward_registry_contains_only_root_identity_hash():
    receipt = steward_root_manifest(count_per_family=4)
    assert receipt["root_count"] == 4 * (len(FAMILIES) - 1)
    assert receipt["labels_generated"] == 0
    assert receipt["worlds_opened"] == 0
    assert receipt["training_ready"] is False
    assert receipt["not_cryptographically_sealed"] is True
    assert all(x not in receipt for x in ("program", "source_text", "oracle", "labels"))


def test_four_arm_generators_reuse_single_source_root_and_zero_prior_sft():
    motifs = _frozen_motifs()
    assert motifs
    rows = [build_row("lineage", 512, arm, motifs) for arm in ARMS]
    assert len({row["root"] for row in rows}) == 1
    assert len({row["source_world_sha256"] for row in rows}) == 1
    assert len({row["arm"] for row in rows}) == len(ARMS)
    assert {row["presentation_number"] for row in rows} == set(range(len(ARMS)))
    assert all(row["training_ready"] is False for row in rows)
    assert all(row["oracle_next_edit"]["op"] in {"add", "remove", "stop"} for row in rows)
    assert FROZEN_MOSES_PROGRAM.key.startswith("or/")
    with pytest.raises(ValueError, match="reserved"):
        build_row("lineage", STEWARD_RANGE.start, "uniform", motifs)
    with pytest.raises(ValueError, match="reserved"):
        build_row("lineage", PUBLIC_TEST_RANGE.start, "uniform", motifs)


def test_full_audit_replays_all_train_and_public_dev_labels(tmp_path):
    manifest = build_bundle(tmp_path, train_roots_per_family=3,
                            dev_roots_per_family=2)
    audit = audit_bundle(tmp_path, full_replay=True)
    assert manifest["train_roots"] == 15
    assert manifest["development_roots"] == 10
    assert manifest["train_presentations_total"] == 15 * len(ARMS)
    assert audit["train_source_roots"] == 15
    assert audit["development_source_roots"] == 10
    assert audit["synthetic_research_fit_eligible"] is True
    assert audit["general_corpus_training_ready"] is False
    assert audit["public_test_labels_used"] == 0
    assert audit["steward_labels_used"] == 0
    assert all(n == 15 for n in audit["train_rows_per_arm"].values())
    assert all(n == 10 for n in audit["development_rows_per_arm"].values())
    reg = json.loads((tmp_path / "public_test_registry.json").read_text())
    assert reg["worlds_opened"] == 0 and reg["teacher_labels_written"] == 0
    assert reg["root_count"] == len(PUBLIC_TEST_RANGE) * (len(FAMILIES) - 1)


def test_tamper_protected_teacher_labels_even_if_manifest_rehashed(tmp_path):
    build_bundle(tmp_path, train_roots_per_family=2, dev_roots_per_family=1)
    file = tmp_path / "train_candidate.jsonl"
    rows = [json.loads(x) for x in file.read_text().splitlines() if x]
    rows[0]["oracle_next_edit"] = {"op": "stop", "id": "none"}
    file.write_text("".join(canonical(r) + "\n" for r in rows))
    manifest_file = tmp_path / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["files"][file.name] = sha256_bytes(file.read_bytes())
    manifest_file.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="oracle replay mismatch"):
        audit_bundle(tmp_path, full_replay=True)


def test_duplicate_arm_cannot_expand_root_training_presentations(tmp_path):
    build_bundle(tmp_path, train_roots_per_family=2, dev_roots_per_family=1)
    file = tmp_path / "train_candidate.jsonl"
    content = file.read_text()
    first = content.splitlines()[0]
    file.write_text(content + first + "\n")
    manifest_file = tmp_path / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["files"][file.name] = sha256_bytes(file.read_bytes())
    manifest_file.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="duplicate source-root"):
        audit_bundle(tmp_path, full_replay=False)


def test_tampered_steward_public_registry_fails_without_labels(tmp_path):
    build_bundle(tmp_path, train_roots_per_family=1, dev_roots_per_family=1)
    path = tmp_path / "steward_registry.json"
    data = json.loads(path.read_text())
    data["labels"] = {"secret": "answer"}
    path.write_text(json.dumps(data))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"][path.name] = sha256_bytes(path.read_bytes())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="registry contains"):
        audit_bundle(tmp_path, full_replay=False)
