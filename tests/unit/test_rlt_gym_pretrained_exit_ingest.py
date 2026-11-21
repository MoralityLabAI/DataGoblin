"""No-network fixture checks for BPD TRM archive ingestion and train-only custody."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from datagoblin.serialization import canonical_json
from datagoblin.training_gym.rlt_gym.pretrained_exit_ingest import (
    FEATURES, MODEL_SHA, _digest, ingest,
)


def _solution(index: int) -> str:
    return "".join(
        str(((r * 3 + r // 3 + c + index) % 9) + 1)
        for r in range(9) for c in range(9)
    )


def _fixture(folder: Path) -> Path:
    folder.mkdir(parents=True)
    source_sha = "f" * 64
    sft, future, repairs = [], [], []
    for index in range(2):
        solution = _solution(index)
        q = "." + solution[1:]
        first = solution if index == 0 else "1" * 81
        base = {
            "root_id": f"train-sudoku-{100 + index}",
            "task_digest": _digest({"question": q}),
            "model_sha256": MODEL_SHA,
            "family": "sudoku9", "split": "train",
            "training_eligible": True,
            "teacher_hidden_state_access": False,
            "source_sha256": source_sha,
        }
        sft.append({
            **base, "schema_version": "rlt_gym.controller_sft.v1",
            "kind": "verified_stop_decision",
            "input": {
                "task_question": q,
                "first_candidate": first,
                "logits_only_feature_names": list(FEATURES),
                "logits_only_features": [float(index)] * len(FEATURES),
            },
            "target": {"decision": "stop" if index == 0 else "continue"},
            "verification": {"first_pass_verified": index == 0},
        })
        future.append({
            **base, "schema_version": "rlt_gym.controller_outcome.v1",
            "kind": "future_verifier_outcome",
            "first_pass_verified": index == 0,
            "second_pass_verified": True,
            "second_pass_rescue": index == 1,
            "observed_second_candidate": solution,
        })
        if index == 1:
            repairs.append({
                **base, "schema_version": "rlt_gym.controller_preference.v1",
                "kind": "verified_repair",
                "input": {"task_question": q},
                "chosen": {"candidate": solution},
                "rejected": {"candidate": first},
                "verification": {"chosen": True, "rejected": False},
            })
    paths = {
        "controller_sft.jsonl": sft,
        "outcomes.jsonl": future,
        "repairs.jsonl": repairs,
    }
    bytes_by_name = {
        name: ("".join(canonical_json(row) + "\n" for row in rows)).encode()
        for name, rows in paths.items()
    }
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in bytes_by_name.items()}
    manifest = {
        "schema_version": "rlt_gym.pretrained_trm_train_export.v1",
        "model_sha256": MODEL_SHA, "train_only": True,
        "development_rows_exported": 0,
        "training_root_count": 2, "verified_repair_pairs": 1,
        "source_train_sha256": source_sha, "files_sha256": hashes,
    }
    for name, data in bytes_by_name.items():
        (folder / name).write_bytes(data)
    (folder / "manifest.json").write_text(canonical_json(manifest))
    return folder


def test_zip_train_only_ingest(tmp_path: Path) -> None:
    source = _fixture(tmp_path / "source")
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for file in source.iterdir():
            stream.write(file, f"datagoblin_train/{file.name}")
    report = ingest(archive, tmp_path / "parsed")
    assert report["training_roots"] == 2
    assert report["verified_repairs"] == 1
    assert report["first_pass_correct"] == 1
    assert report["second_pass_correct"] == 2
    assert report["development_rows"] == 0
    content = (tmp_path / "parsed" / "controller_train.jsonl").read_text()
    assert '"action":"stop"' in content
    assert '"action":"continue"' in content
    assert '"training_eligible":true' in content


def test_directory_receipt_hash_tamper_rejected(tmp_path: Path) -> None:
    source = _fixture(tmp_path / "source")
    path = source / "controller_sft.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="corrupt"):
        ingest(source, tmp_path / "bad")


def test_cross_split_leakage_rejected(tmp_path: Path) -> None:
    source = _fixture(tmp_path / "source")
    path = source / "controller_sft.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]["split"] = "development"
    text = "".join(canonical_json(x) + "\n" for x in rows)
    path.write_text(text)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"][path.name] = hashlib.sha256(text.encode()).hexdigest()
    manifest_path.write_text(canonical_json(manifest))
    with pytest.raises(ValueError, match="custody"):
        ingest(source, tmp_path / "bad")


def test_missing_repair_and_wrong_causal_target_rejected(tmp_path: Path) -> None:
    source = _fixture(tmp_path / "source")
    repair_path = source / "repairs.jsonl"
    repair_path.write_text("")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["verified_repair_pairs"] = 0
    manifest["files_sha256"]["repairs.jsonl"] = hashlib.sha256(b"").hexdigest()
    manifest_path.write_text(canonical_json(manifest))
    with pytest.raises(ValueError, match="missing verified positive"):
        ingest(source, tmp_path / "bad")
