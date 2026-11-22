"""Independent native Mathlib evidence auditor rejection contracts."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.audit_rlt_hard02 import Reader, audit, expected_menu_exhaustion, length_first_control


def test_reject_unsafe_relative_zip_paths(tmp_path: Path):
    path = tmp_path / "unsafe.zip"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("../private/source.py", b"bad")
    with pytest.raises(ValueError, match="unsafe"):
        Reader(path)


def test_reject_duplicate_zip_members(tmp_path: Path):
    path = tmp_path / "dup.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("report.json", "{}")
            archive.writestr("report.json", "{}")
    with pytest.raises(ValueError, match="duplicate"):
        Reader(path)


def test_reject_unpinned_mathlib_claims(tmp_path: Path):
    path = tmp_path / "report.zip"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("report.json", json.dumps({
            "schema": "datagoblin.rlt_hard02_mathlib_first_dev.v1",
            "environment": {"mathlib_commit": "0" * 40},
        }))
    with pytest.raises(ValueError, match="source pin"):
        audit(path)


def test_reject_missing_source_and_proof_digests(tmp_path: Path):
    path = tmp_path / "evidence"
    path.mkdir()
    (path / "report.json").write_text(json.dumps({
        "schema": "datagoblin.rlt_hard02_mathlib_first_dev.v1",
        "environment": {
            "mathlib_commit": "809c3fb3b5c8f5d7dace56e200b426187516535a",
            "lean_version": "Lean (version 4.12.0)",
        },
    }))
    with pytest.raises(ValueError, match="digest manifest"):
        audit(path)


def test_length_first_uses_only_verified_executed_candidate_receipts():
    source = {"menu": ["rfl", "exact True.intro", "intro x\nexact some_very_long_witness x"]}
    row = {"attempts": [
        {"candidate_id": 0, "receipt": {"passed": False}},
        {"candidate_id": 2, "receipt": {
            "passed": True, "code": "kernel_certified",
            "evidence": {"kernel_ok": True},
        }},
    ]}
    result = length_first_control(source, row)
    assert result["observed"] and result["certified"]
    assert result["kernel_verified"]
    assert result["candidate_id"] == 2
    assert result["length"] > result["largest_other_length"]

    unknown = length_first_control(source, {"attempts": [
        {"candidate_id": 0, "receipt": {"passed": True}},
    ]})
    assert not unknown["observed"]
    assert unknown["certified"] is None


def test_length_first_tie_is_index_deterministic():
    source = {"menu": ["alpha", "bravo", "charl"]}
    result = length_first_control(source, {"attempts": [
        {"candidate_id": 0, "receipt": {"passed": False}}
    ]})
    assert result["candidate_id"] == 0
    assert result["observed"]
    assert result["certified"] is False


def test_preregistered_exhausted_root_indices_are_parsed_from_source_only():
    for index in range(8):
        source = {"public_task": {
            "source_file": f"source_roots/train/MathlibRoot_{index:02d}_" + "a" * 20 + ".lean",
        }}
        assert expected_menu_exhaustion(source) == (index % 4 == 3)
    with pytest.raises(ValueError, match="source root"):
        expected_menu_exhaustion({"public_task": {"source_file": "invalid.lean"}})
