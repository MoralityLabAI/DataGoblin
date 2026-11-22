"""Tamper/replay-contract rejection tests for the RLT-HARD-02E auditor."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.audit_rlt_hard02e import Reader, audit


def test_reject_wrong_schema(tmp_path: Path):
    (tmp_path / "report.json").write_text(json.dumps({"schema": "fake"}))
    with pytest.raises(ValueError, match="schema"):
        audit(tmp_path)


def test_reject_wrong_mathlib_source(tmp_path: Path):
    (tmp_path / "report.json").write_text(json.dumps({
        "schema": "datagoblin.rlt_hard02e_verified_list_stop.v1",
        "upstream": {"mathlib_commit": "0" * 40, "lean_version": "4.12.0"},
    }))
    with pytest.raises(ValueError, match="Mathlib"):
        audit(tmp_path)


def test_reject_unsafe_zip_paths_and_duplicates(tmp_path: Path):
    evil = tmp_path / "evil.zip"
    with ZipFile(evil, "w", ZIP_DEFLATED) as archive:
        archive.writestr("../private/report.json", "{}")
    with pytest.raises(ValueError, match="unsafe"):
        Reader(evil)

    duplicate = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(duplicate, "w", ZIP_DEFLATED) as archive:
            archive.writestr("report.json", "{}")
            archive.writestr("report.json", "{}")
    with pytest.raises(ValueError, match="duplicate"):
        Reader(duplicate)


def test_reject_missing_hashes_and_model_files(tmp_path: Path):
    (tmp_path / "report.json").write_text(json.dumps({
        "schema": "datagoblin.rlt_hard02e_verified_list_stop.v1",
        "upstream": {
            "mathlib_commit": "809c3fb3b5c8f5d7dace56e200b426187516535a",
            "lean_version": "Lean (version 4.12.0)",
        },
        "verifier_sha256": "a" * 64,
    }))
    with pytest.raises(ValueError, match="manifest"):
        audit(tmp_path)


def test_reader_refuses_parent_path_access(tmp_path: Path):
    reader = Reader(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        reader.read("../outside.json")
    reader.close()
