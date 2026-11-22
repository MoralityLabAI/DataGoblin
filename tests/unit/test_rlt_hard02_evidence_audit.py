"""Independent native Mathlib evidence auditor rejection contracts."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.audit_rlt_hard02 import Reader, audit


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
