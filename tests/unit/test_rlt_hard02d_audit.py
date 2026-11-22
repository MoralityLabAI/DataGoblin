"""Rejection tests for the independent RLT-HARD-02D kernel evidence auditor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_rlt_hard02d_list import audit


def _report(directory: Path, **kwargs):
    directory.mkdir(parents=True, exist_ok=True)
    data = {
        "schema": "datagoblin.rlt_hard02d_strict_list_qa.v1",
        "upstream_commit": "809c3fb3b5c8f5d7dace56e200b426187516535a",
        "environment": {
            "mathlib_commit": "809c3fb3b5c8f5d7dace56e200b426187516535a",
            "lean_version": "Lean (version 4.12.0)",
        },
    }
    data.update(kwargs)
    (directory / "report.json").write_text(json.dumps(data))


def test_wrong_native_evidence_schema_is_rejected(tmp_path: Path):
    _report(tmp_path / "wrong", schema="unverified")
    with pytest.raises(ValueError, match="schema"):
        audit(tmp_path / "wrong")


def test_mismatched_source_and_compiler_are_rejected(tmp_path: Path):
    _report(tmp_path / "sha", upstream_commit="a" * 40)
    with pytest.raises(ValueError, match="revision"):
        audit(tmp_path / "sha")
    _report(tmp_path / "lean", environment={
        "mathlib_commit": "809c3fb3b5c8f5d7dace56e200b426187516535a",
        "lean_version": "Lean (version 4.22.0)",
    })
    with pytest.raises(ValueError, match="toolchain"):
        audit(tmp_path / "lean")


def test_missing_or_tampered_file_sha256_manifest_is_rejected(tmp_path: Path):
    _report(tmp_path / "missing")
    with pytest.raises(ValueError, match="manifest"):
        audit(tmp_path / "missing")
    _report(tmp_path / "altered", files_sha256={"evidence.json": "0" * 64},
            source_sha256={}, certified_source_sha256={})
    (tmp_path / "altered" / "evidence.json").write_text("{}")
    with pytest.raises(ValueError, match="tampered"):
        audit(tmp_path / "altered")
