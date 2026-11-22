"""Integrity rejection checks for OxiBonsai real-inference/native-Lean evidence."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.audit_rlt_hard02f import Reader, audit


def test_reject_unsafe_zip_members_and_duplicate_archive_entries(tmp_path: Path):
    escape = tmp_path / "escape.zip"
    with ZipFile(escape, "w", ZIP_DEFLATED) as z:
        z.writestr("../secret_model.gguf", b"unsafe")
    with pytest.raises(ValueError, match="unsafe"):
        Reader(escape)

    dup = tmp_path / "dup.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(dup, "w", ZIP_DEFLATED) as z:
            z.writestr("report.json", "{}")
            z.writestr("report.json", "{}")
    with pytest.raises(ValueError, match="duplicate"):
        Reader(dup)


def test_reject_model_schema_and_checkpoint_identity_tamper(tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"schema": "fixture"}))
    with pytest.raises(ValueError, match="schema"):
        audit(tmp_path)
    report.write_text(json.dumps({
        "schema": "datagoblin.rlt_hard02f_oxibonsai_q1_native.v1",
        "upstream_mathlib_commit": "809c3fb3b5c8f5d7dace56e200b426187516535a",
        "oxibonsai_repo_commit": "1d10c48e14754ff6999158f8fa13c7c92b40b0dc",
        "model_identity": {"model_id": "prism-ml/Bonsai-8B-gguf", "model_sha256": "a" * 64},
    }))
    (tmp_path / "model_binding.json").write_text(json.dumps({
        "model_id": "prism-ml/Bonsai-8B-gguf", "model_sha256": "a" * 64,
    }))
    with pytest.raises(ValueError, match="checkpoint"):
        audit(tmp_path)


def test_reader_rejects_parent_directory_path(tmp_path: Path):
    r = Reader(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        r.read("../original_checkpoint.gguf")
    r.close()
