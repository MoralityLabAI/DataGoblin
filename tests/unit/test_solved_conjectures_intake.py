"""Synthetic repository smoke checks for the pinned Lean intake."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from stage_solved_conjectures import stage, triage  # noqa: E402


def test_pinned_intake(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()

    git("init", "-q")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    for rel, body in {
        "lean/A.lean": "theorem a : True := by trivial\n",
        "lean/B.lean": "theorem a : True := by trivial\n",
        "lean/FranklData.lean": "def x := 1\n",
        "docs/archive.lean": "def y := 2\n",
    }.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "fixture")
    pin, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    out = tmp_path / "first"
    result = stage(repo, out, [], pin=pin, expected_tree=tree, expected_counts=None)
    assert result["primary_paths"] == 3
    assert result["distinct_content_sha256"] == 2
    assert result["duplicates"] == {"within_source_exact": 1}
    assert result["training_ready"] is False
    rows = [json.loads(line) for line in (out / "candidates.jsonl").read_text().splitlines()]
    assert all(row["split"] is None and row["disposition"] == "QUARANTINE" for row in rows)
    assert result["review_queue_files"] == 2
    result2 = stage(
        repo, tmp_path / "second", [out / "candidates.jsonl"],
        pin=pin, expected_tree=tree, expected_counts=None,
    )
    assert result2["duplicates"] == {"reference_exact": 3}
    assert result2["review_queue_files"] == 0
    with pytest.raises(ValueError, match="tree mismatch"):
        stage(repo, tmp_path / "wrong", [], pin=pin, expected_tree="0" * 40)
    with pytest.raises(FileExistsError):
        stage(repo, out, [], pin=pin, expected_tree=tree, expected_counts=None)


def test_triage():
    assert triage("lean/Conjectures/LonelinessData.lean", 10) == (
        "certificate_family_review", "Loneliness"
    )
    assert triage("lean/Conjectures/CKZUpstream/A.lean", 10)[0] == "imported_rights_review"
    assert triage("lean/Other.lean", 100001)[0] == "large_file_review"
