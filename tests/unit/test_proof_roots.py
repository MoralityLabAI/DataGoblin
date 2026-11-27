"""Unit tests for name-free candidate root clustering and split leakage checks."""
from __future__ import annotations

from training.corpus_intake.proof_roots import group_statements


def test_matches_across_named_theorem_copies() -> None:
    rows = [
        {
            "kind": "theorem", "source_lane": "lean_research", "repository": "r1",
            "path": "a.lean", "split_role": "training_candidate",
            "name_free_statement_sha256": "abc",
        },
        {
            "kind": "lemma", "source_lane": "lean_research", "repository": "r2",
            "path": "b.lean", "split_role": "protected_eval",
            "name_free_statement_sha256": "abc",
        },
        {
            "kind": "theorem", "source_lane": "general_lean_development", "repository": "r3",
            "path": "c.lean", "split_role": "unassigned",
            "name_free_statement_sha256": "different",
        },
    ]
    result = group_statements(rows)
    assert result["theorem_like_occurrences"] == 3
    assert result["name_free_unique_statement_syntaxes"] == 2
    assert result["repeated_syntax_clusters"] == 1
    assert result["train_vs_eval_overlap_candidates"] == 1
    assert result["duplicate_clusters"][0]["review_required"]


def test_ignores_non_theorems_and_missing_fingerprints() -> None:
    rows = [
        {"kind": "def", "name_free_statement_sha256": "x"},
        {"kind": "theorem", "source_lane": "lean_research", "split_role": "unassigned"},
    ]
    result = group_statements(rows)
    assert result["theorem_like_occurrences"] == 1
    assert result["name_free_unique_statement_syntaxes"] == 0
    assert result["train_vs_eval_overlap_candidates"] == 0


def test_no_leakage_when_train_and_train_share_proof() -> None:
    rows = [
        {"kind": "theorem", "split_role": "training_candidate", "name_free_statement_sha256": "x"},
        {"kind": "theorem", "split_role": "training_candidate", "name_free_statement_sha256": "x"},
    ]
    result = group_statements(rows)
    assert result["repeated_syntax_clusters"] == 1
    assert result["train_vs_eval_overlap_candidates"] == 0
