"""Source-level corpus-audit invariants; runs offline without external repositories."""
from __future__ import annotations

from training.corpus_intake.audit_owned_math import (
    category,
    decl_rows,
    markdown_candidates,
    strip_lean_comments,
    split_role,
)


def token_stub(text: str) -> int:
    return len(text.split())


def test_generated_instances_remain_external_evidence() -> None:
    assert category("semaev-index-calculus", "lean/SemaevIC/InstanceD6m19R3s2.lean") == "generated_lean_instance"
    assert category("semaev-index-calculus", "lean/SemaevIC/ChainStandardD7.lean") == "generated_lean_template"
    assert category("semaev-index-calculus", "lean/SemaevIC/FreeExtension.lean") == "general_lean_development"


def test_nested_comments_are_excluded_from_declarations() -> None:
    source = """/- theorem counterfeit : False := by
      /- lemma second : False := by sorry -/
    -/
    theorem valid : True := by
      trivial
    -- theorem another : False := by sorry
    theorem incomplete : True := by
      sorry
    """
    stripped = strip_lean_comments(source)
    rows = decl_rows("toy", "toy.lean", source, token_stub)
    assert "counterfeit" not in stripped
    assert [x["name"] for x in rows] == ["valid", "incomplete"]
    assert [x["has_sorry_or_admit"] for x in rows] == [False, True]
    assert rows[0]["line"] == 4
    assert all(x["verification"] == "source_only_no_kernel_replay" for x in rows)


def test_episodic_sections_are_candidates_without_verifier_promotion() -> None:
    text = """# A
    Successful preliminary identity.
    ## Counterexample
    This failed because the proposed universal assumption is false.
    ## Construction
    We can construct a test.
    """
    rows = markdown_candidates("semaev", "notes/argument.md", text, token_stub)
    assert len(rows) == 1
    assert rows[0]["heading"].strip() == "## Counterexample"
    assert rows[0]["status"] == "heuristic_candidate_requires_human_review"


def test_cross_repo_identical_files_require_content_hashes() -> None:
    import hashlib
    first = hashlib.sha256(b"theorem q : True := by trivial").hexdigest()
    second = hashlib.sha256(b"theorem q : True := by trivial").hexdigest()
    assert first == second


def test_source_groups() -> None:
    assert category("EvalMath", "lean/FeedbackChannel/Example.lean") == "lean_research"
    assert category("BPD", "experiments/bpd_lean_curriculum_v2/Example.lean") == "lean_fixture_or_curriculum"
    assert category("semaev-index-calculus", "dialogue/2026-10-25-solver.md") == "research_dialogues"


def test_split_roles_and_name_free_theorem_fingerprint() -> None:
    assert split_role("fixtures/bpd_lean_curriculum_v2/train_negatives.lean") == "training_candidate"
    assert split_role("fixtures/bpd_lean_curriculum_v2/test_positives.lean") == "protected_eval"
    assert split_role("fixtures/bpd_lean_curriculum_v2/dev_negatives.lean") == "development"
    source = """theorem named_first : True := by
  trivial

  theorem renamed_second : True := by
    trivial
"""
    rows = decl_rows("EvalMath", "lean/Fixture.lean", source, token_stub)
    assert len(rows) == 2
    assert rows[0]["name_free_statement_sha256"] == rows[1]["name_free_statement_sha256"]
    assert rows[0]["statement_sha256"] != rows[1]["statement_sha256"]
