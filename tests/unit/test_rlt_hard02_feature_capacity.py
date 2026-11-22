"""Source-only feature capacity audit for the frozen matched Mathlib cohort."""

from __future__ import annotations

from datagoblin.training_gym.rlt_gym.hard02_mathlib import SPLITS
from scripts.audit_rlt_hard02_feature_capacity import inspect


def test_matched_action_semantics_are_unidentifiable_without_position():
    result = inspect()
    assert result["source_seed"] == 2026112203
    assert result["all_32_sources_position_only"] is True
    assert len(result["per_root"]) == 32
    for split in SPLITS:
        cell = result["by_split"][split]
        assert cell["position_only_roots"] == 8
        assert cell["semantically_distinguishable_roots"] == 0
        assert cell["equal_candidate_length_roots"] == 8


def test_proof_candidate_rows_are_separately_indexed_but_lexically_collapsed():
    result = inspect()
    for row in result["per_root"]:
        assert row["full_distinct_action_signatures"] == 5
        assert row["nonposition_distinct_action_signatures"] == 1
        assert row["feature_dimension"] > 6
    assert result["position_column"] == 6
    assert result["vocabulary_size"] > 0


def test_audit_is_reproducible_and_source_seed_changes_root_ids():
    a = inspect()
    b = inspect()
    fresh = inspect(seed=2026112204)
    assert a == b
    assert {r["root_id"] for r in a["per_root"]}.isdisjoint(
        r["root_id"] for r in fresh["per_root"]
    )
    assert all(fresh["by_split"][s]["roots"] == 8 for s in SPLITS)
