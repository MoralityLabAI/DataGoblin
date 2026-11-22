"""EvidenceID and lineage custody tests for the original PLN-style shadow ledger."""
from __future__ import annotations

import json

import pytest

from MeTTa.pln_lineage import EvidenceClaim, summarize_lineage, synthetic_demo


def test_repeated_identical_evidence_id_cannot_increase_confidence():
    once = [EvidenceClaim("a", "lineage-A", 1, True)]
    repeated = [*once, *once, *once]
    first = summarize_lineage(once)
    second = summarize_lineage(repeated)
    assert first["strength"] == second["strength"] == 1.0
    assert first["confidence"] == second["confidence"] == 0.333333
    assert first["stamp_sha256"] == second["stamp_sha256"]
    assert second["replayed_identical_evidence_ids"] == 2
    assert second["claim_ids_unique"] == 1


def test_same_lineage_multiple_evidence_ids_count_once():
    claims = [EvidenceClaim("replica1", "rootA", 2, True),
              EvidenceClaim("replica2", "rootA", 2, True)]
    score = summarize_lineage(claims)
    assert score["claim_ids_unique"] == 2
    assert score["declared_lineages"] == 1
    assert score["independent_support"] == 1
    assert score["confidence"] == 0.333333
    assert score["shared_lineage_repeats_ignored"] == 1


def test_independent_counterevidence_changes_strength_and_confidence():
    claims = [EvidenceClaim("a", "sourceA", 1, True),
              EvidenceClaim("b", "sourceB", 1, False)]
    score = summarize_lineage(claims)
    assert score["independent_support"] == 1
    assert score["independent_refute"] == 1
    assert score["strength"] == 0.5
    assert score["confidence"] == 0.5
    assert score["needs_review"] is False


def test_latest_revision_supersedes_older_claims_in_shadow_only():
    claims = [EvidenceClaim("old", "sourceA", 1, True),
              EvidenceClaim("new", "sourceA", 2, False)]
    score = summarize_lineage(claims)
    assert score["old_revisions_superseded"] == 1
    assert score["independent_support"] == 0
    assert score["independent_refute"] == 1
    assert score["strength"] == 0.0
    assert score["authorization"] == "none"


def test_conflicting_same_epoch_lineage_excluded_and_flagged():
    claims = [EvidenceClaim("a", "sourceA", 3, True),
              EvidenceClaim("b", "sourceA", 3, False),
              EvidenceClaim("c", "sourceB", 3, True)]
    report = summarize_lineage(claims)
    assert report["conflicted_lineages_excluded"] == 1
    assert report["independent_support"] == 1
    assert report["independent_refute"] == 0
    assert report["confidence"] == 0.333333
    assert report["needs_review"] is True


def test_no_evidence_keeps_strength_unknown():
    result = summarize_lineage([])
    assert result["strength"] is None
    assert result["confidence"] == 0.0
    assert result["needs_review"] is True
    assert result["native_pln_executed"] is False
    assert result["confidence_calibrated"] is False


def test_order_invariant_lineage_stamp_and_no_input_ids_in_result():
    claims = [EvidenceClaim("secret-A", "secret-lineage-A", 1, True),
              EvidenceClaim("secret-B", "secret-lineage-B", 1, False)]
    a = summarize_lineage(claims)
    b = summarize_lineage(list(reversed(claims)))
    assert a == b
    assert "secret" not in json.dumps(a)


def test_one_id_conflicting_content_refused():
    claims = [EvidenceClaim("same", "x", 1, True),
              EvidenceClaim("same", "x", 1, False)]
    with pytest.raises(ValueError, match="conflicting contents"):
        summarize_lineage(claims)


def test_strict_resource_horizon_and_type_contract():
    with pytest.raises(ValueError, match="claim horizon"):
        summarize_lineage([EvidenceClaim("a", "a", 1, True),
                           EvidenceClaim("b", "b", 1, False)], max_claims=1)
    with pytest.raises(ValueError, match="lineage horizon"):
        summarize_lineage([EvidenceClaim("a", "a", 1, True),
                           EvidenceClaim("b", "b", 1, False)], max_lineages=1)
    with pytest.raises(ValueError, match="confidence prior"):
        summarize_lineage([], prior_equivalent_count=0)
    with pytest.raises(ValueError, match="invalid computation"):
        summarize_lineage([], max_claims=0)
    with pytest.raises(ValueError, match="supports"):
        EvidenceClaim("a", "a", 1, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="revision"):
        EvidenceClaim("a", "a", -1, True)
    with pytest.raises(ValueError, match="typed evidence"):
        summarize_lineage([{"evidence_id": "a"}])  # type: ignore[list-item]


def test_fixture_has_duplicate_evidence_without_promotion():
    report = synthetic_demo()
    assert report["demo_only"] is True
    assert report["claim_ids_unique"] == 3
    assert report["replayed_identical_evidence_ids"] == 1
    assert report["declared_lineages"] == 2
    assert report["independent_support"] == 1
    assert report["independent_refute"] == 1
    assert report["strength"] == 0.5
    assert report["confidence"] == 0.5
    assert report["training_ready"] is False
    assert report["authorization"] == "none"
    assert report["native_pln_executed"] is False
    assert len(report["stamp_sha256"]) == 64
