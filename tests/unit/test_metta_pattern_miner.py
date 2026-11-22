"""CPU contracts for MeTTa Pattern Miner-inspired synthetic curation proposals."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from MeTTa.looped_sft import root_id, split_for
from MeTTa.pattern_miner import (
    REPORT_SCHEMA,
    generate_observations,
    make_report,
    mine_motifs,
    render_advisory_metta,
    structural_features,
)
from MeTTa.sft_oracle import FAMILIES, OOD_FAMILY, first_state, make_task


def test_structural_features_use_only_public_facts():
    task = make_task("dependency", 7)
    initial = first_state(task, 7)
    features = structural_features(task, initial)
    assert tuple(sorted(set(features))) == features
    assert any(x.startswith("selected_") for x in features)
    assert "optimal_rank" not in json.dumps(features)
    assert "teacher" not in json.dumps(features)


def test_train_only_one_root_one_observation():
    observations = generate_observations(roots_per_family=24)
    assert observations
    assert len({x.root for x in observations}) == len(observations)
    assert {x.family for x in observations} <= set(FAMILIES) - {OOD_FAMILY}
    assert all(x.split == "train_candidate" for x in observations)
    for item in observations:
        family, seed = item.root.split(":")[1:]
        assert item.root == root_id(family, int(seed))
        assert split_for(family, int(seed)) == "train_candidate"


def test_miner_refuses_duplicate_and_reserved_lineages():
    observations = generate_observations(roots_per_family=14)
    with pytest.raises(ValueError, match="repeated source lineage"):
        mine_motifs([*observations, observations[0]], min_support=2)
    with pytest.raises(ValueError, match="reserved root"):
        mine_motifs([replace(observations[0], split="test_reserved"), *observations[1:]], min_support=2)
    with pytest.raises(ValueError, match="reserved root"):
        mine_motifs([replace(observations[0], family=OOD_FAMILY), *observations[1:]], min_support=2)


def test_bounded_motifs_are_deterministic_and_training_only():
    observations = generate_observations(roots_per_family=36)
    one = mine_motifs(observations, min_support=2, top_k=8)
    two = mine_motifs(observations, min_support=2, top_k=8)
    assert one == two
    assert one["schema_version"] == REPORT_SCHEMA
    assert one["root_count"] == len(observations)
    assert one["discovery_roots"] + one["confirmation_roots"] == len(observations)
    assert one["source_split"] == "train_candidate"
    assert one["reserved_root_labels_read"] == 0
    assert one["new_sft_presentations_created"] == 0
    assert one["training_ready"] is False
    assert one["native_hyperon_miner_executed"] is False
    assert all(1 <= len(r["features"]) <= 2 for r in one["candidates"])
    assert all(len(r["source_lineage_sha256"]) == 64 for r in one["candidates"])
    assert all(r["candidate_only"] is True for r in one["candidates"])


def test_source_lineage_stamp_deduplicates_presentations():
    obs = generate_observations(roots_per_family=40)
    one = mine_motifs(obs, min_support=3, top_k=8)
    reshuffled = list(reversed(obs))
    two = mine_motifs(reshuffled, min_support=3, top_k=8)
    assert one == two
    for motif in one["candidates"]:
        assert motif["discovery"]["support"] <= one["discovery_roots"]


def test_fake_label_and_readiness_cannot_enter_metta_export():
    report = mine_motifs(generate_observations(roots_per_family=18), min_support=2)
    metta = render_advisory_metta(report)
    assert metta.startswith("; Advisory synthetic motif")
    assert "(ProposedMotif" in metta or not report["candidates"]
    assert "optimal_subset" not in metta
    assert "verifier_receipt" not in metta
    with pytest.raises(ValueError, match="invalid or released"):
        render_advisory_metta({**report, "training_ready": True})


def test_full_oracle_corpus_remains_separate_from_proposals():
    report = make_report(roots_per_family=24, min_support=2, top_k=10)
    assert report["roots_per_family"] == 24
    assert report["generation_hashes"]["miner_sha256"]
    assert report["generation_hashes"]["task_oracle_sha256"]
    assert report["generation_hashes"]["source_split_sha256"]
    assert report["source"] == "synthetic_python_oracle_initial_state"


def test_reject_invalid_miner_budgets_and_unbounded_horizon():
    with pytest.raises(ValueError, match="frozen"):
        generate_observations(roots_per_family=513)
    with pytest.raises(ValueError, match="invalid mining"):
        mine_motifs(generate_observations(roots_per_family=24), min_support=1)
