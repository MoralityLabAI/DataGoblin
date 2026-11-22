"""Tests for original bounded symbolic program search on train-only MeTTa roots."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from MeTTa.moses_control import (
    SCHEMA,
    Program,
    _validate_rows,
    advisory_metta,
    confusion,
    evolve,
    make_program,
)
from MeTTa.pattern_miner import generate_observations


def test_boolean_grammar_and_canonical_program_size():
    program = make_program("and", ("x", "!y"))
    assert program.nodes == 3
    assert program.predict(("x",)) is True
    assert program.predict(("x", "y")) is False
    assert Program("true").predict(()) is True
    assert Program("false").predict(("x",)) is False
    assert Program("lit", ("!x",)).predict(()) is True
    assert make_program("or", ("x", "!y")).predict(("y",)) is False
    with pytest.raises(ValueError, match="arity"):
        Program("and", ("x",))
    with pytest.raises(ValueError, match="canonical"):
        Program("or", ("z", "a"))
    with pytest.raises(ValueError, match="unknown program"):
        Program("eval", ())


def test_train_root_custody_rejects_reserved_and_forged_split():
    obs = generate_observations(roots_per_family=28)
    assert obs
    d, c = _validate_rows(obs)
    assert d and c
    assert not ({r.root for r in d} & {r.root for r in c})
    with pytest.raises(ValueError, match="source roots"):
        _validate_rows([*obs, obs[0]])
    with pytest.raises(ValueError, match="reserved"):
        _validate_rows([replace(obs[0], split="test_reserved"), *obs[1:]])
    with pytest.raises(ValueError, match="source-root"):
        _validate_rows([replace(obs[0], root="fake:source:root"), *obs[1:]])


def test_full_small_evolution_deterministic_and_split_safe():
    obs = generate_observations(roots_per_family=36)
    a = evolve(obs, seed=23, population_size=20, generations=6)
    b = evolve(list(reversed(obs)), seed=23, population_size=20, generations=6)
    assert a == b
    assert a["schema_version"] == SCHEMA
    assert a["discovery_roots"] + a["confirmation_roots"] == len(obs)
    assert a["reserved_root_labels_accessed"] == 0
    assert a["source_split"] == "train_candidate"
    assert a["new_training_presentations"] == 0
    assert a["training_ready"] is False
    assert a["upstream_metta_moses_executed"] is False
    assert a["model_trained"] is False
    assert a["best_program"]["nodes"] <= 3
    assert a["candidate_programs_evaluated"] >= 20
    assert len(a["discovery_lineage_sha256"]) == 64
    assert len(a["confirmation_lineage_sha256"]) == 64
    assert a["discovery_lineage_sha256"] != a["confirmation_lineage_sha256"]
    assert len(a["history"]) == 6
    assert all(0 <= x["best_discovery_balanced_accuracy"] <= 1 for x in a["history"])


def test_frozen_controls_use_same_confirmation_partition():
    obs = generate_observations(roots_per_family=32)
    report = evolve(obs, seed=11, population_size=16, generations=5)
    n = report["confirmation_roots"]
    for arm in ("internal_confirmation",):
        stats = report[arm]
        assert stats["roots"] == n
        assert stats["tp"] + stats["tn"] + stats["fp"] + stats["fn"] == n
    for arm in ("majority_baseline", "cost_pressure_baseline"):
        stats = report[arm]["metrics"]
        assert stats["roots"] == n
        assert stats["tp"] + stats["tn"] + stats["fp"] + stats["fn"] == n
    assert isinstance(report["confirmation_minus_majority_bacc"], float)


def test_syntactic_metta_program_is_advisory():
    report = evolve(generate_observations(roots_per_family=26), seed=2,
                    population_size=16, generations=3)
    text = advisory_metta(report)
    assert "(= (ProposeInvalid $world)" in text
    assert "unverified by native interpreter" in text
    assert "training_ready" not in text
    assert "oracle" in text
    with pytest.raises(ValueError, match="safe frozen"):
        advisory_metta({**report, "training_ready": True})


def test_invalid_search_bounds_and_no_reserved_labels():
    observations = generate_observations(roots_per_family=25)
    with pytest.raises(ValueError, match="bounded"):
        evolve(observations, population_size=2)
    with pytest.raises(ValueError, match="bounded"):
        evolve(observations, generations=65)
    with pytest.raises(ValueError, match="reserved"):
        evolve([replace(observations[0], family="conflict"), *observations[1:]])
    with pytest.raises(ValueError, match="source-root"):
        evolve([replace(observations[0], root="metta-sft-v1:nonexistent:999999"), *observations[1:]])


def test_confusion_reports_unknown_class_accuracy_with_stable_denominator():
    obs = generate_observations(roots_per_family=36)
    stats = confusion(Program("false"), obs)
    assert stats["roots"] == len(obs)
    assert stats["tp"] == 0 and stats["fp"] == 0
    assert stats["tn"] + stats["fn"] == len(obs)
    assert len(json.dumps(stats)) > 20
