"""Contracts for the post-hoc length-confound repair development cohort."""

from __future__ import annotations

from collections import Counter

import pytest

from datagoblin.training_gym.rlt_gym.hard02_mathlib import (
    BUDGET,
    SPLITS,
    _witness,
    generate,
    matched_candidate_menu,
)


def test_pinned_old_positive_control_is_unchanged():
    roots = generate(per_split=8, seed=2026112202)
    assert len(roots) == 32
    assert all(set(root.menu).issuperset({"rfl", "simp", "constructor", "exact True.intro"})
               for root in roots)


def test_matched_menu_source_diversity_and_zero_byte_length_shortcuts():
    roots = generate(per_split=8, seed=2026112203, matched_menus=True)
    assert len(roots) == 32
    assert Counter(root.split for root in roots) == dict.fromkeys(SPLITS, 8)
    assert len({r.root_id for r in roots}) == len(roots)
    assert len({r.statement for r in roots}) == len(roots)
    assert len({r.source_file for r in roots}) == len(roots)
    for root in roots:
        assert len(root.menu) == BUDGET and len(set(root.menu)) == BUDGET
        assert len({len(action.encode("utf-8")) for action in root.menu}) == 1
        assert all(len(action.splitlines()) >= 1 for action in root.menu)
        assert all("sorry" not in action.lower() and "admit" not in action.lower() for action in root.menu)
        assert root.public_task()["import"] == "Mathlib"
        assert "witness" not in str(root.public_task())


def test_registered_witness_present_only_on_answerable_cases():
    seed = 2026112203
    roots = generate(per_split=8, seed=seed, matched_menus=True)
    for split in SPLITS:
        cohort = [r for r in roots if r.split == split]
        assert len(cohort) == 8
        witness_count = 0
        for i, root in enumerate(cohort):
            stmt, prefix, witness = _witness(
                split, i, numeric_shift=17 + seed % 41, depth_shift=1 + seed % 3
            )
            assert root.statement == stmt
            assert root.prefix == prefix
            found = sum(script.strip() == witness for script in root.menu)
            assert found == int(i % 4 != 3)
            witness_count += found
        assert witness_count == 6


def test_matched_cohort_is_fresh_by_theorem_statement():
    first = generate(per_split=8, seed=2026112202)
    second = generate(per_split=8, seed=2026112203, matched_menus=True)
    third = generate(per_split=8, seed=2026112204, matched_menus=True)
    for left, right in ((first, second), (second, third)):
        assert {r.root_id for r in left}.isdisjoint(r.root_id for r in right)
        assert {r.statement for r in left}.isdisjoint(r.statement for r in right)


def test_near_miss_candidates_reject_bad_source_contract():
    _, _, witness = _witness("train", 0)
    with pytest.raises(ValueError, match="unregistered"):
        matched_candidate_menu("fiction", witness, witnessed=True)
    with pytest.raises(ValueError, match="malformed"):
        matched_candidate_menu("train", "rfl", witnessed=True)
