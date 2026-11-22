"""Contracts for the pinned Mathlib-native RLT-HARD-02A development intake."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from datagoblin.training_gym.rlt_gym.hard02_mathlib import (
    BUDGET,
    MATHLIB_COMMIT,
    SPLITS,
    THRESHOLDS,
    UPSTREAM_FILES,
    _witness,
    calibrate,
    export_sources,
    fit_policy,
    generate,
    order,
    sha,
    verify_upstream,
)


def test_source_roots_distinct_and_upstream_family_disjoint():
    roots = generate(per_split=8)
    assert len(roots) == 32
    assert Counter(x.split for x in roots) == {split: 8 for split in SPLITS}
    assert len({x.root_id for x in roots}) == 32
    assert len({x.source_file for x in roots}) == 32
    assert len({x.statement for x in roots}) == 32
    assert len(set(UPSTREAM_FILES.values())) == 4
    assert {x.family for x in roots} == {f"mathlib_{s}" for s in SPLITS}
    for root in roots:
        public = root.public_task()
        assert len(root.menu) == BUDGET
        assert public["statement"] == root.statement
        assert public["import"] == "Mathlib"
        assert public["upstream_module"] == UPSTREAM_FILES[root.split]
        assert "witness" not in json.dumps(public)
        assert "answerable" not in json.dumps(public)
        assert all(all(bad not in action.lower() for bad in ("sorry", "admit", "run_tac"))
                   for action in root.menu)


def test_preregistered_finite_exhaustion_not_mathematical_unprovability():
    roots = generate(per_split=8)
    for split in SPLITS:
        cohort = [root for root in roots if root.split == split]
        assert sum("exact False.elim (by assumption)" in root.menu for root in cohort) == 2
        for i, root in enumerate(cohort):
            statement, prefix, witness = _witness(split, i)
            assert root.statement == statement and root.prefix == prefix
            if i % 4 != 3:
                assert witness in root.menu
            else:
                assert witness not in root.menu


def test_generated_open_goal_files_are_create_only(tmp_path: Path):
    roots = generate(per_split=4)
    manifest = export_sources(tmp_path, roots)
    assert len(manifest) == 16
    for rel, checksum in manifest.items():
        text = (tmp_path / rel).read_text()
        assert "import Mathlib" in text
        assert "trace_state" in text
        assert "sorry" not in text
        assert hashlib.sha256(text.encode()).hexdigest() == checksum
    with pytest.raises(FileExistsError):
        export_sources(tmp_path, roots)


def test_source_registration_seed_disjoint_and_nontrivial_prefix():
    a = generate(per_split=8, seed=100)
    b = generate(per_split=8, seed=101)
    assert {x.root_id for x in a}.isdisjoint(x.root_id for x in b)
    assert all("intro " in r.prefix for r in a)
    assert all(r.statement.count("→") >= 1 or "∣" in r.statement for r in a)
    with pytest.raises(ValueError, match="per_split"):
        generate(per_split=3)


def test_policy_fit_blocks_nontrain_sources(monkeypatch):
    # Exercise custody first, avoiding optional Autograd installation in source-only tests.
    roots = generate(per_split=4)
    empty = [{"split": "calibration", "goal_pp": "⊢ True", "menu": roots[0].menu, "checks": []}]
    with pytest.raises(ValueError, match="TRAIN only"):
        fit_policy(empty, ())


def test_threshold_gate_blocks_noncal_and_no_native_fabrication():
    roots = generate(per_split=4)
    assert THRESHOLDS == tuple(sorted(THRESHOLDS))
    with pytest.raises(ValueError, match="CAL"):
        calibrate([{"split": "train"}], np.zeros(6), ())
    with pytest.raises(ValueError, match="unknown"):
        order(roots[0], "invented", "⊢ True", np.zeros(6), ())


def test_mathlib_source_pin_requires_native_checkout(tmp_path: Path):
    assert len(MATHLIB_COMMIT) == 40
    assert sha(b"proof") == hashlib.sha256(b"proof").hexdigest()
    with pytest.raises(ValueError, match="Mathlib"):
        verify_upstream(tmp_path / "missing", tmp_path / "lake")
