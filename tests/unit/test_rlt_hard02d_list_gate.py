"""Source-custody and malicious-certificate contracts for RLT-HARD-02D."""

from __future__ import annotations

from collections import Counter

import pytest

from datagoblin.training_gym.rlt_gym.hard02d_list_gate import (
    BUDGET,
    SPLIT,
    _check,
    _tactic,
    generate,
)


def test_near_miss_list_qa_has_exact_negative_source_roots():
    rows = generate(roots=8, seed=2026112219)
    assert len(rows) == 8
    assert len({r.root_id for r in rows}) == 8
    assert len({r.statement for r in rows}) == 8
    assert len({r.source_file for r in rows}) == 8
    assert Counter(r.split for r in rows) == {SPLIT: 8}
    for i, row in enumerate(rows):
        assert len(row.menu) == BUDGET and len(set(row.menu)) == BUDGET
        assert len({len(x.encode("utf-8")) for x in row.menu}) == 1
        witness = _tactic(1)
        assert (witness in row.menu) == (i % 4 != 3)
        assert "List.length_append" in row.statement or "List.length_append" in witness
        assert row.public_task()["menu_sha256"]
        assert "expected" not in str(row.public_task())
        assert "gold" not in str(row.public_task())


def test_source_seed_changes_theorem_statement_and_root_id():
    a = generate(roots=8, seed=2026112219)
    b = generate(roots=8, seed=2026112220)
    assert {r.root_id for r in a}.isdisjoint({r.root_id for r in b})
    assert {r.statement for r in a}.isdisjoint({r.statement for r in b})


def test_candidate_actions_use_exact_type_compatibility():
    row = generate(roots=4)[0]
    assert _tactic(1) in row.menu
    assert "simpa" not in _tactic(1)
    for n in (0, 2, 3, 4, 5):
        assert _tactic(n) != _tactic(1)
        assert "simp only" in _tactic(n)
        assert "exact congrArg" in _tactic(n)


def test_invalid_source_count_and_empty_seed():
    with pytest.raises(ValueError, match="roots"):
        generate(roots=3)
    with pytest.raises(ValueError, match="nonnegative"):
        generate(seed=-1)


def test_runner_rejects_unchecked_fake_kernel_success():
    root = generate(roots=4)[0]

    def dishonest_checker(_task, _candidate):
        return {
            "passed": True,
            "code": "kernel_certified",
            "evidence": {"kernel_invoked": False, "kernel_ok": True, "axioms": []},
        }

    with pytest.raises(ValueError, match="kernel"):
        _check(dishonest_checker, root, 0)

    def bad_axiom(_task, _candidate):
        return {
            "passed": True,
            "code": "kernel_certified",
            "evidence": {"kernel_invoked": True, "kernel_ok": True,
                         "axioms": ["sorryAx"]},
        }

    with pytest.raises(ValueError, match="kernel"):
        _check(bad_axiom, root, 0)
