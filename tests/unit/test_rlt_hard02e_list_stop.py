"""Offline source, oracle-custody and policy contracts for RLT-HARD-02E."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from datagoblin.training_gym.rlt_gym.hard02e_list_stop import (
    ARMS,
    BUDGET,
    SEED,
    SPLITS,
    _calibrate,
    _check,
    _labels_present,
    _online,
    _ordered,
    _tactic,
    generate,
    rule_candidate,
    syntax_length,
)


def _candidate_k(tactic: str) -> int:
    import re

    found = re.search(r"m \+ ([0-9])\) h$", tactic)
    assert found
    return int(found.group(1))


def test_fresh_source_disjoint_splits_and_real_negative_menu_contract():
    roots = generate()
    assert len(roots) == 32
    assert Counter(r.split for r in roots) == dict.fromkeys(SPLITS, 8)
    assert len({r.root_id for r in roots}) == 32
    assert len({r.source_file for r in roots}) == 32
    assert len({r.statement for r in roots}) == 32
    assert len(ARMS) == 6
    for split in SPLITS:
        group = [r for r in roots if r.split == split]
        assert sum(r.expected_witness for r in group) == 6
        assert sum(not r.expected_witness for r in group) == 2
        for root in group:
            assert len(root.menu) == BUDGET
            assert len(set(root.menu)) == BUDGET
            assert len({len(t.encode("utf-8")) for t in root.menu}) == 1
            assert root.public_task()["import"] == "Mathlib"
            assert root.public_task()["upstream_module"] == "Mathlib/Data/List/Basic.lean"
            assert "expected_witness" not in str(root.public_task())
            assert "suffix_len" not in str(root.public_task())
            assert syntax_length(root) == root.suffix_len
            assert (_tactic(root.suffix_len) in root.menu) == root.expected_witness
            assert (rule_candidate(root) is not None) == root.expected_witness
            assert all(_candidate_k(t) <= 9 for t in root.menu)
    assert all(6 <= r.suffix_len <= 8 for r in roots if r.split == "ood_test")
    assert all(1 <= r.suffix_len <= 5 for r in roots if r.split == "train")


def test_fresh_seeds_prevent_statement_reuse_and_duplicate_actions():
    a = generate(seed=SEED)
    b = generate(seed=SEED + 1)
    assert {r.statement for r in a}.isdisjoint(r.statement for r in b)
    assert {r.root_id for r in a}.isdisjoint(r.root_id for r in b)
    with pytest.raises(ValueError, match="per_split"):
        generate(per_split=5)
    with pytest.raises(ValueError, match="nonnegative"):
        generate(seed=-4)
    with pytest.raises(ValueError, match="one-digit"):
        _tactic(10)


def test_rule_stops_on_negative_without_kernel_or_private_flag():
    negative = next(r for r in generate() if not r.expected_witness)
    positive = next(r for r in generate() if r.expected_witness)

    def checker(task, candidate):
        assert task["statement"]
        is_correct = candidate["id"] == rule_candidate(positive)
        return {
            "passed": is_correct,
            "task_digest": "overridden",
            "candidate_digest": "overridden",
        }

    # No verifier calls for the public-symbolic prediction of an exhausted menu.
    def never_used(*_args):
        return pytest.fail("unexpected verifier invocation")
    result = _online(negative, f"⊢ {negative.statement}", never_used,
                     "suffix_length_rule", np.zeros(208), np.zeros(30), (), {})
    assert result["attempts"] == []
    assert not result["certified"]
    assert result["stopped_before_budget"]
    assert rule_candidate(positive) in range(5)


def test_real_receipt_custody_rejects_untrusted_certificates():
    root = generate()[0]

    def checker(_task, _candidate):
        from datagoblin.training_gym.rlt_gym.external_checked_exit import digest

        return {
            "task_digest": digest(root.public_task()),
            "candidate_digest": digest({"id": 0}),
            "passed": True, "code": "kernel_certified",
            "evidence": {"kernel_invoked": False, "kernel_ok": True, "axioms": []},
        }

    with pytest.raises(ValueError, match="kernel"):
        _check(checker, root, 0)


def test_train_cal_validation_rejects_noncal():
    train = [r for r in generate() if r.split == "train"]
    bad = [{"root": train[0]}]
    with pytest.raises(ValueError, match="CAL"):
        _calibrate(bad, np.zeros(208), semantic=True, words=())


def test_native_labels_cannot_be_made_from_generator_only():
    roots = generate(per_split=4)
    p = next(r for r in roots if r.expected_witness)
    n = next(r for r in roots if not r.expected_witness)
    correct_p = rule_candidate(p)
    assert correct_p is not None
    good = [{"candidate_id": i, "receipt": {"passed": i == correct_p}} for i in range(5)]
    assert _labels_present(p, good)
    bad = [{"candidate_id": i, "receipt": {"passed": False}} for i in range(5)]
    assert _labels_present(n, bad)
    assert not _labels_present(p, bad)
    assert not _labels_present(n, good)


def test_candidate_order_and_random_control_are_fixed_by_source_id():
    root = generate()[0]
    x = np.zeros(208)
    assert _ordered(root, "random_full", "⊢ True", x, x, ())[0] == _ordered(
        root, "random_full", "⊢ True", x, x, ()
    )[0]
    assert _ordered(root, "fixed_full", "⊢ True", x, x, ())[0] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="unregistered"):
        _ordered(root, "fictional_arm", "⊢ True", x, x, ())
