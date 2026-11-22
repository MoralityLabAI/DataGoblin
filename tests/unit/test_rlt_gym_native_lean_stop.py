"""Offline contracts for the new native Lean trained-ranker experiment."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from datagoblin.training_gym.rlt_gym.external_checked_exit import digest
from datagoblin.training_gym.rlt_gym.native_lean_stop import (
    FAMILY_PATTERNS,
    SCRIPT_PATTERNS,
    SCRIPTS,
    SPLITS,
    _online,
    _policy_ids,
    calibrate,
    features,
    fit_ranker,
    generate_roots,
    probabilities,
    simulate,
)


def _fixture_receipt(root, index):
    witness = next((j for j, tactic in enumerate(root.tactics) if tactic == SCRIPTS[
        _observed_family(root.statement)
    ]), None)
    passed = index == witness
    return {
        "id": index,
        "receipt": {"passed": passed, "code": "kernel_certified" if passed else "open_goals"},
    }


def _observed_family(statement: str) -> int:
    if "(a + b) * (c +" in statement:
        return 0
    if "(b + a)" in statement:
        return 2
    if "(a * c + b * c)" in statement:
        return 3
    return 1


def _checked(root):
    return [_fixture_receipt(root, i) for i in range(len(root.tactics))]


def test_unique_sources_and_mixed_witnessed_exhausted_roots():
    roots = generate_roots(per_split=8)
    assert len(roots) == 32
    assert len({root.root_id for root in roots}) == 32
    assert len({root.statement for root in roots}) == 32
    assert Counter(root.split for root in roots) == dict.fromkeys(SPLITS, 8)
    for split in SPLITS:
        cohort = [root for root in roots if root.split == split]
        assert sum(any(_fixture_receipt(root, i)["receipt"]["passed"]
                       for i in range(5)) for root in cohort) == 6
        assert sum(not any(_fixture_receipt(root, i)["receipt"]["passed"]
                           for i in range(5)) for root in cohort) == 2
    assert all(len(root.tactics) == 5 and len(set(root.tactics)) == 5 for root in roots)
    assert all("answerable" not in str(root.public_task)
               and "gold" not in str(root.public_task) for root in roots)


def test_true_ood_numeric_shift_and_determinism():
    a = generate_roots(per_split=8, seed=100)
    b = generate_roots(per_split=8, seed=100)
    c = generate_roots(per_split=8, seed=101)
    assert a == b
    assert {r.root_id for r in a}.isdisjoint(r.root_id for r in c)
    assert all("400" not in r.statement for r in a if r.split == "train")
    assert all(any(ch.isdigit() for ch in r.statement) for r in a)
    with pytest.raises(ValueError, match="per_split"):
        generate_roots(per_split=5)


def test_feature_privacy_and_recurrence():
    roots = generate_roots(per_split=8)
    assert len(FAMILY_PATTERNS) == 4 and len(SCRIPT_PATTERNS) == 5
    x = features(roots[0].statement, roots[0].tactics[0])
    assert x.shape == (30,) and np.isfinite(x).all()
    assert x[0] == 1
    assert all("sorry" not in t.lower() and "unsafe" not in t.lower()
               for root in roots for t in root.tactics)


def test_fit_and_calibrate_train_only():
    roots = generate_roots(per_split=8)
    train = [(r, _checked(r)) for r in roots if r.split == "train"]
    cal = [(r, _checked(r)) for r in roots if r.split == "calibration"]
    w = fit_ranker(train)
    assert w.shape == (30,)
    assert np.isfinite(w).all()
    p = probabilities(train[0][0], w)
    assert p.shape == (5,) and np.all((0 < p) & (p < 1))
    threshold, frontier = calibrate(cal, w)
    assert threshold in {r["threshold"] for r in frontier}
    assert all(0 <= entry["certified"] <= 8 for entry in frontier)
    with pytest.raises(ValueError, match="TRAIN"):
        fit_ranker(cal)
    with pytest.raises(ValueError, match="CAL"):
        calibrate(train, w)


def test_simulated_policy_never_uses_future_outcomes_for_choice():
    roots = generate_roots(per_split=8)
    root = roots[0]
    w = np.zeros(30)
    ids = _policy_ids(root, "learned", w)
    checked = {i: _fixture_receipt(root, i)["receipt"] for i in range(5)}
    simulation = simulate(root, checked, w, threshold=0.1)
    expected_hit = any(entry["passed"] for entry in checked.values())
    assert simulation["certified"] == expected_hit
    assert simulation["attempts"] <= 5
    assert len(simulation["order"]) == simulation["attempts"]


def test_online_mock_native_verifier_blocks_unchecked_success():
    root = generate_roots(per_split=8)[0]
    statement_hash = digest(root.public_task)

    def checker(task, candidate):
        assert digest(task) == statement_hash
        choice = candidate["id"]
        valid = _fixture_receipt(root, choice)["receipt"]["passed"]
        return {
            "task_digest": statement_hash,
            "passed": valid,
            "code": "kernel_certified" if valid else "open_goals",
            "evidence": {"kernel_invoked": valid, "kernel_ok": valid, "axioms": []},
        }

    for arm in ("fixed_one", "fixed_full", "random_full", "learned"):
        output = _online(root, checker, arm=arm, weights=np.zeros(30), threshold=0.5)
        assert output["arm"] == arm
        assert 1 <= len(output["attempts"]) <= 5
        if output["certified"]:
            assert output["attempts"][-1]["receipt"]["passed"]
    with pytest.raises(ValueError, match="unknown"):
        _policy_ids(root, "invalid_arm", None)


def test_new_generated_scripts_have_multiline_proofs_and_registered_tactics():
    assert all("constructor" in s and "\n· " in s for s in SCRIPTS)
    assert all("intro a b c" in s for s in SCRIPTS)
