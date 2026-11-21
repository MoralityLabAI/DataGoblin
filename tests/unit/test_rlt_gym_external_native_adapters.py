"""Mock-only adapters; actual Lean and MCP evidence requires live pinned runtimes."""

from __future__ import annotations

import hashlib

import pytest

from datagoblin.training_gym.rlt_gym.external_checked_exit import (
    SCHEMA, digest, replay_external_episode,
)
from datagoblin.training_gym.rlt_gym.external_native_adapters import (
    LeanKernelChecker, MCPStateChecker,
)

SHA = hashlib.sha256(b"test-only lean certificate").hexdigest()
MODEL = hashlib.sha256(b"test-only model").hexdigest()
SOURCE = hashlib.sha256(b"test-only lean source").hexdigest()


class FakeLean:
    def __init__(self):
        self.runs = []
        self.kernels = []

    def run(self, sid, tactic):
        self.runs.append((sid, tactic))
        return {
            "ok": True,
            "state": {
                "closed": tactic == "exact good",
                "goals": [] if tactic == "exact good" else ["open"],
                "sid": 101,
            },
        }

    def kernel(self, sid):
        self.kernels.append(sid)
        return {"ok": True, "axioms": ["propext"]}


def test_lean_kernel_and_candidate_menu():
    task = {"goal": "closed", "visible_goal": "Nat.true"}
    lean = FakeLean()
    checker = LeanKernelChecker(
        lean=lean,
        states={digest(task): {"root_sid": 9, "menu": {1: "exact bad", 2: "exact good"}}},
        verifier_sha=SHA,
    )
    invalid = checker(task, {"id": 99})
    assert invalid["code"] == "candidate_id_rejected"
    assert lean.runs == []
    failed = checker(task, {"id": 1})
    assert failed["passed"] is False
    assert lean.kernels == []
    certified = checker(task, {"id": 2})
    assert certified["passed"] is True
    assert certified["code"] == "kernel_certified"
    assert lean.kernels == [101]
    candidates = ({"id": 1}, {"id": 2})
    ep = {
        "schema_version": SCHEMA, "kind": "lean", "split": "train",
        "root_id": "fresh_root", "source_revision": SOURCE,
        "model_sha256": MODEL, "verifier_sha256": SHA,
        "source_exposure": "fresh", "evidence_kind": "native_operator_verified",
        "public_task": task, "task_digest": digest(task),
        "planned_visits": 2, "verified_stop": False,
        "attempts": [
            {"visit": i, "candidate": c, "verification": checker(task, c)}
            for i, c in enumerate(candidates)
        ],
    }
    row = replay_external_episode(ep, checker=checker, trusted_verifier_sha=SHA,
        trusted_source_revision=SOURCE, training_authorized=True)
    assert row["future_rescue"] is True
    assert row["cost_until_first_verified_or_budget"] == 2


def test_mcp_effect_uses_separate_judge_and_frozen_snapshot():
    task = {"issue": "locate parser symbol", "snapshot": "repo-pin"}
    token = digest(task)
    cases = {token: {"menu": {1: "read_a", 2: "read_b"}, "snapshot_sha": "pinned_repo"}}

    def execute_fresh(_token, choice):
        return {
            "snapshot_sha": "pinned_repo",
            "action_id": choice,
            "effect": {"symbols": ["parser"]} if choice == 2 else {"symbols": ["wrong"]},
        }

    def independent_judge(_token, result):
        passed = "parser" in result["effect"]["symbols"]
        return passed, "gold_recalled" if passed else "gold_missing"

    check = MCPStateChecker(
        cases=cases, verifier_sha=SHA, execute_fresh=execute_fresh,
        judge=independent_judge,
    )
    assert check(task, {"id": 1})["passed"] is False
    assert check(task, {"id": 2})["passed"] is True
    bad = MCPStateChecker(
        cases=cases, verifier_sha=SHA,
        execute_fresh=lambda _token, choice: {
            "snapshot_sha": "wrong", "action_id": choice, "effect": {},
        },
        judge=independent_judge,
    )
    with pytest.raises(ValueError, match="reset"):
        bad(task, {"id": 2})


def test_lean_forbidden_tactic_and_kernel_axioms():
    task = {"goal": "g"}
    bad_tactic = LeanKernelChecker(
        lean=FakeLean(),
        states={digest(task): {"root_sid": 1, "menu": {1: "sorry"}}},
        verifier_sha=SHA,
    )
    with pytest.raises(ValueError, match="untrusted tactic"):
        bad_tactic(task, {"id": 1})

    class UntrustedLean(FakeLean):
        def kernel(self, sid):
            return {"ok": True, "axioms": ["sorryAx"]}

    bad_kernel = LeanKernelChecker(
        lean=UntrustedLean(),
        states={digest(task): {"root_sid": 1, "menu": {1: "exact good"}}},
        verifier_sha=SHA,
    )
    assert bad_kernel(task, {"id": 1})["passed"] is False
