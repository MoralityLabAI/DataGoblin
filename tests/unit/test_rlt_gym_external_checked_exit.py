"""Source-only external checker fixtures. These are not Lean kernel or MCP tool evidence."""

from __future__ import annotations

import hashlib

import pytest

from datagoblin.training_gym.rlt_gym.external_checked_exit import (
    SCHEMA, digest, replay_corpus, replay_external_episode,
)

SHA = hashlib.sha256(b"test-only independent checker").hexdigest()
MODEL = hashlib.sha256(b"test-only checkpoint").hexdigest()
SOURCE = hashlib.sha256(b"test-only source registry").hexdigest()


def checker(task, candidate):
    outcome = candidate == {"proof": task["expected"]}
    return {
        "passed": outcome, "code": "closed" if outcome else "open_goals",
        "verifier_sha256": SHA, "task_digest": digest(task),
        "candidate_digest": digest(candidate), "kernel_sha": "fixture-only",
    }


def make_episode(split="train", exposure="fresh", name="a"):
    task = {"statement": "Nat.add_comm", "expected": name}
    candidates = [{"proof": "bad"}, {"proof": name}, {"proof": "bad again"}]
    return {
        "schema_version": SCHEMA, "kind": "lean", "split": split,
        "root_id": name, "source_revision": SOURCE,
        "model_sha256": MODEL, "verifier_sha256": SHA,
        "source_exposure": exposure, "evidence_kind": "native_operator_verified",
        "public_task": task, "task_digest": digest(task),
        "planned_visits": 3, "verified_stop": False,
        "attempts": [
            {"visit": i, "candidate": c, "verification": checker(task, c)}
            for i, c in enumerate(candidates)
        ],
    }


def test_replay_train_and_historical_custody():
    record = replay_external_episode(make_episode(), checker=checker, trusted_verifier_sha=SHA,
        trusted_source_revision=SOURCE, training_authorized=True)
    assert record["future_rescue"] is True
    assert record["first_verified_visit"] == 1
    assert record["cost_until_first_verified_or_budget"] == 2
    assert record["training_eligible"] is True
    assert record["observable_after_first"]["checker_code"] == "open_goals"
    pack = replay_corpus(
        [make_episode(), make_episode(split="calibration", exposure="historical", name="b")],
        checkers={"lean": checker}, trusted_shas={"lean": SHA},
        trusted_source_revisions={"lean": SOURCE}, approved_train_roots=frozenset({"a"}),
    )
    assert pack["verified_roots"] == 2
    assert len(pack["train"]) == 1
    assert len(pack["evaluation_only"]) == 1


def test_receipt_tamper_and_untrusted_verifier():
    record = make_episode()
    record["attempts"][1]["verification"]["passed"] = False
    with pytest.raises(ValueError, match="receipt"):
        replay_external_episode(record, checker=checker, trusted_verifier_sha=SHA,
                                trusted_source_revision=SOURCE)
    with pytest.raises(ValueError, match="verifier"):
        replay_external_episode(make_episode(), checker=checker, trusted_verifier_sha="0" * 64,
            trusted_source_revision=SOURCE)


def test_protected_split_and_reused_root():
    with pytest.raises(ValueError, match="protected"):
        replay_external_episode(
            make_episode(split="sealed_poison"), checker=checker, trusted_verifier_sha=SHA,
            trusted_source_revision=SOURCE,
        )
    with pytest.raises(ValueError, match="reused"):
        replay_corpus(
            [make_episode(), make_episode()],
            checkers={"lean": checker}, trusted_shas={"lean": SHA},
            trusted_source_revisions={"lean": SOURCE}, approved_train_roots=frozenset({"a"}),
        )


def test_mcp_checker_hook_and_no_future_rescue():
    record = make_episode()
    record["kind"] = "mcp"
    assert replay_external_episode(
        record, checker=checker, trusted_verifier_sha=SHA,
        trusted_source_revision=SOURCE,
    )["kind"] == "mcp"
    for i, step in enumerate(record["attempts"]):
        step["candidate"] = {"proof": f"fail{i}"}
        step["verification"] = checker(record["public_task"], step["candidate"])
    checked = replay_external_episode(record, checker=checker, trusted_verifier_sha=SHA,
                                      trusted_source_revision=SOURCE)
    assert checked["future_rescue"] is False
    assert checked["cost_until_first_verified_or_budget"] == 3


def test_missing_external_verifier_is_rejected():
    with pytest.raises(ValueError, match="unregistered"):
        replay_corpus([make_episode()], checkers={}, trusted_shas={}, trusted_source_revisions={})
