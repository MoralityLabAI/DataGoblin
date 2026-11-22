"""The original PR #3 Bonsai adapter and native Q1 tactic-evaluation contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from datagoblin.training_gym.rlt_gym import hard02f_oxibonsai as f
from datagoblin.training_gym.rlt_gym.external_checked_exit import digest
from datagoblin.training_gym.rlt_gym.hard02e_list_stop import generate, rule_candidate


class FakeLeanChecker:
    def __init__(self, root):
        self.root = root
        self.calls = []

    def __call__(self, task, candidate):
        assert task == self.root.public_task()
        idx = candidate["id"]
        assert type(idx) is int and 0 <= idx < 5
        self.calls.append(idx)
        truth = (rule_candidate(self.root) == idx)
        return {
            "task_digest": digest(task),
            "candidate_digest": digest(candidate),
            "verifier_sha256": "a" * 64,
            "passed": truth,
            "code": "kernel_certified" if truth else "tactic_rejected",
            "evidence": (
                {"kernel_invoked": True, "kernel_ok": True, "axioms": []}
                if truth else {}
            ),
        }


class FakeBonsaiModel:
    def __init__(self, choices):
        self.choices = iter(choices)
        self.calls = []

    def choose(self, statement, menu, failed):
        assert statement.startswith("∀ (xs ys : List Nat)")
        assert 1 in menu and menu[1].startswith("STOP")
        assert 1 <= len(menu) <= 6
        self.calls.append((menu.copy(), failed.copy()))
        return {
            "choice": next(self.choices), "usage": {
                "prompt_tokens": 123, "completion_tokens": 1,
            },
            "wall_s": 1.5,
            "request_sha256": "b" * 64,
            "response_sha256": "c" * 64,
        }


def test_original_adapter_identity_and_fixed_1bit_checkpoint():
    assert f.CHECKPOINT == "prism-ml/Bonsai-8B-gguf"
    assert f.CHECKPOINT_SHA256 == "284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54"
    assert f.UPSTREAM_OXI_SHA == "1d10c48e14754ff6999158f8fa13c7c92b40b0dc"
    assert f.ARMS == ("oxibonsai_q1", "fixed_full", "random_full", "suffix_length_rule")


def test_bonsai_accepts_exhausted_negative_menu_stop_without_kernel_calls():
    root = next(x for x in generate() if x.split == "ood_test" and not x.expected_witness)
    engine, checker = FakeBonsaiModel([1]), FakeLeanChecker(root)
    episode = f.bonsai_episode(root, engine, checker, max_attempts=4)
    assert episode["result"] == "model_stop"
    assert not episode["certified"]
    assert episode["candidate_checks"] == 0
    assert episode["model_calls"] == 1
    assert episode["model_wall_seconds"] == 1.5
    assert checker.calls == []
    assert len(engine.calls) == 1
    assert f.control_episode(root, checker, "suffix_length_rule")["candidate_checks"] == 0


def test_bonsai_retries_after_failed_lean_tactic_and_recovers():
    root = next(x for x in generate() if x.split == "ood_test" and x.expected_witness)
    correct = rule_candidate(root)
    assert correct is not None
    wrong = next(i for i in range(5) if i != correct)
    engine = FakeBonsaiModel([wrong + 2, correct + 2])
    checker = FakeLeanChecker(root)
    episode = f.bonsai_episode(root, engine, checker, max_attempts=4)
    assert episode["certified"]
    assert episode["result"] == "kernel_certified"
    assert episode["candidate_checks"] == 2
    assert checker.calls == [wrong, correct]
    assert list(engine.calls[1][1]) == [wrong + 2]
    assert wrong + 2 not in engine.calls[1][0]
    assert episode["model_wall_seconds"] == 3.0
    assert all("model_proposals" not in a for a in episode["attempts"])


def test_invalid_choice_is_recorded_separately_from_explicit_stop():
    root = next(x for x in generate() if x.split == "ood_test" and x.expected_witness)
    checker = FakeLeanChecker(root)
    episode = f.bonsai_episode(root, FakeBonsaiModel([None]), checker)
    assert episode["result"] == "invalid_model_output"
    assert episode["invalid_model_choices"] == 1
    assert episode["candidate_checks"] == 0

    def hostile_choice(_statement, _menu, _failed):
        return {"choice": 77, "wall_s": 0}

    class HostileModel:
        choose = staticmethod(hostile_choice)

    with pytest.raises(ValueError, match="out-of-menu"):
        f.bonsai_episode(root, HostileModel(), checker)


def test_online_bonsai_and_controls_have_the_same_candidate_budget():
    root = next(x for x in generate() if x.split == "ood_test" and x.expected_witness)
    checker = FakeLeanChecker(root)
    for arm in ("fixed_full", "random_full"):
        ep = f.control_episode(root, checker, arm, max_attempts=2)
        assert ep["candidate_checks"] <= 2
    symbol = f.control_episode(root, checker, "suffix_length_rule", max_attempts=2)
    assert symbol["certified"] and symbol["candidate_checks"] == 1
    assert symbol["model_calls"] == 0
    with pytest.raises(ValueError, match="unregistered"):
        f.control_episode(root, checker, "fiction", max_attempts=2)


def test_model_preflight_checks_exact_checkpoint_and_binary_ids(tmp_path: Path, monkeypatch):
    binary, model, tokenizer = [tmp_path / k for k in ("oxibonsai", "Bonsai-8B.gguf", "tokenizer.json")]
    binary.write_bytes(b"cpu-executable-fixture")
    model.write_bytes(b"fixture-model")
    tokenizer.write_bytes(b"fixture-tokenizer")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        f.preflight(binary=binary, model=model, tokenizer=tokenizer,
                    transport="stdio", port=18783, timeout=90)
    monkeypatch.setattr(f, "CHECKPOINT_SHA256", f.sha256_file(model))

    def fake_process(args, **kwargs):
        assert args == [str(binary), "--version"]
        return subprocess.CompletedProcess(args, 0, "oxibonsai 0.2.3\n", "")

    monkeypatch.setattr(f.subprocess, "run", fake_process)
    policy, identity = f.preflight(
        binary=binary, model=model, tokenizer=tokenizer,
        transport="stdio", port=18783, timeout=90,
    )
    assert isinstance(policy, f.DirectOxiBonsai)
    receipt = identity.public_receipt()
    assert receipt["transport"] == "stdio"
    assert receipt["server_model_id"] is None
    assert receipt["model_sha256"] == f.sha256_file(model)
    assert receipt["tokenizer_sha256"] == f.sha256_file(tokenizer)
    assert receipt["runtime_binary_sha256"] == f.sha256_file(binary)
    assert receipt["upstream_oxibonsai_sha256_commit"] == f.UPSTREAM_OXI_SHA
    json.dumps(receipt)


def test_native_checker_rejects_unverified_success():
    root = generate()[0]

    def fake(_task, _candidate):
        return {
            "passed": True, "code": "kernel_certified",
            "task_digest": digest(root.public_task()),
            "candidate_digest": digest({"id": 0}),
            "evidence": {
                "kernel_invoked": False, "kernel_ok": True, "axioms": [],
            },
        }

    with pytest.raises(ValueError, match="kernel"):
        f._check(fake, root, 0)
