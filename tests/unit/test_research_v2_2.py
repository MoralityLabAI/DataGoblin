"""Unit checks for the v2.2 experimental controls (no external endpoints)."""

from types import SimpleNamespace

import numpy as np
import pytest

from datagoblin.research_v2.suite import LEARNERS
from datagoblin.research_v2_1.controllers import _generate_controller_training_data
from datagoblin.research_v2_2.adaptive_exit import AdaptiveExit, choose_threshold, execute
from datagoblin.research_v2_2.controllers import SelectiveCorrector, rule_class, schema_repair
from datagoblin.research_v2_2.distill_bridge import _digest, admit_record
from datagoblin.research_v2_2.gates import validate_continuation, validate_specialist_accounting
from datagoblin.research_v2_2.jinn_adapter import native_step, verify_disjoint


def test_rule_exact_on_fresh_synthetic_states():
    x, _, y = _generate_controller_training_data(seed=2026112807, n_samples=4096)
    assert np.array_equal(np.asarray([rule_class(row) for row in x]), y)


def test_schema_only_preserves_config():
    family = next(k for k, v in LEARNERS.items() if v == "trm_style")
    raw = '{"action":"run_experiment","config":{"trm_width":12,"steps":4}}'
    result = schema_repair({"family": family}, raw, {"experiments": 1})
    import json
    assert json.loads(result.executed_text)["config"] == {"width": 12, "steps": 4}
    assert result.intervened and not result.skip_model_call
    assert schema_repair({"family": family}, "bad", {}).executed_text == "bad"


def test_selective_correction_accounting():
    low = SimpleNamespace(predict=lambda x: (0, np.array([.31, .30, .13, .13, .13])))
    high = SimpleNamespace(predict=lambda x: (0, np.array([.9, .025, .025, .025, .025])))
    correction = SimpleNamespace(predict=lambda x: (1, np.eye(5)[1]), passes=3)
    for router, expected in ((low, 1), (high, 0)):
        head = SelectiveCorrector(router, correction)
        assert head.predict(np.zeros(14))[0] == expected
        assert head.accounting()["rlt_passes"] == 3 * expected
        assert head.accounting()["trm_forwards"] == 1 + expected


def test_early_exit_executes_only_counted_passes():
    calls = []
    def step(cur, k, horizon):
        calls.append((k, horizon))
        return cur + np.array([.01, 0., 0.])
    head = AdaptiveExit(np.zeros(6), 10.0, .5)
    result, k = execute(np.zeros(3), step, head)
    assert k == len(calls) == 3
    assert all(horizon == 24 for _, horizon in calls)
    assert result[0] == pytest.approx(.03)
    assert choose_threshold([(1., .8, 24, 2), (.9, .798, 4, 2), (.8, .81, 3, 3)]) == .9


def test_native_step_matches_residual_and_disjoint():
    head = SimpleNamespace(alphas=[.25, .5, 1.0], w1=np.zeros((6, 12)),
                           b1=np.zeros(12), w2=np.zeros((12, 3)), b2=np.ones(3))
    assert np.allclose(native_step(head, .5, 0)(np.zeros(3), 1, 3), .25)
    with pytest.raises(ValueError):
        verify_disjoint(["a"], ["a"], ["b"])
    verify_disjoint(["a", "a"], ["b"], ["c"])


def test_continuation_rejects_lineage_break():
    fields = ("weights", "optimizer", "scheduler", "rng", "cursor", "corpus", "tokenizer")
    receipt = {f"{field}_sha256": "a" * 64 for field in fields}
    receipt.update(stage_positions=20_000_000, resume_positions=10_000_000,
                   context_tokens=2048, truncated_prompts=0, scheduler_resume_lr=1.25e-5,
                   optimizer_resume_lr=1.25e-5, optimizer_state_restored=True,
                   rng_cursor_restored=True, excluded_root_overlap=0)
    validate_continuation(receipt)
    receipt["optimizer_resume_lr"] = 3e-4
    with pytest.raises(ValueError, match="learning-rate"):
        validate_continuation(receipt)


def test_specialist_requires_both_budgets():
    receipt = dict(truncated_prompts=0, missing_eos_targets=0, pool_epochs_complete=2,
                   specialist_loss_tokens=423812, specialist_forward_tokens=1_000_000,
                   forward_control_tokens=1_000_000, specialist_loss_tokens_total=500_000,
                   loss_control_tokens=500_000)
    validate_specialist_accounting(receipt)
    receipt["loss_control_tokens"] *= 2
    with pytest.raises(ValueError, match="unmatched"):
        validate_specialist_accounting(receipt)


def test_distill_admits_verified_train_records_only():
    packet = {"packet_id": "dev-train", "split": "train", "allowed_evidence_ids": ["e1"], "max_budget": 2}
    record = dict(schema="research-decision.v1", packet_id=packet["packet_id"],
                  packet_sha256=_digest(packet), decision="fit", hypothesis="h", control="baseline",
                  falsifier="zero gain", expected_observation="delta", budget=1,
                  evidence_ids=["e1"], failure_recovery="repair", rationale="paired comparison")
    receipt = dict(record_sha256=_digest(record), packet_sha256=_digest(packet), exact_grade_pass=True)
    assert admit_record(record, public_packet=packet, verified_grade=receipt) == _digest(record)
    with pytest.raises(ValueError):
        admit_record(record, public_packet={**packet, "split": "held"}, verified_grade=receipt)
    with pytest.raises(ValueError):
        admit_record(record, public_packet=packet, verified_grade={**receipt, "exact_grade_pass": False})
