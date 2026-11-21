"""Message-bank pass arithmetic and provenance controls, offline."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from datagoblin.training_gym.rlt_gym.nnsight_message_bank import (
    BankPlan,
    check_message_calls,
    message_call_index,
    run_bank_control_suite,
)


def _calls(features: tuple[int, ...], passes: int = 3, width: int = 12) -> list[dict]:
    return [
        {
            "call_index": i, "pass_index": 1 + i // 12, "layer_index": 6 + i % 12,
            "width": width,
            "bank_0": {"l2": 1.0, "selected_features": [float(x) for x in features]},
            "bank_1": {"l2": 2.0, "selected_features": [float(x + 1) for x in features]},
        }
        for i in range(12 * (passes - 1))
    ]


def test_message_bank_calls_cross_pass_boundaries() -> None:
    assert message_call_index(1, 6) == 0
    assert message_call_index(1, 17) == 11
    assert message_call_index(2, 6) == 12
    assert message_call_index(2, 12) == 18
    with pytest.raises(ValueError, match="first pass"):
        message_call_index(0, 12)
    plan = BankPlan("model", "revision", "task", layer_index=12, feature_ids=(1, 3))
    check_message_calls(_calls(plan.feature_ids), 3, plan.feature_ids)
    with pytest.raises(RuntimeError, match="missing"):
        check_message_calls(_calls(plan.feature_ids)[:-1], 3, plan.feature_ids)
    with pytest.raises(ValueError, match="require 2"):
        BankPlan("model", "revision", "task", expected_passes=1)


def test_message_bank_donor_controls_stay_unverified() -> None:
    plan = BankPlan("model", "revision", "task", feature_ids=(1,), target_token_id=4)
    count = 0

    def fake_trace(_model, prompt, candidate, *, remote, donor_values=None):
        nonlocal count
        count += 1
        assert remote is False
        if candidate.kind == "donor_patch":
            assert donor_values == ((1.0,), (2.0,))
        return {
            "plan": asdict(candidate), "observations": _calls(candidate.feature_ids),
            "readout": {"top_token_id": 3 if candidate.kind == "lesion" else 4,
                        "target_logprob": -2.0 if candidate.kind == "lesion" else -1.0},
            "verifier_status": "unverified", "student_training_eligible": False,
        }

    result = run_bank_control_suite(None, "target", plan, trace_fn=fake_trace,
                                    donor_prompt="donor", donor_task_digest="donor")
    assert count == 6
    assert result["target_message_call"] == 18 - 12  # Pass 2 layer 12: call 6.
    assert result["comparisons"]["lesion"]["target_logprob_minus_sham"] == -1.0
    assert len(result["sha256"]) == 64
    assert len(result["random_feature_ids"]) == len(plan.feature_ids)
    assert result["verifier_status"] == "unverified"
    assert all(arm["student_training_eligible"] is False for arm in result["arms"].values())


def test_actual_nnsight_message_bank_fixture() -> None:
    torch = pytest.importorskip("torch")
    nnsight = pytest.importorskip("nnsight")

    class TinyBank(torch.nn.Module):
        def forward(self, first, second):
            return first + second

    class TinyLoop(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.message_input = TinyBank()
            self.readout = torch.nn.Linear(4, 8, bias=False)

        def forward(self, x):
            for _ in range(2):  # Calls only for passes 2 and 3.
                for _ in range(12):
                    x = x + 0.005 * self.message_input(x, x)
            return self.readout(x)

    torch.manual_seed(17)
    model = nnsight.NNsight(TinyLoop().eval())
    plan = BankPlan("toy", "fixture", "task", site_path="message_input", logits_path="output",
                    feature_ids=(0,), target_token_id=2)
    result = run_bank_control_suite(model, torch.ones(1, 1, 4), plan)
    assert len(result["arms"]["observe"]["observations"]) == 24
    assert result["arms"]["observe"]["readout"] == result["arms"]["sham"]["readout"]

def test_pytorch_causal_message_bank_fixture_without_nnsight() -> None:
    """A real recurrent forward uses 24 bank calls; a targeted patch changes output."""
    torch = pytest.importorskip("torch")

    class Bank(torch.nn.Module):
        def forward(self, block0, block1):
            return block0 + block1

    class Loop(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.message_input = Bank()

        def forward(self, x):
            for _ in range(2):
                for _ in range(12):
                    x = x + self.message_input(x, x) * 0.01
            return x

    model = Loop().eval()
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    baseline = model(x)
    visits = []

    def patch(_module, inputs):
        visit = len(visits)
        visits.append(visit)
        a, b = inputs
        if visit != message_call_index(2, 12):
            return None
        a, b = a.clone(), b.clone()
        a[0, -1, 0] = 0
        b[0, -1, 0] = 0
        return a, b

    handle = model.message_input.register_forward_pre_hook(patch)
    try:
        changed = model(x)
    finally:
        handle.remove()
    assert len(visits) == 24
    assert changed[0, 0, 0] < baseline[0, 0, 0]
    assert torch.equal(changed[0, 0, 1:], baseline[0, 0, 1:])
