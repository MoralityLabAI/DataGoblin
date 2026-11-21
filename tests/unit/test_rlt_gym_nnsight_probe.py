"""Offline checks for pass identity and intervention provenance."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from datagoblin.training_gym.rlt_gym.nnsight_probe import (
    ProbePlan,
    _configuration_passes,
    _trace_context,
    check_passes,
    run_control_suite,
    sample_random_features,
    trace_once,
)


def _rows(features: tuple[int, ...], width: int = 16) -> list[dict]:
    return [
        {
            "pass_index": i, "width": width,
            "activation_l2": float(i + 1),
            "selected_features": [float(f + i) for f in features],
        }
        for i in range(3)
    ]


def test_probe_validation_and_exact_occurrence_check() -> None:
    plan = ProbePlan("model", "revision", "task", feature_ids=(1, 3), site_path="shared")
    check_passes(_rows(plan.feature_ids), 3, plan.feature_ids)
    with pytest.raises(RuntimeError, match="incomplete"):
        check_passes(_rows(plan.feature_ids)[:2], 3, plan.feature_ids)
    with pytest.raises(RuntimeError, match="reordered"):
        check_passes(_rows(plan.feature_ids)[::-1], 3, plan.feature_ids)
    with pytest.raises(ValueError, match="zero-based"):
        ProbePlan("model", "revision", "task", intervention_pass=3)
    with pytest.raises(ValueError, match="distinct"):
        ProbePlan("model", "revision", "task", feature_ids=(1, 1))
    with pytest.raises(ValueError, match="donor"):
        ProbePlan("model", "revision", "task", kind="donor_patch")


def test_random_controls_are_reproducible_disjoint_and_matched() -> None:
    sample = sample_random_features(20, (1, 4, 7), seed=13)
    assert sample == sample_random_features(20, (1, 4, 7), seed=13)
    assert len(sample) == 3 and set(sample).isdisjoint({1, 4, 7})
    with pytest.raises(ValueError, match="too few"):
        sample_random_features(3, (0, 1), seed=13)


def test_suite_keeps_synthetic_readouts_out_of_training() -> None:
    plan = ProbePlan("fixture", "rev", "task", site_path="shared", logits_path="output",
                     feature_ids=(1, 2), target_token_id=4)
    calls = []

    def fake_trace(_model, prompt, probe, *, remote, donor_values=None):
        assert remote is False
        calls.append((prompt, probe.kind))
        if probe.kind == "donor_patch":
            assert donor_values == (2.0, 3.0)
        readout = {"top_token_id": 4, "target_logprob": -1.0}
        if probe.kind == "lesion":
            readout = {"top_token_id": 7, "target_logprob": -2.0}
        return {
            "plan": asdict(probe), "observations": _rows(probe.feature_ids),
            "readout": readout, "verifier_status": "unverified",
            "student_training_eligible": False,
        }

    result = run_control_suite(None, "target", plan, trace_fn=fake_trace,
                               donor_prompt="donor", donor_task_digest="donor-digest")
    assert set(result["arms"]) == {"observe", "sham", "lesion", "random", "donor_patch"}
    assert result["comparisons"]["lesion"]["target_logprob_minus_sham"] == -1.0
    assert result["comparisons"]["lesion"]["top_token_changed_from_sham"]
    assert set(result["random_feature_ids"]).isdisjoint(plan.feature_ids)
    assert len(result["sha256"]) == 64
    assert len(calls) == 6  # Includes the donor observation.
    assert result["student_training_eligible"] is False
    assert all(not x["student_training_eligible"] for x in result["arms"].values())


def test_actual_nnsight_tiny_recurrent_fixture() -> None:
    """Three fires of one shared module must survive a single trace."""
    torch = pytest.importorskip("torch")
    nnsight = pytest.importorskip("nnsight")
    nn = torch.nn

    class TinyRecurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(4, 4, bias=False)
            self.readout = nn.Linear(4, 8, bias=False)

        def forward(self, x):
            for _ in range(3):
                x = self.shared(x).tanh()
            return self.readout(x)

    torch.manual_seed(17)
    model = nnsight.NNsight(TinyRecurrent().eval())
    prompt = torch.ones(1, 1, 4)
    plan = ProbePlan("toy-recurrent", "fixture", "toy", site_path="shared", logits_path="output",
                     feature_ids=(0,), target_token_id=0)
    result = run_control_suite(model, prompt, plan, trace_fn=trace_once)
    assert len(result["arms"]["observe"]["observations"]) == 3
    assert result["arms"]["sham"]["readout"] == result["arms"]["observe"]["readout"]
    assert result["arms"]["observe"]["configured_passes"] is None
    assert result["verifier_status"] == "unverified"

def test_inkling_runtime_configuration_precedes_wrapper_metadata() -> None:
    wrapped = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(loop_num_passes=3)),
        model=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace(loop_num_passes=5))),
    )
    assert _configuration_passes(wrapped) == 5
    assert _configuration_passes(SimpleNamespace(config=SimpleNamespace(loop_num_passes=3))) == 3
    assert _configuration_passes(SimpleNamespace(config=SimpleNamespace())) is None


def test_base_nnsight_trace_does_not_receive_remote_false() -> None:
    class TinyWrapper:
        def trace(self, prompt, **kwargs):
            assert prompt == "fixture"
            return kwargs

    model = TinyWrapper()
    assert _trace_context(model, "fixture", False) == {}
    assert _trace_context(model, "fixture", True) == {"remote": True}
