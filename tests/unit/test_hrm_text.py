from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from datagoblin.experiments.recipe_verdict_dataset import generate_verdict_dataset
from datagoblin.hrm_text import availability
from datagoblin.hrm_text.calibration import fit_isotonic_calibration, fit_logistic_calibration
from datagoblin.hrm_text.cycle_metrics import (
    compute_cycle_features,
    jensen_shannon_from_logits,
    safe_relative_update,
)
from datagoblin.hrm_text.instrumented_forward import (
    DeterministicMockRecurrentModel,
    TraceOptions,
    run_instrumented_forward,
)
from datagoblin.hrm_text.trace_protocol import TRACE_ORDER


def _trace(options: TraceOptions | None = None):
    model = DeterministicMockRecurrentModel()
    return run_instrumented_forward(
        model,
        "minimal deterministic prompt",
        authoritative_decision=None,
        schema_complexity=1.0,
        options=options,
    )


def test_trace_order_and_instrumented_logit_equivalence() -> None:
    summary = _trace()
    assert summary.trace_order == TRACE_ORDER
    assert summary.instrumented_logits_max_abs_error == 0.0


def test_instrumented_logit_regression_gate_fails_closed() -> None:
    class MismatchedModel(DeterministicMockRecurrentModel):
        def trace_forward(self, prompt, observer):
            output = super().trace_forward(prompt, observer)
            return replace(output, final_logits=output.final_logits + 1.0)

    with pytest.raises(ValueError, match="instrumented logits differ"):
        run_instrumented_forward(
            MismatchedModel(),
            "mismatch",
            authoritative_decision=None,
            schema_complexity=1.0,
        )


def test_cycle_metrics_are_stable_and_handle_zero_norms() -> None:
    zero = np.zeros((2, 3))
    assert safe_relative_update(zero, zero) == 0.0
    assert jensen_shannon_from_logits(np.zeros(3), np.zeros(3)) == pytest.approx(0.0)
    states = {name: zero.copy() for name in TRACE_ORDER}
    first = compute_cycle_features(states, step_updates=[0.0] * 8, h1_logits=np.zeros(3), h2_logits=np.zeros(3))
    second = compute_cycle_features(states, step_updates=[0.0] * 8, h1_logits=np.zeros(3), h2_logits=np.zeros(3))
    assert first == second
    assert all(np.isfinite(value) for value in first.to_jsonable().values())


def test_no_full_activation_persistence_by_default_and_budget_is_enforced() -> None:
    persisted = _trace().to_jsonable()
    assert "sampled_activations" not in persisted
    assert not any("state" in key or "activation" in key for key in persisted)
    sampled = _trace(TraceOptions(store_sampled_activations=True, activation_byte_budget=64))
    assert sum(len(values) for values in sampled.sampled_activations.values()) * 8 <= 64
    with pytest.raises(ValueError, match="output byte budget"):
        _trace(TraceOptions(output_byte_budget=100))


def test_calibration_fit_rejects_test_rows() -> None:
    calibration = [
        {"example_id": "a", "split": "calibration", "score": 0.1, "other": 1.0, "error_label": 0.0},
        {"example_id": "b", "split": "calibration", "score": 0.9, "other": 0.0, "error_label": 1.0},
    ]
    assert fit_logistic_calibration(calibration, ("score", "other")).fit_split == "calibration"
    assert fit_isotonic_calibration(calibration, "score").fit_split == "calibration"
    invalid = [dict(calibration[0], split="ordinary_test")]
    with pytest.raises(ValueError, match="calibration split"):
        fit_logistic_calibration(invalid, ("score",))


def test_dataset_is_deterministic_and_sealed_labels_are_isolated(tmp_path: Path) -> None:
    recipe = Path(__file__).parents[2] / "examples" / "recipe.good.yaml"
    first = generate_verdict_dataset(recipe, tmp_path / "first", seed=17)
    second = generate_verdict_dataset(recipe, tmp_path / "second", seed=17)
    assert first.to_jsonable()["splits"] == second.to_jsonable()["splits"]
    sealed_path = tmp_path / "first" / "sealed_poison.jsonl"
    rows = [json.loads(line) for line in sealed_path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all("authoritative_label" not in row for row in rows)
    assert all("receipt_hash" not in row for row in rows)
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "first" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected = {
        "evaluation_leakage": "evaluation_leakage",
        "missing_verifier": "missing_independent_verifier",
        "builder_evaluator_identity_collision": "builder_evaluator_identity",
        "unbounded_budget": "unbounded_budget",
        "missing_falsifier": "missing_falsifier",
        "trajectory_mixture_errors": "trajectory_mixture_invalid",
        "self_certified_evidence": "self_certified_environment_evidence",
        "metric_mutation": "metric_mutated_after_freeze",
        "hidden_seed_exposure": "evaluation_leakage",
        "diversity_collapse": "curriculum_diversity_collapse",
        "cherry_picked_seeds": "cherry_picked_seeds",
    }
    for row in train_rows:
        if row["category"] == "valid":
            assert row["authoritative_label"]["decision"] == "ACCEPT"
        else:
            assert expected[row["category"]] in row["authoritative_label"]["violations"]


def test_availability_check_does_not_import_optional_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(availability.importlib.util, "find_spec", lambda _name: None)
    status = availability.detect_hrm_availability()
    assert not status.torch_available
    assert not status.transformers_available
    assert not status.weights_loaded
    assert not status.download_attempted
