"""Train-only STOP head and conservative recurrent stopping regression tests."""
from __future__ import annotations

import numpy as np
import pytest

from MeTTa.airis02_stop import StopHead
from MeTTa.airis02_student import INPUT_DIM, LinearStudent, RecurrentStudent, evaluate
from MeTTa.airis02_fresh import _frozen_motifs, build_row


def test_separate_binary_stop_gate_is_deterministic_and_bounded():
    rng = np.random.default_rng(77)
    x = rng.normal(size=(64, INPUT_DIM))
    y = np.zeros(64, dtype=np.int64)
    y[::4] = 1
    left = StopHead(17)
    right = StopHead(17)
    a = left.fit(x, y, steps=32, lr=0.18)
    b = right.fit(x, y, steps=32, lr=0.18)
    assert a == b
    assert left.digest() == right.digest()
    assert left.params == INPUT_DIM + 1
    assert 0.0 <= a["train_false_stop_rate"] <= 0.05
    assert a["threshold_selected_on"] == "training_only"
    assert a["no_development_label_calibration"] is True
    assert a["train_positive"] == 16
    assert a["train_negative"] == 48
    assert left.should_stop(x[0:1]) in (True, False)


def test_stop_gate_requires_learner_fit_and_both_labels():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(12, INPUT_DIM))
    h = StopHead(3)
    with pytest.raises(ValueError, match="must be fitted"):
        h.should_stop(x[0:1])
    with pytest.raises(ValueError, match="both positive"):
        h.fit(x, np.zeros(12, dtype=np.int64), steps=5, lr=0.1)
    with pytest.raises(ValueError, match="false-positive target"):
        h.fit(x, np.array([0, 1] * 6), steps=5, lr=0.1, max_false_stop_rate=0.3)
    with pytest.raises(ValueError, match="shape"):
        h.fit(x[:, :-1], np.array([0, 1] * 6), steps=5, lr=0.1)
    with pytest.raises(ValueError, match="binary"):
        h.fit(x, np.array([0, 2] * 6), steps=5, lr=0.1)


def test_learned_stop_evaluation_preserves_exact_verifier_and_source_custody():
    motifs = _frozen_motifs()
    row = build_row("budget", 512, "uniform", motifs)
    model = RecurrentStudent(42, hidden=8)
    x = np.random.default_rng(42).normal(size=(40, INPUT_DIM))
    labels = np.array([0, 1] * 20, dtype=np.int64)
    head = StopHead(8)
    head.fit(x, labels, steps=8, lr=0.12)
    raw = evaluate(model, [row], loops=4)
    guarded = evaluate(model, [row], loops=4, stop_head=head)
    assert raw["cases"] == guarded["cases"] == 1
    assert raw["mean_learned_stop_head_calls"] == 0
    assert guarded["mean_learned_stop_head_calls"] >= 1
    assert guarded["inapplicable_actions"] == 0
    assert guarded["solved_and_stopped"] <= guarded["ever_reached_optimum"]
    assert 0 <= guarded["premature_stops"] <= 1
    assert guarded["mean_remaining_canonical_edits"] >= 0
    assert isinstance(guarded["mean_inference_iterations"], float)


def test_stop_head_compatible_with_simple_direct_student():
    motifs = _frozen_motifs()
    rows = [build_row("coverage", 512, "uniform", motifs)]
    head = StopHead(1)
    x = np.random.default_rng(21).normal(size=(12, INPUT_DIM))
    y = np.array([0, 1] * 6)
    head.fit(x, y, steps=3, lr=0.1)
    direct = LinearStudent(1)
    result = evaluate(direct, rows, loops=1, stop_head=head)
    assert result["cases"] == 1
    assert result["mean_learned_stop_head_calls"] >= 1
    assert result["inapplicable_actions"] == 0
