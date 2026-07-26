from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any


def _evaluator_namespace() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return runpy.run_path(str(root / "scripts/evaluate_ldt_array_hard_surface.py"))


def _prediction(
    array_id: str,
    left_index: int,
    right_index: int,
    preference: str,
) -> dict[str, object]:
    return {
        "array_id": array_id,
        "left_index": left_index,
        "right_index": right_index,
        "preference": preference,
    }


def test_array_reducer_keeps_abstention_distinct_from_tradeoff() -> None:
    derive = _evaluator_namespace()["_derive_predicted_array_statuses"]
    array_id = "array-test"
    arrays = {
        array_id: {
            "candidates": [
                {"safety_qualified": True},
                {"safety_qualified": True},
                {"safety_qualified": True},
                {"safety_qualified": False},
            ]
        }
    }
    predictions = [
        _prediction(array_id, 0, 1, "ABSTAIN"),
        _prediction(array_id, 1, 0, "ABSTAIN"),
        _prediction(array_id, 0, 2, "LEFT"),
        _prediction(array_id, 2, 0, "RIGHT"),
        _prediction(array_id, 1, 2, "TRADEOFF"),
        _prediction(array_id, 2, 1, "TRADEOFF"),
    ]
    assert derive(predictions, arrays)[array_id] == [
        "UNRESOLVED",
        "UNRESOLVED",
        "DOMINATED",
        "UNSAFE",
    ]


def test_array_reducer_marks_orientation_disagreement_unresolved() -> None:
    derive = _evaluator_namespace()["_derive_predicted_array_statuses"]
    array_id = "array-test"
    arrays = {
        array_id: {
            "candidates": [
                {"safety_qualified": True},
                {"safety_qualified": True},
            ]
        }
    }
    predictions = [
        _prediction(array_id, 0, 1, "LEFT"),
        _prediction(array_id, 1, 0, "LEFT"),
    ]
    assert derive(predictions, arrays)[array_id] == [
        "UNRESOLVED",
        "UNRESOLVED",
    ]


def test_pareto_rule_uses_parameters_only_for_metric_equivalence() -> None:
    dominates = _evaluator_namespace()["_dominates"]
    names = (
        "preference_macro_f1",
        "abstain_recall",
        "rationale_macro_f1",
        "status_macro_f1",
        "exact_array_accuracy",
        "selected_set_exact_accuracy",
    )

    def aggregate(value: float, parameters: int) -> dict[str, object]:
        return {
            "metrics": {name: {"mean": value} for name in names},
            "trainable_parameters": parameters,
        }

    compact = aggregate(1.0, 297)
    larger = aggregate(1.0, 313)
    cheaper_but_worse = aggregate(0.98, 10)
    assert dominates(compact, larger, 0.01)
    assert not dominates(cheaper_but_worse, compact, 0.01)
