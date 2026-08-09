from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from datagoblin.experiments.ldt_set_metric_robustness_v10 import (
    EXACT_SELECTED_SET_ACCURACY,
    EXACT_STATUS_VECTOR_ACCURACY,
    SetMetricContractError,
    build_development_observations,
    score_exact_set_metrics,
)
from datagoblin.serialization import load_structured

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = (
    ROOT
    / "configs"
    / "preregistrations"
    / "ldt_set_metric_robustness_v10.yaml"
)


def test_exact_metrics_align_by_candidate_id() -> None:
    oracle = (
        ("a", "SELECTED"),
        ("b", "DOMINATED"),
        ("c", "UNSAFE"),
    )
    assert score_exact_set_metrics(oracle, tuple(reversed(oracle))) == {
        EXACT_STATUS_VECTOR_ACCURACY: 1.0,
        EXACT_SELECTED_SET_ACCURACY: 1.0,
    }
    wrong_nonselected_status = (
        ("a", "SELECTED"),
        ("b", "UNRESOLVED"),
        ("c", "UNSAFE"),
    )
    assert score_exact_set_metrics(oracle, wrong_nonselected_status) == {
        EXACT_STATUS_VECTOR_ACCURACY: 0.0,
        EXACT_SELECTED_SET_ACCURACY: 1.0,
    }


@pytest.mark.parametrize(
    "prediction",
    [
        (("a", "SELECTED"), ("a", "DOMINATED")),
        (("a", "SELECTED"),),
        (("a", "NOT_A_STATUS"), ("b", "DOMINATED")),
    ],
)
def test_exact_metrics_reject_ambiguous_or_invalid_assignments(
    prediction: tuple[tuple[str, str], ...],
) -> None:
    oracle = (("a", "SELECTED"), ("b", "DOMINATED"))
    with pytest.raises(SetMetricContractError):
        score_exact_set_metrics(oracle, prediction)


def test_preregistered_probe_matrix_is_complete_and_independent() -> None:
    spec = load_structured(SPEC_PATH)
    metric_ids = {metric["id"] for metric in spec["metrics"]}
    counts = Counter(
        (metric_id, probe["probe_type"])
        for probe in spec["probes"]
        for metric_id in probe["metric_ids"]
    )
    assert metric_ids == {
        EXACT_STATUS_VECTOR_ACCURACY,
        EXACT_SELECTED_SET_ACCURACY,
    }
    assert set(probe["probe_type"] for probe in spec["probes"]) == {
        "invariance",
        "sensitivity",
        "monotonicity",
        "anti_gaming",
        "clean_control",
    }
    assert set(counts.values()) == {2}
    assert len({probe["probe_family_id"] for probe in spec["probes"]}) == 10
    assert all(probe["split"] == "development" for probe in spec["probes"])


def test_public_development_probes_match_frozen_expectations() -> None:
    rows = build_development_observations(load_structured(SPEC_PATH))
    assert len(rows) == 20
    assert len({(row["probe_id"], row["metric_id"]) for row in rows}) == 20
    for row in rows:
        expected = row["expected_behavior"]
        if expected == "stable":
            assert row["baseline_value"] == row["staged_value"]
        elif expected == "decrease":
            assert row["staged_value"] <= row["baseline_value"]
            assert (
                row["baseline_value"] - row["staged_value"]
                >= row["required_delta"]
            )
        elif expected == "detect_violation":
            assert row["observed_violation"] is True
        else:
            assert expected == "no_violation"
            assert row["observed_violation"] is False
