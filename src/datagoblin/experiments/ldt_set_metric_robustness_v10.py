from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datagoblin.experiments.ldt_array_hard_surface import ARRAY_STATUSES
from datagoblin.serialization import canonical_json, load_structured

EXACT_STATUS_VECTOR_ACCURACY = "exact_status_vector_accuracy"
EXACT_SELECTED_SET_ACCURACY = "exact_selected_set_accuracy"
SUPPORTED_METRICS = (
    EXACT_STATUS_VECTOR_ACCURACY,
    EXACT_SELECTED_SET_ACCURACY,
)

StatusAssignment = Sequence[tuple[str, str]]


class SetMetricContractError(ValueError):
    """Raised when an array cannot be aligned without ambiguity."""


def _index_assignment(
    assignment: StatusAssignment,
    *,
    role: str,
) -> dict[str, str]:
    if not assignment:
        raise SetMetricContractError(f"{role} assignment is empty")
    indexed: dict[str, str] = {}
    for candidate_id, status in assignment:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise SetMetricContractError(f"{role} candidate ID is empty")
        if candidate_id in indexed:
            raise SetMetricContractError(
                f"{role} candidate ID is duplicated: {candidate_id}"
            )
        if status not in ARRAY_STATUSES:
            raise SetMetricContractError(
                f"{role} status is invalid for {candidate_id}: {status}"
            )
        indexed[candidate_id] = status
    return indexed


def score_exact_set_metrics(
    oracle: StatusAssignment,
    predicted: StatusAssignment,
) -> dict[str, float]:
    """Score one array after strict candidate-ID alignment.

    Both metrics are binary at the array level. Candidate order is a nuisance,
    while duplicate, missing, or extra candidate IDs are contract violations.
    """

    oracle_by_id = _index_assignment(oracle, role="oracle")
    predicted_by_id = _index_assignment(predicted, role="predicted")
    if set(oracle_by_id) != set(predicted_by_id):
        missing = sorted(set(oracle_by_id) - set(predicted_by_id))
        extra = sorted(set(predicted_by_id) - set(oracle_by_id))
        raise SetMetricContractError(
            f"candidate inventory mismatch; missing={missing}, extra={extra}"
        )
    exact_status = all(
        predicted_by_id[candidate_id] == oracle_status
        for candidate_id, oracle_status in oracle_by_id.items()
    )
    oracle_selected = {
        candidate_id
        for candidate_id, status in oracle_by_id.items()
        if status == "SELECTED"
    }
    predicted_selected = {
        candidate_id
        for candidate_id, status in predicted_by_id.items()
        if status == "SELECTED"
    }
    return {
        EXACT_STATUS_VECTOR_ACCURACY: float(exact_status),
        EXACT_SELECTED_SET_ACCURACY: float(
            predicted_selected == oracle_selected
        ),
    }


_ORACLE: tuple[tuple[str, str], ...] = (
    ("candidate-a", "SELECTED"),
    ("candidate-b", "SELECTED"),
    ("candidate-c", "DOMINATED"),
    ("candidate-d", "UNRESOLVED"),
    ("candidate-e", "UNSAFE"),
    ("candidate-f", "DOMINATED"),
)


def _replace(
    assignment: StatusAssignment,
    replacements: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (candidate_id, replacements.get(candidate_id, status))
        for candidate_id, status in assignment
    )


def _score_pair(
    baseline_oracle: StatusAssignment,
    baseline_prediction: StatusAssignment,
    staged_oracle: StatusAssignment,
    staged_prediction: StatusAssignment,
) -> dict[str, dict[str, float]]:
    baseline = score_exact_set_metrics(
        baseline_oracle,
        baseline_prediction,
    )
    staged = score_exact_set_metrics(staged_oracle, staged_prediction)
    return {
        metric_id: {
            "baseline_value": baseline[metric_id],
            "staged_value": staged[metric_id],
        }
        for metric_id in SUPPORTED_METRICS
    }


def _detect_attack(
    oracle: StatusAssignment,
    prediction: StatusAssignment,
) -> dict[str, dict[str, bool]]:
    try:
        scores = score_exact_set_metrics(oracle, prediction)
    except SetMetricContractError:
        return {
            metric_id: {"observed_violation": True}
            for metric_id in SUPPORTED_METRICS
        }
    return {
        metric_id: {"observed_violation": scores[metric_id] < 1.0}
        for metric_id in SUPPORTED_METRICS
    }


def _clean_control(
    oracle: StatusAssignment,
    prediction: StatusAssignment,
) -> dict[str, dict[str, bool]]:
    detected = _detect_attack(oracle, prediction)
    return {
        metric_id: {
            "observed_violation": detected[metric_id][
                "observed_violation"
            ]
        }
        for metric_id in SUPPORTED_METRICS
    }


def _fixture_outcomes() -> dict[str, dict[str, dict[str, float | bool]]]:
    reversed_oracle = tuple(reversed(_ORACLE))
    renamed = tuple(
        (f"renamed-{index}", status)
        for index, (_, status) in enumerate(_ORACLE, start=1)
    )
    one_removal = _replace(_ORACLE, {"candidate-a": "DOMINATED"})
    two_removals = _replace(
        _ORACLE,
        {"candidate-a": "DOMINATED", "candidate-b": "DOMINATED"},
    )
    one_insertion = _replace(_ORACLE, {"candidate-c": "SELECTED"})
    two_insertions = _replace(
        _ORACLE,
        {"candidate-c": "SELECTED", "candidate-f": "SELECTED"},
    )
    duplicate_identity = (*_ORACLE[:-1], ("candidate-e", "DOMINATED"))
    prefix = "fixture://datagoblin-v10-development/"
    return {
        prefix + "candidate-order-permutation": _score_pair(
            _ORACLE,
            _ORACLE,
            _ORACLE,
            reversed_oracle,
        ),
        prefix + "candidate-id-alpha-renaming": _score_pair(
            _ORACLE,
            _ORACLE,
            renamed,
            renamed,
        ),
        prefix + "selected-member-removal": _score_pair(
            _ORACLE,
            _ORACLE,
            _ORACLE,
            one_removal,
        ),
        prefix + "selected-nonmember-insertion": _score_pair(
            _ORACLE,
            _ORACLE,
            _ORACLE,
            one_insertion,
        ),
        prefix + "compounded-selected-removals": _score_pair(
            _ORACLE,
            one_removal,
            _ORACLE,
            two_removals,
        ),
        prefix + "compounded-selected-insertions": _score_pair(
            _ORACLE,
            one_insertion,
            _ORACLE,
            two_insertions,
        ),
        prefix + "factorized-majority-decoy": _detect_attack(
            _ORACLE,
            one_removal,
        ),
        prefix + "duplicate-candidate-identity": _detect_attack(
            _ORACLE,
            duplicate_identity,
        ),
        prefix + "exact-canonical-control": _clean_control(
            _ORACLE,
            _ORACLE,
        ),
        prefix + "exact-reordered-control": _clean_control(
            _ORACLE,
            reversed_oracle,
        ),
    }


def build_development_observations(
    spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Execute the public development fixtures from a staged v10 spec."""

    if spec.get("schema_version") != "1.0":
        raise ValueError("unsupported robustness specification")
    if spec.get("evidence_scope") != "development":
        raise ValueError("public fixtures provide development evidence only")
    declared_metrics = {
        str(metric["id"]) for metric in spec.get("metrics", ())
    }
    if declared_metrics != set(SUPPORTED_METRICS):
        raise ValueError("v10 metric declaration differs from implementation")
    fixtures = _fixture_outcomes()
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for probe in spec.get("probes", ()):
        source_ref = str(probe["source_ref"])
        if source_ref not in fixtures:
            raise ValueError(f"unknown development fixture: {source_ref}")
        for raw_metric_id in probe["metric_ids"]:
            metric_id = str(raw_metric_id)
            key = (str(probe["probe_id"]), metric_id)
            if key in seen_keys:
                raise ValueError(f"duplicate probe/metric key: {key}")
            seen_keys.add(key)
            outcome = fixtures[source_ref][metric_id]
            row = {
                "schema_version": "1.0",
                "probe_id": probe["probe_id"],
                "probe_family_id": probe["probe_family_id"],
                "metric_id": metric_id,
                "probe_type": probe["probe_type"],
                "split": probe["split"],
                "expected_behavior": probe["expected_behavior"],
                **outcome,
            }
            for field in ("tolerance", "required_delta"):
                if field in probe:
                    row[field] = probe[field]
            rows.append(row)
    return rows


def emit_development_observations(
    spec_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    rows = build_development_observations(load_structured(spec_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{canonical_json(row)}\n" for row in rows),
        encoding="utf-8",
    )
    return rows
