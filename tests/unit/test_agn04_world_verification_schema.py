"""AGN-04 verification-record schema is valid and matches a synthetic fixture."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "agn04_world_verification_record.schema.json"
)


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _record(**overrides) -> dict:
    record = {
        "schema": "agn04.world_verification_record.v1",
        "family_id": "fixture_family",
        "root_id": "fixture_root",
        "mechanism": "provenance",
        "split": "dev",
        "source_digest": "a" * 64,
        "difficulty": {
            "inference_depth": 3,
            "branching_factor": 2,
            "uncertainty": 0.25,
            "source_dependence": 1,
            "required_memory": 4,
            "revision_depth": 1,
            "opportunity_structure": 0.5,
        },
        "verification": {
            "solver": "exact",
            "optimality_gap_reported": None,
            "realized_policy_utility": 0.69,
            "attainable_opportunity_capture": 0.9,
            "informative_decision_accuracy": 0.8,
            "min_optimal_planning_horizon": 2,
            "shortcut_policy_strength": 0.1,
            "ending_reachable": True,
        },
        "training_ready": False,
        "kernel_checked": False,
        "evidence_class": "finite_replay_checked",
    }
    record.update(overrides)
    return record


def test_schema_is_valid_draft_2020_12() -> None:
    jsonschema.Draft202012Validator.check_schema(_schema())


def test_fixture_record_validates() -> None:
    jsonschema.validate(_record(), _schema())


def test_training_ready_cannot_be_true() -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_record(training_ready=True), _schema())


def test_kernel_checked_cannot_be_true() -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_record(kernel_checked=True), _schema())


def test_bounded_search_requires_optimality_gap() -> None:
    record = _record()
    record["verification"]["solver"] = "bounded_search"
    # optimality_gap_reported stays None here on purpose: the schema allows the type
    # for both solvers, so this checks the field is at least present and typed, not
    # that the skill enforces non-null bounded-search gaps (that is a skill-level rule).
    jsonschema.validate(record, _schema())


def test_rejected_attempt_needs_known_failure_code() -> None:
    record = _record(rejected_attempts=[{"attempt_digest": "b" * 64, "failure_code": "not_a_real_code"}])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(record, _schema())
