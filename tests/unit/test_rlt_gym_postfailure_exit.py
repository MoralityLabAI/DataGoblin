"""Verifier-backed candidate replay and safe continuation calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datagoblin.training_gym.rlt_gym.postfailure_exit import (
    WIDTH, _policy, campaign, choose_threshold, evaluate, fit, predict,
)
from datagoblin.training_gym.rlt_gym.runner import run_suite
from datagoblin.training_gym.rlt_gym.tasks import write_suite


def _case(split: str, idx: int, *, success: bool = False, rescue: bool = False):
    vector = [0.0] * WIDTH
    vector[0] = 1.0 if rescue else -1.0
    vector[1] = idx / 20
    return {
        "root_id": f"{split}:root:{idx}", "task_digest": f"digest-{split}-{idx}",
        "family": "graph_route" if idx % 2 == 0 else "arithmetic_program",
        "split": split, "model_digest": "test-model",
        "verification_code": "route_endpoints", "visit_budget": 4,
        "first_passed": success, "future_rescue": rescue,
        "first_valid_visit_zero_based": 0 if success else 2 if rescue else None,
        "postfailure_features": None if success else vector,
        "oracle_full_depth_checks": 1 if success else 3 if rescue else 4,
        "evidence_class": "fixture_not_model_inference", "trainable": split == "train",
    }


def test_calibrate_only_verified_future_rescue_and_cost() -> None:
    train = [_case("train", i, rescue=(i % 4 in {0, 1})) for i in range(24)]
    model = fit(train)
    positives = [_case("calibration", i, rescue=(i % 4 in {0, 1})) for i in range(12)]
    threshold = choose_threshold(positives, model, missed_rescue_allowance=0)
    assert threshold["calibration_policy"]["missed_future_rescues"] == 0
    assert threshold["calibration_first_valid"]["solved"] == 6
    result = evaluate(
        [_case("ordinary_test", i, rescue=(i % 4 in {0, 1})) for i in range(12)],
        model, threshold,
    )
    assert result["policies"]["first_valid"]["solved"] == 6
    assert result["policies"]["fixed_one"]["solved"] == 0
    assert result["policies"]["learned_continue"]["solved"] <= 6
    assert result["model_is_reexecuted_with_dynamic_stop"] is False
    assert all(0 <= predict(row, model) <= 1 for row in train)


def test_strict_splits_and_duplicate_training_roots() -> None:
    train = [_case("train", 0, rescue=True), _case("train", 1, rescue=False)]
    with pytest.raises(ValueError, match="TRAIN"):
        fit([*train, _case("calibration", 99, rescue=False)])
    with pytest.raises(ValueError, match="duplicate"):
        fit([train[0], train[0]])
    model = fit(train)
    with pytest.raises(ValueError, match="calibration"):
        choose_threshold([_case("train", 1, rescue=True)], model)
    with pytest.raises(ValueError, match="ordinary/OOD"):
        evaluate([_case("calibration", 2)], model, {"threshold": 1.0})


def test_per_root_oracle_call_count() -> None:
    rows = [
        _case("ordinary_test", 0, success=True),
        _case("ordinary_test", 1, rescue=True),
        _case("ordinary_test", 2),
    ]
    result = _policy(rows, {row["root_id"]: True for row in rows})
    assert result["solved"] == 2
    assert result["verifier_calls"] == 1 + 3 + 4
    assert result["rescued_after_first_failure"] == 1
    early = _policy(rows, {row["root_id"]: False for row in rows})
    assert early["solved"] == 1
    assert early["verifier_calls"] == 3
    assert early["missed_future_rescues"] == 1


def test_real_exact_replay_seven_family_rlt_suite(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    write_suite(suite, seed=104, per_family=1)
    run = tmp_path / "run"
    result = run_suite(
        suite, run, seeds=(7,), modes=("direct",),
        training_modes=("direct",), depths=(2,),
        splits=("train", "calibration", "ordinary_test", "ood_test"),
        training_depth=2, verified_stop=False, include_controls=False,
        max_wall_seconds=120,
    )
    assert result["expected_episodes"] == 28
    report = campaign(suite, run, tmp_path / "output", planned_depth=2)
    assert report["provenance"]["source_roots"] == 28
    assert report["provenance"]["split_counts"] == {
        "calibration": 7, "ood_test": 7, "ordinary_test": 7, "train": 7,
    }
    assert report["evaluations"]["ordinary_test"]["roots"] == 7
    assert report["evaluations"]["ood_test"]["roots"] == 7
    assert report["lean_mcp_transfer_status"] == "awaiting_verifier_backed_independent_source_adapters"
    assert (tmp_path / "output" / "controller.json").is_file()
    assert (tmp_path / "output" / "selection.json").is_file()
    assert json.loads((tmp_path / "output" / "report.json").read_text())["schema_version"] == (
        "rlt_gym.postfailure_exit.v1"
    )
    with pytest.raises(ValueError, match="output directory exists"):
        campaign(suite, run, tmp_path / "output", planned_depth=2)
