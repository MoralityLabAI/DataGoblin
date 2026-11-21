"""Source-only family-prior verifier-cost controls, with root-level CAL custody."""

from __future__ import annotations

import pytest

from datagoblin.training_gym.rlt_gym.family_exit_baseline import (
    choose_family_threshold,
    evaluate_family,
    fit_family,
    score,
)
from datagoblin.training_gym.rlt_gym.postfailure_exit import FAMILIES


def _case(split, family, index, *, first=False, rescue=False):
    return {
        "root_id": f"{split}-{family}-{index}",
        "split": split,
        "family": family,
        "first_passed": first,
        "future_rescue": rescue,
        "oracle_full_depth_checks": 1 if first else 2 if rescue else 4,
    }


def test_train_only_priors_and_pseudocounts():
    rows = [
        _case("train", "table_transform", i, rescue=i < 3) for i in range(7)
    ] + [
        _case("train", "graph_route", i, rescue=False) for i in range(9)
    ]
    model = fit_family(rows)
    assert model["family_failures"]["table_transform"] == 7
    assert model["family_future_rescues"]["table_transform"] == 3
    assert score(rows[0], model) > score(rows[-1], model)
    assert all(0 < model["family_probabilities"][f] < 1 for f in FAMILIES)


def test_no_calibration_or_repeated_train_roots():
    train = [_case("train", "graph_route", 0), _case("train", "graph_route", 1)]
    model = fit_family(train)
    with pytest.raises(ValueError, match="TRAIN"):
        fit_family(train + [_case("calibration", "graph_route", 2)])
    with pytest.raises(ValueError, match="duplicate"):
        fit_family([train[0], train[0]])
    with pytest.raises(ValueError, match="CALIBRATION"):
        choose_family_threshold(train, model)
    with pytest.raises(ValueError, match="ordinary/OOD"):
        evaluate_family([train[0]], model, {"threshold": 0.5})


def test_calibration_retains_all_future_rescues():
    train = [
        _case("train", "table_transform", i, rescue=i < 4) for i in range(12)
    ] + [
        _case("train", "graph_route", i, rescue=False) for i in range(12)
    ]
    model = fit_family(train)
    cal = [
        _case("calibration", "table_transform", i, rescue=i < 2) for i in range(8)
    ] + [
        _case("calibration", "graph_route", i, rescue=False) for i in range(8)
    ]
    selection = choose_family_threshold(cal, model, missed_rescue_allowance=0)
    assert selection["calibration_policy"]["missed_future_rescues"] == 0
    assert selection["calibration_policy"]["solved"] == 2
    assert selection["calibration_first_valid"]["solved"] == 2
    assert selection["calibration_policy"]["verifier_calls"] < selection["calibration_first_valid"]["verifier_calls"]
    dev = [
        _case("ordinary_test", "table_transform", i, rescue=i < 2) for i in range(6)
    ] + [
        _case("ordinary_test", "graph_route", i, rescue=False) for i in range(6)
    ]
    scorecard = evaluate_family(dev, model, selection)
    assert scorecard["policies"]["family_continue"]["solved"] == 2
    assert scorecard["policies"]["first_valid"]["solved"] == 2
    assert scorecard["measured_runtime_savings"] is False


def test_unsafe_or_unknown_family_rejected():
    model = fit_family([_case("train", "graph_route", 0)])
    with pytest.raises(ValueError, match="unknown family"):
        score(_case("ordinary_test", "not_registered", 0), model)
    with pytest.raises(ValueError, match="missed-rescue allowance"):
        choose_family_threshold(
            [_case("calibration", "graph_route", 0)], model,
            missed_rescue_allowance=-1,
        )
