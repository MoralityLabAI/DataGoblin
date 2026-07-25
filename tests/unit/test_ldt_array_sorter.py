from __future__ import annotations

import math
from array import array

from datagoblin.experiments.ldt_array_sorter import (
    FEATURE_WHITELIST,
    INVERT_PREFERENCE,
    METRIC_FIELDS,
    LinearJointArray,
    LoopedJointArray,
    PairEncoder,
    safety_qualification,
    swap_pair,
    taste_policy,
)


def _side(**overrides: object) -> dict[str, object]:
    side: dict[str, object] = {
        "game": "court",
        "fixture_pressure": "balanced",
        "context_mode": "compact_context",
        "optimization_pressure": "parameter_efficiency_subject_to_safety",
        "trainable_parameter_bucket": "small",
        "safety_qualified": True,
        "ordinary_exact_record_accuracy": 0.9,
        "ordinary_mapping_f1": 0.9,
        "ood_exact_record_accuracy": 0.8,
        "ood_mapping_f1": 0.8,
    }
    side.update(overrides)
    assert set(side) == set(FEATURE_WHITELIST)
    return side


def _row(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    preference, rationale = taste_policy({"left": left, "right": right})
    return {
        "schema_version": "curation-taste-pair.v1",
        "example_id": "taste-test",
        "meta_split": "train",
        "lineage_group_sha256": "a" * 64,
        "source_tournament_report_sha256": "b" * 64,
        "source_kind": "environment_counterfactual_control",
        "input": {"left": left, "right": right},
        "target": {"preference": preference, "rationale_code": rationale},
        "authority": "ENVIRONMENT_OWNED",
    }


def test_taste_policy_applies_safety_pareto_efficiency_and_tradeoff() -> None:
    safe = _side()
    unsafe = _side(safety_qualified=False, **{field: 1.0 for field in METRIC_FIELDS})
    assert taste_policy({"left": safe, "right": unsafe}) == ("LEFT", "safety_gate")

    dominated = _side(**{field: 0.5 for field in METRIC_FIELDS})
    assert taste_policy({"left": safe, "right": dominated}) == ("LEFT", "pareto_dominance")

    expensive = _side(trainable_parameter_bucket="large")
    assert taste_policy({"left": safe, "right": expensive}) == ("LEFT", "parameter_efficiency")

    ordinary = _side(
        ordinary_exact_record_accuracy=1.0,
        ordinary_mapping_f1=1.0,
        ood_exact_record_accuracy=0.5,
        ood_mapping_f1=0.5,
        trainable_parameter_bucket="medium",
    )
    robust = _side(
        ordinary_exact_record_accuracy=0.5,
        ordinary_mapping_f1=0.5,
        ood_exact_record_accuracy=1.0,
        ood_mapping_f1=1.0,
        trainable_parameter_bucket="medium",
    )
    assert taste_policy({"left": ordinary, "right": robust}) == (
        "TRADEOFF",
        "irreducible_tradeoff",
    )


def test_swap_augmentation_preserves_lineage_and_inverts_only_direction() -> None:
    row = _row(_side(), _side(safety_qualified=False))
    swapped = swap_pair(row)
    assert swapped["lineage_group_sha256"] == row["lineage_group_sha256"]
    assert swapped["input"]["left"] == row["input"]["right"]
    assert swapped["target"]["preference"] == INVERT_PREFERENCE[row["target"]["preference"]]
    assert swapped["target"]["rationale_code"] == row["target"]["rationale_code"]
    assert taste_policy(swapped["input"]) == (
        swapped["target"]["preference"],
        swapped["target"]["rationale_code"],
    )


def test_encoder_is_train_fitted_and_emits_fixed_dense_arrays() -> None:
    row = _row(_side(), _side(context_mode="normal_full_context"))
    encoder = PairEncoder.fit([row])
    raw = encoder.transform(row["input"], view="raw")
    typed = encoder.transform(row["input"], view="typed")
    unknown = _row(
        _side(context_mode="never_seen"),
        _side(context_mode="also_never_seen"),
    )
    unknown_typed = encoder.transform(unknown["input"], view="typed")
    assert isinstance(raw, array)
    assert isinstance(typed, array)
    assert len(raw) == len(encoder.feature_names("raw"))
    assert len(typed) == len(encoder.feature_names("typed"))
    assert len(unknown_typed) == len(typed)
    assert len(typed) == len(raw) + 6


def test_weight_tied_loops_add_compute_without_adding_parameters() -> None:
    width = 42
    two_loop = LoopedJointArray(width, hidden_width=4, loop_count=2, seed=11)
    four_loop = LoopedJointArray(width, hidden_width=4, loop_count=4, seed=11)
    one_pass = LoopedJointArray(width, hidden_width=4, loop_count=1, seed=11)
    linear = LinearJointArray(width, seed=11)
    assert two_loop.parameter_count == four_loop.parameter_count
    assert one_pass.parameter_count < two_loop.parameter_count
    assert linear.parameter_count > two_loop.parameter_count
    features = array("d", [0.0]) * width
    features[0] = 1.0
    for model in (linear, one_pass, two_loop, four_loop):
        preference, rationale = model.predict_proba(features)
        assert math.isclose(sum(preference), 1.0)
        assert math.isclose(sum(rationale), 1.0)


def test_safety_qualification_rejects_one_sided_majority_behavior() -> None:
    assert safety_qualification(1.0, 1.0)
    assert not safety_qualification(1.0, 0.0)
    assert not safety_qualification(0.99, 1.0)
