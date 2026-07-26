from __future__ import annotations

import json
import math
import runpy
from array import array
from pathlib import Path

import pytest

from datagoblin.experiments.ldt_array_confirmation import (
    ConfirmationPairEncoder,
    derive_confirmation_array_target,
    generate_confirmation_dataset,
    load_confirmation_dataset,
    partial_interval_taste_policy,
    reduce_confirmation_array,
    swap_confirmation_pair,
)
from datagoblin.experiments.ldt_array_hard_surface import HARD_INVERT_PREFERENCE
from datagoblin.experiments.ldt_array_sorter import METRIC_FIELDS, make_model
from datagoblin.experiments.ldt_target_independent_v7 import (
    generate_target_independent_dataset,
)


def _candidate(
    values: tuple[float, float, float, float],
    *,
    safe: bool = True,
    bucket: str = "medium",
    lower_missing: int | None = None,
    upper_missing: int | None = None,
    width: float = 0.0,
) -> dict[str, object]:
    result: dict[str, object] = {
        "game": "court",
        "fixture_pressure": "test",
        "context_mode": "test",
        "optimization_pressure": "test",
        "trainable_parameter_bucket": bucket,
        "safety_qualified": safe,
    }
    for index, (field, value) in enumerate(
        zip(METRIC_FIELDS, values, strict=True)
    ):
        result[field] = {
            "lower": None if lower_missing == index else value - width,
            "upper": None if upper_missing == index else value + width,
        }
    return result


def test_partial_policy_is_conservative_without_discarding_robust_bounds() -> None:
    baseline = _candidate((0.7, 0.7, 0.7, 0.7))
    lower_only_strong = _candidate(
        (0.85, 0.85, 0.85, 0.85),
        upper_missing=0,
    )
    upper_only = _candidate(
        (0.85, 0.85, 0.85, 0.85),
        lower_missing=0,
    )
    assert partial_interval_taste_policy(
        {"left": lower_only_strong, "right": baseline}
    ) == ("LEFT", "pareto_dominance")
    assert partial_interval_taste_policy(
        {"left": upper_only, "right": baseline}
    ) == ("ABSTAIN", "insufficient_evidence")


def test_confirmation_swap_inverts_bindings_and_direction() -> None:
    row = {
        "left_index": 1,
        "right_index": 3,
        "left_tournament_report_sha256": "a" * 64,
        "right_tournament_report_sha256": "b" * 64,
        "input": {
            "left": _candidate((0.8, 0.8, 0.8, 0.8)),
            "right": _candidate((0.7, 0.7, 0.7, 0.7)),
        },
        "target": {
            "preference": "LEFT",
            "rationale_code": "pareto_dominance",
        },
    }
    swapped = swap_confirmation_pair(row)
    assert swapped["left_index"] == 3
    assert swapped["right_index"] == 1
    assert swapped["left_tournament_report_sha256"] == "b" * 64
    assert swapped["target"]["preference"] == HARD_INVERT_PREFERENCE["LEFT"]


def test_confirmation_array_reducer_preserves_status_precedence() -> None:
    candidates = [
        _candidate((0.8, 0.8, 0.8, 0.8)),
        _candidate((0.7, 0.7, 0.7, 0.7)),
        _candidate((0.9, 0.9, 0.9, 0.9), lower_missing=0),
        _candidate((0.99, 0.99, 0.99, 0.99), safe=False),
    ]
    target = derive_confirmation_array_target(candidates, tolerance=0.01)
    assert target["statuses"] == [
        "UNRESOLVED",
        "DOMINATED",
        "UNRESOLVED",
        "UNSAFE",
    ]
    preferences = {
        (0, 1): "LEFT",
        (1, 0): "RIGHT",
        (0, 2): "LEFT",
        (2, 0): "LEFT",
        (1, 2): "TRADEOFF",
        (2, 1): "TRADEOFF",
    }
    assert reduce_confirmation_array(candidates, preferences)["statuses"] == [
        "UNRESOLVED",
        "DOMINATED",
        "UNRESOLVED",
        "UNSAFE",
    ]


def test_confirmation_generator_is_variable_split_safe_and_content_free(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    report = generate_confirmation_dataset(
        root / "configs/preregistrations/ldt_array_confirmation_v3.yaml",
        tmp_path / "dataset",
    )
    assert report["generation_passed"]
    manifest_path = tmp_path / "dataset/public_dataset/dataset_manifest.json"
    (
        manifest,
        train_pairs,
        heldout_pairs,
        train_arrays,
        heldout_arrays,
        train_reports,
        heldout_reports,
    ) = load_confirmation_dataset(manifest_path)
    assert (len(train_pairs), len(heldout_pairs)) == (496, 248)
    assert (len(train_arrays), len(heldout_arrays)) == (24, 12)
    assert (len(train_reports), len(heldout_reports)) == (120, 60)
    assert {row["candidate_count"] for row in train_arrays + heldout_arrays} == {
        4,
        5,
        6,
    }
    assert all("target" not in row for row in heldout_pairs + heldout_arrays)
    assert all(
        not row["contains_row_level_game_records"]
        and not row["contains_hidden_reasoning"]
        for row in train_reports + heldout_reports
    )
    public_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert public_manifest["report_binding_audit"]["complete"]
    assert not public_manifest["heldout_label_custody"][
        "paths_disclosed_to_candidate"
    ]
    assert manifest["category_shift_audit"]["overlap"] == 0


def test_v7_generator_seals_target_independent_confirmation(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    report = generate_target_independent_dataset(
        root
        / "configs/preregistrations/ldt_target_independent_v7.yaml",
        tmp_path / "v7",
    )
    manifest = json.loads(
        (
            tmp_path / "v7/public_dataset/dataset_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert report["generation_passed"]
    assert report["sealed_confirmation_created_before_training"]
    assert manifest["lineage_audit"]["cross_split_groups"] == 0
    assert manifest["lineage_audit"]["prior_overlap"] == 0
    audit = manifest["target_independence_audit"]
    assert audit["feature_count"] == 45
    assert audit["forbidden_feature_hits"] == []
    assert not audit["executes_verifier_policy"]
    assert not audit["includes_context_or_pressure_categories"]
    assert all(
        not split["targets_in_file"]
        for name, split in manifest["splits"].items()
        if name.startswith("heldout_")
    )
    assert (
        tmp_path / "v7/private_steward/heldout_pair_labels.jsonl"
    ).is_file()
    assert (
        tmp_path / "v7/private_steward/heldout_array_labels.jsonl"
    ).is_file()


def test_confirmation_encoder_supports_dynamic_joint_heads() -> None:
    row = {
        "input": {
            "left": _candidate(
                (0.85, 0.85, 0.85, 0.85), upper_missing=0
            ),
            "right": _candidate((0.7, 0.7, 0.7, 0.7)),
        }
    }
    encoder = ConfirmationPairEncoder.fit([row])
    raw = encoder.transform(row["input"], view="raw")
    typed = encoder.transform(row["input"], view="typed")
    assert isinstance(raw, array)
    assert len(typed) == len(raw) + 10
    model = make_model(
        {
            "family": "latent_array",
            "hidden_width": 4,
            "loop_count": 1,
        },
        len(typed),
        17,
        preference_count=4,
        rationale_count=5,
    )
    preference, rationale = model.predict_proba(typed)
    assert math.isclose(sum(preference), 1.0)
    assert math.isclose(sum(rationale), 1.0)


def test_independent_encoder_never_executes_policy_or_exposes_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "input": {
            "left": _candidate(
                (0.85, 0.85, 0.85, 0.85),
                upper_missing=0,
            ),
            "right": _candidate((0.7, 0.7, 0.7, 0.7)),
        }
    }
    encoder = ConfirmationPairEncoder.fit([row])

    def forbidden_policy(*args: object, **kwargs: object) -> None:
        raise AssertionError("independent features executed target policy")

    monkeypatch.setattr(
        "datagoblin.experiments.ldt_array_confirmation."
        "partial_interval_taste_policy",
        forbidden_policy,
    )
    encoded = encoder.transform(row["input"], view="independent")
    names = encoder.feature_names("independent")
    forbidden_fragments = (
        "lattice",
        "domin",
        "equivalent",
        "tradeoff",
        "insufficient",
        "context_mode",
        "fixture_pressure",
        "optimization_pressure",
        "delta",
        "mismatch",
    )
    assert len(encoded) == len(names) == 45
    assert not any(
        fragment in name
        for name in names
        for fragment in forbidden_fragments
    )


def test_temperature_calibration_uses_only_frozen_grid() -> None:
    root = Path(__file__).resolve().parents[2]
    trainer = runpy.run_path(
        str(root / "scripts/train_ldt_array_confirmation.py")
    )
    examples = [
        (
            {
                "LEFT": 0.55,
                "RIGHT": 0.15,
                "TRADEOFF": 0.15,
                "ABSTAIN": 0.15,
            },
            "LEFT",
        )
        for _ in range(8)
    ]
    receipt = trainer["_fit_temperature"](
        examples,
        temperatures=(0.5, 1.0, 2.0),
        bins=10,
    )
    assert receipt["fit_split"] == (
        "out_of_fold_development_predictions_only"
    )
    assert receipt["selected_temperature"] == 0.5
    assert {
        row["temperature"] for row in receipt["grid_reports"]
    } == {0.5, 1.0, 2.0}


def test_v7_selection_excludes_zero_support_abstain_folds() -> None:
    root = Path(__file__).resolve().parents[2]
    selector = runpy.run_path(
        str(root / "scripts/select_ldt_target_independent_v7.py")
    )
    receipt = {
        "cv_fold_metrics": [
            {
                "preference_macro_f1": 0.4,
                "abstain_recall": None,
            },
            {
                "preference_macro_f1": 0.5,
                "abstain_recall": 0.75,
            },
        ],
        "cv_mean_metrics": {"preference_macro_f1": 0.45},
        "calibration": {
            "selected_metrics": {
                "multiclass_negative_log_likelihood": 1.0,
            }
        },
        "trainable_parameters": 20,
        "seed": 7,
    }
    rank = selector["_rank"](receipt)
    assert rank[0:3] == (-0.4, -0.45, -0.75)


def test_confirmation_evaluator_calibration_and_pareto_rules() -> None:
    root = Path(__file__).resolve().parents[2]
    evaluator = runpy.run_path(
        str(root / "scripts/evaluate_ldt_array_confirmation.py")
    )
    prediction = {
        "preference": "LEFT",
        "preference_probabilities": {
            "LEFT": 0.75,
            "RIGHT": 0.10,
            "TRADEOFF": 0.10,
            "ABSTAIN": 0.05,
        },
    }
    ece = evaluator["_expected_calibration_error"](
        ["LEFT"],
        [prediction],
        bins=10,
    )
    assert math.isclose(ece, 0.25)

    maximize = (
        "preference_macro_f1",
        "abstain_recall",
        "rationale_macro_f1",
        "status_macro_f1",
        "exact_array_accuracy",
        "selected_set_exact_accuracy",
    )
    minimize = (
        "multiclass_brier",
        "multiclass_negative_log_likelihood",
        "expected_calibration_error",
    )

    def aggregate(
        *,
        quality: float,
        calibration: float,
        parameters: int,
    ) -> dict[str, object]:
        metrics = {
            name: {"mean": quality}
            for name in maximize
        }
        metrics.update(
            {name: {"mean": calibration} for name in minimize}
        )
        return {
            "metrics": metrics,
            "trainable_parameters": parameters,
        }

    compact = aggregate(quality=1.0, calibration=0.01, parameters=345)
    larger = aggregate(quality=1.0, calibration=0.01, parameters=361)
    cheaper_but_uncalibrated = aggregate(
        quality=1.0,
        calibration=0.04,
        parameters=10,
    )
    assert evaluator["_dominates"](compact, larger, 0.01)
    assert not evaluator["_dominates"](
        cheaper_but_uncalibrated,
        compact,
        0.01,
    )
