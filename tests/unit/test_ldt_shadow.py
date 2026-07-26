from __future__ import annotations

import copy
import math
from array import array
from pathlib import Path

import pytest

from datagoblin.experiments.ldt_array_hard_surface import (
    HARD_PREFERENCES,
    HARD_RATIONALES,
)
from datagoblin.experiments.ldt_array_sorter import make_model
from datagoblin.experiments.ldt_shadow import (
    _expected_calibration_error,
    _supported_macro_f1,
    generate_shadow_fixture,
    load_shadow_fixture,
    restore_latent_model,
)


def _latent_payload() -> dict[str, object]:
    model = make_model(
        {
            "family": "latent_array",
            "hidden_width": 4,
            "loop_count": 1,
        },
        12,
        17,
        preference_count=len(HARD_PREFERENCES),
        rationale_count=len(HARD_RATIONALES),
    )
    return {
        "family": "latent_array",
        "feature_view": "typed",
        "trainable_parameters": model.parameter_count,
        "state": model.state_dict(),
    }


def test_shadow_generator_is_fresh_content_free_and_target_sealed(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    report = generate_shadow_fixture(
        root / "configs/preregistrations/ldt_shadow_v4.yaml",
        tmp_path / "fixture",
    )
    assert report["generation_passed"]
    manifest_path = tmp_path / "fixture/public_shadow/shadow_manifest.json"
    manifest, pairs, arrays, reports = load_shadow_fixture(manifest_path)
    assert (len(pairs), len(arrays), len(reports)) == (248, 12, 60)
    assert all("target" not in row for row in pairs + arrays)
    assert all(
        not row["contains_row_level_game_records"]
        and not row["contains_hidden_reasoning"]
        for row in reports
    )
    assert manifest["lineage_audit"]["overlap"] == 0
    assert manifest["report_binding_audit"]["complete"]
    assert manifest["shadow_decision_effect"] == "NONE"
    assert not manifest["training_authorized"]
    assert not manifest["corpus_promotion_authorized"]


def test_restore_latent_model_round_trips_frozen_arrays() -> None:
    payload = _latent_payload()
    original = make_model(
        {
            "family": "latent_array",
            "hidden_width": 4,
            "loop_count": 1,
        },
        12,
        0,
        preference_count=len(HARD_PREFERENCES),
        rationale_count=len(HARD_RATIONALES),
    )
    state = payload["state"]
    original.input_weights[:] = state["input_weights"]
    original.hidden_bias[:] = state["hidden_bias"]
    original.preference_weights[:] = state["preference_weights"]
    original.preference_bias[:] = state["preference_bias"]
    original.rationale_weights[:] = state["rationale_weights"]
    original.rationale_bias[:] = state["rationale_bias"]
    restored = restore_latent_model(payload)
    features = array("d", [index / 10.0 for index in range(12)])
    expected_preference, expected_rationale = original.predict_proba(features)
    preference, rationale = restored.predict_proba(features)
    assert all(
        math.isclose(left, right)
        for left, right in zip(
            expected_preference,
            preference,
            strict=True,
        )
    )
    assert all(
        math.isclose(left, right)
        for left, right in zip(
            expected_rationale,
            rationale,
            strict=True,
        )
    )


def test_restore_rejects_recurrent_or_nonfinite_state() -> None:
    recurrent = _latent_payload()
    recurrent["state"]["recurrent_weights"] = [[1.0]]
    with pytest.raises(ValueError, match="recurrent"):
        restore_latent_model(recurrent)
    nonfinite = copy.deepcopy(_latent_payload())
    nonfinite["state"]["hidden_bias"][0] = float("nan")
    with pytest.raises(ValueError, match="hidden_bias"):
        restore_latent_model(nonfinite)


def test_shadow_ece_uses_recommendation_confidence() -> None:
    recommendations = [
        {"preference": "LEFT", "confidence": 0.8},
        {"preference": "RIGHT", "confidence": 0.8},
    ]
    assert math.isclose(
        _expected_calibration_error(
            ["LEFT", "LEFT"],
            recommendations,
            bins=10,
        ),
        0.3,
    )


def test_shadow_slice_macro_f1_ignores_absent_classes() -> None:
    metrics = {
        "per_class": {
            "SELECTED": {"support": 3, "f1": 1.0},
            "DOMINATED": {"support": 2, "f1": 1.0},
            "UNRESOLVED": {"support": 1, "f1": 1.0},
            "UNSAFE": {"support": 0, "f1": 0.0},
        }
    }
    assert _supported_macro_f1(metrics) == 1.0
