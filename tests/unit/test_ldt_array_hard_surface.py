from __future__ import annotations

import json
import math
from array import array
from pathlib import Path

from datagoblin.experiments.ldt_array_hard_surface import (
    ARRAY_STATUSES,
    HARD_INVERT_PREFERENCE,
    HARD_PREFERENCES,
    HARD_RATIONALES,
    HardPairEncoder,
    derive_array_target,
    generate_hard_surface_dataset,
    interval_taste_policy,
    load_hard_surface_dataset,
    swap_hard_pair,
)
from datagoblin.experiments.ldt_array_sorter import METRIC_FIELDS, make_model


def _candidate(
    values: tuple[float, float, float, float],
    *,
    safe: bool = True,
    bucket: str = "medium",
    width: float = 0.0,
    missing: int | None = None,
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
        result[field] = (
            {"lower": None, "upper": None}
            if missing == index
            else {"lower": value - width, "upper": value + width}
        )
    return result


def test_interval_policy_distinguishes_tradeoff_from_uncertainty() -> None:
    baseline = _candidate((0.7, 0.7, 0.7, 0.7))
    dominant = _candidate((0.8, 0.8, 0.8, 0.8), bucket="large")
    unsafe = _candidate((0.99, 0.99, 0.99, 0.99), safe=False)
    equivalent_cheaper = _candidate(
        (0.804, 0.804, 0.804, 0.804), bucket="small"
    )
    specialist = _candidate((0.9, 0.9, 0.6, 0.6))
    robust = _candidate((0.6, 0.6, 0.9, 0.9))
    uncertain = _candidate((0.8, 0.8, 0.8, 0.8), width=0.03)
    missing = _candidate((0.8, 0.8, 0.8, 0.8), missing=2)
    assert interval_taste_policy({"left": baseline, "right": unsafe}) == (
        "LEFT",
        "safety_gate",
    )
    assert interval_taste_policy({"left": dominant, "right": baseline}) == (
        "LEFT",
        "pareto_dominance",
    )
    assert interval_taste_policy(
        {"left": dominant, "right": equivalent_cheaper}
    ) == ("RIGHT", "parameter_efficiency")
    assert interval_taste_policy({"left": specialist, "right": robust}) == (
        "TRADEOFF",
        "irreducible_tradeoff",
    )
    assert interval_taste_policy({"left": dominant, "right": uncertain}) == (
        "ABSTAIN",
        "insufficient_evidence",
    )
    assert interval_taste_policy({"left": dominant, "right": missing}) == (
        "ABSTAIN",
        "insufficient_evidence",
    )


def test_hard_swap_inverts_direction_and_preserves_abstention() -> None:
    left = _candidate((0.8, 0.8, 0.8, 0.8))
    right = _candidate((0.8, 0.8, 0.8, 0.8), missing=0)
    row = {
        "input": {"left": left, "right": right},
        "left_index": 1,
        "right_index": 4,
        "target": {
            "preference": "ABSTAIN",
            "rationale_code": "insufficient_evidence",
        },
    }
    swapped = swap_hard_pair(row)
    assert swapped["left_index"] == 4
    assert swapped["right_index"] == 1
    assert swapped["target"]["preference"] == HARD_INVERT_PREFERENCE["ABSTAIN"]
    assert interval_taste_policy(swapped["input"]) == (
        "ABSTAIN",
        "insufficient_evidence",
    )


def test_array_target_keeps_unresolved_separate_from_tradeoff() -> None:
    candidates = [
        _candidate((0.7, 0.7, 0.7, 0.7)),
        _candidate((0.8, 0.8, 0.8, 0.8)),
        _candidate((0.9, 0.9, 0.6, 0.6)),
        _candidate((0.99, 0.99, 0.99, 0.99), safe=False),
        _candidate((0.8, 0.8, 0.8, 0.8), missing=1),
    ]
    target = derive_array_target(candidates, tolerance=0.01)
    assert set(target["statuses"]) <= set(ARRAY_STATUSES)
    assert target["statuses"][0] == "DOMINATED"
    assert target["statuses"][3] == "UNSAFE"
    assert target["statuses"][4] == "UNRESOLVED"


def test_generator_emits_balanced_split_safe_public_custody(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    report = generate_hard_surface_dataset(
        root / "configs/preregistrations/ldt_array_hard_surface_v2.yaml",
        tmp_path / "dataset",
    )
    assert report["generation_passed"]
    manifest_path = tmp_path / "dataset/public_dataset/dataset_manifest.json"
    manifest, train_pairs, heldout_pairs, train_arrays, heldout_arrays = (
        load_hard_surface_dataset(manifest_path)
    )
    assert (len(train_pairs), len(heldout_pairs)) == (320, 240)
    assert (len(train_arrays), len(heldout_arrays)) == (16, 12)
    assert all("target" not in row for row in heldout_pairs + heldout_arrays)
    assert manifest["orientation_audit"]["complete"]
    for array_row in train_arrays:
        directions = {
            (row["left_index"], row["right_index"])
            for row in train_pairs
            if row["array_id"] == array_row["array_id"]
        }
        assert len(directions) == 20
        assert all((right, left) in directions for left, right in directions)
    public_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert public_manifest["heldout_label_custody"]["paths_disclosed_to_candidate"] is False


def test_hard_encoder_and_dynamic_output_heads_are_dense_arrays() -> None:
    left = _candidate((0.8, 0.8, 0.8, 0.8))
    right = _candidate((0.8, 0.8, 0.8, 0.8), width=0.03)
    row = {"input": {"left": left, "right": right}}
    encoder = HardPairEncoder.fit([row])
    raw = encoder.transform(row["input"], view="raw")
    typed = encoder.transform(row["input"], view="typed")
    assert isinstance(raw, array)
    assert len(typed) == len(raw) + 9
    spec = {
        "family": "weight_tied_looped_array",
        "hidden_width": 4,
        "loop_count": 3,
    }
    model = make_model(
        spec,
        len(typed),
        13,
        preference_count=len(HARD_PREFERENCES),
        rationale_count=len(HARD_RATIONALES),
    )
    preference, rationale = model.predict_proba(typed)
    assert len(preference) == len(HARD_PREFERENCES)
    assert len(rationale) == len(HARD_RATIONALES)
    assert math.isclose(sum(preference), 1.0)
    assert math.isclose(sum(rationale), 1.0)
