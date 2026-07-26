from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

from datagoblin.experiments.ldt_array_confirmation import (
    ConfirmationPairEncoder,
)
from datagoblin.experiments.ldt_array_hard_surface import (
    HARD_INVERT_PREFERENCE,
    HARD_PREFERENCES,
    HARD_RATIONALES,
)
from datagoblin.experiments.ldt_array_sorter import make_model
from datagoblin.experiments.ldt_shift_builder import (
    _materialize,
    generate_shift_fixture,
    load_shift_fixture,
)
from datagoblin.experiments.ldt_shift_challenge import (
    FrozenShiftModel,
    _model_features,
    _nonlearned_prediction,
)
from datagoblin.experiments.ldt_shift_verifier import (
    shift_verifier_array_target,
    shift_verifier_target,
)
from datagoblin.serialization import load_structured


def _contract() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    return load_structured(
        root
        / "configs/preregistrations/ldt_independent_shift_v6.yaml"
    )


def _development_contract() -> dict[str, object]:
    contract = copy.deepcopy(_contract())
    contract["experiment_id"] = (
        "datagoblin-ldt-independent-shift-v6-development"
    )
    contract["shift_surface"]["seed"] = 3001
    contract["shift_surface"]["context_prefix"] = (
        "independent_shift_v6_development"
    )
    return contract


def test_independent_builder_is_target_free_fresh_and_shifted(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    report = generate_shift_fixture(
        root
        / "configs/preregistrations/ldt_independent_shift_v6.yaml",
        tmp_path / "fixture",
    )
    assert report["generation_passed"]
    assert report["counts"] == {
        "pairs": 248,
        "arrays": 12,
        "reports": 60,
    }
    assert report["surface_counts"] == {
        "missing_bound_count": 28,
        "complete_interval_metric_count": 4,
        "near_tolerance_directed_metric_count": 96,
    }
    manifest_path = (
        tmp_path / "fixture/public_shift/shift_manifest.json"
    )
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest, pairs, arrays, reports = load_shift_fixture(manifest_path)
    assert "private_verifier" not in manifest_text
    assert all("target" not in row for row in pairs + arrays)
    assert all(
        not row["contains_row_level_game_records"]
        and not row["contains_hidden_reasoning"]
        for row in reports
    )
    assert manifest["lineage_audit"]["overlap"] == 0
    assert not manifest["training_authorized"]
    assert not manifest["model_selection_authorized"]


def test_independent_verifier_covers_every_declared_class_on_development() -> None:
    contract = _development_contract()
    pairs, arrays, _ = _materialize(contract)
    tolerance = float(
        contract["shift_surface"]["metric_equivalence_tolerance"]
    )
    pair_targets = [
        shift_verifier_target(row["input"], tolerance=tolerance)
        for row in pairs
    ]
    array_targets = [
        shift_verifier_array_target(
            [
                candidate["features"]
                for candidate in row["candidates"]
            ],
            tolerance=tolerance,
        )
        for row in arrays
    ]
    assert set(value[0] for value in pair_targets) == set(
        HARD_PREFERENCES
    )
    assert set(value[1] for value in pair_targets) == set(
        HARD_RATIONALES
    )
    assert Counter(value[0] for value in pair_targets) == {
        "LEFT": 88,
        "RIGHT": 88,
        "TRADEOFF": 56,
        "ABSTAIN": 16,
    }
    assert set(
        status
        for target in array_targets
        for status in target["statuses"]
    ) == {"SELECTED", "DOMINATED", "UNRESOLVED", "UNSAFE"}


def test_policy_lattice_mask_removes_only_policy_derived_features() -> None:
    contract = _development_contract()
    pairs, _, _ = _materialize(contract)
    encoder = ConfirmationPairEncoder.fit(pairs, tolerance=0.01)
    width = len(encoder.feature_names("typed"))
    model = make_model(
        {"family": "latent_array", "hidden_width": 4, "loop_count": 1},
        width,
        17,
        preference_count=len(HARD_PREFERENCES),
        rationale_count=len(HARD_RATIONALES),
    )
    bundle = FrozenShiftModel(
        payload={"feature_view": "typed"},
        model=model,
        encoder=encoder,
        model_sha256="a" * 64,
        temperature=1.0,
    )
    row = next(
        row
        for row in pairs
        if row["input"]["left"]["safety_qualified"]
        != row["input"]["right"]["safety_qualified"]
    )
    full = _model_features(
        bundle,
        row["input"],
        policy_lattice_mode="full_diagnostic_control",
    )
    masked = _model_features(
        bundle,
        row["input"],
        policy_lattice_mode="masked",
    )
    names = encoder.feature_names("typed")
    for name, full_value, masked_value in zip(
        names,
        full,
        masked,
        strict=True,
    ):
        if name.startswith("lattice:"):
            assert masked_value == 0.0
        else:
            assert masked_value == full_value
    assert any(
        full_value != 0.0
        for name, full_value in zip(names, full, strict=True)
        if name.startswith("lattice:")
    )


def test_nonlearned_baseline_is_orientation_consistent() -> None:
    contract = _development_contract()
    pairs, _, _ = _materialize(contract)
    row = pairs[0]
    prediction = _nonlearned_prediction(
        row,
        tolerance=0.01,
        scalar_margin=0.01,
    )
    assert prediction["swap_consistent"]
    assert (
        prediction["swapped_preference"]
        == HARD_INVERT_PREFERENCE[prediction["preference"]]
    )
    assert json.loads(
        json.dumps(prediction["preference_probabilities"])
    ) == prediction["preference_probabilities"]
