from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np

from datagoblin.experiments.ldt_array_confirmation import (
    load_confirmation_dataset,
)
from datagoblin.experiments.ldt_array_sorter import METRIC_FIELDS
from datagoblin.experiments.ldt_edge_set_array_v10 import (
    EdgeDifferenceCandidateEncoder,
    SetCoupledLoopedStatusModel,
    generate_edge_set_array_dataset,
    train_edge_status_model,
)
from datagoblin.serialization import dump_structured

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    ROOT
    / "configs"
    / "preregistrations"
    / "ldt_edge_set_array_v10.yaml"
)


def _candidate(offset: float) -> dict[str, object]:
    result: dict[str, object] = {
        "trainable_parameter_bucket": "medium",
        "safety_qualified": True,
    }
    for index, field in enumerate(METRIC_FIELDS):
        value = offset + index * 0.05
        result[field] = {"lower": value, "upper": value + 0.01}
    return result


def test_edge_encoder_is_permutation_equivariant() -> None:
    encoder = EdgeDifferenceCandidateEncoder()
    candidates = [_candidate(0.2), _candidate(0.4), _candidate(0.7)]
    original = encoder.transform_array(candidates)
    order = [2, 0, 1]
    permuted = encoder.transform_array([candidates[index] for index in order])
    assert original.shape == (3, 111)
    assert len(encoder.feature_names()) == 111
    assert np.allclose(permuted, original[order], atol=1e-12)
    receipt = encoder.receipt()
    assert receipt["permutation_equivariant"] is True
    assert receipt["uses_targets"] is False
    assert receipt["executes_verifier_policy"] is False


def test_soft_dice_gradient_is_coupled_across_candidates() -> None:
    model = SetCoupledLoopedStatusModel(
        input_width=3,
        hidden_width=2,
        loop_count=1,
        seed=7,
        set_loss_weight=0.5,
    )
    targets = np.asarray([0, 1], dtype=np.int64)
    probabilities_a = np.asarray(
        [[0.6, 0.2, 0.1, 0.1], [0.3, 0.4, 0.2, 0.1]],
        dtype=np.float64,
    )
    probabilities_b = probabilities_a.copy()
    probabilities_b[1] = [0.05, 0.65, 0.2, 0.1]
    _, gradient_a = model._soft_dice_loss_and_probability_gradient(
        probabilities_a,
        targets,
    )
    _, gradient_b = model._soft_dice_loss_and_probability_gradient(
        probabilities_b,
        targets,
    )
    assert not math.isclose(gradient_a[0], gradient_b[0], abs_tol=1e-12)


def test_v10_surface_and_set_coupled_head_are_trainable(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "v10"
    report = generate_edge_set_array_dataset(
        PREREGISTRATION,
        dataset_root,
    )
    manifest_path = dataset_root / "public_dataset/dataset_manifest.json"
    manifest, _, _, train_arrays, heldout_arrays, _, _ = (
        load_confirmation_dataset(manifest_path)
    )
    assert report["generation_passed"]
    assert report["counts"] == {
        "train_pairs": 552,
        "heldout_pairs": 350,
        "train_arrays": 24,
        "heldout_arrays": 16,
        "train_reports": 126,
        "heldout_reports": 82,
    }
    assert all("target" not in row for row in heldout_arrays)
    assert manifest["lineage_audit"]["forbidden_prior_groups"] == 54
    assert manifest["lineage_audit"]["prior_overlap"] == 0

    trainer = runpy.run_path(
        str(ROOT / "scripts" / "train_ldt_array_confirmation.py")
    )
    exact_metrics = trainer["_array_labeled_metrics"](
        [],
        [],
        train_arrays,
        direct_statuses={
            str(row["array_id"]): list(row["target"]["statuses"])
            for row in train_arrays
        },
    )
    assert exact_metrics["exact_array_accuracy"] == 1.0
    assert exact_metrics["selected_set_exact_accuracy"] == 1.0

    encoder = EdgeDifferenceCandidateEncoder()
    model = SetCoupledLoopedStatusModel(
        input_width=len(encoder.feature_names()),
        hidden_width=6,
        loop_count=2,
        seed=43,
        set_loss_weight=0.5,
    )

    def mean_loss() -> float:
        losses = []
        for row in train_arrays:
            features = encoder.transform_array(
                [candidate["features"] for candidate in row["candidates"]]
            )
            targets = [
                manifest["array_status_labels"].index(str(status))
                for status in row["target"]["statuses"]
            ]
            losses.append(model.loss(features, targets))
        return sum(losses) / len(losses)

    before = mean_loss()
    steps = train_edge_status_model(
        model,
        encoder,
        train_arrays,
        epochs=40,
        learning_rate=0.02,
        l2=0.0005,
        gradient_clip=5.0,
        seed=43,
        checkpoint_every_epochs=20,
    )
    assert steps == 40 * len(train_arrays)
    assert mean_loss() < before

    prepare = runpy.run_path(
        str(ROOT / "scripts" / "prepare_ldt_array_confirmation_run.py")
    )
    plan = prepare["prepare"](
        PREREGISTRATION,
        manifest_path,
        tmp_path / "trainer-plan.json",
    )
    assert plan["preregistration_schema_version"] == "10.0"
    audit = plan["feature_independence_audit"]["listwise_status"]
    assert audit["set_coupled_ablation_matched"] is True
    assert audit["metric_gate"]["heldout_status"] == (
        "supported_on_declared_holdout"
    )


def test_supersession_comparator_requires_behavioral_equivalence(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    superseding = tmp_path / "superseding"
    candidate_receipts = [{"candidate_id": "candidate", "seed": 31}]
    trainer = {
        "experiment_id": "datagoblin-ldt-edge-set-array-v10",
        "preregistration_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "predictions_sha256": "c" * 64,
        "array_predictions_sha256": "d" * 64,
        "candidate_receipts": candidate_receipts,
        "heldout_pair_targets_seen": False,
        "heldout_array_targets_seen": False,
    }
    selection = {
        "candidate_selection": {"candidate": {"selected_seed": 31}},
        "selected_pair_predictions_sha256": "e" * 64,
        "selected_array_predictions_sha256": "f" * 64,
        "primary_candidate_id": None,
        "primary_seed": None,
        "no_qualified_primary": True,
        "confirmation_join_authorized": False,
        "confirmation_labels_read": False,
    }
    for root in (original, superseding):
        dump_structured(root / "merged/trainer_summary.json", trainer)
        dump_structured(root / "selected/selection_receipt.json", selection)

    comparator = runpy.run_path(
        str(ROOT / "scripts" / "compare_ldt_v10_superseding_runs.py")
    )
    report = comparator["compare"](
        original,
        superseding,
        tmp_path / "supersession.json",
    )
    assert report["status"] == "superseding_evaluator_equivalent"
    assert all(report["checks"].values())

    changed = dict(selection)
    changed["confirmation_join_authorized"] = True
    dump_structured(
        superseding / "selected/selection_receipt.json",
        changed,
    )
    with np.testing.assert_raises_regex(ValueError, "same_primary_decision"):
        comparator["compare"](
            original,
            superseding,
            tmp_path / "rejected.json",
        )
