from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.experiments.ldt_edge_set_array_v10 import (
    CrossBoundEdgeCandidateEncoder,
)
from datagoblin.experiments.ldt_exact_array_likelihood_v12 import (
    ExactArrayLikelihoodModel,
)
from datagoblin.experiments.ldt_structured_array_v8 import (
    generate_structured_array_dataset,
)
from datagoblin.serialization import load_structured

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    ROOT
    / "configs"
    / "preregistrations"
    / "ldt_adapter_receipt_sorter_v16.yaml"
)


def _receipt(
    candidate_id: str,
    *,
    exact: float,
    selected: float,
    worst_exact: float,
    worst_selected: float,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "cv_mean_metrics": {
            "exact_array_accuracy": exact,
            "selected_set_exact_accuracy": selected,
        },
        "cv_fold_metrics": [
            {
                "exact_array_accuracy": worst_exact,
                "selected_set_exact_accuracy": worst_selected,
            }
        ],
    }


def _qualified_receipt(
    candidate_id: str,
    *,
    seed: int,
    exact: float,
    selected: float,
    worst_exact: float,
    worst_selected: float,
) -> dict[str, Any]:
    fold = {
        "safety_gate_accuracy": 1.0,
        "swap_consistency": 1.0,
        "preference_macro_f1": 0.90,
        "abstain_recall": 1.0,
        "status_macro_f1": 0.90,
        "exact_array_accuracy": worst_exact,
        "selected_set_exact_accuracy": worst_selected,
    }
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "trainable_parameters": 100,
        "cv_mean_metrics": {
            **fold,
            "exact_array_accuracy": exact,
            "selected_set_exact_accuracy": selected,
        },
        "cv_fold_metrics": [fold],
        "calibration": {
            "selected_metrics": {
                "expected_calibration_error": 0.05,
                "multiclass_negative_log_likelihood": 0.20,
            }
        },
    }


def test_v16_contract_is_fresh_receipt_domain_and_comparative() -> None:
    config = load_structured(PREREGISTRATION)
    comparison = config["architecture_comparison"]
    selection = config["development_selection"]
    assert config["schema_version"] == "16.0"
    assert config["status"] == "frozen_before_dataset_generation"
    assert config["source_evidence"]["v2_heldout_confirmation_spent"]
    assert config["source_evidence"]["forbidden_design_inputs"] == [
        "v2_heldout_row_labels",
        "v2_heldout_per_array_targets",
        "v2_heldout_error_identifiers",
    ]
    assert len(config["surface"]["forbidden_prior_lineage_sha256"]) == 22
    assert config["training"]["seeds"] == [311, 367, 421]
    assert config["dataset_contract"]["required_train_arrays"] == 36
    assert comparison["cross_bound_global_primary_id"] == (
        "exact-crossbound-h4-global-v16"
    )
    assert selection["primary_candidate_ids"] == [
        "exact-crossbound-h4-global-v16"
    ]
    assert selection["comparative_primary_gates"] == {
        "baseline_candidate_id": "exact-edge-h4-global-v16",
        "minimum_mean_exact_array_accuracy_delta": 0.05,
        "minimum_mean_selected_set_exact_accuracy_delta": 0.05,
        "minimum_worst_fold_exact_array_accuracy_delta": 0.0,
        "minimum_worst_fold_selected_set_exact_accuracy_delta": 0.0,
    }
    assert not any(
        control["eligible_for_selection"]
        for control in config["transfer_controls"]
    )


def test_cross_bound_encoder_is_target_free_and_permutation_equivariant(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    report = generate_structured_array_dataset(PREREGISTRATION, fixture)
    assert report["generation_passed"]
    arrays = [
        json.loads(line)
        for line in (
            fixture / "public_dataset/train_arrays.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line
    ]
    candidates = [
        row["features"] for row in arrays[0]["candidates"]
    ]
    encoder = CrossBoundEdgeCandidateEncoder()
    encoded = encoder.transform_array(candidates)
    permutation = list(reversed(range(len(candidates))))
    permuted = encoder.transform_array(
        [candidates[index] for index in permutation]
    )
    np.testing.assert_allclose(permuted, encoded[permutation])
    category_changed = copy.deepcopy(candidates)
    for candidate in category_changed:
        candidate["game"] = "not-a-game-feature"
        candidate["fixture_pressure"] = "not-a-pressure-feature"
        candidate["context_mode"] = "not-an-architecture-feature"
        candidate["optimization_pressure"] = "not-an-identity-feature"
    np.testing.assert_allclose(
        encoder.transform_array(category_changed),
        encoded,
    )
    receipt = encoder.receipt()
    assert receipt["feature_count"] == 74
    assert receipt["permutation_equivariant"]
    assert not receipt["executes_verifier_policy"]
    assert not receipt["uses_metric_tolerance"]
    assert not receipt["uses_targets"]


def test_v16_generation_and_plan_bind_support_and_custody(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    report = generate_structured_array_dataset(PREREGISTRATION, fixture)
    manifest = load_structured(
        fixture / "public_dataset/dataset_manifest.json"
    )
    prepare = runpy.run_path(
        str(ROOT / "scripts/prepare_ldt_array_confirmation_run.py")
    )
    plan = prepare["prepare"](
        PREREGISTRATION,
        fixture / "public_dataset/dataset_manifest.json",
        tmp_path / "trainer_plan.json",
    )
    assert report["generation_passed"]
    assert report["counts"] == {
        "train_pairs": 888,
        "heldout_pairs": 888,
        "train_arrays": 36,
        "heldout_arrays": 36,
        "train_reports": 196,
        "heldout_reports": 196,
    }
    assert manifest["lineage_audit"] == {
        "cross_split_groups": 0,
        "forbidden_prior_groups": 22,
        "heldout_groups": 6,
        "prior_overlap": 0,
        "train_groups": 6,
    }
    assert all(
        count == 6
        for axis in manifest["development_support_audit"][
            "lineage_support"
        ].values()
        for count in axis.values()
    )
    encoders = plan["feature_independence_audit"]["listwise_status"][
        "candidate_encoders"
    ]
    assert encoders["cross_bound_edge_message"]["feature_count"] == 74
    assert not plan["heldout_pair_label_access"]
    assert not plan["heldout_array_label_access"]


def test_v16_trainer_builds_cross_bound_exact_model() -> None:
    config = load_structured(PREREGISTRATION)
    specs = {
        row["model_id"]: row["listwise_status"]
        for row in config["models"]["candidates"]
    }
    trainer = runpy.run_path(
        str(ROOT / "scripts/train_ldt_array_confirmation.py")
    )
    spec = specs["exact-crossbound-h4-global-v16"]
    encoder = trainer["_candidate_encoder"](spec)
    model = trainer["_make_status_model"](
        spec,
        input_width=len(encoder.feature_names()),
        seed=311,
    )
    assert isinstance(encoder, CrossBoundEdgeCandidateEncoder)
    assert isinstance(model, ExactArrayLikelihoodModel)
    assert model.decoder == "global_selected_count_map"


def test_v16_fixed_safety_membrane_is_explicit_and_not_learned() -> None:
    trainer = runpy.run_path(
        str(ROOT / "scripts/train_ldt_array_confirmation.py")
    )
    prediction = {
        "preference": "RIGHT",
        "rationale_code": "irreducible_tradeoff",
        "confidence": 0.6,
        "preference_probabilities": {
            "LEFT": 0.2,
            "RIGHT": 0.6,
            "TRADEOFF": 0.1,
            "ABSTAIN": 0.1,
        },
        "swapped_preference": "LEFT",
        "swap_consistent": True,
        "loop_delta": 0.1,
    }
    row = {
        "input": {
            "left": {"safety_qualified": True},
            "right": {"safety_qualified": False},
        }
    }
    routed = trainer["_apply_fixed_pair_safety_membrane"](
        row,
        prediction,
    )
    assert routed["preference"] == "LEFT"
    assert routed["rationale_code"] == "safety_gate"
    assert routed["confidence"] == 1.0
    assert routed["swap_consistent"]
    assert routed["preference_probabilities"]["LEFT"] == 1.0


def test_v16_primary_domain_summary_is_support_weighted() -> None:
    evaluator = runpy.run_path(
        str(ROOT / "scripts/evaluate_ldt_array_confirmation.py")
    )
    pressure_slices = {
        "a": {
            "support": {"array_count": 2},
            "metrics": {
                "exact_array_accuracy": 0.5,
                "selected_set_exact_accuracy": 1.0,
            },
        },
        "b": {
            "support": {"array_count": 1},
            "metrics": {
                "exact_array_accuracy": 1.0,
                "selected_set_exact_accuracy": 0.0,
            },
        },
    }
    evaluation = {
        "minimum_primary_domain_exact_status_vector_accuracy": 0.65,
        "minimum_primary_domain_selected_set_exact_accuracy": 0.65,
    }
    summary = evaluator["_primary_domain_summary"](
        pressure_slices,
        evaluation,
        ["a", "b"],
    )
    assert summary["exact_status_vector_accuracy"] == 2 / 3
    assert summary["exact_selected_set_accuracy"] == 2 / 3
    assert summary["qualified"]


def test_v16_comparative_gate_requires_mean_and_worst_improvement() -> None:
    selector = runpy.run_path(
        str(ROOT / "scripts/select_ldt_listwise_array_v9.py")
    )
    gates = load_structured(PREREGISTRATION)["development_selection"][
        "comparative_primary_gates"
    ]
    baseline = _receipt(
        "exact-edge-h4-global-v16",
        exact=0.70,
        selected=0.75,
        worst_exact=0.50,
        worst_selected=0.50,
    )
    primary = _receipt(
        "exact-crossbound-h4-global-v16",
        exact=0.76,
        selected=0.81,
        worst_exact=0.50,
        worst_selected=0.50,
    )
    checks = selector["_comparative_primary_checks"](
        primary,
        baseline,
        gates,
    )
    assert all(checks.values())
    primary["cv_mean_metrics"]["exact_array_accuracy"] = 0.74
    checks = selector["_comparative_primary_checks"](
        primary,
        baseline,
        gates,
    )
    assert not checks["mean_exact_array_accuracy_delta"]


def test_v16_packet_auditor_interface_cannot_accept_labels() -> None:
    auditor = runpy.run_path(
        str(
            ROOT
            / "scripts/audit_ldt_adapter_receipt_sorter_v16_packet.py"
        )
    )
    parser = auditor["build_parser"]()
    assert "--pair-labels" not in parser.format_help()
    assert "--array-labels" not in parser.format_help()
    assert auditor["_target_free"](
        {"prediction": "SELECTED", "rationale_code": "safety_gate"}
    )
    assert not auditor["_target_free"](
        {"prediction": "SELECTED", "target": "DOMINATED"}
    )


def test_v16_packet_auditor_reproduces_comparative_stop() -> None:
    auditor = runpy.run_path(
        str(
            ROOT
            / "scripts/audit_ldt_adapter_receipt_sorter_v16_packet.py"
        )
    )
    config = load_structured(PREREGISTRATION)
    receipts = []
    for candidate_id in (
        "exact-independent-h4-global-v16",
        "exact-edge-h4-global-v16",
        "exact-crossbound-h4-factorized-v16",
        "exact-crossbound-h4-global-v16",
    ):
        for seed in (311, 367, 421):
            receipts.append(
                _qualified_receipt(
                    candidate_id,
                    exact=0.80,
                    selected=0.85,
                    worst_exact=0.60,
                    worst_selected=0.60,
                    seed=seed,
                )
            )
    reproduced = auditor["_reproduce_selection"](
        config,
        {"candidate_receipts": receipts},
    )
    assert not reproduced["authorized"]
    assert reproduced["primary_id"] is None
    assert not reproduced["comparative_checks"][
        "mean_exact_array_accuracy_delta"
    ]
    assert not reproduced["comparative_checks"][
        "mean_selected_set_exact_accuracy_delta"
    ]
