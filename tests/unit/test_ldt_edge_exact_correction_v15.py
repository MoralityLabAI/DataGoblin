from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from datagoblin.experiments.ldt_array_sorter import sha256_file
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
    / "ldt_edge_exact_correction_v15.yaml"
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _qualified_receipt() -> dict[str, Any]:
    fold = {
        "preference_macro_f1": 0.80,
        "abstain_recall": 0.80,
        "status_macro_f1": 0.85,
        "exact_array_accuracy": 0.50,
        "selected_set_exact_accuracy": 0.50,
    }
    return {
        "candidate_id": "exact-edge-h4-global-v15",
        "seed": 167,
        "cv_mean_metrics": {
            "safety_gate_accuracy": 1.0,
            "swap_consistency": 1.0,
            "preference_macro_f1": 0.80,
            "abstain_recall": 0.80,
            "status_macro_f1": 0.90,
            "exact_array_accuracy": 0.70,
            "selected_set_exact_accuracy": 0.80,
        },
        "cv_fold_metrics": [fold],
        "calibration": {
            "selected_metrics": {
                "expected_calibration_error": 0.05,
                "multiclass_negative_log_likelihood": 0.50,
            }
        },
        "trainable_parameters": 768,
    }


def _selection_receipt(
    candidate_id: str,
    seed: int,
    *,
    qualified: bool,
) -> dict[str, Any]:
    receipt = _qualified_receipt()
    receipt["candidate_id"] = candidate_id
    receipt["seed"] = seed
    receipt["trained"] = True
    receipt["listwise_objective"] = {"candidate_id": candidate_id}
    if not qualified:
        receipt["cv_mean_metrics"]["exact_array_accuracy"] = 0.50
        receipt["cv_mean_metrics"]["selected_set_exact_accuracy"] = 0.50
    return receipt


def test_v15_contract_is_fresh_edge_exact_and_primary_only() -> None:
    config = load_structured(PREREGISTRATION)
    forbidden = config["surface"]["forbidden_prior_lineage_sha256"]
    models = [
        row["model_id"] for row in config["models"]["candidates"]
    ]
    contract = config["dataset_contract"]
    selection = config["development_selection"]
    heldout = config["heldout_evaluation"]
    assert config["schema_version"] == "15.0"
    assert config["status"] == "frozen_before_dataset_generation"
    assert config["source_evidence"]["confirmation_spent"]
    assert len(forbidden) == len(set(forbidden)) == 104
    assert config["training"]["seeds"] == [167, 223, 269]
    assert models == [
        "looped-h4-pair-control-v15",
        "exact-independent-h4-global-v15",
        "exact-edge-h4-factorized-v15",
        "exact-edge-h4-global-v15",
    ]
    assert contract["required_train_arrays_by_size"] == {4: 5, 5: 9, 6: 10}
    assert contract["required_heldout_arrays_by_size"] == {4: 3, 5: 6, 6: 7}
    assert selection["selection_semantics"] == (
        "gate_then_rank_per_architecture"
    )
    assert selection["primary_candidate_ids"] == [
        "exact-edge-h4-global-v15"
    ]
    assert selection["minimum_worst_fold_exact_array_accuracy"] == 0.50
    assert heldout["primary_only_confirmation_packet"]
    assert heldout["minimum_exact_status_vector_accuracy"] == 0.65
    assert heldout["minimum_selected_set_exact_accuracy"] == 0.75


def test_v15_generation_and_plan_bind_edge_and_size_contract(
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
    plan_path = tmp_path / "trainer_plan.json"
    plan = prepare["prepare"](
        PREREGISTRATION,
        fixture / "public_dataset/dataset_manifest.json",
        plan_path,
    )
    listwise = plan["feature_independence_audit"]["listwise_status"]
    encoders = listwise["candidate_encoders"]
    assert report["generation_passed"]
    assert report["checks"]["array_size_counts_match_preregistration"]
    assert manifest["lineage_audit"] == {
        "cross_split_groups": 0,
        "forbidden_prior_groups": 104,
        "heldout_groups": 4,
        "prior_overlap": 0,
        "train_groups": 6,
    }
    assert manifest["array_size_audit"]["train_distribution"] == {
        "4": 5,
        "5": 9,
        "6": 10,
    }
    assert manifest["array_size_audit"]["heldout_distribution"] == {
        "4": 3,
        "5": 6,
        "6": 7,
    }
    assert encoders["independent_candidate"]["feature_count"] == 23
    assert encoders["edge_difference_message"]["feature_count"] == 111
    assert listwise["edge_decoder_ablation_matched"]
    assert listwise["independent_global_control_present"]
    assert not plan["heldout_pair_label_access"]
    assert not plan["heldout_array_label_access"]


def test_v15_exact_model_uses_the_declared_edge_encoder() -> None:
    config = load_structured(PREREGISTRATION)
    specs = {
        row["model_id"]: row["listwise_status"]
        for row in config["models"]["candidates"]
    }
    trainer = runpy.run_path(
        str(ROOT / "scripts/train_ldt_array_confirmation.py")
    )
    edge_encoder = trainer["_candidate_encoder"](
        specs["exact-edge-h4-global-v15"]
    )
    independent_encoder = trainer["_candidate_encoder"](
        specs["exact-independent-h4-global-v15"]
    )
    edge_model = trainer["_make_status_model"](
        specs["exact-edge-h4-global-v15"],
        input_width=len(edge_encoder.feature_names()),
        seed=167,
    )
    independent_model = trainer["_make_status_model"](
        specs["exact-independent-h4-global-v15"],
        input_width=len(independent_encoder.feature_names()),
        seed=167,
    )
    assert edge_encoder.receipt()["feature_view"] == (
        "edge_difference_message"
    )
    assert independent_encoder.receipt()["feature_view"] == (
        "independent_candidate"
    )
    assert isinstance(edge_model, ExactArrayLikelihoodModel)
    assert edge_model.decoder == "global_selected_count_map"
    assert edge_model.parameter_count > independent_model.parameter_count


def test_v15_gate_first_qualification_includes_worst_exact_vector() -> None:
    selector = runpy.run_path(
        str(ROOT / "scripts/select_ldt_listwise_array_v9.py")
    )
    contract = load_structured(PREREGISTRATION)["development_selection"]
    receipt = _qualified_receipt()
    assert selector["_qualified"](receipt, contract)
    receipt["cv_fold_metrics"][0]["exact_array_accuracy"] = 0.25
    assert not selector["_qualified"](receipt, contract)


def test_v15_selector_emits_only_the_qualified_edge_primary(
    tmp_path: Path,
) -> None:
    config = load_structured(PREREGISTRATION)
    candidates = [
        row["model_id"] for row in config["models"]["candidates"]
    ]
    seeds = config["training"]["seeds"]
    primary_id = "exact-edge-h4-global-v15"
    receipts = [
        _selection_receipt(
            candidate_id,
            seed,
            qualified=(candidate_id == primary_id and seed == 167),
        )
        for candidate_id in candidates
        for seed in seeds
    ]
    pair_rows = [
        {
            "candidate_id": candidate_id,
            "seed": seed,
            "example_id": f"pair-{candidate_id}-{seed}",
        }
        for candidate_id in candidates
        for seed in seeds
    ]
    array_rows = [
        {
            "candidate_id": candidate_id,
            "seed": seed,
            "array_id": f"array-{candidate_id}-{seed}",
        }
        for candidate_id in candidates
        for seed in seeds
    ]
    pair_path = tmp_path / "pairs.jsonl"
    array_path = tmp_path / "arrays.jsonl"
    _write_jsonl(pair_path, pair_rows)
    _write_jsonl(array_path, array_rows)
    trainer_path = tmp_path / "trainer.json"
    trainer_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "heldout_pair_targets_seen": False,
                "heldout_array_targets_seen": False,
                "preregistration_sha256": sha256_file(PREREGISTRATION),
                "predictions_sha256": sha256_file(pair_path),
                "array_predictions_sha256": sha256_file(array_path),
                "candidate_receipts": receipts,
            }
        ),
        encoding="utf-8",
    )
    selector = runpy.run_path(
        str(ROOT / "scripts/select_ldt_listwise_array_v9.py")
    )
    out = tmp_path / "selected"
    selection = selector["select"](
        preregistration_path=PREREGISTRATION,
        trainer_summary_path=trainer_path,
        pair_predictions_path=pair_path,
        array_predictions_path=array_path,
        out_dir=out,
    )
    assert selection["confirmation_join_authorized"]
    assert selection["primary_candidate_id"] == primary_id
    assert selection["primary_seed"] == 167
    assert selection["confirmation_packet"]["candidate_count"] == 1
    assert len(
        load_structured(out / "trainer_summary.primary.json")[
            "candidate_receipts"
        ]
    ) == 1


def test_v15_evaluator_refuses_unauthorized_join_before_labels(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    generate_structured_array_dataset(PREREGISTRATION, fixture)
    trainer_path = tmp_path / "trainer.json"
    trainer_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "heldout_pair_targets_seen": False,
                "heldout_array_targets_seen": False,
                "development_selection": {
                    "confirmation_join_authorized": False,
                },
            }
        ),
        encoding="utf-8",
    )
    evaluator = runpy.run_path(
        str(ROOT / "scripts/evaluate_ldt_array_confirmation.py")
    )
    with pytest.raises(
        ValueError,
        match="did not authorize confirmation",
    ):
        evaluator["evaluate"](
            SimpleNamespace(
                preregistration=PREREGISTRATION,
                dataset_manifest=(
                    fixture / "public_dataset/dataset_manifest.json"
                ),
                predictions=tmp_path / "missing-pairs.jsonl",
                array_predictions=tmp_path / "missing-arrays.jsonl",
                trainer_summary=trainer_path,
                pair_labels=tmp_path / "missing-pair-labels.jsonl",
                array_labels=tmp_path / "missing-array-labels.jsonl",
                out=tmp_path / "never-created",
            )
        )


def test_v15_auditor_reproduces_worst_exact_and_heldout_gates() -> None:
    auditor = runpy.run_path(
        str(ROOT / "scripts/audit_ldt_edge_exact_correction_v15.py")
    )
    config = load_structured(PREREGISTRATION)
    receipt = _qualified_receipt()
    assert not auditor["_gate_failures"](
        receipt,
        config["development_selection"],
    )
    receipt["cv_fold_metrics"][0]["exact_array_accuracy"] = 0.25
    assert "worst_fold_exact_array_accuracy" in auditor["_gate_failures"](
        receipt,
        config["development_selection"],
    )

    aggregate: dict[str, Any] = {
        "all_seeds_safety_qualified": True,
        "all_seeds_preference_qualified": True,
        "all_seeds_orientation_qualified": True,
        "all_seeds_uncertainty_qualified": True,
        "all_seeds_calibration_qualified": True,
        "all_seeds_status_qualified": True,
        "all_seeds_exact_array_qualified": True,
        "all_seeds_selected_set_qualified": True,
        "metrics": {
            "exact_array_accuracy": {"mean": 0.65},
            "selected_set_exact_accuracy": {"mean": 0.75},
        },
    }
    assert auditor["_heldout_pass"](
        aggregate,
        config["heldout_evaluation"],
    )
    aggregate["metrics"]["exact_array_accuracy"]["mean"] = 0.625
    assert not auditor["_heldout_pass"](
        aggregate,
        config["heldout_evaluation"],
    )
