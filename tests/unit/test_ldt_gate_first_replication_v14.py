from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from datagoblin.experiments.ldt_array_sorter import sha256_file
from datagoblin.experiments.ldt_structured_array_v8 import (
    generate_structured_array_dataset,
)
from datagoblin.serialization import load_structured

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    ROOT
    / "configs"
    / "preregistrations"
    / "ldt_gate_first_replication_v14.yaml"
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _receipt(
    candidate_id: str,
    seed: int,
    *,
    qualified: bool,
) -> dict[str, object]:
    exact = 0.75 if qualified else 0.50
    selected = 0.75 if qualified else 0.50
    mean = {
        "safety_gate_accuracy": 1.0,
        "swap_consistency": 1.0,
        "preference_macro_f1": 0.80,
        "abstain_recall": 0.80,
        "status_macro_f1": 0.90,
        "exact_array_accuracy": exact,
        "selected_set_exact_accuracy": selected,
    }
    fold = {
        "preference_macro_f1": 0.80,
        "abstain_recall": 0.80,
        "status_macro_f1": 0.85,
        "exact_array_accuracy": 0.50,
        "selected_set_exact_accuracy": 0.50,
    }
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "trained": True,
        "cv_mean_metrics": mean,
        "cv_fold_metrics": [fold],
        "calibration": {
            "selected_metrics": {
                "expected_calibration_error": 0.05,
                "multiclass_negative_log_likelihood": 0.50,
            }
        },
        "listwise_objective": {"candidate_id": candidate_id},
        "trainable_parameters": 416,
    }


def test_v14_contract_is_fresh_gate_first_and_primary_only() -> None:
    config = load_structured(PREREGISTRATION)
    forbidden = config["surface"]["forbidden_prior_lineage_sha256"]
    models = [
        row["model_id"] for row in config["models"]["candidates"]
    ]
    selection = config["development_selection"]
    heldout = config["heldout_evaluation"]
    assert config["schema_version"] == "14.0"
    assert config["status"] == "frozen_before_dataset_generation"
    assert len(forbidden) == len(set(forbidden)) == 94
    assert config["training"]["seeds"] == [61, 107, 149]
    assert models == [
        "looped-h4-pair-control-v14",
        "looped-h4-recurrent-factorized-v14",
        "exact-array-h4-factorized-v14",
        "exact-array-h4-global-map-v14",
    ]
    assert selection["selection_semantics"] == (
        "gate_then_rank_per_architecture"
    )
    assert selection["primary_candidate_ids"] == [
        "exact-array-h4-global-map-v14"
    ]
    assert heldout["primary_only_confirmation_packet"]
    assert heldout["minimum_abstain_recall"] == 0.75
    assert heldout["minimum_exact_status_vector_accuracy"] == 0.65
    assert heldout["minimum_selected_set_exact_accuracy"] == 0.75


def test_v14_generation_and_plan_bind_gate_first_contract(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    report = generate_structured_array_dataset(
        PREREGISTRATION,
        fixture,
    )
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
    policy = plan["feature_independence_audit"]["selection_policy"]
    assert report["generation_passed"]
    assert manifest["lineage_audit"] == {
        "cross_split_groups": 0,
        "forbidden_prior_groups": 94,
        "heldout_groups": 4,
        "prior_overlap": 0,
        "train_groups": 6,
    }
    assert policy == {
        "semantics": "gate_then_rank_per_architecture",
        "primary_candidate_ids": ["exact-array-h4-global-map-v14"],
        "primary_only_confirmation_packet": True,
        "exact_status_vector_gate": 0.65,
        "exact_selected_set_gate": 0.75,
        "safety_gate": 1.0,
        "swap_gate": 1.0,
        "preference_macro_f1_gate": 0.75,
        "abstain_recall_gate": 0.75,
        "status_macro_f1_gate": 0.80,
    }
    assert not plan["heldout_pair_label_access"]
    assert not plan["heldout_array_label_access"]


def test_v14_selector_emits_only_the_qualified_primary_packet(
    tmp_path: Path,
) -> None:
    config = load_structured(PREREGISTRATION)
    candidates = [
        row["model_id"] for row in config["models"]["candidates"]
    ]
    seeds = config["training"]["seeds"]
    primary_id = "exact-array-h4-global-map-v14"
    receipts = [
        _receipt(
            candidate_id,
            seed,
            qualified=(candidate_id == primary_id and seed == 61),
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
    packet = selection["confirmation_packet"]
    primary_summary = load_structured(
        out / "trainer_summary.primary.json"
    )
    assert selection["confirmation_join_authorized"]
    assert selection["primary_candidate_id"] == primary_id
    assert selection["primary_seed"] == 61
    assert packet["candidate_count"] == 1
    assert packet["candidate_id"] == primary_id
    assert not packet["private_labels_read"]
    assert len(primary_summary["candidate_receipts"]) == 1
    assert primary_summary["confirmation_packet_candidate_count"] == 1
    assert len(
        (out / "heldout_pair_predictions.primary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 1
    assert len(
        (out / "heldout_array_predictions.primary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 1


def test_v14_evaluator_refuses_unauthorized_join_before_labels(
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


def test_v14_evaluator_binds_array_source_to_frozen_candidate_spec() -> None:
    evaluator = runpy.run_path(
        str(ROOT / "scripts/evaluate_ldt_array_confirmation.py")
    )
    config = load_structured(PREREGISTRATION)
    expected_sources = evaluator["_expected_array_prediction_sources"](
        config
    )
    candidate_id = "exact-array-h4-global-map-v14"
    assert expected_sources[candidate_id] == (
        "exact_likelihood_global_selected_count_map"
    )

    candidates = [{"candidate_id": f"candidate-{index}"} for index in range(4)]
    heldout_array = {
        "array_id": "heldout-array",
        "lineage_group_sha256": "a" * 64,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    statuses = ["SELECTED", "UNSAFE", "DOMINATED", "UNRESOLVED"]
    status_probabilities = [
        {
            "UNSAFE": 0.50,
            "DOMINATED": 0.05,
            "UNRESOLVED": 0.05,
            "SELECTED": 0.40,
        },
        {
            "UNSAFE": 0.60,
            "DOMINATED": 0.05,
            "UNRESOLVED": 0.05,
            "SELECTED": 0.30,
        },
        {
            "UNSAFE": 0.10,
            "DOMINATED": 0.70,
            "UNRESOLVED": 0.10,
            "SELECTED": 0.10,
        },
        {
            "UNSAFE": 0.10,
            "DOMINATED": 0.10,
            "UNRESOLVED": 0.70,
            "SELECTED": 0.10,
        },
    ]
    prediction = {
        "schema_version": "ldt-listwise-array-prediction.v9",
        "candidate_id": candidate_id,
        "seed": 107,
        "array_id": heldout_array["array_id"],
        "lineage_group_sha256": heldout_array["lineage_group_sha256"],
        "candidate_count": len(candidates),
        "candidate_input_sha256": evaluator["sha256_json"](candidates),
        "dataset_manifest_sha256": "b" * 64,
        "prediction_source": expected_sources[candidate_id],
        "statuses": statuses,
        "status_probabilities": status_probabilities,
        "authority": "MODEL_RECOMMENDATION_ONLY",
    }
    grouped = evaluator["_validate_array_predictions"](
        [prediction],
        [heldout_array],
        manifest_sha256="b" * 64,
        expected_prediction_sources=expected_sources,
    )
    assert list(grouped) == [(candidate_id, 107)]

    mismatched = dict(prediction)
    mismatched["prediction_source"] = (
        "exact_likelihood_factorized_status_argmax"
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        evaluator["_validate_array_predictions"](
            [mismatched],
            [heldout_array],
            manifest_sha256="b" * 64,
            expected_prediction_sources=expected_sources,
        )

    wrong_selected_set = dict(prediction)
    wrong_selected_set["statuses"] = [
        "UNSAFE",
        "SELECTED",
        "DOMINATED",
        "UNRESOLVED",
    ]
    with pytest.raises(
        ValueError,
        match="global selected-count decoder output is inconsistent",
    ):
        evaluator["_validate_array_predictions"](
            [wrong_selected_set],
            [heldout_array],
            manifest_sha256="b" * 64,
            expected_prediction_sources=expected_sources,
        )


def test_v14_public_auditor_reproduces_both_exact_gates() -> None:
    auditor = runpy.run_path(
        str(ROOT / "scripts/audit_ldt_gate_first_replication_v14.py")
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
            "exact_array_accuracy": {"mean": 0.60},
            "selected_set_exact_accuracy": {"mean": 0.80},
        },
    }
    heldout = load_structured(PREREGISTRATION)["heldout_evaluation"]
    assert not auditor["_heldout_pass"](aggregate, heldout)
    aggregate["metrics"]["exact_array_accuracy"]["mean"] = 0.70
    aggregate["metrics"]["selected_set_exact_accuracy"]["mean"] = 0.70
    assert not auditor["_heldout_pass"](aggregate, heldout)
    aggregate["metrics"]["selected_set_exact_accuracy"]["mean"] = 0.80
    assert auditor["_heldout_pass"](aggregate, heldout)
