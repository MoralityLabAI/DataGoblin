from __future__ import annotations

import json
from pathlib import Path

import pytest

from datagoblin.adapter_verifier.architecture_tournament import (
    apply_v15_shadow_sorter,
    build_architecture_tournament_fixture,
    load_architecture_tournament_contract,
    train_architecture_candidate,
    validate_architecture_candidate,
)
from datagoblin.experiments.ldt_edge_set_array_v10 import EdgeDifferenceCandidateEncoder
from datagoblin.experiments.ldt_exact_array_likelihood_v12 import ExactArrayLikelihoodModel
from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured, load_structured

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/preregistrations/game_adapter_architecture_tournament_v2.yaml"


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def test_v2_contract_freezes_matched_architecture_array() -> None:
    config = load_architecture_tournament_contract(CONFIG)
    assert config["status"] == "frozen_before_dataset_generation"
    assert config["fixture_plan"]["required_arrays_per_lane"] == 12
    assert config["fixture_plan"]["required_candidates_per_array"] == 4
    assert config["model_seed_plan"]["seeds"] == [31, 73, 127]
    assert [arm["role"] for arm in config["arms"]] == [
        "unchanged_deterministic_control",
        "low_rank_adapter",
        "recurrence_matched_one_pass_control",
        "hrm_text_style_recurrent_surrogate",
    ]
    assert not config["metrics"]["universal_scalar_winner"]
    assert config["source_evidence"]["environment_verifier_remains_decision_owner"]
    assert config["resource_contract"]["caps"]["swap_bytes"] == 0


def test_v2_generation_separates_candidate_and_steward_custody(tmp_path: Path) -> None:
    out = tmp_path / "fixture"
    report = build_architecture_tournament_fixture(CONFIG, out)
    assert report["generation_passed"]
    assert all(report["checks"].values())
    public = load_structured(out / "public_dataset/tournament_manifest.json")
    release = load_structured(out / "candidate_release/release_manifest.json")
    custody = load_structured(out / "private_steward/custody_manifest.json")
    assert len(public["fixtures"]["development"]) == 3
    assert len(public["fixtures"]["heldout"]) == 3
    assert public["lineage_audit"]["cross_lane_overlap"] == 0
    assert public["record_audit"]["cross_lane_overlap"] == 0
    assert {entry["surface_id"] for entry in release["entries"]} == {
        entry["surface_id"] for entry in public["fixtures"]["development"]
    }
    assert not release["evaluation_labels_included"]
    assert not release["heldout_labels_included"]
    assert custody["heldout_reveal_count"] == 0
    assert custody["heldout_reveal_budget"] == 1
    assert not custody["heldout_labels_exposed_to_candidate"]
    assert not any("ground_truth" in path.name for path in (out / "public_dataset").rglob("*"))


def test_v2_deterministic_control_never_reads_training_or_heldout_labels(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    build_architecture_tournament_fixture(CONFIG, fixture)
    run = tmp_path / "control"
    candidate = train_architecture_candidate(
        CONFIG,
        fixture / "public_dataset/tournament_manifest.json",
        fixture / "candidate_release/release_manifest.json",
        run,
        candidate_id="frozen-exact-key-control-v2",
        seed=0,
    )
    assert not candidate["trainable"]
    assert candidate["training_splits"] == []
    assert candidate["calibration_splits"] == []
    assert not candidate["evaluation_label_access"]
    assert not candidate["heldout_label_access"]
    fit = load_structured(run / "fit_receipt.json")
    assert fit["training_splits"] == []
    assert fit["calibration_splits"] == []
    assert not fit["development_evaluation_labels_seen"]
    assert not fit["heldout_labels_seen"]
    preflight = validate_architecture_candidate(
        CONFIG,
        fixture / "public_dataset/tournament_manifest.json",
        run,
        require_resource_receipt=False,
    )
    assert preflight["valid"]

    candidate["heldout_label_access"] = True
    dump_structured(run / "candidate_manifest.json", candidate)
    with pytest.raises(ValueError, match="candidate preflight failed"):
        validate_architecture_candidate(
            CONFIG,
            fixture / "public_dataset/tournament_manifest.json",
            run,
            require_resource_receipt=False,
        )


def test_v2_shadow_sorter_is_hash_bound_and_recommendation_only(tmp_path: Path) -> None:
    config = load_structured(CONFIG)
    encoder = EdgeDifferenceCandidateEncoder()
    model = ExactArrayLikelihoodModel(
        input_width=len(encoder.feature_names()),
        hidden_width=4,
        loop_count=3,
        seed=167,
        max_candidates=6,
        decoder="global_selected_count_map",
    )
    model_path = tmp_path / "retained-model.json"
    dump_structured(
        model_path,
        {
            "schema_version": "ldt-array-confirmation-model.v3",
            "candidate_id": "exact-edge-h4-global-v15",
            "seed": 167,
            "authority": "MODEL_RECOMMENDATION_ONLY",
            "heldout_targets_seen": False,
            "listwise_state": model.state_dict(),
        },
    )
    config["source_evidence"]["v15_model_sha256"] = sha256_file(model_path)
    config["benchmark_config"] = str(
        ROOT / "configs/preregistrations/game_adapter_schema_drift_v1.yaml"
    )
    config_path = tmp_path / "config.yaml"
    dump_structured(config_path, config)
    candidate_ids = [arm["candidate_id"] for arm in config["arms"]]
    arrays = []
    for index in range(12):
        candidates = []
        for offset, (candidate_id, arm) in enumerate(zip(candidate_ids, config["arms"], strict=True)):
            value = 0.60 + 0.06 * offset
            interval = {"lower": value, "upper": value}
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "features": {
                        "game": config["games"][index % 4],
                        "fixture_pressure": config["fixture_plan"]["development"][index // 4]["pressure"],
                        "context_mode": arm["context_mode"],
                        "optimization_pressure": arm["optimization_pressure"],
                        "trainable_parameter_bucket": arm["trainable_parameter_bucket"],
                        "safety_qualified": True,
                        "ordinary_exact_record_accuracy": interval,
                        "ordinary_mapping_f1": interval,
                        "ood_exact_record_accuracy": interval,
                        "ood_mapping_f1": interval,
                    },
                }
            )
        arrays.append(
            {
                "schema_version": "game-adapter-architecture-array.v2",
                "array_id": f"array-{index:02d}",
                "fixture_pressure": candidates[0]["features"]["fixture_pressure"],
                "game": candidates[0]["features"]["game"],
                "candidate_ids": candidate_ids,
                "candidates": candidates,
            }
        )
    arrays_path = tmp_path / "arrays.jsonl"
    _jsonl(arrays_path, arrays)
    development_path = tmp_path / "development.json"
    dump_structured(
        development_path,
        {
            "tournament_config_sha256": sha256_file(config_path),
            "public_arrays_sha256": sha256_file(arrays_path),
            "heldout_labels_read": False,
            "heldout_reveal_count": 0,
        },
    )
    receipt = apply_v15_shadow_sorter(
        config_path,
        development_path,
        arrays_path,
        model_path,
        tmp_path / "shadow",
    )
    assert receipt["array_count"] == 12
    assert receipt["authority"] == "MODEL_RECOMMENDATION_ONLY"
    assert not receipt["heldout_metrics_seen"]
    assert not receipt["heldout_targets_seen"]
    predictions = [
        json.loads(line)
        for line in (tmp_path / "shadow/shadow_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(predictions) == 12
    assert all(row["candidate_ids"] == candidate_ids for row in predictions)
    assert all(row["authority"] == "MODEL_RECOMMENDATION_ONLY" for row in predictions)
