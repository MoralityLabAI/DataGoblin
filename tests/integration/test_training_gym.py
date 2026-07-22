from __future__ import annotations

import json
from pathlib import Path

import pytest

from datagoblin.serialization import dump_structured, load_structured
from datagoblin.training_gym.contract import load_gym_contract
from datagoblin.training_gym.demo import run_training_gym_demo
from datagoblin.training_gym.evaluation import evaluate_candidate


def test_training_gym_mock_demo_is_sealed_bounded_and_not_promotable(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    out = tmp_path / "gym"
    comparison = run_training_gym_demo(
        root / "configs/preregistrations/training_gym_v1.yaml",
        root / "examples/recipe.good.yaml",
        out,
        seed=7,
    )
    assert comparison["winner"] is None
    assert {row["candidate_id"] for row in comparison["rows"]} == {
        "constant-accept-negative-control",
        "structural-two-loop-smoke-control",
    }
    assert all(not row["promotion_eligible"] for row in comparison["rows"])
    assert all(not row["promotion_checks"]["all_evaluation_tasks_executable"] for row in comparison["rows"])
    summary = load_structured(out / "summary.json")
    assert summary["status"] == "mock_evaluation_completed"
    assert not summary["weights_loaded"]
    assert summary["cleanup_passed"]
    lifecycle_artifacts = {
        "trainer_plan.json",
        "events.example.jsonl",
        "summary.json",
        "cleanup_summary.json",
        "comparison.json",
        "comparison.md",
    }
    assert lifecycle_artifacts <= {path.name for path in out.iterdir()}
    scored_path = out / "evaluations/structural_smoke/scored_rows.jsonl"
    sealed_rows = [
        json.loads(line)
        for line in scored_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["split"] == "sealed_poison"
    ]
    assert sealed_rows
    assert all("authoritative_label" not in row for row in sealed_rows)
    assert all(row["label_available_to_evaluator"] for row in sealed_rows)

    candidate_path = out / "candidates/structural_smoke/candidate_manifest.json"
    candidate = load_structured(candidate_path)
    candidate["training_splits"] = ["ordinary_test"]
    leaked_path = tmp_path / "leaked_candidate.json"
    dump_structured(leaked_path, candidate)
    with pytest.raises(ValueError, match="evaluation data in training splits"):
        evaluate_candidate(
            out / "frozen_contract.yaml",
            out / "public_dataset/dataset_manifest.json",
            leaked_path,
            out / "candidates/structural_smoke/predictions.jsonl",
            tmp_path / "rejected-evaluation",
        )

    real_without_receipts = load_structured(candidate_path)
    dataset_manifest = load_structured(out / "public_dataset/dataset_manifest.json")
    real_without_receipts["mock"] = False
    real_without_receipts["weights_loaded"] = True
    real_without_receipts["training_splits"] = ["train"]
    real_without_receipts["training_split_hashes"] = {"train": dataset_manifest["splits"]["train"]["sha256"]}
    unreceipted_path = tmp_path / "unreceipted_candidate.json"
    dump_structured(unreceipted_path, real_without_receipts)
    with pytest.raises(ValueError, match="resource summary missing fields"):
        evaluate_candidate(
            out / "frozen_contract.yaml",
            out / "public_dataset/dataset_manifest.json",
            unreceipted_path,
            out / "candidates/structural_smoke/predictions.jsonl",
            tmp_path / "unreceipted-evaluation",
        )


def test_training_gym_contract_rejects_metric_overlap(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    contract = load_structured(root / "configs/preregistrations/training_gym_v1.yaml")
    contract["metrics"]["selection"].append(contract["metrics"]["evidence"][0])
    path = tmp_path / "invalid-contract.yaml"
    dump_structured(path, contract)
    with pytest.raises(ValueError, match="must be disjoint"):
        load_gym_contract(path)
