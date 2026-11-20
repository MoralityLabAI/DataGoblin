"""Development-only gym examples. These are explicit labels, not sealed evaluation."""
from __future__ import annotations

from pathlib import Path

import pytest

from training.gyms.semaev_redteam.grade_dev import evaluate, load_tasks


def task_file() -> Path:
    return Path(__file__).resolve().parents[2] / "training/gyms/semaev_redteam/tasks_dev.json"


def test_all_task_roots_are_unique_and_references_are_named() -> None:
    tasks = load_tasks(task_file())
    assert len(tasks) == 6
    for task in tasks.values():
        assert task["source_paths"]
        assert task["required_obligations"]


def test_gold_response_gets_full_score() -> None:
    tasks = load_tasks(task_file())
    task = tasks["RMP-02"]
    response = {
        "task_id": "RMP-02",
        "claim_status": task["gold_status"],
        "missing_obligations": task["required_obligations"],
        "evidence_paths": task["source_paths"],
        "next_check_kind": task["next_check_kind"],
        "confidence": 1.0,
    }
    report = evaluate(tasks, [response])
    assert report["mean_score"] == 1.0
    assert report["mean_status_brier"] == 0.0
    assert report["status"] == "development_only_unsealed"


def test_overconfident_incorrect_claim_is_penalized() -> None:
    tasks = load_tasks(task_file())
    report = evaluate(tasks, [{
        "task_id": "SEC-04",
        "claim_status": "scope_limited",
        "missing_obligations": [],
        "evidence_paths": [],
        "next_check_kind": "lean_proof",
        "confidence": 1,
    }])
    assert report["mean_score"] == 0.0
    assert report["mean_status_brier"] == 1.0


def test_duplicate_responses_rejected() -> None:
    tasks = load_tasks(task_file())
    response = {
        "task_id": "GDL-01", "claim_status": "scope_limited",
        "missing_obligations": [],
        "evidence_paths": [],
        "next_check_kind": "formal_scope_audit", "confidence": 0.5,
    }
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(tasks, [response, response])
