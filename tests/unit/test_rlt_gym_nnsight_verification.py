"""Identity and verifier custody tests for post-hoc Inkling probe joins."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from datagoblin.training_gym.rlt_gym.nnsight_verification import join_verified_candidates, validate_probe_report


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _receipt():
    task = {"task_id": "task-1", "task_digest": "task-digest", "split": "calibration"}
    model_id, revision, prompt_hash = "model", "sha", "promptsha"
    arms = {}
    for arm in ("observe", "sham", "lesion", "random"):
        arms[arm] = {
            "plan": {"model_id": model_id, "revision": revision, "task_digest": "task-digest", "kind": arm},
            "verifier_status": "unverified", "student_training_eligible": False,
            "prompt_sha256": prompt_hash,
        }
    report = {
        "schema_version": "rlt_gym.nnsight_message_bank.v1", "model_id": model_id,
        "revision": revision, "task_digest": "task-digest", "verifier_status": "unverified",
        "student_training_eligible": False, "arms": arms,
    }
    report["sha256"] = hashlib.sha256(_canonical(report).encode("utf-8")).hexdigest()
    rows = [
        {"arm": arm, "task_id": "task-1", "task_digest": "task-digest",
         "probe_report_sha256": report["sha256"], "model_id": model_id,
         "revision": revision, "prompt_sha256": prompt_hash,
         "candidate": {"value": 7 if arm != "lesion" else 9}}
        for arm in arms
    ]
    return report, rows, task


def _verify(task, candidate):
    return {
        "passed": candidate == {"value": 7}, "status": "passed" if candidate == {"value": 7} else "failed",
        "task_digest": task["task_digest"], "verifier_id": "fixture-verifier",
    }


def test_verifier_join_preserves_independent_receipts() -> None:
    report, rows, task = _receipt()
    result = join_verified_candidates(report, rows, task, _verify)
    assert result["arms"]["lesion"]["verification"]["passed"] is False
    assert result["matched_controls"]["lesion"]["differs_from_sham"] is True
    assert result["causal_effect_verified"] is False
    assert result["model_execution_attested"] is False
    assert result["student_training_eligible"] is False


def test_verifier_join_rejects_tampering_and_split_leakage() -> None:
    report, rows, task = _receipt()
    with pytest.raises(ValueError, match="hash"):
        altered = deepcopy(report)
        altered["arms"]["lesion"]["plan"]["revision"] = "tampered"
        validate_probe_report(altered, task)
    with pytest.raises(ValueError, match="sealed"):
        validate_probe_report(report, {**task, "split": "sealed_poison"})
    with pytest.raises(ValueError, match="duplicate"):
        join_verified_candidates(report, [*rows[:3], rows[0]], task, _verify)
    with pytest.raises(ValueError, match="provenance"):
        altered_rows = deepcopy(rows)
        altered_rows[1]["revision"] = "other"
        join_verified_candidates(report, altered_rows, task, _verify)
    with pytest.raises(RuntimeError, match="verifier"):
        join_verified_candidates(report, rows, task, lambda *_: {"passed": True})


def test_real_rlt_verifier_on_frozen_calibration_task(tmp_path) -> None:
    """Uses the repository's actual world generator/verifier when available."""
    tasks = pytest.importorskip("datagoblin.training_gym.rlt_gym.tasks")
    suite = tmp_path / "suite"
    tasks.write_suite(suite, seed=7, per_family=1)
    _, all_tasks = tasks.load_suite(suite / "manifest.json")
    task = next(item for item in all_tasks if item["split"] == "calibration")
    report, rows, _ = _receipt()
    report["task_digest"] = task["task_digest"]
    for name, arm in report["arms"].items():
        arm["plan"]["task_digest"] = task["task_digest"]
    report_body = {key: value for key, value in report.items() if key != "sha256"}
    report["sha256"] = hashlib.sha256(_canonical(report_body).encode()).hexdigest()
    for row in rows:
        row["task_id"] = task["task_id"]
        row["task_digest"] = task["task_digest"]
        row["probe_report_sha256"] = report["sha256"]
        row["candidate"] = tasks.reference_answer(task)
    scored = join_verified_candidates(report, rows, task, tasks.verify)
    assert all(row["verification"]["passed"] for row in scored["arms"].values())
    assert all(row["verification"]["verifier_id"] for row in scored["arms"].values())
    assert scored["model_execution_attested"] is False
