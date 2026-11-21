"""Native TRM exit trace import fixtures, deliberately training-ineligible."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json
from datagoblin.training_gym.rlt_gym.native_exit_bridge import (
    MODEL_SHA256,
    _digest,
    import_native_exit,
)


def _stage(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    solution = "".join(str((r * 3 + r // 3 + c) % 9 + 1) for r in range(9) for c in range(9))
    question = "." + solution[1:]
    root = "development-sudoku-root1"
    fixed = {
        "trace": [
            {"outer_step": 1, "candidate": "1" * 81, "sudoku_checker_passed": False,
             "mean_blank_logit_margin": 0.1, "min_blank_logit_margin": 0.1},
            {"outer_step": 2, "candidate": solution, "sudoku_checker_passed": True,
             "mean_blank_logit_margin": 3.0, "min_blank_logit_margin": 3.0},
        ],
    }
    src = {
        "schema_version": "bpd.native_trm_verifier_exit.v1",
        "model": "gaoxin492", "model_sha256": MODEL_SHA256,
        "evidence_class": "native_pretrained_development",
        "split": "development", "student_training_eligible": False,
        "check_uses_planted_solution": False, "roots": 1, "holes": 1,
        "max_outer_steps": 2,
        "records": [{
            "root_id": root, "question": question,
            "model_sha256": MODEL_SHA256,
            "task_digest": _digest({"question": question}),
            "policies": {
                "fixed_2": fixed,
                "verifier_exit": {"trace": [dict(v) for v in fixed["trace"]]},
            },
        }],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(canonical_json(src), encoding="utf-8")
    audit = {
        "schema_version": "bpd.native_trm_verifier_exit_audit.v1",
        "receipt_sha256": sha256_file(source_path),
        "checkpoint_sha256": MODEL_SHA256, "root_and_checker_replay_passed": True,
        "source_roots": 1, "depth_one_verified": 0,
        "fixed_solved": 1, "adaptive_solved": 1,
        "fixed_steps": 2, "adaptive_steps": 2,
        "step_fraction_saved": 0, "measured_native_fraction_saved": 0,
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(canonical_json(audit), encoding="utf-8")
    return source_path, audit_path


def test_import_rechecks_verifier_and_preserves_development_custody(tmp_path: Path) -> None:
    src, audited = _stage(tmp_path)
    manifest = import_native_exit(src, audited, tmp_path / "out")
    assert manifest["root_count"] == 1
    assert manifest["transition_count"] == 2
    assert manifest["training_eligible"] is False
    assert manifest["verifier_exit_correct"] == 1
    rows = [
        json.loads(line) for line in (tmp_path / "out" / "depth_transitions.jsonl").read_text().splitlines()
    ]
    assert [r["target"]["exit_decision"] for r in rows] == ["continue", "stop"]
    assert all(row["under_online_adaptive_trajectory"] for row in rows)
    assert all(r["training_eligible"] is False and r["split"] == "development" for r in rows)


def test_import_rejects_changed_checker_receipts(tmp_path: Path) -> None:
    src, audited = _stage(tmp_path)
    value = json.loads(src.read_text())
    value["records"][0]["policies"]["fixed_2"]["trace"][1]["sudoku_checker_passed"] = False
    src.write_text(canonical_json(value), encoding="utf-8")
    audit_value = json.loads(audited.read_text())
    audit_value["receipt_sha256"] = sha256_file(src)
    audited.write_text(canonical_json(audit_value), encoding="utf-8")
    with pytest.raises(ValueError, match="verifier outcome"):
        import_native_exit(src, audited, tmp_path / "out")


def test_import_rejects_unpaired_audit_or_train_leakage(tmp_path: Path) -> None:
    src, audited = _stage(tmp_path)
    audit = json.loads(audited.read_text())
    audit["receipt_sha256"] = "0" * 64
    audited.write_text(canonical_json(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="audit"):
        import_native_exit(src, audited, tmp_path / "mismatch")
    src2, audited2 = _stage(tmp_path / "second")
    value = json.loads(src2.read_text())
    value["student_training_eligible"] = True
    src2.write_text(canonical_json(value), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        import_native_exit(src2, audited2, tmp_path / "not_training")


def test_import_rejects_changed_rlt_prefix(tmp_path: Path) -> None:
    src, audited = _stage(tmp_path)
    value = json.loads(src.read_text())
    value["records"][0]["policies"]["verifier_exit"]["trace"][0]["candidate"] = "2" * 81
    src.write_text(canonical_json(value), encoding="utf-8")
    audit_value = json.loads(audited.read_text())
    audit_value["receipt_sha256"] = sha256_file(src)
    audited.write_text(canonical_json(audit_value), encoding="utf-8")
    with pytest.raises(ValueError, match="prefix"):
        import_native_exit(src, audited, tmp_path / "out")



def test_import_marks_counterfactual_fixed_depth_after_early_verified_stop(tmp_path: Path) -> None:
    src, audited = _stage(tmp_path)
    value = json.loads(src.read_text())
    record = value["records"][0]
    solved = record["policies"]["fixed_2"]["trace"][1]["candidate"]
    record["policies"]["fixed_2"]["trace"][0]["candidate"] = solved
    record["policies"]["fixed_2"]["trace"][0]["sudoku_checker_passed"] = True
    record["policies"]["verifier_exit"]["trace"] = [
        dict(record["policies"]["fixed_2"]["trace"][0])
    ]
    src.write_text(canonical_json(value), encoding="utf-8")
    a = json.loads(audited.read_text())
    a["receipt_sha256"] = sha256_file(src)
    a["depth_one_verified"] = 1
    a["adaptive_steps"] = 1
    a["step_fraction_saved"] = 0.5
    audited.write_text(canonical_json(a), encoding="utf-8")
    import_native_exit(src, audited, tmp_path / "out")
    lines = (tmp_path / "out" / "depth_transitions.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert rows[0]["under_online_adaptive_trajectory"] is True
    assert rows[1]["under_online_adaptive_trajectory"] is False
    assert rows[1]["counterfactual_fixed_depth_only"] is True
    assert all(row["training_eligible"] is False for row in rows)
