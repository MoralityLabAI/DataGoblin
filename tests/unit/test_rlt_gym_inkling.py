"""Offline-only regressions for architectural loop sweeps and student selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured
from datagoblin.training_gym.rlt_gym.inkling import capture, hillclimb, score, stage
from datagoblin.training_gym.rlt_gym.runner import read_jsonl
from datagoblin.training_gym.rlt_gym.tasks import load_suite, reference_answer, write_suite


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _setup(tmp_path: Path) -> tuple[Path, Path, dict]:
    suite = tmp_path / "suite"
    write_suite(suite, seed=7, per_family=1)
    _, tasks = load_suite(suite / "manifest.json")
    frozen = {task["task_id"]: task for task in tasks}
    staged = tmp_path / "stage"
    result = stage(suite, staged, passes=(1, 3), max_tasks=7)
    assert result["network_calls"] == 0
    assert len(result["passes"]) == 2
    assert len(read_jsonl(staged / "p01" / "requests.jsonl")) == 7
    return suite, staged, frozen


def _fixture_capture(staged: Path, captures: Path, frozen: dict, passes: int) -> None:
    key = f"p{passes:02d}"
    requests = read_jsonl(staged / key / "requests.jsonl")
    rows = [
        {
            "request_id": row["request_id"],
            "task_digest": row["task_digest"],
            "candidate": (
                reference_answer(frozen[row["task_id"]]) if passes == 3 else {"not_the_answer": True}
            ),
            "public_steps": ["check with environment"],
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
        for row in requests
    ]
    folder = captures / key
    _write_jsonl(folder / "responses.jsonl", rows)
    dump_structured(
        folder / "manifest.json",
        {
            "schema_version": "rlt_gym.inkling_capture.v1",
            "source_kind": "synthetic_fixture",
            "passes": passes,
            "requests_sha256": sha256_file(staged / key / "requests.jsonl"),
            "responses_sha256": sha256_file(folder / "responses.jsonl"),
            "server_configuration_verified": False,
        },
    )


def test_stage_dry_run_and_endpoint_restriction(tmp_path: Path) -> None:
    suite, staged, _ = _setup(tmp_path)
    plan = capture(staged, tmp_path / "not_created", passes=3, server_loop_passes=3, dry_run=True)
    assert plan["network_calls"] == 0
    assert not (tmp_path / "not_created").exists()
    with pytest.raises(ValueError, match="pass count"):
        capture(staged, tmp_path / "absent", passes=1, server_loop_passes=3, dry_run=True)
    with pytest.raises(ValueError, match="loopback"):
        capture(
            staged, tmp_path / "absent", passes=1,
            server_loop_passes=1, base_url="https://external.example/v1", dry_run=True,
        )
    with pytest.raises(ValueError, match="splits"):
        stage(suite, tmp_path / "bad_split", split="ood_test")


def test_checked_depth_preference_and_fixture_provenance(tmp_path: Path) -> None:
    suite, staged, frozen = _setup(tmp_path)
    captures = tmp_path / "captures"
    for passes in (1, 3):
        _fixture_capture(staged, captures, frozen, passes)
    with pytest.raises(ValueError, match="fixture"):
        score(staged, captures, suite, tmp_path / "reject")
    result = score(staged, captures, suite, tmp_path / "score", include_fixtures=True)
    assert result["training_rows"] == 7
    assert result["preference_pairs"] == 7
    manifest = json.loads((tmp_path / "score" / "manifest.json").read_text())
    assert manifest["student_training_eligible_rows"] == 0
    records = read_jsonl(tmp_path / "score" / "sft.jsonl")
    assert {row["teacher_loop_passes"] for row in records} == {3}
    assert all(row["fixture"] and not row["student_training_eligible"] for row in records)
    assert all("public_steps" not in row["target"] and row["teacher_public_steps_unverified"] for row in records)
    pairs = read_jsonl(tmp_path / "score" / "preferences.jsonl")
    assert all(row["chosen_verification"]["passed"] for row in pairs)
    assert all(not row["rejected_verification"]["passed"] for row in pairs)


def test_hillclimb_selection_never_reads_test_splits(tmp_path: Path) -> None:
    suite, _, frozen = _setup(tmp_path)
    cal = [task for task in frozen.values() if task["split"] == "calibration"]
    rows = []
    for model_id in ("rlt_depth4", "ldt_tiny"):
        for task in cal:
            rows.append({
                "model_id": model_id,
                "task_id": task["task_id"],
                "task_digest": task["task_digest"],
                "visits": 2 if model_id == "ldt_tiny" else 4,
                "candidate": reference_answer(task),
            })
    source = tmp_path / "predictions.jsonl"
    _write_jsonl(source, rows)
    result = hillclimb(suite, source, tmp_path / "ranks")
    assert result["winner"] == "ldt_tiny"
    report = json.loads((tmp_path / "ranks" / "rank.json").read_text())
    assert report["selection_only"] is True
    assert report["requires_fresh_confirmation"] == ["ordinary_test", "ood_test"]
    bad = rows + [{
        "model_id": "rlt_depth4", "task_id": next(t for t in frozen if frozen[t]["split"] == "ood_test"),
        "task_digest": next(t["task_digest"] for t in frozen.values() if t["split"] == "ood_test"),
        "visits": 1, "candidate": {},
    }]
    _write_jsonl(tmp_path / "invalid.jsonl", bad)
    with pytest.raises(ValueError, match="calibration"):
        hillclimb(suite, tmp_path / "invalid.jsonl", tmp_path / "invalid_ranks")


def test_response_parser_is_bounded() -> None:
    from datagoblin.training_gym.rlt_gym.inkling import _decode_candidate

    assert _decode_candidate('{"candidate":{"value":3},"public_steps":["multiply"]}') == (
        True, {"value": 3}, ["multiply"],
    )
    assert _decode_candidate('{"value":3}') == (True, {"value": 3}, [])
    assert _decode_candidate("nonsense")[0] is False
    assert _decode_candidate("x" * 300_000)[0] is False
