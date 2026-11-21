"""Offline-only regressions for architectural loop sweeps and student selection."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured
from datagoblin.training_gym.rlt_gym import inkling
from datagoblin.training_gym.rlt_gym.inkling import capture, hillclimb, score, stage
from datagoblin.training_gym.rlt_gym.inkling_instrumentation import LoopBoundary, stability_sample
from datagoblin.training_gym.rlt_gym.inkling_rlt_trace import fit_feedback
from datagoblin.training_gym.rlt_gym.inkling_sae import fit_sae
from datagoblin.training_gym.rlt_gym.inkling_trace import (
    WIDTH,
    boundary_sites,
    export_sae_inputs,
    seal_trace,
    stage_traces,
)
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
    assert manifest["stream_rows"]["outcomes"] == 14
    assert manifest["stream_rows"]["repairs"] == 7
    assert manifest["stream_rows"]["aleph"] == 0
    records = read_jsonl(tmp_path / "score" / "sft.jsonl")
    assert {row["teacher_loop_passes"] for row in records} == {3}
    assert all(row["fixture"] and not row["student_training_eligible"] for row in records)
    assert all("public_steps" not in row["target"] and row["teacher_public_steps_unverified"] for row in records)
    pairs = read_jsonl(tmp_path / "score" / "preferences.jsonl")
    assert all(row["chosen_verification"]["passed"] for row in pairs)
    assert all(not row["rejected_verification"]["passed"] for row in pairs)


def test_capture_binds_model_identity_and_preserves_spent_tokens(tmp_path: Path, monkeypatch) -> None:
    suite, staged, frozen = _setup(tmp_path)
    captures = tmp_path / "captures"
    first = read_jsonl(staged / "p01" / "requests.jsonl")[0]
    answer = canonical_json(reference_answer(frozen[first["task_id"]]))
    replies = iter([
        (inkling.MODEL, '{"not_the_answer":true}'), ("unexpected-model", answer),
        (inkling.MODEL, answer), (inkling.MODEL, None),
    ])

    def open_response(request, timeout):
        body = json.loads(request.data)
        assert body["model"] == inkling.MODEL and timeout > 0
        model, content = next(replies)
        return BytesIO(canonical_json({
            "model": model, "usage": {"prompt_tokens": 25, "completion_tokens": 10},
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        }).encode())

    monkeypatch.setattr(inkling.urllib.request, "build_opener", lambda *_: SimpleNamespace(open=open_response))
    for passes in (1, 3):
        capture(staged, captures / f"p{passes:02d}", passes=passes, server_loop_passes=passes,
                max_requests=2, weights_revision="fixture-in-test")
    out = tmp_path / "score"
    assert score(staged, captures, suite, out)["preference_pairs"] == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert [arm["reported_tokens"] for arm in manifest["per_pass"]] == [70, 70]
    assert [arm["verified_outcomes"] for arm in manifest["per_pass"]] == [1, 1]
    assert manifest["paired_depths"][0]["paired_verified_tasks"] == 1
    assert manifest["paired_depths"][0]["higher_only"] == 1
    rows = read_jsonl(out / "aleph.jsonl")
    assert {row["kind"] for row in rows} == {"sft", "outcomes", "repairs"}
    assert all(row["loss"]["boundary"] == "response_only" for row in rows)
    observation_path = captures / "p01" / "observations.jsonl"
    observation_path.write_bytes(observation_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="observation digest"):
        score(staged, captures, suite, tmp_path / "reject")


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
    bad = [*rows, {
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


def test_instrumented_linear_loop_gain() -> None:
    matrix = np.diag([0.25, 0.75])
    probe = SimpleNamespace(
        activation=lambda point: np.array([2.0, 3.0]),
        advance=lambda point, state: matrix @ state,
        jvp=lambda point, state, tangent: matrix @ tangent,
        vjp=lambda point, state, cotangent: matrix.T @ cotangent,
    )
    metrics = stability_sample(probe, LoopBoundary("task", 1, 3, 17, 0, "message"))
    assert metrics["spectral_gain_estimate"] == pytest.approx(0.75)
    assert metrics["finite_dose_directional_gain"] == pytest.approx(0.75)
    assert metrics["contraction_certified"] is False


def test_boundary_trace_preserves_site_tokens_and_lineages(tmp_path: Path) -> None:
    suite, train, _ = _setup(tmp_path)
    cal = tmp_path / "cal"
    stage(suite, cal, passes=(1, 3), split="calibration")
    prompts = tmp_path / "prompts"
    plan = stage_traces(suite, [train, cal], prompts, per_stage=1)
    assert [row["split"] for row in plan["samples"]] == ["train", "calibration"]
    assert len({row["lineage_id"] for row in plan["samples"]}) == 2
    sites = boundary_sites(3)
    assert sites == {0: "l_out-5", 1: "l_out-17", 2: "l_out-35", 3: "l_out-47"}
    result = tmp_path / "native"
    result.mkdir()
    index = result / "rows.tsv"
    lines = ["sample_key\tpass_index\ttensor\ttoken_index\n"]
    lines.extend(f"{row['sample_key']}\t{depth}\t{tensor}\t{token}\n"
                 for row in plan["samples"] for depth, tensor in sites.items() for token in range(16))
    index.write_text("".join(lines[:-1]), encoding="utf-8")
    vectors = np.random.default_rng(7).normal(size=(128, WIDTH)).astype("<f4")
    vectors.tofile(result / "activations.f32")
    (result / "native-complete.txt").write_text("128\n", encoding="utf-8")
    provenance = {"passes": 3, "weights_sha256": "1" * 64, "runtime_sha256": "2" * 64}
    with pytest.raises(ValueError, match="token positions"):
        seal_trace(prompts, result, **provenance)
    index.write_text("".join(lines), encoding="utf-8")
    report = seal_trace(prompts, result, **provenance)
    assert report["rows"] == 128
    assert report["tokens_by_sample"] == {"000": 16, "001": 16}
    exported = tmp_path / "sae"
    streams = export_sae_inputs(prompts, result, exported)["streams"]
    assert streams["train"]["rows"] == streams["calibration"]["rows"] == 48
    assert streams["train"]["roots"] == streams["calibration"]["roots"] == 1
    with np.load(exported / "calibration.npz") as archive:
        assert archive["inkling_core_prefill"].shape == (48, WIDTH)
    dictionary = tmp_path / "dictionary"
    fitted = fit_sae(exported, dictionary, features=8, top_k=2, steps=2)
    assert fitted["calibration"]["rows"] == 48
    feedback = fit_feedback(prompts, result, exported, dictionary, tmp_path / "heads", max_tokens=16)
    assert feedback["train_tokens"] == feedback["calibration_tokens"] == 16
    assert [row["eta"] for row in feedback["calibration"]["rlt"]] == [1 / 3, 0.5, 1]
    assert feedback["calibration"]["matched_damping"][0]["eta"] == pytest.approx(5 / 12)
    index.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed receipt"):
        export_sae_inputs(prompts, result, tmp_path / "tampered")
