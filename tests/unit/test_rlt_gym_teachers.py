from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from datagoblin.serialization import canonical_json
from datagoblin.training_gym.rlt_gym.tasks import FAMILIES, load_suite, reference_answer, verify, write_suite
from datagoblin.training_gym.rlt_gym.teachers import (
    continue_teacher_batch,
    export_dataset,
    import_teacher_responses,
    prepare_teacher_batch,
    prepare_teacher_repairs,
)


@pytest.fixture
def suite(tmp_path):
    manifest = write_suite(tmp_path / "suite", seed=77, per_family=2)
    _, tasks = load_suite(manifest)
    return manifest, {task["task_id"]: task for task in tasks}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def batch(suite, out, *, provider="codex", **kwargs):
    manifest, _ = suite
    result = prepare_teacher_batch(manifest, out, provider=provider, model="operator-supplied-model", **kwargs)
    requests = read_rows(result["requests_path"])
    return result, requests


def response(request, task, *, correct=True):
    candidate = (
        reference_answer(task) if correct else next(c for c in task["candidates"] if not verify(task, c)["passed"])
    )
    return {
        "request_id": request["request_id"],
        "task_digest": request["task_digest"],
        "candidate": candidate,
        "public_steps": ["Computed the requested visible result."],
        "confidence": 0.9,
        "usage": {"input_tokens": 40, "output_tokens": 20},
    }


def import_rows(suite, batch_result, rows, out):
    source = out.parent / (out.name + "_responses.jsonl")
    write_rows(source, rows)
    return import_teacher_responses(suite[0], Path(batch_result["requests_path"]), source, out)


def test_batch_is_balanced_and_has_no_evaluator_fields(suite, tmp_path):
    result, requests = batch(suite, tmp_path / "batch", max_tasks=len(FAMILIES))
    assert len({request["context"]["task"]["family"] for request in requests}) == len(FAMILIES)
    assert result["network_calls"] == 0
    assert result["split_counts"] == {"train": len(FAMILIES)}
    for request in requests:
        assert set(request["context"]["task"]) == {"prompt", "input", "family", "evidence_epoch"}
        assert "candidates" not in request["context"]["task"]
        assert "split" not in request and "seed" not in request
        assert request["context"]["history"] == []
    for split in ("ordinary_test", "ood_test", "sealed_poison"):
        with pytest.raises(ValueError, match="train or calibration"):
            batch(suite, tmp_path / split, split=split)


def test_task_id_filter_cannot_open_evidence_split(suite, tmp_path):
    selected = next(task for task in suite[1].values() if task["split"] == "ordinary_test")
    with pytest.raises(ValueError, match="selected development split"):
        batch(suite, tmp_path / "batch", task_ids=[selected["task_id"]])


def test_success_is_checked_and_usage_preserved(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=1)
    request = requests[0]
    imported = import_rows(suite, prepared, [response(request, suite[1][request["task_id"]])], tmp_path / "import")
    episode = read_rows(imported["rollouts_path"])[0]
    assert episode["outcome"] == {
        "status": "passed",
        "passed": True,
        "realized_visits": 1,
        "stop_reason": "verified_stop",
    }
    assert episode["backend"]["dynamics_kind"] == "black_box_refinement"
    assert episode["steps"][0]["usage"] == {"input_tokens": 40, "output_tokens": 20}
    assert imported["accepted_responses"] == 1
    exported = export_dataset(suite[0], Path(imported["rollouts_path"]), tmp_path / "dataset")
    assert exported["counts"]["sft"] == 1
    assert exported["counts"]["outcomes"] == 1
    assert read_rows(exported["sft_path"])[0]["teacher_explanations_are_claims"]


def test_failure_repair_retains_history_and_does_not_create_false_preference(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=1, max_visits=2)
    task = suite[1][requests[0]["task_id"]]
    imported = import_rows(suite, prepared, [response(requests[0], task, correct=False)], tmp_path / "round1")
    continued = continue_teacher_batch(suite[0], Path(imported["rollouts_path"]), tmp_path / "batch2")
    followup = read_rows(continued["requests_path"])[0]
    assert followup["visit"] == 1
    assert followup["context"]["history"][0]["feedback"]["status"] == "failed"
    assert "answer" not in followup["context"]["history"][0]["feedback"]
    imported2 = import_rows(suite, continued, [response(followup, task)], tmp_path / "round2")
    exported = export_dataset(suite[0], Path(imported2["rollouts_path"]), tmp_path / "dataset")
    assert exported["counts"] == {"sft": 1, "transitions": 2, "outcomes": 1, "repairs": 1, "preferences": 0}
    stopped = continue_teacher_batch(suite[0], Path(imported2["rollouts_path"]), tmp_path / "finished")
    assert stopped["request_count"] == 0


def test_preference_requires_same_visible_history(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch1", max_tasks=1)
    prepared2, requests2 = batch(suite, tmp_path / "batch2", provider="opus", max_tasks=1)
    assert requests[0]["context"] == requests2[0]["context"]
    task = suite[1][requests[0]["task_id"]]
    first = import_rows(suite, prepared, [response(requests[0], task)], tmp_path / "round1")
    second = import_rows(suite, prepared2, [response(requests2[0], task, correct=False)], tmp_path / "round2")
    combined = tmp_path / "combined.jsonl"
    write_rows(combined, read_rows(first["rollouts_path"]) + read_rows(second["rollouts_path"]))
    exported = export_dataset(suite[0], combined, tmp_path / "dataset")
    pairs = read_rows(exported["preferences_path"])
    assert len(pairs) == 1
    assert pairs[0]["chosen_verification"]["passed"]
    assert not pairs[0]["rejected_verification"]["passed"]
    assert pairs[0]["chosen_backend"]["provider"] == "codex"
    assert pairs[0]["rejected_backend"]["provider"] == "opus"
    assert pairs[0]["chosen_source_receipt"]["step_provenance"]["responses_sha256"] == first["responses_sha256"]
    assert pairs[0]["rejected_source_receipt"]["request_id"] == requests2[0]["request_id"]


def test_authority_claims_and_hidden_state_claims_are_parse_failures(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=2)
    rows = [response(request, suite[1][request["task_id"]]) for request in requests]
    rows[0]["passed"] = True
    rows[1]["hidden_states"] = [0.1, 0.2]
    imported = import_rows(suite, prepared, rows, tmp_path / "import")
    assert imported["outcome_counts"] == {"parse_failure": 2}
    assert imported["rejected_rows"] == 2
    exported = export_dataset(suite[0], Path(imported["rollouts_path"]), tmp_path / "dataset")
    assert exported["counts"]["sft"] == 0
    assert exported["counts"]["outcomes"] == 2
    assert all(row["censored"] for row in read_rows(exported["outcomes_path"]))


def test_duplicate_unknown_and_missing_responses_are_accounted(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=2)
    row = response(requests[0], suite[1][requests[0]["task_id"]])
    unknown = {**row, "request_id": "unknown"}
    imported = import_rows(suite, prepared, [row, row, unknown], tmp_path / "import")
    assert imported["accepted_responses"] == 0
    assert imported["missing_responses"] == 1
    assert imported["rejected_rows"] == 3
    assert imported["outcome_counts"] == {"missing_response": 1, "parse_failure": 1}


def test_budget_exhaustion_cannot_queue_more_calls(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=1, max_visits=1)
    task = suite[1][requests[0]["task_id"]]
    imported = import_rows(suite, prepared, [response(requests[0], task, correct=False)], tmp_path / "import")
    episode = read_rows(imported["rollouts_path"])[0]
    assert episode["outcome"]["stop_reason"] == "budget_exhausted"
    continued = continue_teacher_batch(suite[0], Path(imported["rollouts_path"]), tmp_path / "continue")
    assert continued["request_count"] == 0


def test_export_rechecks_receipts_and_frozen_split(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=1)
    imported = import_rows(
        suite, prepared, [response(requests[0], suite[1][requests[0]["task_id"]])], tmp_path / "import"
    )
    episode = read_rows(imported["rollouts_path"])[0]
    for field in ("receipt", "split"):
        altered = copy.deepcopy(episode)
        altered.pop("episode_digest")
        if field == "receipt":
            altered["steps"][0]["verification"]["candidate_digest"] = "0" * 64
        else:
            altered["split"] = "ordinary_test"
        path = tmp_path / f"altered-{field}.jsonl"
        write_rows(path, [altered])
        with pytest.raises(ValueError):
            export_dataset(suite[0], path, tmp_path / f"export-{field}")


def test_calibration_never_enters_training_export(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=1, split="calibration")
    imported = import_rows(
        suite, prepared, [response(requests[0], suite[1][requests[0]["task_id"]])], tmp_path / "import"
    )
    exported = export_dataset(suite[0], Path(imported["rollouts_path"]), tmp_path / "dataset")
    assert all(count == 0 for count in exported["counts"].values())
    assert exported["excluded"] == {"split:calibration": 1}


def test_controls_require_explicit_training_export_flag(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=1)
    imported = import_rows(
        suite, prepared, [response(requests[0], suite[1][requests[0]["task_id"]])], tmp_path / "import"
    )
    episode = read_rows(imported["rollouts_path"])[0]
    episode.pop("episode_digest")
    episode["backend"] = {"kind": "candidate_order", "dynamics_kind": "discrete_search"}
    episode["steps"][0]["producer"] = episode["backend"]
    path = tmp_path / "control.jsonl"
    write_rows(path, [episode])
    excluded = export_dataset(suite[0], path, tmp_path / "excluded")
    included = export_dataset(suite[0], path, tmp_path / "included", include_controls=True)
    assert excluded["counts"]["sft"] == 0
    assert included["counts"]["sft"] == 1
    assert read_rows(included["sft_path"])[0]["is_control"]


def test_batch_mutation_and_nonfinite_reply_are_detected(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=1)
    source = tmp_path / "responses.jsonl"
    source.write_text('{"request_id":"x","candidate":NaN}\n', encoding="utf-8")
    imported = import_teacher_responses(suite[0], Path(prepared["requests_path"]), source, tmp_path / "import")
    assert imported["rejected_rows"] == 1
    assert imported["missing_responses"] == 1
    requests[0]["context"]["task"]["prompt"] = "mutated"
    write_rows(prepared["requests_path"], requests)
    with pytest.raises(ValueError, match="requests changed"):
        import_teacher_responses(suite[0], Path(prepared["requests_path"]), source, tmp_path / "mutated")


def test_native_terminal_regression_and_abort_stay_distinct(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=1)
    request = requests[0]
    task = suite[1][request["task_id"]]
    imported = import_rows(suite, prepared, [response(request, task)], tmp_path / "import")
    episode = read_rows(imported["rollouts_path"])[0]
    episode.pop("episode_digest")
    episode["backend"] = {"kind": "frozen_random_reservoir", "dynamics_kind": "continuous_recurrent"}
    episode["steps"][0]["producer"] = episode["backend"]
    episode["config"]["verified_stop"] = False
    wrong = response(request, task, correct=False)["candidate"]
    episode["steps"].append({"visit": 1, "candidate": wrong, "verification": verify(task, wrong)})
    episode["outcome"] = {"status": "failed", "passed": False, "realized_visits": 2, "stop_reason": "terminal"}
    path = tmp_path / "regressed.jsonl"
    write_rows(path, [episode])
    exported = export_dataset(suite[0], path, tmp_path / "regressed-export", include_controls=True)
    assert exported["counts"]["sft"] == 1
    assert read_rows(exported["outcomes_path"])[0]["outcome"]["passed"] is False
    episode["steps"].pop()
    episode["outcome"] = {
        "status": "resource_abort",
        "passed": False,
        "realized_visits": 1,
        "stop_reason": "resource_budget",
    }
    write_rows(path, [episode])
    aborted = export_dataset(suite[0], path, tmp_path / "aborted-export", include_controls=True)
    assert read_rows(aborted["outcomes_path"])[0]["censored"]


def test_parse_failure_cannot_hide_fabricated_teacher_activations(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=1)
    row = response(requests[0], suite[1][requests[0]["task_id"]])
    row["passed"] = True
    imported = import_rows(suite, prepared, [row], tmp_path / "import")
    episode = read_rows(imported["rollouts_path"])[0]
    episode.pop("episode_digest")
    episode["steps"][0]["hidden_states"] = [1, 2]
    path = tmp_path / "altered.jsonl"
    write_rows(path, [episode])
    with pytest.raises(ValueError, match="hidden-state"):
        export_dataset(suite[0], path, tmp_path / "export")


def test_outcome_forecast_excludes_the_current_verifier_result(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=1)
    request = requests[0]
    task = suite[1][request["task_id"]]
    imported = import_rows(suite, prepared, [response(request, task)], tmp_path / "import")
    exported = export_dataset(suite[0], Path(imported["rollouts_path"]), tmp_path / "dataset")
    outcome = read_rows(exported["outcomes_path"])[0]
    assert outcome["prediction_point"] == "before_final_verifier"
    assert outcome["context"]["history"] == []
    assert outcome["context"]["candidate_to_evaluate"] == reference_answer(task)
    assert set(outcome["context"]) == {"task", "history", "candidate_to_evaluate"}
    assert "verification" not in outcome["context"]
    assert "outcome" not in outcome["context"]
    assert "status" not in outcome["context"]


def test_raw_response_archive_preserves_malformed_bytes_and_deep_json(suite, tmp_path):
    import hashlib

    prepared, _ = batch(suite, tmp_path / "batch", max_tasks=1)
    nested = b'{"request_id":"unknown","candidate":' + b"[" * 1500 + b"0" + b"]" * 1500 + b"}\r\n"
    raw = nested + b"invalid UTF-8 \xff\n" + b"{broken JSON}\n"
    source = tmp_path / "source.jsonl"
    source.write_bytes(raw)
    imported = import_teacher_responses(suite[0], Path(prepared["requests_path"]), source, tmp_path / "import")
    source.unlink()
    assert Path(imported["raw_responses_path"]).read_bytes() == raw
    assert imported["raw_responses_sha256"] == hashlib.sha256(raw).hexdigest()
    assert imported["raw_responses_training_eligible"] is False
    assert imported["rejected_rows"] == 3
    assert imported["missing_responses"] == 1


def test_known_nonfinite_reply_consumes_one_visit(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=1, max_visits=1)
    request = requests[0]
    source = tmp_path / "nan.jsonl"
    source.write_text(
        json.dumps(
            {"request_id": request["request_id"], "task_digest": request["task_digest"], "candidate": float("nan")}
        )
        + "\n"
    )
    imported = import_teacher_responses(suite[0], Path(prepared["requests_path"]), source, tmp_path / "import")
    assert imported["missing_responses"] == 0
    assert imported["teacher_attempts_observed"] == 1
    episode = read_rows(imported["rollouts_path"])[0]
    assert episode["outcome"]["status"] == "parse_failure"
    assert episode["outcome"]["realized_visits"] == 1
    assert episode["outcome"]["stop_reason"] == "budget_exhausted"
    continued = continue_teacher_batch(suite[0], Path(imported["rollouts_path"]), tmp_path / "continue")
    assert continued["request_count"] == 0


def test_native_diagnostics_stay_outside_prediction_context(suite, tmp_path):
    prepared, requests = batch(suite, tmp_path / "batch", max_tasks=1)
    request = requests[0]
    imported = import_rows(suite, prepared, [response(request, suite[1][request["task_id"]])], tmp_path / "import")
    episode = read_rows(imported["rollouts_path"])[0]
    episode.pop("episode_digest")
    episode["backend"] = {"kind": "frozen_random_reservoir", "dynamics_kind": "continuous_recurrent"}
    episode["steps"][0]["producer"] = episode["backend"]
    episode["steps"][0].update(
        {
            "state_summaries": {"answer": {"norm": 0.5}},
            "eta": 0.25,
            "readout_site": "carried_answer",
            "intervention": None,
            "block_calls": 1,
            "evidence_epoch": episode["evidence_epoch"],
            "segment_id": "segment0",
        }
    )
    source = tmp_path / "native.jsonl"
    write_rows(source, [episode])
    exported = export_dataset(suite[0], source, tmp_path / "dataset", include_controls=True)
    transition = read_rows(exported["transitions_path"])[0]
    assert transition["native_observation"]["state_summaries"] == {"answer": {"norm": 0.5}}
    assert transition["native_observation"]["eta"] == 0.25
    assert "native_observation" not in transition["context"]
    assert "state_summaries" not in canonical_json(transition["context"])
    outcome = read_rows(exported["outcomes_path"])[0]
    assert "native_observation" not in outcome
    assert "state_summaries" not in canonical_json(outcome["context"])


def test_response_import_has_a_hard_byte_bound(suite, tmp_path, monkeypatch):
    from datagoblin.training_gym.rlt_gym import teachers

    prepared, _ = batch(suite, tmp_path / "batch", max_tasks=1)
    source = tmp_path / "oversized.jsonl"
    source.write_bytes(b"x" * 65)
    monkeypatch.setattr(teachers, "MAX_RESPONSE_BYTES", 64)
    with pytest.raises(ValueError, match="import bound"):
        import_teacher_responses(suite[0], Path(prepared["requests_path"]), source, tmp_path / "import")
    assert not (tmp_path / "import").exists()


def native_failure(task, *, status="failed", kind="frozen_random_reservoir"):
    wrong = next(candidate for candidate in task["candidates"] if not verify(task, candidate)["passed"])
    return {
        "schema_version": "rlt_gym.rollout.v1",
        "episode_id": "native-" + task["task_id"],
        **{
            key: task[key]
            for key in ("task_id", "task_digest", "family", "generator_id", "lineage_id", "split", "evidence_epoch")
        },
        "backend": {
            "kind": kind,
            "dynamics_kind": "continuous_recurrent",
            "provider": "local_cpu",
            "model": "native-test-control",
        },
        "config": {"planned_depth": 1, "mode": "direct", "readout_site": "carried_answer", "seed": 7},
        "steps": [
            {
                "visit": 0,
                "candidate": wrong,
                "verification": verify(task, wrong),
                "state_summaries": {"answer": {"norm": 0.75}},
                "eta": 1.0,
            }
        ],
        "outcome": {"status": status, "passed": False, "realized_visits": 1, "stop_reason": "visit_budget"},
    }


def test_native_failure_to_teacher_repair_has_per_step_provenance(suite, tmp_path):
    task = next(task for task in suite[1].values() if task["split"] == "train")
    source = tmp_path / "native.jsonl"
    native = native_failure(task)
    write_rows(source, [native])
    prepared = prepare_teacher_repairs(
        suite[0], source, tmp_path / "batch", provider="codex", model="teacher-test", max_tasks=1, max_visits=3
    )
    request = read_rows(prepared["requests_path"])[0]
    context_episode = read_rows(tmp_path / "batch" / "contexts.jsonl")[0]
    assert request["visit"] == 1
    assert request["context"]["history"][0]["producer"]["kind"] == "frozen_random_reservoir"
    assert "state_summaries" not in canonical_json(request)
    assert "state_summaries" not in context_episode["steps"][0]
    assert context_episode["origin"]["source_task_digest"] == task["task_digest"]
    assert context_episode["origin"]["source_episode_id"] == native["episode_id"]
    assert prepared["selection"]["maximum_teacher_replies_per_episode"] == 2
    imported = import_rows(suite, prepared, [response(request, task)], tmp_path / "import")
    mixed = read_rows(imported["rollouts_path"])[0]
    assert mixed["steps"][0]["producer"]["kind"] == "frozen_random_reservoir"
    assert mixed["steps"][1]["producer"]["kind"] == "teacher"
    assert mixed["steps"][1]["producer"]["model"] == "teacher-test"
    exported = export_dataset(suite[0], Path(imported["rollouts_path"]), tmp_path / "dataset")
    assert exported["counts"] == {"sft": 1, "transitions": 1, "outcomes": 1, "repairs": 1, "preferences": 0}
    transition = read_rows(exported["transitions_path"])[0]
    assert transition["backend"]["kind"] == "teacher"
    assert transition["visit"] == 1
    assert transition["context"]["history"][0]["candidate"] == native["steps"][0]["candidate"]
    assert transition["is_control"] is False
    assert "native_observation" not in transition
    assert exported["excluded"] == {"control_step:frozen_random_reservoir": 1}
    included = export_dataset(
        suite[0], Path(imported["rollouts_path"]), tmp_path / "with-controls", include_controls=True
    )
    rows = read_rows(included["transitions_path"])
    assert len(rows) == 2
    assert rows[0]["producer"]["kind"] == "frozen_random_reservoir" and rows[0]["is_control"]
    assert rows[0]["context_only"]
    assert rows[0]["step_provenance"]["source_task_digest"] == task["task_digest"]


def test_repair_queue_excludes_evidence_censored_and_oracle_sources(suite, tmp_path):
    train = [task for task in suite[1].values() if task["split"] == "train"]
    heldout = next(task for task in suite[1].values() if task["split"] == "ordinary_test")
    source = tmp_path / "native.jsonl"
    write_rows(
        source,
        [
            native_failure(train[0]),
            native_failure(train[1], status="resource_abort"),
            native_failure(train[2], kind="reference_solver"),
            native_failure(heldout),
        ],
    )
    prepared = prepare_teacher_repairs(suite[0], source, tmp_path / "batch", provider="opus", model="repair-model")
    requests = read_rows(prepared["requests_path"])
    assert len(requests) == 1 and requests[0]["task_id"] == train[0]["task_id"]
    assert prepared["selection"]["excluded"] == {
        "outcome:resource_abort": 1,
        "source_backend": 1,
        "split:ordinary_test": 1,
    }
    assert prepared["split_counts"] == {"train": 1}
    with pytest.raises(ValueError, match="max_visits"):
        prepare_teacher_repairs(suite[0], source, tmp_path / "bad-depth", provider="codex", model="m", max_visits=1)


def test_repair_queue_balances_families_and_deduplicates_task(suite, tmp_path):
    train = [task for task in suite[1].values() if task["split"] == "train"]
    sources = [native_failure(task) for task in train]
    source = tmp_path / "native.jsonl"
    write_rows(source, [*sources, sources[0]])
    prepared = prepare_teacher_repairs(
        suite[0], source, tmp_path / "batch", provider="codex", model="m", max_tasks=len(FAMILIES)
    )
    requests = read_rows(prepared["requests_path"])
    assert len(requests) == len(FAMILIES)
    assert len({request["context"]["task"]["family"] for request in requests}) == len(FAMILIES)
    assert prepared["selection"]["excluded"] == {"duplicate_task": 1}
