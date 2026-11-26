"""Complete-denominator rollout checks and evidence-compatible comparisons."""

import copy
import hashlib
import json
import sys
from collections import Counter
from types import ModuleType

import pytest

from datagoblin.research_v2 import environment, learners, runner, suite


def _write_report(out, cases, scores, candidate="candidate-a", role="checkpoint"):
    out.mkdir()
    rows = []
    for case, score in zip(cases, scores, strict=True):
        row = {
            "schema": suite.VERSION,
            "task_id": case["id"],
            "root_id": case["root_id"],
            "family": case["family"],
            "split": case["split"],
            "primary_score": score,
            "success": score >= 0.75,
            "integrity_pass": True,
            "completed": True,
            "closed": True,
            "metrics": {},
            "cost": {"errors": 0},
        }
        row["result_sha256"] = suite.digest(row)
        rows.append(row)
    metadata = {
        "candidate_id": candidate,
        "role": role,
        "model_evaluated": role == "checkpoint",
        "suite_sha256": "a" * 64,
        "comparison": copy.deepcopy(runner.COMPARISON),
        "comparison_sha256": suite.digest(runner.COMPARISON),
        "checkpoint_rank_eligible": role == "checkpoint",
    }
    if role == "checkpoint":
        identity = {
            "registry_entry": {"id": candidate, "model_id": "test/" + candidate, "revision": "b" * 40},
            "binding": {"registry_id": candidate},
        }
        metadata.update(identity=identity, identity_sha256=suite.digest(identity))
    runner._finish(out, rows, metadata)
    return out


def _mutate_report(path, field, value):
    file = path / "report.json"
    report = json.loads(file.read_text())
    report[field] = value
    report["report_sha256"] = suite.digest({k: v for k, v in report.items() if k != "report_sha256"})
    file.write_text(suite.canon(report) + "\n")


def _mutate_rows(path, mutate, *, repair_row_hash=True):
    file = path / "results.jsonl"
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    mutate(rows)
    if repair_row_hash:
        for row in rows:
            row["result_sha256"] = suite.digest({k: v for k, v in row.items() if k != "result_sha256"})
    file.write_text("".join(suite.canon(row) + "\n" for row in rows))
    _mutate_report(path, "results_sha256", hashlib.sha256(file.read_bytes()).hexdigest())


def _endpoint_fixture(tmp_path, monkeypatch, request_action, *, max_requests=64, wall_seconds=900):
    # Transport identity and HTTP parsing have their own tests. Here an explicit
    # fake transport controls failure timing while the real runner handles costs.
    transport = ModuleType("datagoblin.research_v2.transport")
    transport.request_action = request_action
    transport.validate_binding = lambda _entry, binding: binding
    monkeypatch.setitem(sys.modules, "datagoblin.research_v2.transport", transport)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        suite.canon(
            {
                "schema": "datagoblin-research-v2-model-registry-1",
                "models": [{"id": "test-model", "model_id": "test/model", "revision": "b" * 40}],
            }
        )
    )
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(
        suite.canon(
            {
                "resource_caps": {"max_requests": max_requests, "wall_seconds": wall_seconds},
                "served_model": "test/model",
                "registry_id": "test-model",
            }
        )
    )
    suite_path = tmp_path / "suite"
    suite.write_suite(suite_path, seed=17, per_family=1)
    return {
        "suite_path": suite_path,
        "model_id": "test-model",
        "binding_path": binding_path,
        "registry_path": registry_path,
    }


def test_family_balancing_is_independent_of_source_order_and_keeps_complete_rounds():
    cases = suite.make_suite(seed=17, per_family=3)
    selected = runner.select_cases(list(reversed(cases)), 16)
    assert len(selected) == 16
    assert Counter(row["family"] for row in selected) == dict.fromkeys(suite.FAMILIES, 2)
    assert [row["family"] for row in selected[:8]] == list(suite.FAMILIES)
    assert [row["family"] for row in selected[8:]] == list(suite.FAMILIES)
    assert len({row["id"] for row in selected}) == 16
    scarce = [case for case in cases if case["family"] != "trm_training"] + [cases[0]]
    assert len(runner.select_cases(scarce, 24)) == 8
    with pytest.raises(ValueError, match="every"):
        runner.select_cases([case for case in cases if case["family"] != "trm_training"], 8)


@pytest.mark.parametrize("maximum", [0, 7, 9, 129, True, 8.0])
def test_nonbalanced_episode_caps_are_rejected(maximum):
    with pytest.raises(ValueError):
        runner.select_cases(suite.make_suite(), maximum)


def test_aggregate_keeps_failed_and_unfinished_cases_in_the_denominator():
    rows = []
    for family in suite.FAMILIES:
        for completed, score in [(True, 1), (False, 0)]:
            rows.append(
                {
                    "family": family,
                    "completed": completed,
                    "primary_score": score,
                    "success": completed,
                    "integrity_pass": completed,
                    "cost": {"errors": int(not completed)},
                }
            )
    result = runner.aggregate(rows)
    assert result["cases"] == 16
    assert result["macro_score"] == 0.5
    assert result["success_rate"] == 0.5
    assert all(group["cases"] == 2 and group["completed"] == 1 for group in result["families"].values())
    assert all(group["format_or_tool_errors"] == 1 for group in result["families"].values())
    for bad in [True, float("nan"), float("inf"), -0.1, 1.1]:
        with pytest.raises(ValueError):
            runner.aggregate([{**rows[0], "primary_score": bad}])


@pytest.mark.parametrize(
    "field,value",
    [
        ("suite_sha256", "f" * 64),
        ("implementation_sha256", "f" * 64),
        ("comparison_sha256", "f" * 64),
        ("role", "control"),
    ],
)
def test_comparison_rejects_changed_suite_code_budget_or_role(tmp_path, field, value):
    cases = runner.select_cases(suite.make_suite(), 8)
    first = _write_report(tmp_path / "first", cases, [0] * 8)
    second = _write_report(tmp_path / "second", cases, [1] * 8, candidate="candidate-b")
    _mutate_report(second, field, value)
    with pytest.raises(ValueError, match=r"identical|comparison contract"):
        runner.compare([first, second])


def test_comparison_rejects_different_tasks_even_with_identical_suite_label(tmp_path):
    first_cases = runner.select_cases(suite.make_suite(seed=17), 8)
    second_cases = runner.select_cases(suite.make_suite(seed=18), 8)
    first = _write_report(tmp_path / "first", first_cases, [0] * 8)
    second = _write_report(tmp_path / "second", second_cases, [1] * 8, candidate="candidate-b")
    with pytest.raises(ValueError, match="identical"):
        runner.compare([first, second])


def test_comparison_rejects_changed_paired_root(tmp_path):
    cases = runner.select_cases(suite.make_suite(), 8)
    first = _write_report(tmp_path / "first", cases, [0] * 8)
    second = _write_report(tmp_path / "second", cases, [1] * 8, candidate="candidate-b")
    _mutate_rows(second, lambda rows: rows[0].update(root_id="f" * 64))
    with pytest.raises(ValueError, match="root"):
        runner.compare([first, second])


def test_report_hash_result_hash_row_hash_and_aggregate_are_all_checked(tmp_path):
    cases = runner.select_cases(suite.make_suite(), 8)
    report_changed = _write_report(tmp_path / "report_changed", cases, [0] * 8)
    file = report_changed / "report.json"
    report = json.loads(file.read_text())
    report["candidate_id"] = "unbound-substitution"
    file.write_text(suite.canon(report))
    with pytest.raises(ValueError, match="report digest"):
        runner.read_report(report_changed)

    bytes_changed = _write_report(tmp_path / "bytes_changed", cases, [0] * 8)
    with (bytes_changed / "results.jsonl").open("a") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="bytes"):
        runner.read_report(bytes_changed)

    row_changed = _write_report(tmp_path / "row_changed", cases, [0] * 8)
    _mutate_rows(row_changed, lambda rows: rows[0].update(primary_score=1), repair_row_hash=False)
    with pytest.raises(ValueError, match="row digest"):
        runner.read_report(row_changed)

    aggregate_changed = _write_report(tmp_path / "aggregate_changed", cases, [0] * 8)
    report, _ = runner.read_report(aggregate_changed)
    report["summary"]["macro_score"] = 1
    _mutate_report(aggregate_changed, "summary", report["summary"])
    with pytest.raises(ValueError, match="aggregate"):
        runner.read_report(aggregate_changed)


def test_missing_case_cannot_be_silently_removed_from_a_bound_report(tmp_path):
    cases = runner.select_cases(suite.make_suite(), 8)
    path = _write_report(tmp_path / "missing", cases, [0] * 8)
    _mutate_rows(path, lambda rows: rows.pop())
    with pytest.raises(ValueError, match="membership"):
        runner.read_report(path)


def test_smoke_interval_is_descriptive_and_controls_never_mix_with_checkpoints(tmp_path):
    cases = runner.select_cases(suite.make_suite(), 8)
    baseline = _write_report(tmp_path / "control", cases, [0] * 8, candidate="control/noop", role="control")
    reference = _write_report(tmp_path / "reference", cases, [1] * 8, candidate="control/reference", role="control")
    report = runner.compare([baseline, reference])
    assert report["paired"][0]["interval_status"] == "descriptive_only"
    assert report["paired"][0]["complete_denominator"] == 8
    assert report["paired"][0]["rank_eligible"] is False
    assert report["winner"] is None
    checkpoint = _write_report(tmp_path / "checkpoint", cases, [1] * 8, candidate="candidate-b")
    with pytest.raises(ValueError, match="role"):
        runner.compare([baseline, checkpoint])


def test_paired_bootstrap_preserves_family_balance_and_has_nontrivial_uncertainty(tmp_path):
    cases = runner.select_cases(suite.make_suite(seed=17, per_family=4), 32)
    first = _write_report(tmp_path / "first", cases, [0] * 32)
    scores = [0] * 8 + [1] * 8 + [0] * 8 + [1] * 8
    second = _write_report(tmp_path / "second", cases, scores, candidate="candidate-b")
    paired = runner.compare([first, second])["paired"][0]
    assert paired["macro_delta"] == 0.5
    assert paired["family_deltas"] == dict.fromkeys(suite.FAMILIES, 0.5)
    low, high = paired["paired_root_bootstrap_95"]
    assert 0 < low < 0.5 < high < 1
    assert paired["interval_status"] == "paired_within_family"
    assert paired["complete_denominator"] == 32
    assert runner.compare([first, second])["paired"][0] == paired


def test_duplicate_checkpoint_identity_cannot_be_ranked_against_itself(tmp_path):
    cases = runner.select_cases(suite.make_suite(), 8)
    first = _write_report(tmp_path / "first", cases, [0] * 8)
    second = _write_report(tmp_path / "second", cases, [1] * 8)
    with pytest.raises(ValueError, match="duplicate candidate"):
        runner.compare([first, second])


def test_endpoint_failures_keep_all_requested_cases_with_zero_scores(tmp_path, monkeypatch):
    calls = []

    def failing_request(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("stub_endpoint_failure")

    settings = _endpoint_fixture(tmp_path, monkeypatch, failing_request)
    out = tmp_path / "run"
    report = runner.run_endpoint(**settings, out=out)
    checked, rows = runner.read_report(out)
    assert report == checked
    assert len(calls) == report["attempted_actions"] == report["request_failures"] == 8
    assert report["inference_requests"] == 0
    assert report["model_evaluated"] is False
    assert len(rows) == report["summary"]["cases"] == 8
    assert Counter(row["family"] for row in rows) == dict.fromkeys(suite.FAMILIES, 1)
    assert all(row["primary_score"] == 0 and not row["completed"] for row in rows)
    assert report["checkpoint_rank_eligible"] is False


@pytest.mark.parametrize("known", [True, False])
def test_failed_generation_retains_reported_or_explicitly_unknown_usage(tmp_path, monkeypatch, known):
    class FailedGeneration(RuntimeError):
        def __init__(self):
            super().__init__("fixture_failure")
            self.receipt = {"http_requests": 2, "inference_requests": 1}
            if known:
                self.receipt["usage"] = {"prompt_tokens": 100, "completion_tokens": 512, "total_tokens": 612}

    def failed_request(*_args, **_kwargs):
        raise FailedGeneration()

    settings = _endpoint_fixture(tmp_path, monkeypatch, failed_request, max_requests=2)
    report = runner.run_endpoint(**settings, out=tmp_path / "run")
    assert report["inference_requests"] == 1
    assert report["http_requests"] == 2
    assert report["max_completion_tokens_reserved"] == 512
    assert report["inference_requests_with_unknown_usage"] == int(not known)
    assert report["usage"] == {"prompt_tokens": 100 if known else 0, "completion_tokens": 512 if known else 0}
    assert report["model_evaluated"] is False
    assert report["summary"]["cases"] == 8


def test_owned_weight_artifacts_are_saved_and_bound_to_the_run(tmp_path, monkeypatch):
    def local_worker(request, **_kwargs):
        artifact = learners.fit(request["kind"], request["seed"], request["config"])
        return {
            "artifact": artifact,
            "calibration": learners.evaluate(artifact, request["calibration_seed"], "calibration"),
        }

    monkeypatch.setattr(environment, "run_worker", local_worker)
    case = suite.make_suite()[0]
    session = environment.ResearchSession(case)
    session.step_json('{"action":"run_experiment","config":{"steps":1}}')
    session.save_artifacts(tmp_path / "artifacts")
    runner._finish(
        tmp_path,
        [session.result()],
        {
            "candidate_id": "control/artifact",
            "role": "control",
            "model_evaluated": False,
            "suite_sha256": "a" * 64,
            "comparison": runner.COMPARISON,
            "comparison_sha256": suite.digest(runner.COMPARISON),
            "checkpoint_rank_eligible": False,
        },
    )
    report, rows = runner.read_report(tmp_path)
    assert len(report["artifact_files"]) == len(rows[0]["artifact_hashes"]) == 1
    artifact_path = tmp_path / next(iter(report["artifact_files"]))
    artifact_path.write_text(artifact_path.read_text() + " ")
    with pytest.raises(ValueError, match="artifact file digest"):
        runner.read_report(tmp_path)


def test_request_cap_keeps_unattempted_cases_and_never_calls_beyond_cap(tmp_path, monkeypatch):
    calls = []

    def endless_inspect(*args, **kwargs):
        calls.append((args, kwargs))
        return {"text": '{"action":"inspect"}', "usage": {"prompt_tokens": 10, "completion_tokens": 3}}

    settings = _endpoint_fixture(tmp_path, monkeypatch, endless_inspect, max_requests=4)
    out = tmp_path / "run"
    report = runner.run_endpoint(**settings, out=out)
    _, rows = runner.read_report(out)
    assert len(calls) == report["inference_requests"] == 2
    assert report["http_requests"] == 4
    assert report["usage"] == {"prompt_tokens": 20, "completion_tokens": 6}
    assert len(rows) == 8
    assert all(row["primary_score"] == 0 for row in rows)
    assert all(row["rollout_stop_reason"] == "run_budget_exhausted" for row in rows)
    assert [row["cost"]["turns"] for row in rows] == [2, 0, 0, 0, 0, 0, 0, 0]
    assert report["checkpoint_rank_eligible"] is False


def test_endpoint_can_write_only_to_a_fresh_output(tmp_path, monkeypatch):
    def never_requested(*_args, **_kwargs):
        raise AssertionError("existing output triggered inference")

    settings = _endpoint_fixture(tmp_path, monkeypatch, never_requested)
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        runner.run_endpoint(**settings, out=out)


def test_control_failure_retains_every_case_without_model_claims(tmp_path, monkeypatch):
    def broken_reference(_case):
        raise RuntimeError("fixture_worker_failed")

    monkeypatch.setattr(runner, "reference_actions", broken_reference)
    suite_path = tmp_path / "suite"
    suite.write_suite(suite_path, seed=17, per_family=1)
    out = tmp_path / "control"
    report = runner.run_control(suite_path, out)
    checked, rows = runner.read_report(out)
    assert checked == report
    assert len(rows) == 8
    assert all(row["primary_score"] == 0 and row["closed"] and not row["completed"] for row in rows)
    assert report["role"] == "control"
    assert report["model_evaluated"] is False
    assert report["inference_requests"] == 0
    assert report["checkpoint_rank_eligible"] is False


def test_global_deadline_is_passed_to_environment_workers(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])

    def proposing_request(*_args, **_kwargs):
        clock[0] += 2
        return {
            "text": '{"action":"run_experiment","config":{"steps":0}}',
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }

    settings = _endpoint_fixture(tmp_path, monkeypatch, proposing_request, wall_seconds=3)
    calls = []

    def slow_worker(request, *, timeout_seconds):
        calls.append(timeout_seconds)
        artifact = learners.fit(request["kind"], request["seed"], request["config"])
        calibration = learners.evaluate(artifact, request["calibration_seed"], "calibration")
        clock[0] += 2
        return {"artifact": artifact, "calibration": calibration}

    monkeypatch.setattr(environment, "run_worker", slow_worker)
    out = tmp_path / "run"
    report = runner.run_endpoint(**settings, out=out)
    _, rows = runner.read_report(out)
    assert calls == [1.0]
    assert report["inference_requests"] == 1
    assert len(rows) == 8
    assert all(row["primary_score"] == 0 for row in rows)


def test_compare_rejects_different_effective_endpoint_budgets(tmp_path, monkeypatch):
    def failed_request(*_args, **_kwargs):
        raise RuntimeError("stub_endpoint_failure")

    settings = _endpoint_fixture(tmp_path, monkeypatch, failed_request, max_requests=2)
    first = tmp_path / "first"
    runner.run_endpoint(**settings, out=first)
    binding = json.loads(settings["binding_path"].read_text())
    binding["resource_caps"]["max_requests"] = 8
    binding["registry_id"] = "another-checkpoint"
    settings["binding_path"].write_text(suite.canon(binding))
    registry = json.loads(settings["registry_path"].read_text())
    registry["models"].append({"id": "another-checkpoint", "model_id": "test/another", "revision": "c" * 40})
    settings["registry_path"].write_text(suite.canon(registry))
    settings["model_id"] = "another-checkpoint"
    second = tmp_path / "second"
    runner.run_endpoint(**settings, out=second)
    with pytest.raises(ValueError, match=r"identical|budget"):
        runner.compare([first, second])


def test_repeated_roots_cannot_supply_fake_bootstrap_replications(tmp_path):
    cases = runner.select_cases(suite.make_suite(seed=17, per_family=2), 16)
    path = _write_report(tmp_path / "duplicated", cases, [0] * 16)
    _mutate_rows(path, lambda rows: rows[8].update(root_id=rows[0]["root_id"]))
    with pytest.raises(ValueError, match="root"):
        runner.read_report(path)


def test_checkpoint_name_cannot_be_detached_from_attested_identity(tmp_path):
    cases = runner.select_cases(suite.make_suite(), 8)
    name_changed = _write_report(tmp_path / "name_changed", cases, [0] * 8)
    _mutate_report(name_changed, "candidate_id", "renamed-checkpoint")
    with pytest.raises(ValueError, match="identity"):
        runner.read_report(name_changed)

    identity_changed = _write_report(tmp_path / "identity_changed", cases, [0] * 8)
    report, _ = runner.read_report(identity_changed)
    report["identity"]["registry_entry"]["revision"] = "f" * 40
    _mutate_report(identity_changed, "identity", report["identity"])
    with pytest.raises(ValueError, match="identity"):
        runner.read_report(identity_changed)

    binding_changed = _write_report(tmp_path / "binding_changed", cases, [0] * 8)
    report, _ = runner.read_report(binding_changed)
    report["identity"]["binding"]["registry_id"] = "different-binding"
    _mutate_report(binding_changed, "identity", report["identity"])
    _mutate_report(binding_changed, "identity_sha256", suite.digest(report["identity"]))
    with pytest.raises(ValueError, match="identity"):
        runner.read_report(binding_changed)


def test_wrapped_core_receipt_is_verified_after_transport_layer_rehash(tmp_path, monkeypatch):
    def failed_request(*_args, **_kwargs):
        raise RuntimeError("stub_endpoint_failure")

    settings = _endpoint_fixture(tmp_path, monkeypatch, failed_request)
    out = tmp_path / "run"
    runner.run_endpoint(**settings, out=out)
    _mutate_rows(out, lambda rows: rows[0]["metrics"].update(unexecuted_claim=True))
    with pytest.raises(ValueError, match="core result"):
        runner.read_report(out)


def _future_entry():
    return {
        "id": "baby-goblin-future-test",
        "model_id": "AlephFunk/future-checkpoint-fixture",
        "weights_sha256": "b" * 64,
        "training_index_sha256": "c" * 64,
        "training_positions": 20000000,
        "parameters": 999352320,
        "context_length": 2048,
        "precision": "fp32-native",
        "protocol": "raw-completion",
        "template": "raw task completion",
        "recurrence": {"core_passes": 2},
    }


def _registration_files(tmp_path, entry):
    registry_path = tmp_path / "source.json"
    registry_path.write_text(
        suite.canon(
            {
                "schema": "datagoblin-research-v2-model-registry-1",
                "models": [{"id": "existing-checkpoint", "model_id": "test/existing", "weights_sha256": "a" * 64}],
            }
        )
    )
    entry_path = tmp_path / "entry.json"
    entry_path.write_text(suite.canon(entry))
    return entry_path, registry_path


def test_future_checkpoint_registration_preserves_old_identity_and_requires_separate_binding(tmp_path):
    entry = _future_entry()
    entry_path, registry_path = _registration_files(tmp_path, entry)
    original = registry_path.read_bytes()
    out = tmp_path / "future.json"
    receipt = runner.register_checkpoint(entry_path, registry_path, out)
    assert receipt["registered"] == entry["id"]
    assert registry_path.read_bytes() == original
    values = runner.registry(out)
    assert values["existing-checkpoint"]["weights_sha256"] == "a" * 64
    assert values[entry["id"]]["weights_sha256"] == entry["weights_sha256"]
    assert values[entry["id"]]["training_positions"] == 20000000
    assert values[entry["id"]]["endpoint_status"] == "unbound"
    assert receipt["registry_sha256"] == suite.digest(json.loads(out.read_text()))


def test_future_checkpoint_cannot_reassign_an_existing_name_or_overwrite_output(tmp_path):
    entry = _future_entry()
    entry["id"] = "existing-checkpoint"
    entry_path, registry_path = _registration_files(tmp_path, entry)
    out = tmp_path / "future.json"
    with pytest.raises(ValueError, match="immutable"):
        runner.register_checkpoint(entry_path, registry_path, out)
    assert not out.exists()
    entry_path.write_text(suite.canon(_future_entry()))
    out.write_text("retained output")
    with pytest.raises(ValueError, match="fresh"):
        runner.register_checkpoint(entry_path, registry_path, out)
    assert out.read_text() == "retained output"


@pytest.mark.parametrize(
    "field,value",
    [
        ("weights_sha256", "unknown"),
        ("training_index_sha256", "f" * 63),
        ("training_positions", -1),
        ("training_positions", True),
        ("parameters", 0),
        ("context_length", 1.5),
        ("protocol", "execute-arbitrary-code"),
    ],
)
def test_future_checkpoint_rejects_unbound_hashes_and_invalid_counts(tmp_path, field, value):
    entry = _future_entry()
    entry[field] = value
    entry_path, registry_path = _registration_files(tmp_path, entry)
    out = tmp_path / "future.json"
    with pytest.raises(ValueError):
        runner.register_checkpoint(entry_path, registry_path, out)
    assert not out.exists()
