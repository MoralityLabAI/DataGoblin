"""Adversarial protocol, custody, paired-artifact and bounded-worker checks."""

import json

import pytest

from datagoblin.research_v2 import environment, learners, suite


def _case(family):
    return next(case for case in suite.make_suite() if case["family"] == family)


def _dispatch(session, action):
    return session.step_json(suite.canon(action))


def _inline_worker(request, **_kwargs):
    # Unit tests replace process transport only. Learners and grading stay real.
    if request["op"] == "fit":
        artifact = learners.fit(request["kind"], request["seed"], request["config"])
        return {
            "artifact": artifact,
            "calibration": learners.evaluate(artifact, request["calibration_seed"], "calibration"),
        }
    return learners.evaluate(request["artifact"], request["seed"], request["split"])


def _fit_pair(session, config=None):
    config = config or {"steps": 32}
    assert _dispatch(session, {"action": "run_experiment", "config": {**config, "steps": 0}})["created"] == "a1"
    assert _dispatch(session, {"action": "run_experiment", "config": config})["created"] == "a2"
    rows = session.artifact_summaries()
    before, after = rows[0]["calibration_accuracy"], rows[1]["calibration_accuracy"]
    return {
        "artifact_id": "a2", "baseline_id": "a1",
        "claim": "ties" if after == before else "improves" if after > before else "worse",
    }


@pytest.mark.parametrize("raw", [
    '{"action":"inspect","action":"submit"}',
    '{"action":"submit","answer":{"x":0,"x":1}}',
    '{"action":"run_experiment","config":{"learning_rate":NaN}}',
    '{"action":"run_experiment","config":{"learning_rate":Infinity}}',
    '{"action":"run_experiment","config":{"learning_rate":1e10000}}',
    "[]", "null", "true", "1", "invalid-json",
    '{"action":"inspect","claimed_score":1}',
    '{"action":["inspect"]}',
    '{"action":"run_experiment","config":{"steps":true}}',
])
def test_malformed_or_claimed_actions_spend_a_turn_without_scoring(raw):
    session = environment.ResearchSession(_case("ldt_filter"))
    response = session.step_json(raw)
    assert "error" in response
    assert response["remaining"]["turns"] == environment.LIMITS["turns"] - 1
    report = session.result()
    assert report["primary_score"] == 0
    assert report["cost"]["experiments"] == 0
    assert report["cost"]["optimizer_updates_reserved"] == 0


def test_response_byte_limit_and_turn_limit_close_without_training():
    session = environment.ResearchSession(_case("ldt_filter"))
    for _ in range(environment.LIMITS["turns"]):
        response = session.step_json("x" * (environment.LIMITS["response_bytes"] + 1))
        assert "error" in response
    assert response["done"] is True
    closed = session.result()
    assert closed["primary_score"] == 0
    assert _dispatch(session, {"action": "inspect"})["status"] == "episode_closed"
    assert session.result() == closed


@pytest.mark.parametrize("family", [f for f in suite.FAMILIES if f not in suite.LEARNERS])
def test_exact_finite_controls_verify_and_completed_results_are_immutable(family):
    case = _case(family)
    session = environment.ResearchSession(case)
    reply = _dispatch(session, {"action": "submit", "answer": suite.reference_answer(case)})
    assert reply["status"] == "submitted"
    report = session.result()
    assert report["success"] is True
    assert report["primary_score"] == 1
    assert suite.digest({k: v for k, v in report.items() if k != "result_sha256"}) == report["result_sha256"]
    assert _dispatch(session, {"action": "submit", "answer": {}})["status"] == "episode_closed"
    assert session.result() == report


def test_boolean_witness_requires_integer_and_equivalence_requires_boolean():
    case = _case("boolean_repair")
    case["public"] = {"bits": 4, "reference": [[1, 0]], "proposal": [[1, 1]]}
    assert suite.check_answer(case, {"equivalent": False, "witness": 0})
    assert not suite.check_answer(case, {"equivalent": False, "witness": False})
    assert not suite.check_answer(case, {"equivalent": 0, "witness": 0})
    assert not suite.check_answer(case, {"equivalent": "false", "witness": 0})
    assert not suite.check_answer(case, {"equivalent": False, "witness": 16})
    assert not suite.check_answer(case, {"equivalent": False, "witness": 0, "score": 1})


def test_nontrain_source_row_holds_all_descendants_and_every_record_is_accounted_for():
    case = _case("evidence_custody")
    template = {
        "split": "train", "rights": "internal_training", "authority": "environment", "receipt_bound": True,
    }
    case["public"] = {"held_roots": ["reserved"], "records": [
        {**template, "id": "clean", "root": "fresh"},
        {**template, "id": "derived_train", "root": "source_eval"},
        {**template, "id": "source", "root": "source_eval", "split": "test"},
        {**template, "id": "held_descendant", "root": "reserved"},
    ]}
    expected = {"admit": ["clean"], "quarantine": ["derived_train", "held_descendant", "source"]}
    assert suite.reference_answer(case) == expected
    assert suite.check_answer(case, expected)
    assert not suite.check_answer(case, {"admit": ["clean", "derived_train"], "quarantine": ["source"]})
    assert not suite.check_answer(case, {"admit": ["clean", "clean"], "quarantine": expected["quarantine"]})
    assert not suite.check_answer(case, {"admit": ["clean"], "quarantine": expected["quarantine"][:-1]})


def test_finite_checks_are_limited_and_learned_cases_have_no_test_query_tool():
    case = _case("cursor_recovery")
    session = environment.ResearchSession(case)
    request = {"action": "check", "answer": suite.reference_answer(case)}
    assert _dispatch(session, request)["check_passed"] is True
    assert _dispatch(session, request)["check_passed"] is True
    assert "error" in _dispatch(session, request)
    learned = environment.ResearchSession(_case("ldt_filter"))
    assert "error" in _dispatch(learned, {"action": "check", "answer": {}})
    assert learned.result()["cost"]["checks"] == 0


def test_public_packet_and_experiment_observations_exclude_private_evaluation_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, "run_worker", _inline_worker)
    out = tmp_path / "suite"
    suite.write_suite(out, seed=123456789, per_family=1)
    cases = suite.load_suite(out)
    public_text = (out / "tasks.public.jsonl").read_text()
    for case in cases:
        assert '"private"' not in public_text
        for name in ("calibration_seed", "evaluation_seed", "ood_seed"):
            assert name not in public_text
            assert str(case["private"][name]) not in public_text
    session = environment.ResearchSession(next(case for case in cases if case["family"] == "ldt_filter"))
    observed = _dispatch(session, {"action": "run_experiment", "config": {"steps": 0}})
    serialized = suite.canon(observed)
    assert "evaluation" not in serialized
    assert "weights" not in serialized
    assert "private" not in serialized
    for value in session.case["private"].values():
        assert str(value) not in serialized


def test_copied_case_and_checked_manifest_reject_mutation(tmp_path):
    original = _case("cursor_recovery")
    session = environment.ResearchSession(original)
    original["public"]["total_positions"] = 1
    assert session.case["public"]["total_positions"] == 10000000
    with pytest.raises(ValueError, match="changed"):
        environment.ResearchSession(original)
    out = tmp_path / "suite"
    suite.write_suite(out, seed=17, per_family=1)
    with (out / "tasks.public.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="changed"):
        suite.load_suite(out)


def test_held_suites_stay_outside_git_and_registered_roots_are_excluded(tmp_path):
    repository = tmp_path / "repo"
    (repository / ".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="outside"):
        suite.write_suite(repository / "held", seed=17, per_family=1, split="heldout")
    exclusions = {case["root_id"] for case in suite.make_suite(seed=17, per_family=1)}
    with pytest.raises(ValueError, match="overlap"):
        suite.write_suite(tmp_path / "held", seed=17, per_family=1, split="heldout", exclusions=exclusions)
    held = suite.make_suite(seed=17, split="heldout")
    with pytest.raises(ValueError, match="development"):
        suite.reference_actions(held[0])


def test_injected_artifacts_and_claimed_measurements_cannot_enter_experiment_store():
    session = environment.ResearchSession(_case("ldt_filter"))
    fabricated = {"artifact_id": "uploaded-sha", "baseline_id": "baseline-sha", "claim": "improves"}
    assert "error" in _dispatch(session, {"action": "submit", "answer": fabricated})
    assert session.result()["primary_score"] == 0
    assert session.result()["closed"] is True
    assert session.result()["artifact_hashes"] == []
    another = environment.ResearchSession(_case("ldt_filter"))
    forged = {**fabricated, "metrics": {"accuracy": 1.0}, "weights": [[1.0]]}
    assert "error" in _dispatch(another, {"action": "submit", "answer": forged})
    assert another.result()["completed"] is False


def test_failed_worker_keeps_reserved_budget_and_cannot_be_retried_for_free(monkeypatch):
    requests = []

    def failing_worker(request, **_kwargs):
        requests.append(request)
        raise RuntimeError("worker_timeout")

    monkeypatch.setattr(environment, "run_worker", failing_worker)
    session = environment.ResearchSession(_case("ldt_filter"))
    for _ in range(2):
        assert "worker_timeout" in _dispatch(session, {"action": "run_experiment", "config": {"steps": 128}})["error"]
    report = session.result()
    assert report["cost"]["experiments"] == 2
    assert report["cost"]["optimizer_updates_reserved"] == 256
    assert report["cost"]["training_cell_calls_reserved"] == 256 * 96
    assert report["artifact_hashes"] == []
    assert "budget" in _dispatch(session, {"action": "run_experiment", "config": {"steps": 1}})["error"]
    assert len(requests) == 2


def test_experiment_and_training_cell_budgets_fail_before_launch(monkeypatch):
    monkeypatch.setattr(environment, "run_worker", _inline_worker)
    session = environment.ResearchSession(_case("ldt_filter"))
    for index in range(4):
        assert _dispatch(session, {"action": "run_experiment", "config": {"steps": 0}})["created"] == f"a{index + 1}"
    assert "unavailable" in _dispatch(session, {"action": "run_experiment", "config": {"steps": 0}})["error"]
    trm = environment.ResearchSession(_case("trm_training"))
    action = {"action": "run_experiment", "config": {"steps": 128, "train_examples": 256, "recursions": 8}}
    assert "cell budget" in _dispatch(trm, action)["error"]
    assert trm.result()["cost"]["experiments"] == 0


def test_selected_artifact_requires_its_own_matched_zero_update_baseline(monkeypatch):
    monkeypatch.setattr(environment, "run_worker", _inline_worker)
    session = environment.ResearchSession(_case("ldt_filter"))
    _dispatch(session, {"action": "run_experiment", "config": {"steps": 0, "train_examples": 32}})
    _dispatch(session, {"action": "run_experiment", "config": {"steps": 32, "train_examples": 64}})
    answer = {"artifact_id": "a2", "baseline_id": "a1", "claim": "improves"}
    assert "matched" in _dispatch(session, {"action": "submit", "answer": answer})["error"]
    assert session.result()["primary_score"] == 0
    assert session.result()["closed"] is True


def test_claimed_improvement_and_veto_are_checked_independently(monkeypatch):
    monkeypatch.setattr(environment, "run_worker", _inline_worker)
    for wrong_claim, disabled_veto in [(True, False), (False, True)]:
        session = environment.ResearchSession(_case("ldt_filter"))
        answer = _fit_pair(session, {"steps": 32, "critical_veto": not disabled_veto})
        if wrong_claim:
            answer["claim"] = "worse" if answer["claim"] != "worse" else "improves"
        reply = _dispatch(session, {"action": "submit", "answer": answer})
        assert reply["status"] == "submitted"
        result = session.result()
        assert result["primary_score"] == 0
        assert result["integrity_pass"] is False
        if disabled_veto:
            assert result["metrics"]["critical_veto_pass"] is False
        else:
            assert result["metrics"]["claim_supported"] is False


def test_persona_memory_preserves_owner_scope_revision_and_revocation(monkeypatch):
    monkeypatch.setattr(environment, "run_worker", _inline_worker)
    case = _case("persona_externalization")
    expected = suite.persona_memory(case["public"])
    assert expected == {"tone": next(event["value"] for event in case["public"]["events"] if event["revision"] == 1)}
    session = environment.ResearchSession(case)
    answer = _fit_pair(session)
    answer["memory"] = {"tone": "cruel", "verbosity": "brief"}
    _dispatch(session, {"action": "submit", "answer": answer})
    result = session.result()
    assert result["metrics"]["persona_memory_correct"] is False
    assert result["primary_score"] == 0


def test_final_grading_failure_closes_episode_with_zero_score(monkeypatch):
    monkeypatch.setattr(environment, "run_worker", _inline_worker)
    session = environment.ResearchSession(_case("ldt_filter"))
    answer = _fit_pair(session)

    def failed_evaluator(_request, **_kwargs):
        raise RuntimeError("worker_timeout")

    monkeypatch.setattr(environment, "run_worker", failed_evaluator)
    observed = _dispatch(session, {"action": "submit", "answer": answer})
    assert "error" in observed
    result = session.result()
    assert result["closed"] is True
    assert result["completed"] is False
    assert result["primary_score"] == 0
    assert result["cost"]["optimizer_updates_reserved"] == 32


def test_episode_deadline_limits_final_workers_and_preserves_charged_fit(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(environment.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(environment, "run_worker", _inline_worker)
    session = environment.ResearchSession(_case("ldt_filter"))
    answer = _fit_pair(session)
    clock[0] = session.started + environment.LIMITS["episode_seconds"] - 5
    calls = []

    def overrun(request, *, timeout_seconds):
        calls.append((request["split"], timeout_seconds))
        response = _inline_worker(request)
        clock[0] += 6
        return response

    monkeypatch.setattr(environment, "run_worker", overrun)
    reply = _dispatch(session, {"action": "submit", "answer": answer})
    assert reply["error"] == "episode_timeout"
    assert calls == [("evaluation", 5.0)]
    report = session.result()
    assert report["closed"] is True
    assert report["completed"] is False
    assert report["primary_score"] == 0
    assert report["cost"]["optimizer_updates_reserved"] == 32


def test_expired_episode_cannot_launch_experiments(monkeypatch):
    session = environment.ResearchSession(_case("ldt_filter"))
    monkeypatch.setattr(
        environment.time, "monotonic", lambda: session.started + environment.LIMITS["episode_seconds"] + 1,
    )

    def never_called(_request, **_kwargs):
        raise AssertionError("expired episode launched a worker")

    monkeypatch.setattr(environment, "run_worker", never_called)
    reply = _dispatch(session, {"action": "run_experiment", "config": {"steps": 32}})
    assert reply["status"] == "episode_timeout"
    assert session.result()["cost"]["experiments"] == 0
    assert session.result()["closed"] is True


def test_real_isolated_worker_produces_replayable_trained_artifact():
    # This executes the actual Linux CPU/RAM-capped process, without a provider.
    case = _case("trm_training")
    histories = []
    for _ in range(2):
        session = environment.ResearchSession(case)
        answer = _fit_pair(session, {"steps": 32, "recursions": 8, "train_examples": 32})
        observed = _dispatch(session, {"action": "submit", "answer": answer})
        assert observed["status"] == "submitted"
        report = session.result()
        assert report["integrity_pass"] is True
        assert report["primary_score"] > 0.75
        assert report["metrics"]["weights_changed"] is True
        assert report["metrics"]["delta_from_initialization"] > 0
        histories.append(report)
    assert histories[0] == histories[1]


def test_worker_subprocess_arguments_never_execute_candidate_code(monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        return type("Completed", (), {"returncode": 0, "stdout": "{}"})()

    monkeypatch.setattr(environment.subprocess, "run", fake_run)
    request = {"op": "fit", "config": {"payload": "__import__('os').system('not-executed')"}}
    assert environment.run_worker(request) == {}
    assert isinstance(captured["argv"], list)
    assert "-I" in captured["argv"]
    assert not captured["kwargs"].get("shell", False)
    assert json.loads(captured["kwargs"]["input"]) == request
    assert "payload" not in " ".join(captured["argv"])
