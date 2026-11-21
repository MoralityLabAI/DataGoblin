import pytest

from datagoblin.serialization import dump_structured
from datagoblin.training_gym.rlt_gym.bpd_contract import MATCH_FIELDS, validate_comparison
from datagoblin.training_gym.rlt_gym.campaign import episode_metrics
from datagoblin.training_gym.rlt_gym.codex_teacher import replies
from datagoblin.training_gym.rlt_gym.corpus import build, records
from datagoblin.training_gym.rlt_gym.runner import replay_run, run_suite
from datagoblin.training_gym.rlt_gym.tasks import write_suite


def test_rlt_gym_campaign_exports_only_replayed_train_supervision(tmp_path):
    suite = write_suite(tmp_path / "suite", seed=9534, per_family=1)
    run = tmp_path / "run"
    run_suite(suite, run, seeds=(7,), modes=("direct",), depths=(2,), splits=("train", "ordinary_test"))
    replay_path = tmp_path / "replay.json"
    dump_structured(replay_path, replay_run(suite, run))
    source = {"suite": str(suite), "rollouts": str(run / "rollouts.jsonl"), "receipt": str(replay_path)}
    out = tmp_path / "corpus"
    summary = build({"suites": [str(suite)], "sources": [source, source]}, out)
    assert summary["excluded_rollouts"]["ordinary_test"] > 0
    assert summary["streams"]["sft"]["roots"] == 7
    assert summary["streams"]["research"]["unique_examples"] > 0
    for name in summary["streams"]:
        rows = list(records(out / f"{name}.jsonl"))
        assert len({row["id"] for row in rows}) == len(rows)
        assert all(row["split"] == "train" and row["loss"]["boundary"] == "response_only" for row in rows)
    forecasts = list(records(out / "outcomes.jsonl"))
    assert all("checker_code" not in row["input"] and "passed" in row["target"] for row in forecasts)


def test_rlt_gym_campaign_distinguishes_regressions_from_final_result():
    marks = [True, False, True]
    episode = {"outcome": {"first_verified_visit": 0, "passed": True, "censored": False,
                           "verifier_calls": 3, "adapter_calls": 3},
               "steps": [{"verification": {"passed": passed, "candidate_digest": str(i), "code": "example"}}
                         for i, passed in enumerate(marks)]}
    result = episode_metrics(episode, 0.1)
    assert result["regressed_after_success"] == 1
    assert result["final_regressions"] == 0
    assert result["first_verified_visit_1"] == 1


def test_rlt_gym_codex_binding_retains_unparseable_candidates():
    request = [{"request_id": "r", "task_digest": "t"}]
    result = list(replies({"answers": [{"index": 0, "candidate_json": "bogus", "summary": "test"},
                                       {"index": 2, "candidate_json": "null", "summary": ""}]}, request))
    assert result == [{"request_id": "r", "task_digest": "t", "candidate": None, "revision_summary": "test"}]


def test_rlt_gym_bpd_contract_preserves_matched_dose_and_visit():
    fields = {key: "x" for key in MATCH_FIELDS} | {"visit": 1, "planned_depth": 4}
    arms = {name: {**fields, "realized_absolute_dose": dose}
            for name, dose in (("untreated", 0), ("sham", 0), ("matched_random", 0.2), ("component", 0.2))}
    arms["component"]["donor_receipt_sha256"] = "d"
    comparison = {"arms": arms, "selection_receipt_sha256": "s", "bpd_mechanism_verified": False}
    validate_comparison(comparison)
    arms["matched_random"]["realized_absolute_dose"] = 0.3
    with pytest.raises(ValueError, match="same positive realized dose"):
        validate_comparison(comparison)
