"""End-to-end evidence checks for the bounded RLT rollout gym."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from collections import defaultdict

import pytest

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json, dump_structured, load_structured
from datagoblin.training_gym.rlt_gym.dynamics import NativeRanker
from datagoblin.training_gym.rlt_gym.runner import (
    read_jsonl,
    replay_run,
    rollout,
    run_suite,
    summarize,
    verify_episode,
)
from datagoblin.training_gym.rlt_gym.tasks import load_suite, reference_answer, verify, write_suite


@pytest.fixture
def suite(tmp_path):
    return write_suite(tmp_path / "suite", seed=17, per_family=1)


@pytest.fixture
def completed_run(suite, tmp_path):
    directory = tmp_path / "run"
    result = run_suite(
        suite, directory, seeds=(7,), modes=("direct", "rlt", "matched_damping"),
        training_modes=("direct",), depths=(1, 4), splits=("calibration",), max_wall_seconds=120,
    )
    return suite, directory, result


def test_generated_run_replays_every_trajectory_and_checker(completed_run):
    suite, directory, result = completed_run
    checked = replay_run(suite, directory)
    assert result["episodes"] == checked["episodes"] == 56
    assert checked["passed"] is True
    assert checked["exact_trajectory_replays"] == 56
    assert checked["receipt_only_aborted_episodes"] == 0
    assert checked["independently_rechecked_candidates"] == result["summary"]["verifier_calls"]
    episodes = read_jsonl(directory / "rollouts.jsonl")
    assert len({row["episode_id"] for row in episodes}) == len(episodes)
    reference = [row for row in episodes if row["backend"]["kind"] == "reference_solver"]
    assert len(reference) == 7 and all(row["outcome"]["passed"] for row in reference)
    assert result["summary"]["architecture_promotion_eligible"] is False
    assert result["resources"]["weights_downloaded"] is False


def test_one_visit_runtime_controls_share_states_and_scores(completed_run):
    _, directory, _ = completed_run
    groups = defaultdict(list)
    for episode in read_jsonl(directory / "rollouts.jsonl"):
        if episode["backend"]["kind"] == "frozen_random_reservoir" and episode["config"]["planned_depth"] == 1:
            groups[episode["task_digest"]].append(episode)
    assert len(groups) == 7
    for values in groups.values():
        assert {row["config"]["mode"] for row in values} == {"direct", "rlt", "matched_damping"}
        first = values[0]["steps"][0]
        for episode in values:
            step = episode["steps"][0]
            assert step["eta"] == 1
            assert step["state_summaries"] == first["state_summaries"]
            assert step["candidate_scores"] == first["candidate_scores"]
            assert step["verification"] == first["verification"]


def test_each_training_arm_has_matching_initialization_labels_and_work(suite, tmp_path):
    directory = tmp_path / "trained-arms"
    run_suite(
        suite, directory, seeds=(7,), modes=("direct",),
        training_modes=("direct", "rlt", "matched_damping"), depths=(2,),
        splits=("calibration",), include_controls=False, max_wall_seconds=120,
    )
    manifest = load_structured(directory / "manifest.json")
    models = [NativeRanker.load(directory / artifact["path"]) for artifact in manifest["models"].values()]
    assert len(models) == 3
    assert {model.metadata["training_mode"] for model in models} == {"direct", "rlt", "matched_damping"}
    for field in ("initialization_digest", "training_fingerprint", "training_receipts_digest",
                  "training_readout_examples", "training_block_calls", "training_verifier_calls"):
        assert len({model.metadata[field] for model in models}) == 1
    assert replay_run(suite, directory)["exact_trajectory_replays"] == 21


def test_replay_rejects_changed_artifact_bytes(completed_run):
    suite, directory, _ = completed_run
    path = directory / "rollouts.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="digest"):
        replay_run(suite, directory)


def test_rehashed_receipt_still_requires_independent_verification(completed_run):
    suite, directory, _ = completed_run
    _, task_rows = load_suite(suite)
    by_id = {task["task_id"]: task for task in task_rows}
    episode = copy.deepcopy(read_jsonl(directory / "rollouts.jsonl")[0])
    receipt = episode["steps"][0]["verification"]
    receipt["passed"] = not receipt["passed"]
    receipt["status"] = "passed" if receipt["passed"] else "failed"
    episode["episode_digest"] = sha256_json({key: value for key, value in episode.items() if key != "episode_digest"})
    with pytest.raises(ValueError, match=r"verifier|receipt"):
        verify_episode(by_id[episode["task_id"]], episode)


def test_replay_rejects_an_unplanned_cell_even_with_matching_counts_and_hashes(completed_run):
    suite, directory, _ = completed_run
    manifest = load_structured(directory / "manifest.json")
    _, task_rows = load_suite(suite)
    by_id = {task["task_id"]: task for task in task_rows}
    episodes = read_jsonl(directory / "rollouts.jsonl")
    index = next(i for i, row in enumerate(episodes) if row["backend"]["kind"] == "frozen_random_reservoir")
    original = episodes[index]
    artifact = manifest["models"][original["backend"]["model_digest"]]
    model = NativeRanker.load(directory / artifact["path"])
    # A complete, independently valid trajectory at a depth absent from the frozen plan.
    episodes[index] = rollout(by_id[original["task_id"]], model, mode="direct", planned_depth=3, seed=7)
    (directory / "rollouts.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in episodes), encoding="utf-8",
    )
    dump_structured(directory / "summary.json", summarize(episodes))
    for filename in ("rollouts.jsonl", "summary.json"):
        manifest["files"][filename] = sha256_file(directory / filename)
    manifest["manifest_digest"] = sha256_json({k: v for k, v in manifest.items() if k != "manifest_digest"})
    dump_structured(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match=r"matrix|cell|planned|configuration"):
        replay_run(suite, directory)


class _ScriptedAdapter:
    model_digest = "1" * 64

    def __init__(self, candidates):
        self.metadata = {"model_kind": "engineering_fixture", "training_mode": "direct", "fitted": False}
        self.candidates = candidates
        self.visits_requested = 0

    def visits(self, task, mode, planned_depth, **kwargs):
        for index, candidate in enumerate(self.candidates):
            self.visits_requested += 1
            yield {"visit": index, "candidate": candidate, "planned_depth": planned_depth,
                   "evidence_epoch": task["evidence_epoch"], "segment_id": "fixture-segment"}


def test_expired_deadline_does_not_request_a_model_visit(suite):
    _, task_rows = load_suite(suite)
    task = next(task for task in task_rows if task["split"] == "calibration")
    adapter = _ScriptedAdapter([reference_answer(task)])
    episode = rollout(task, adapter, planned_depth=1, deadline=time.monotonic() - 1)
    assert adapter.visits_requested == 0
    assert episode["steps"] == []
    assert episode["outcome"]["status"] == "resource_abort"
    assert episode["outcome"]["realized_visits"] == 0
    verify_episode(task, episode)


def test_terminal_scoring_retains_an_earlier_success_and_final_failure(suite):
    _, task_rows = load_suite(suite)
    task = next(task for task in task_rows if task["split"] == "calibration")
    good = reference_answer(task)
    bad = next(candidate for candidate in task["candidates"] if not verify(task, candidate)["passed"])
    terminal = rollout(task, _ScriptedAdapter([good, bad]), planned_depth=2, verified_stop=False)
    assert terminal["outcome"]["status"] == "failed"
    assert terminal["outcome"]["passed"] is False
    assert terminal["outcome"]["first_verified_visit"] == 0
    assert len(terminal["steps"]) == 2
    verify_episode(task, terminal)
    first_valid = rollout(task, _ScriptedAdapter([good, bad]), planned_depth=2, verified_stop=True)
    assert first_valid["outcome"]["passed"] is True
    assert first_valid["outcome"]["realized_visits"] == 1
    verify_episode(task, first_valid)


def test_public_cli_creates_runs_and_replays_without_external_services(tmp_path):
    suite = tmp_path / "cli-suite"
    directory = tmp_path / "cli-run"

    def command(*arguments):
        completed = subprocess.run(
            [sys.executable, "-m", "datagoblin", "rlt-gym", *map(str, arguments)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return json.loads(completed.stdout)

    made = command("make-suite", "--seed", "41", "--per-family", "1", "--out", suite)
    assert made["tasks"] == 35
    collected = command("run", "--suite", suite, "--out", directory, "--seeds", "7",
                        "--modes", "direct", "--depths", "1", "--omit-controls")
    assert collected["episodes"] == 7
    checked = command("replay", "--suite", suite, "--run", directory)
    assert checked["passed"] is True
    assert checked["exact_trajectory_replays"] == 7
