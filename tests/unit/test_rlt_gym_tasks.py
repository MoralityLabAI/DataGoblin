from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import canonical_json
from datagoblin.training_gym.rlt_gym.tasks import (
    FAMILIES,
    SPLITS,
    generate_tasks,
    load_suite,
    public_task,
    reference_answer,
    validate_task,
    verify,
    write_suite,
)


@pytest.fixture(scope="module")
def worlds():
    return generate_tasks(seed=19, per_family=1)


def _task(worlds, family, split="train"):
    return copy.deepcopy(next(task for task in worlds if task["family"] == family and task["split"] == split))


def _seal(task):
    task["task_digest"] = sha256_json({key: value for key, value in task.items() if key != "task_digest"})
    return task


def _rewrite_suite(manifest_path: Path, manifest, tasks):
    task_path = manifest_path.parent / "tasks.jsonl"
    task_path.write_text("".join(canonical_json(task) + "\n" for task in tasks), encoding="utf-8")
    manifest["tasks_sha256"] = sha256_file(task_path)
    manifest["manifest_digest"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")


def test_generation_is_reproducible_and_instance_disjoint(worlds):
    assert worlds == generate_tasks(seed=19, per_family=1)
    assert worlds != generate_tasks(seed=20, per_family=1)
    assert len(worlds) == len(FAMILIES) * len(SPLITS)
    assert len({task["lineage_id"] for task in worlds}) == len(worlds)
    assert Counter(task["split"] for task in worlds) == dict.fromkeys(SPLITS, len(FAMILIES))
    development = {t["generator_id"] for t in worlds if t["split"] in {"train", "calibration", "ordinary_test"}}
    assert not development & {t["generator_id"] for t in worlds if t["split"] == "ood_test"}


@pytest.mark.parametrize("family", FAMILIES)
def test_every_world_has_checked_oracle_and_mixed_pool(worlds, family):
    for task in (item for item in worlds if item["family"] == family):
        candidate = reference_answer(task)
        receipt = verify(task, candidate)
        assert receipt["passed"] is True
        assert receipt["candidate_digest"] == sha256_json(candidate)
        assert receipt["task_digest"] == task["task_digest"]
        assert receipt["evidence_epoch"] == task["evidence_epoch"]
        outcomes = [verify(task, item)["passed"] for item in task["candidates"]]
        assert any(outcomes) and not all(outcomes)


def test_public_view_excludes_control_metadata_and_does_not_alias(worlds):
    task = worlds[0]
    view = public_task(task)
    assert set(view) == {"family", "prompt", "input", "evidence_epoch"}
    view["input"].clear()
    assert task["input"]
    assert all("answer" not in item and "seed" not in item for item in worlds)


def test_arithmetic_checker_matches_manual_execution_and_rejects_boolean(worlds):
    task = _task(worlds, "arithmetic_program")
    task["input"] = {"initial": 4, "instructions": [
        {"op": "multiply", "operand": 3}, {"op": "if_even_add", "operand": 5},
        {"op": "if_even_add", "operand": 2}, {"op": "subtract", "operand": 7},
    ]}
    _seal(task)
    assert verify(task, {"value": 8})["passed"]
    assert not verify(task, {"value": 12})["passed"]
    task["input"] = {"initial": 1, "instructions": [{"op": "add", "operand": 0}]}
    _seal(task)
    assert not verify(task, {"value": True})["passed"]
    assert not verify(task, {"value": 1.0})["passed"]
    assert not verify(task, {"value": 1, "claimed_success": True})["passed"]


def test_route_checks_current_edges_cost_and_endpoints(worlds):
    task = _task(worlds, "graph_route")
    task["input"] = {"nodes": ["a", "b", "c"], "edges": [["a", "b", 1], ["b", "c", 1], ["a", "c", 3]],
                     "start": "a", "goal": "c"}
    _seal(task)
    assert verify(task, {"path": ["a", "b", "c"]})["passed"]
    assert verify(task, {"path": ["a", "c"]})["code"] == "nonminimal_route"
    assert not verify(task, {"path": ["a", "b", "a", "c"]})["passed"]
    task["input"]["edges"] = [["a", "b", 1], ["a", "c", 3]]
    task["evidence_epoch"] += 1
    _seal(task)
    assert verify(task, {"path": ["a", "b", "c"]})["code"] == "inactive_edge"


def test_constraint_checker_accepts_alternative_solution_and_rejects_missing_keys(worlds):
    task = _task(worlds, "constraint_assignment")
    task["input"] = {"variables": ["a", "b"], "colors": 3, "different": [["a", "b"]],
                     "fixed": {"a": 0}, "all_different": []}
    _seal(task)
    assert verify(task, {"assignment": {"a": 0, "b": 1}})["passed"]
    assert verify(task, {"assignment": {"a": 0, "b": 2}})["passed"]
    assert not verify(task, {"assignment": {"a": 0}})["passed"]
    assert not verify(task, {"assignment": {"a": False, "b": 1}})["passed"]


def test_evidence_requires_current_source_even_when_old_value_matches(worlds):
    task = _task(worlds, "evidence_retrieval")
    answer = reference_answer(task)
    old = next(doc for doc in task["input"]["documents"] if doc["epoch"] != task["evidence_epoch"])
    old["value"] = answer["value"]
    _seal(task)
    assert not verify(task, {"value": answer["value"], "evidence_ids": [old["id"]]})["passed"]
    task["evidence_epoch"] += 1
    _seal(task)
    with pytest.raises(ValueError, match="epoch"):
        validate_task(task)


def test_table_filter_multiplier_and_integer_shape(worlds):
    task = _task(worlds, "table_transform")
    task["input"] = {"records": [
        {"id": "r1", "group": "g", "amount": 4, "keep": True},
        {"id": "r2", "group": "g", "amount": 8, "keep": False},
        {"id": "r3", "group": "g", "amount": 2, "keep": True},
    ], "minimum_amount": 3, "multipliers": {"g": 2}}
    _seal(task)
    answer = {"rows": [{"group": "g", "total": 8, "count": 1}]}
    assert verify(task, answer)["passed"]
    answer["rows"][0]["count"] = True
    assert not verify(task, answer)["passed"]


def test_experiment_checker_enforces_cost_and_selection_firewall(worlds):
    task = _task(worlds, "experiment_plan")
    answer = reference_answer(task)
    assert not verify(task, {**answer, "replicates": 100})["passed"]
    assert not verify(task, {**answer, "replicates": True})["passed"]
    assert not verify(task, {**answer, "selection_split": "ordinary_test"})["passed"]
    assert not verify(task, {**answer, "independent_verifier": 1})["passed"]


def test_finite_state_checker_executes_actions_and_bounds(worlds):
    task = _task(worlds, "finite_state_plan")
    task["input"] = {"capacities": [3, 5], "initial": [0, 0], "target_container": 1,
                     "target_amount": 4, "max_actions": 6}
    _seal(task)
    answer = {"actions": ["fill:1", "pour:1:0", "empty:0", "pour:1:0", "fill:1", "pour:1:0"]}
    assert verify(task, answer)["passed"]
    assert not verify(task, {"actions": ["fill:9"]})["passed"]
    assert not verify(task, {"actions": [*answer["actions"], "empty:0"]})["passed"]


def test_tampered_task_and_nonfinite_candidate_rejected(worlds):
    task = copy.deepcopy(worlds[0])
    task["prompt"] += " tampered"
    with pytest.raises(ValueError, match="digest"):
        verify(task, {})
    with pytest.raises(ValueError, match="finite"):
        verify(worlds[0], {"value": float("nan")})
    task = _task(worlds, "table_transform")
    task["input"]["records"][0]["group"] = []
    _seal(task)
    with pytest.raises(ValueError, match="world"):
        validate_task(task)


def test_suite_roundtrip_is_immutable_and_hash_bound(tmp_path):
    path = write_suite(tmp_path / "suite", seed=19, per_family=1)
    manifest, tasks = load_suite(path)
    assert tasks == generate_tasks(seed=19, per_family=1)
    assert manifest["task_count"] == 35
    with pytest.raises(FileExistsError):
        write_suite(path.parent, seed=19, per_family=1)
    with (path.parent / "tasks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="digest"):
        load_suite(path)


@pytest.mark.parametrize("tamper", ["lineage", "world", "generator"])
def test_suite_rejects_split_leakage_even_after_receipts_are_resealed(tmp_path, tamper):
    path = write_suite(tmp_path / "suite", seed=19, per_family=1)
    manifest, tasks = load_suite(path)
    train = next(task for task in tasks if task["family"] == "graph_route" and task["split"] == "train")
    destination_split = "ood_test" if tamper == "generator" else "calibration"
    other = next(task for task in tasks if task["family"] == "graph_route" and task["split"] == destination_split)
    if tamper == "lineage":
        other["lineage_id"] = train["lineage_id"]
    elif tamper == "world":
        other["input"] = copy.deepcopy(train["input"])
    else:
        other["generator_id"] = train["generator_id"]
    _seal(other)
    _rewrite_suite(path, manifest, tasks)
    with pytest.raises(ValueError, match="overlap"):
        load_suite(path)


def test_suite_rejects_path_escape_and_manifest_tampering(tmp_path):
    path = write_suite(tmp_path / "suite", per_family=1)
    manifest = json.loads(path.read_text())
    manifest["tasks_file"] = "../elsewhere.jsonl"
    manifest["manifest_digest"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    path.write_text(canonical_json(manifest))
    with pytest.raises(ValueError, match="path"):
        load_suite(path)
    manifest["task_count"] = 10
    path.write_text(canonical_json(manifest))
    with pytest.raises(ValueError, match="digest"):
        load_suite(path)
