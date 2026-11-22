"""AGN-02: exact three-decision semantics, matched student budget and replay."""

from __future__ import annotations

import json

import pytest

from Agonist.agn02 import (
    ARM_NAMES,
    COHORT,
    STUDENT_SEEDS,
    _assign_cohorts,
    _student_decisions,
    check_value,
    construct_roots,
    exact_actions,
    fit_student,
    make_tastes,
    replay,
    run,
    simulate,
)
from Agonist.gym import canonical, read_jsonl, verify_pair, write_jsonl


def test_fifty_independent_taste_families_and_custody():
    tastes = make_tastes(17)
    assert len(tastes) == 50
    assigned = _assign_cohorts(tastes)
    assert {group for group, _ in assigned} == set(COHORT)
    assert {group: sum(name == group for name, _ in assigned) for group in COHORT} == COHORT
    assert len({item["family_id"] for _, item in assigned}) == 50
    assert all(t["split"] == "train" for group, t in assigned if group in ("anchor", "pool"))
    assert all(t["split"] == "dev" for group, t in assigned if group in ("calibration", "evaluation"))


def test_fifty_valid_causal_flips_and_three_decisions():
    roots = construct_roots(17, make_tastes(17))
    assert len(roots) == 50
    for root in roots:
        initial, revised, check = exact_actions(root)
        assert (initial, revised) == verify_pair(
            root["world_a"], root["world_b"], root["intervention"]
        )
        assert initial != revised
        assert check in ("check", "stop")
        assert (check == "check") == (check_value(root) > 0)
        assert root["generation_kind"] == "offline_rule_fixture_no_teacher"


def test_check_has_value_only_when_structural_action_possible():
    roots = construct_roots(17, make_tastes(17))
    structural = next(
        root for root in roots
        if root["world_b"]["state"]["available"] >= root["world_b"]["state"]["required"]
        and root["world_b"]["state"]["safety_approved"]
    )
    state = dict(structural["world_b"]["state"])
    root = {**structural, "check_case": {"hazard_percent": 85, "check_cost": 1, "mistake_cost": 12}}
    assert check_value(root) > 0
    root = {**root, "world_b": {**root["world_b"], "state": {**state, "safety_approved": False}}}
    assert check_value(root) == -100


def test_learned_selector_is_actual_training_and_eval_is_heldout():
    files, report = simulate(seed=17, updates=96, selection_count=12)
    assert len(files["oracle_receipts.jsonl"]) == 50
    assert len(files["selector_training.jsonl"]) == 10
    assert len(files["selections.jsonl"]) == 4
    assert len(files["student_scores.jsonl"]) == 8 * 3 * 4
    assert report["teacher_inference_calls"] == 0
    assert report["student_seeds"] == list(STUDENT_SEEDS)
    assert report["training_ready"] is False
    assert report["sealed_confirmation_run"] is False
    assert report["selector_training"]["calibration_student_fits"] == 11
    assert report["selector_training"]["learned_controller_is_data_goblin_1b"] is False
    selected = {row["arm"]: row for row in files["selections.jsonl"]}
    assert set(selected) == set(ARM_NAMES)
    assert selected["anchor_only"]["train_examples"] == 30
    for arm in ("random", "fixed", "learned"):
        assert selected[arm]["train_examples"] == 66
        assert selected[arm]["selected_pool_roots"] == 12
    training_roots = {r["root"] for r in files["roots.jsonl"] if r["cohort"] in ("anchor", "pool")}
    evaluation_roots = {r["root"] for r in files["roots.jsonl"] if r["cohort"] == "evaluation"}
    assert not (training_roots & evaluation_roots)
    for row in selected.values():
        assert set(row["roots"]) <= training_roots
    assert {r["root"] for r in files["student_scores.jsonl"]} == evaluation_roots


def test_stable_ridge_surrogate_and_genuine_student_fitting():
    files, report = simulate(seed=17, updates=80, selection_count=5)
    assert len(report["selector_training"]["selector_weights"]) == 11
    roots = files["roots.jsonl"]
    student_a = fit_student(roots[:10], seed=11, updates=80)
    student_b = fit_student(roots[:10], seed=11, updates=80)
    assert (student_a == student_b).all()
    assert any(float(abs(x)) > 0 for x in student_a.flatten())
    assert len(_student_decisions(student_a, roots[-1])) == 3


def test_deterministic_archive_replay_and_sha256_tamper_detection(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    a = run(first, seed=17, updates=80, selection_count=5)
    b = run(second, seed=17, updates=80, selection_count=5)
    assert a == b
    assert replay(first) == {
        "verified_roots": 50,
        "student_score_rows": 96,
        "recomputed": True,
        "sealed_confirmation_run": False,
    }
    with pytest.raises(FileExistsError):
        run(first, seed=17, updates=80, selection_count=5)
    path = first / "student_scores.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        replay(first)


def test_source_root_rights_and_external_taste_manifest(tmp_path):
    tastes = make_tastes(29)
    tastes_path = tmp_path / "tastes.jsonl"
    write_jsonl(tastes_path, tastes)
    archive = tmp_path / "archive"
    manifest = run(archive, seed=29, updates=50, selection_count=4, tastes_path=tastes_path)
    assert manifest["source_tastes"] == "operator_supplied"
    assert replay(archive, tastes_path)["verified_roots"] == 50
    with pytest.raises(ValueError, match="required exactly"):
        replay(archive)
    tastes[0]["rights"] = "reference_only"
    write_jsonl(tastes_path, tastes)
    with pytest.raises(ValueError, match="taste input hash mismatch"):
        replay(archive, tastes_path)
    with pytest.raises(ValueError, match="reference-only"):
        _assign_cohorts(tastes)


def test_split_crossing_and_repeated_family_is_rejected():
    tastes = make_tastes(17)
    tastes[1]["family_id"] = tastes[0]["family_id"]
    with pytest.raises(ValueError, match="independent taste families"):
        _assign_cohorts(tastes)
    tastes = make_tastes(17)
    tastes[40]["split"] = "train"
    with pytest.raises(ValueError, match="34 train"):
        _assign_cohorts(tastes)


def test_manifest_gold_cannot_be_relabelled_after_hash_update(tmp_path):
    archive = tmp_path / "run"
    run(archive, seed=17, updates=50)
    receipt_file = archive / "oracle_receipts.jsonl"
    receipts = read_jsonl(receipt_file)
    receipts[0]["actions"][0] = "hold" if receipts[0]["actions"][0] == "deploy" else "deploy"
    write_jsonl(receipt_file, receipts)
    manifest_file = archive / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    from Agonist.gym import sha256_file

    manifest["artifacts"]["oracle_receipts.jsonl"]["sha256"] = sha256_file(receipt_file)
    manifest_file.write_text(canonical(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="independent recomputation disagrees"):
        replay(archive)


def test_archived_eval_scores_are_dev_only_and_completely_paired():
    files, report = simulate(seed=21, updates=60)
    scores = files["student_scores.jsonl"]
    assert all(row["cohort"] == "evaluation" for row in scores)
    assert all(row["checked_decisions"] == 3 for row in scores)
    assert len({(row["root"], row["arm"], row["student_seed"]) for row in scores}) == len(scores)
    assert set(report["paired"]) == {
        "learned_minus_random", "learned_minus_fixed", "learned_minus_anchor_only"
    }
    for value in report["paired"].values():
        lo, hi = value["descriptive_root_bootstrap_95"]
        assert -1 <= lo <= hi <= 1
