"""MeTTa-10 CPU recurrent-vs-typed-lattice hill-climb gym regression tests."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from MeTTa.ram_hillclimb import (
    ARMS,
    DEFAULTS,
    DEV_START,
    FAMILIES,
    OOD_FAMILY,
    OPTIONS,
    PUBLIC_TEST_START,
    TRAIN_START,
    check_config,
    fit_candidate,
    make_bundle,
    metta_proposal,
    peak_rss_mib,
    read_proposals,
    root_name,
    run_gym,
    select_key,
    training_views,
    validate_proposal,
)
from MeTTa.sft_oracle import exact_optimum, make_task, next_edit


def test_disjoint_root_custody_and_four_view_budget() -> None:
    bundle = make_bundle(2, 1)
    assert bundle["train_source_roots"] == len(FAMILIES) * 2
    assert bundle["dev_source_roots"] == len(FAMILIES)
    assert bundle["dev_cases"] == len(FAMILIES)
    assert 1 <= bundle["max_presentations_per_root"] <= 4
    exposures = Counter(row["root"] for row in bundle["train"])
    assert all(1 <= count <= 4 for count in exposures.values())
    assert {r["root"] for r in bundle["train"]}.isdisjoint(
        {r["root"] for r in bundle["dev"]}
    )
    assert set(bundle["public_test_root_names_only"]).isdisjoint(
        {r["root"] for r in bundle["train"] + bundle["dev"]}
    )
    assert root_name(FAMILIES[0], TRAIN_START) in exposures
    assert root_name(FAMILIES[0], DEV_START) in {
        row["root"] for row in bundle["dev"]
    }
    assert root_name(OOD_FAMILY, PUBLIC_TEST_START) in bundle[
        "public_test_root_names_only"
    ]
    assert all(r["training_ready"] is False for r in bundle["train"])
    assert all(r["split"] == "public_development" for r in bundle["dev"])
    assert all(r["label_exposed_to_student"] is False for r in bundle["dev"])
    for row in bundle["train"]:
        task = row["public_world"]
        assert task["program"].endswith("\n")
        assert row["oracle_next_edit"] == next_edit(
            row["selected"], exact_optimum(task)["selected"]
        )


def test_training_views_cover_fixed_point_without_extra_presentations() -> None:
    task = make_task(FAMILIES[0], TRAIN_START)
    views = training_views(task, TRAIN_START)
    assert len(views) == len({tuple(view) for view in views})
    assert views[-1] == exact_optimum(task)["selected"]
    assert len(views) <= 4


def test_only_registered_one_field_training_edits_can_run() -> None:
    configs = {arm: dict(DEFAULTS[arm]) for arm in ARMS}
    action = {
        "arm": "trm", "field": "lr", "value": 0.08,
        "source": "chat_import", "model_id": "user_declared_teacher",
    }
    updated = validate_proposal(action, configs)
    assert updated["lr"] == 0.08
    assert updated["hidden"] == configs["trm"]["hidden"]
    assert configs["trm"]["lr"] == DEFAULTS["trm"]["lr"]
    check_config("trm", updated)
    source = metta_proposal(action, 8192)
    assert "(Candidate trm lr v0_08)" in source
    assert "!(match &self" in source
    assert all(field in OPTIONS["trm"] for field in updated)
    with pytest.raises(ValueError):
        validate_proposal({"arm": "trm", "field": "cuda", "value": 1}, configs)
    with pytest.raises(ValueError):
        validate_proposal({"arm": "trm", "field": "lr", "value": 2.0}, configs)
    with pytest.raises(ValueError):
        validate_proposal({"arm": "trm", "field": "lr", "value": True}, configs)
    with pytest.raises(ValueError):
        validate_proposal({"arm": "trm", "field": "lr", "value": 0.2}, configs)
    with pytest.raises(ValueError):
        validate_proposal({
            "arm": "trm", "field": "lr", "value": 0.08,
            "python": "import os; os.system('whoami')",
        }, configs)


def test_genuine_numpy_training_and_host_lattice_gate() -> None:
    bundle = make_bundle(1, 1)
    for arm in ARMS:
        model, result = fit_candidate(
            arm, dict(DEFAULTS[arm]), bundle["train"], bundle["dev"],
            seed=20261122, ram_limit_mib=8192,
        )
        assert result["model_trained"] is True
        assert len(result["checkpoint_sha256"]) == 64
        assert result["checkpoint_sha256"] == model.digest()
        assert result["parameter_count"] > 0
        assert result["real_cpu_gradient_updates"] == 16
        assert result["training_examples"] == len(bundle["train"])
        assert result["peak_process_rss_mib"] <= 8192
        assert result["dev_aggregate"]["cases"] == len(FAMILIES)
        assert set(result["family_metrics"]) == set(FAMILIES)
        assert len(result["selection_key"]) == 8
        assert result["training_ready"] is False
        if arm == "ldt_proxy":
            assert result["dev_aggregate"]["new_violation_additions"] == 0
            assert result["model_scope"].startswith("linear_learner")
        else:
            assert result["model_scope"].startswith("weight_tied_tanh")
    assert peak_rss_mib() < 8192


def test_train_only_stop_head_is_a_real_cpu_learned_control() -> None:
    bundle = make_bundle(1, 1)
    for arm in ARMS:
        config = dict(DEFAULTS[arm])
        config["stop_policy"] = "train_calibrated"
        model, result = fit_candidate(
            arm, config, bundle["train"], bundle["dev"],
            seed=20261122, ram_limit_mib=8192,
        )
        receipt = result["train_only_stop_calibration"]
        assert receipt is not None
        assert receipt["threshold_selected_on"] == "training_only"
        assert receipt["no_development_label_calibration"] is True
        assert receipt["train_positive"] > 0
        assert receipt["train_negative"] > 0
        assert result["real_cpu_gradient_updates"] == 32
        assert len(receipt["checkpoint_sha256"]) == 64
        assert model.digest() == result["checkpoint_sha256"]
        assert result["training_ready"] is False
        if arm == "ldt_proxy":
            assert result["dev_aggregate"]["new_violation_additions"] == 0


def test_full_hillclimb_replays_exact_train_traces_and_keeps_test_names_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "gym"
        proposal_file = Path(tmp) / "teacher.jsonl"
        proposals = [
            {
                "arm": "trm", "field": "inference_loops", "value": 2,
                "source": "offline_chat", "model_id": "unverified",
            },
            {
                "arm": "ldt_proxy", "field": "lr", "value": 0.08,
                "source": "offline_chat", "model_id": "unverified",
            },
        ]
        proposal_file.write_text(
            "".join(json.dumps(p) + "\n" for p in proposals), encoding="utf-8"
        )
        summary = run_gym(
            output, train_roots=1, dev_roots=1,
            proposals_path=proposal_file, max_proposals=2,
            ram_limit_mib=8192,
        )
        assert summary["model_trained_on_cpu"] is True
        assert summary["gpu_required"] is False
        assert summary["external_teacher_model_invoked"] is False
        assert summary["teacher_mode"] == "imported_jsonl"
        assert summary["proposals_executed"] == 2
        assert summary["baseline_and_search_candidates"] == 4
        assert summary["development_cases"] == len(FAMILIES)
        assert summary["public_test_roots_materialized"] == 0
        assert summary["ood_conflict_roots_materialized"] == 0
        assert summary["max_source_presentations_per_root"] <= 4
        assert summary["train_repair_traces"] == summary["train_cases"]
        assert summary["observed_peak_rss_mib"] < 8192
        assert summary["training_ready"] is False
        assert summary["synthetic_research_fit_eligible"] is True
        assert summary["next_teacher_prompt"]["labels_in_prompt"] is False
        traces = [
            json.loads(line)
            for line in (output / "train_repair_traces.jsonl").read_text().splitlines()
        ]
        assert len(traces) == summary["train_cases"]
        assert set(x["split"] for x in traces) == {"train_candidate"}
        assert set(x["family"] for x in traces) == set(FAMILIES)
        assert all(x["training_ready"] is False for x in traces)
        assert all(x["metta_program"] and x["verified_canonical_next_edit"] for x in traces)
        assert {x["root"] for x in traces}.isdisjoint({
            root_name(FAMILIES[0], DEV_START),
            root_name(FAMILIES[0], PUBLIC_TEST_START),
        })
        steps = [
            json.loads(line)
            for line in (output / "hillclimb_steps.jsonl").read_text().splitlines()
        ]
        assert len(steps) == 2
        assert all(step["public_test_roots_opened"] is False for step in steps)
        assert all(step["training_ready"] is False for step in steps)
        assert all(step["decision"] in {
            "reject_safety_regression", "reject_no_gain", "accept_verified_development_gain"
        } for step in steps)
        for step in steps:
            before = step["incumbent_metrics"]
            after = step["candidate_fit"]["dev_aggregate"]
            if step["accepted"]:
                assert select_key(after) > select_key(before)
                assert after["new_violation_additions"] <= before["new_violation_additions"]
                assert after["unsafe_additions"] <= before["unsafe_additions"]
        assert summary == json.loads((output / "report.json").read_text())
        assert len(summary["train_repair_traces_sha256"]) == 64
        assert len(summary["hillclimb_steps_sha256"]) == 64


def test_teacher_input_file_is_explicit_and_bounded(tmp_path: Path) -> None:
    assert len(read_proposals(None, 3)) == 3
    with pytest.raises(ValueError):
        read_proposals(None, 9)
    huge = tmp_path / "huge.jsonl"
    huge.write_text(" " * 20000)
    with pytest.raises(ValueError):
        read_proposals(huge, 4)
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("[]\n")
    with pytest.raises(ValueError):
        read_proposals(invalid, 4)
