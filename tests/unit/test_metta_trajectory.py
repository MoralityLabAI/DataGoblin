import json
from itertools import pairwise

import pytest

from MeTTa.ram_hillclimb_trajectory import DEFAULT, fit, improves, revise, rollout, run
from MeTTa.ram_trajectory import RepairVerifier, export_trajectories, training_trajectory, typed_proposal
from MeTTa.sft_oracle import evaluate_subset, exact_optimum, make_task


def test_full_trajectory_and_alternate_view_custody(tmp_path):
    row = training_trajectory("dependency", 20001)
    frames = row["transitions"]
    optimum = exact_optimum(row["world"])["selected"]
    assert frames[-1]["selected"] == optimum
    assert [frame["remaining_distance"] for frame in frames] == list(reversed(range(len(frames))))
    assert frames[-1]["stop_target"] == "STOP"
    for before, after in pairwise(frames):
        assert before["receipt"]["selected_after"] == after["selected"]
        assert before["next_edit"] in before["legal_actions"]
        assert before["stop_target"] == "CONTINUE"
    for frame in frames:
        wrong = frame["counterfactual"]
        assert wrong["action"] != frame["next_edit"]
        assert not wrong["receipt"]["accepted"] or wrong["receipt"]["remaining_after"] >= frame["remaining_distance"]
    manifest = export_trajectories([row], tmp_path)
    sft = json.loads((tmp_path / "train_sft_alternate.jsonl").read_text())
    rlt = json.loads((tmp_path / "train_rlt_alternate.jsonl").read_text())
    assert manifest["presentations_per_root"] == 1
    for field in ("root", "source_sha256", "presentation_sha256"):
        assert sft["metadata"][field] == rlt[field]


def test_typed_membrane_recomputes_constraint_and_rejects_widening():
    for seed in range(20):
        world = make_task("revision", seed)
        ineligible = next((item for item in world["items"] if not item["eligible"]), None)
        if ineligible:
            break
    verifier = RepairVerifier(world)
    proposal = typed_proposal({"op": "add", "id": ineligible["id"]})
    proposal["claimed_provenance"] = "env_sound_dead"
    receipt = verifier.refine(proposal, [])
    assert not receipt["accepted"]
    assert receipt["selected_after"] == []
    assert receipt["veto_violations"] == ["ineligible"]
    assert "ineligible" in evaluate_subset(world, [ineligible["id"]])["violations"]
    assert verifier.constraint_checks == 2
    proposal["domain_proposed"] = ["excluded", "included", "unknown"]
    assert verifier.refine(proposal, [])["reason"] == "not_a_singleton_refinement"
    assert verifier.checks == 2


def test_matched_cpu_fits_are_deterministic_and_charge_each_retry():
    trajectories = [training_trajectory(family, 20000) for family in ("budget", "revision", "coverage")]
    models, control = fit(trajectories, "ldt_proxy", DEFAULT, 1101, 16, 8192)
    _, repeat = fit(trajectories, "ldt_proxy", DEFAULT, 1101, 16, 8192)
    _, direct = fit(trajectories, "direct", DEFAULT, 1101, 16, 8192)
    assert control["checkpoint_sha256"] == repeat["checkpoint_sha256"]
    assert direct["action_matmul_mac_budget"] == control["action_matmul_mac_budget"]
    assert direct["action_matmul_macs"] <= direct["action_matmul_mac_budget"]
    assert direct["action_updates"] > control["action_updates"]
    assert control["final_action_loss"] < control["initial_action_loss"]
    row = rollout(models, "ldt_proxy", DEFAULT, "revision", 30000)
    assert row["metrics"]["verifier_calls"] == len(row["events"])
    assert row["metrics"]["checked_refinements"] == sum(event["action"]["op"] != "stop" for event in row["events"])
    assert row["metrics"]["action_iterations"] == 4 * row["metrics"]["score_evaluations"]
    assert row["metrics"]["unsafe_additions"] == 0


def test_grammar_and_safety_precede_development_selection():
    with pytest.raises(ValueError, match="registered"):
        revise(DEFAULT, {"arm": "trm", "field": "train_depth", "value": 3})
    with pytest.raises(ValueError, match="does not apply"):
        revise(DEFAULT, {"arm": "trm", "field": "feedback", "value": True})
    score = dict.fromkeys(("unsafe_additions", "new_violation_additions", "inapplicable_actions"), 0)
    assert not improves(score, {**score, "unsafe_additions": 1, "solved_and_stopped": 9})[0]


def test_freeze_requires_matching_development_checkpoint(tmp_path, monkeypatch):
    # Keep the test independent of process-wide address-space limits.
    monkeypatch.setattr("MeTTa.ram_hillclimb_trajectory.install_memory_ceiling", lambda _: {})
    args = dict(train_roots=1, dev_roots=1, steps=16, limit=8192, proposals=[])
    development = tmp_path / "development"
    report = run(development, **args, freeze=False)
    assert not report["conflict_transfer_executed"]
    assert not (development / "frozen_conflict_report.json").exists()
    assert all(":conflict:" not in name for name in report["development_root_names"])
    assert all(":conflict:" not in row["root"] for row in report["train_sources"])
    frozen = tmp_path / "frozen"
    transfer = run(frozen, **args, freeze=True, expected=development / "report.json")
    assert transfer["champion_checkpoints"] == report["champion_checkpoints"]
    receipts = [json.loads(line) for line in (frozen / "frozen_conflict_receipts.jsonl").read_text().splitlines()]
    assert len(receipts) == 6
    assert all(row["family"] == "conflict" for row in receipts)
