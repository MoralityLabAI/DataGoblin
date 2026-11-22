"""Exact-oracle and split-custody tests for synthetic MeTTa looped SFT."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from MeTTa.collect import scan_forms, sha256_bytes
from MeTTa.looped_sft import audit_bundle, checked_training_row, make_bundle, prompt, root_id, split_for
from MeTTa.sft_oracle import (
    FAMILIES,
    edit,
    evaluate_subset,
    exact_optimum,
    first_state,
    make_task,
    next_edit,
    replay,
)


@pytest.mark.parametrize("family", FAMILIES)
def test_exhaustive_oracle_selects_best_feasible_evidence(family: str) -> None:
    for seed in range(4):
        task = make_task(family, seed)
        assert task == make_task(family, seed)
        result = exact_optimum(task)
        selected = result["selected"]
        assert evaluate_subset(task, selected)["valid"] is True
        assert evaluate_subset(task, selected)["rank"] == result["value"]
        items = [x["id"] for x in task["items"]]
        assert result["checked_subsets"] == sum(
            1 for k in range(task["max_items"] + 1) for _ in itertools.combinations(items, k)
        )
        # Independent enumeration of candidate scores in the test harness
        # checks the oracle's maximality and deterministic tie-break.
        alternatives = [
            tuple(combo)
            for k in range(task["max_items"] + 1)
            for combo in itertools.combinations(items, k)
            if evaluate_subset(task, combo)["valid"]
        ]
        best = min(alternatives, key=lambda c: (
            -evaluate_subset(task, c)["rank"][0],
            -evaluate_subset(task, c)["rank"][1],
            -evaluate_subset(task, c)["rank"][2],
            c,
        ))
        assert tuple(selected) == best


@pytest.mark.parametrize("family", FAMILIES)
def test_metta_program_roundtrips_as_lexical_facts(family: str) -> None:
    task = make_task(family, 11)
    forms = scan_forms(task["program"])
    assert forms[-1][2] == task["query"]
    assert any(f[2].startswith("(Provides ") for f in forms)
    assert len([f for f in forms if f[2].startswith("(Candidate ")]) == len(task["items"])
    assert task["program"] == make_task(family, 11)["program"]


@pytest.mark.parametrize("family", FAMILIES)
def test_recurrent_trajectory_follows_one_checked_edit_per_visit(family: str) -> None:
    task = make_task(family, 31)
    oracle = exact_optimum(task)
    initial = first_state(task, 31)
    trace = replay(task, initial, oracle["selected"])
    assert trace[0]["selected"] == initial
    assert trace[-1]["selected"] == oracle["selected"]
    assert trace[-1]["action"] == {"op": "stop", "id": "none"}
    assert all(a["remaining_canonical_edits"] == b["remaining_canonical_edits"] + 1
               for a, b in zip(trace, trace[1:]))
    assert all(next_edit(t["selected"], oracle["selected"]) == t["action"] for t in trace)
    assert edit(oracle["selected"], {"op": "stop", "id": "none"}) == oracle["selected"]


def test_looped_bundle_is_deterministic_and_unlabeled_reserves(tmp_path: Path) -> None:
    first = make_bundle(tmp_path / "first", roots_per_family=16)
    second = make_bundle(tmp_path / "second", roots_per_family=16)
    assert first["root_count"] == 16 * len(FAMILIES)
    assert first["files"] == second["files"]
    assert first["policy"]["native_metta_verified"] is False
    assert first["policy"]["eval_labels_written"] == 0
    assert first["max_presentations_per_root"] <= 4
    assert first["continue_targets"] > 0
    assert first["stop_targets"] > 0
    assert first["split_roots"]["ood_reserved"] == 16
    assert first["training_presentations"] > first["split_roots"]["train_candidate"]
    audit = audit_bundle(tmp_path / "first", full_replay=True)
    assert audit["valid"] and audit["training_ready"] is False

    seen = {}
    for line in (tmp_path / "first" / "train_candidates.jsonl").read_text().splitlines():
        row = json.loads(line)
        seen.setdefault(row["root"], set()).add(row["presentation"])
        assert row["split"] == "train_candidate"
        assert row["evidence_type"] == "SIMULATED"
        assert row["training_ready"] is False
        assert "verifier_receipt" in row
        assert len(row["messages"]) == 2
        assert row["preference"]["chosen"] != row["preference"]["rejected"]
        assert row["recurrent_supervision"]["synthetic_trace"] is True
        assert row["verifier_receipt"]["target_matches_first_canonical_edit"] is True
    assert all(1 <= len(v) <= 4 for v in seen.values())

    for line in (tmp_path / "first" / "reserved_public.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert row["split"] != "train_candidate"
        assert row["teacher_labels_included"] is False
        assert row["training_ready"] is False
        assert "messages" not in row
        assert "preference" not in row
        assert "oracle" not in row
        assert "verifier_receipt" not in row


def test_heldout_family_is_disjoint_from_train() -> None:
    assert all(split_for("conflict", seed) == "ood_reserved" for seed in range(100))
    assert all(root_id(f, n) != root_id(g, n) for f in FAMILIES for g in FAMILIES if f != g
               for n in range(5))


def test_teacher_action_rejects_noncanonical_move() -> None:
    task = make_task("dependency", 5)
    oracle = exact_optimum(task)
    selected = first_state(task, 5)
    action = {"op": "stop", "id": "none"}
    if next_edit(selected, oracle["selected"]) == action:
        pytest.skip("seed starts at optimum")
    with pytest.raises(ValueError, match="canonical next action"):
        checked_training_row(task, root_id("dependency", 5), {
            "selected": selected, "action": action, "step": 0,
        }, oracle)


def test_audit_rejects_forged_teacher_even_with_updated_content_hash(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    make_bundle(output, roots_per_family=8)
    path = output / "train_candidates.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows
    row = rows[0]
    row["recurrent_supervision"]["canonical_edits_remaining_after"] = 777
    path.write_text("\n".join(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                              for x in rows) + "\n")
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"]["train_candidates.jsonl"] = sha256_bytes(path.read_bytes())
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="oracle target"):
        audit_bundle(output, full_replay=True)


def test_checked_training_row_keeps_answer_out_of_prompt() -> None:
    task = make_task("coverage", 12)
    oracle = exact_optimum(task)
    initial = first_state(task, 12)
    event = replay(task, initial, oracle["selected"])[0]
    row = checked_training_row(task, root_id("coverage", 12), event, oracle)
    user = json.loads(row["messages"][0]["content"])
    assert user == prompt(task, initial)
    assert "optimal_subset" not in row["messages"][0]["content"]
    assert "canonical_edits_remaining" not in row["messages"][0]["content"]
    assert row["preference"]["chosen"] == json.loads(row["messages"][1]["content"])


def test_cannot_write_candidate_labels_inside_repo_source(tmp_path: Path) -> None:
    from MeTTa.collect import ROOT

    with pytest.raises(ValueError, match="outside committed source"):
        make_bundle(ROOT / "MeTTa" / "generated", roots_per_family=2)
    with pytest.raises(ValueError, match="outside committed source"):
        make_bundle(ROOT / "training" / "metta", roots_per_family=2)
