"""Independent constraint-verifier veto tests for AIRIS-02 research rollouts."""
from __future__ import annotations

import numpy as np

from MeTTa.airis02_fresh import _frozen_motifs, build_row
from MeTTa.airis02_student import (
    ACTION_CLASSES,
    LinearStudent,
    _guarded_action,
    action_index,
    evaluate,
)
from MeTTa.sft_oracle import first_state, make_task, render_metta, validate_task


class RankedDummy:
    def __init__(self, first: int, second: int):
        self.first = first
        self.second = second

    def logits(self, x: np.ndarray, *, loops: int = 1) -> np.ndarray:
        result = np.full((len(x), ACTION_CLASSES), -100.0)
        result[:, self.first] = 10.0
        result[:, self.second] = 9.0
        result[:, 20] = -10.0
        return result


def test_veto_prevents_new_budget_violation_and_falls_back_to_remove():
    task = make_task("budget", 512)
    selected = first_state(task, 512)
    present = {item["id"] for item in task["items"]}
    chosen = {item["id"]: item for item in task["items"] if item["id"] in selected}
    task["token_budget"] = sum(item["cost"] for item in chosen.values())
    task["program"] = render_metta(task)
    validate_task(task)
    unselected = sorted(present - set(selected))[0]
    first = action_index({"op": "add", "id": unselected})
    fallback = action_index({"op": "remove", "id": selected[0]})
    scorer = RankedDummy(first, fallback)
    action, vetoes, checks = _guarded_action(
        scorer, task, selected, loops=1, stop_head=None
    )
    assert action == {"op": "remove", "id": selected[0]}
    assert vetoes >= 1
    assert checks >= 2
    assert first < 10 and fallback >= 10


def test_guarded_rollout_preserves_root_authority_and_zero_new_violations():
    motifs = _frozen_motifs()
    rows = [build_row("dependency", 512, "uniform", motifs),
            build_row("lineage", 512, "motif", motifs)]
    model = LinearStudent(21)
    result = evaluate(model, rows, loops=1, verifier_guard=True)
    assert result["cases"] == 2
    assert result["new_violation_additions"] == 0
    assert result["inapplicable_actions"] == 0
    assert result["verifier_vetoes"] >= 0
    assert result["verifier_queries"] >= result["verifier_vetoes"]
    assert result["solved_and_stopped"] <= result["ever_reached_optimum"]
    assert result["abstentions"] in (0, 1, 2)


def test_unassisted_rollout_remains_a_separate_baseline():
    motifs = _frozen_motifs()
    row = build_row("budget", 512, "uniform", motifs)
    scorer = LinearStudent(3)
    no_veto = evaluate(scorer, [row], loops=1)
    guarded = evaluate(scorer, [row], loops=1, verifier_guard=True)
    assert no_veto["verifier_queries"] == 0
    assert no_veto["verifier_vetoes"] == 0
    assert guarded["new_violation_additions"] == 0
    assert guarded["solved_and_stopped"] <= guarded["ever_reached_optimum"]
