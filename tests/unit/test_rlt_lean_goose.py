import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from datagoblin.training_gym.rlt_gym.lean_goose import ROOTS, episode, parse_tactic, probe, proposed_action


@pytest.mark.parametrize(
    ("answer", "action"),
    [
        (" exact Eq.symm h\n", "exact Eq.symm h"),
        ("rw [← h]", "rw [← h]"),
        ("STOP", "STOP"),
        ("```lean\nrfl\n```", None),
        ("exact by sorry", None),
        ("exact sorryAx True true", None),
        ("rfl\n#eval 1", None),
        ("constructor <;> assumption", None),
    ],
)
def test_tactics_stay_within_the_host_grammar(answer, action):
    assert parse_tactic(answer) == action


def test_tactic_list_spends_one_action_and_cannot_hide_a_command():
    assert proposed_action("constructor; exact h.2; exact h.1") == "constructor"
    assert proposed_action("rfl;\n#eval true") is None


@pytest.mark.skipif(not os.environ.get("DATAGOBLIN_LEAN_412"), reason="requires native Lean 4.12.0")
def test_native_open_transition_and_sft_custody(tmp_path):
    lean = Path(os.environ["DATAGOBLIN_LEAN_412"]).resolve()
    root = ROOTS[0]
    for registered in ROOTS:
        assert probe(lean, registered.statement, [], tmp_path / "Root.lean")["status"] == "open"
    assert probe(lean, root.statement, ["rfl"], tmp_path / "Rejected.lean")["status"] == "rejected"

    def model(*tactics):
        engine = Mock()
        engine.infer.side_effect = [
            {"answer": tactic, "prompt_token_parity": True, "response_sha256": str(i)}
            for i, tactic in enumerate(tactics)
        ]
        return engine

    trained = episode(model("intro p q hp h", "exact h hp"), lean, root, tmp_path,
                      attempts=2, source_revision="test")
    assert trained["status"] == "kernel_certified"
    assert trained["sft_rows"] == 2
    assert (tmp_path / trained["certificate"]["path"]).is_file()
    sft = [json.loads(line) for line in (tmp_path / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["messages"][-1]["content"] for row in sft] == ["intro p q hp h", "exact h hp"]

    held = episode(model("intro p q r hp", "exact Or.inl (Or.inr hp)"), lean, ROOTS[3], tmp_path,
                   attempts=2, source_revision="test")
    assert held["status"] == "kernel_certified"
    assert held["sft_rows"] == 0
    assert len((tmp_path / "sft.jsonl").read_text(encoding="utf-8").splitlines()) == 2
