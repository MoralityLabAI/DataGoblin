import json
import os
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest

from datagoblin.hashing import sha256_file
from datagoblin.training_gym.rlt_gym.external_checked_exit import digest
from datagoblin.training_gym.rlt_gym.hard02e_list_stop import generate, rule_candidate
from datagoblin.training_gym.rlt_gym.lean_goose import ROOTS, SCHEMA, episode, parse_tactic, probe, proposed_action
from datagoblin.training_gym.rlt_gym.lean_goose_replay import _first_turns


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


def test_plan_replay_binds_complete_registry_and_model_receipts(tmp_path):
    transitions = tmp_path / "transitions.jsonl"
    manifest = tmp_path / "SHA256.json"
    revision = digest([asdict(root) for root in ROOTS])
    model = {"answer": "intro a b h; exact h.symm", "max_tokens": 64, "prompt_token_parity": True,
             "preflight_prompt_tokens": 10, "provider": {"usage": {"input_tokens": 10}}}
    receipt = {**model, "request_sha256": digest({"prompt": "test", "max_tokens": 64}),
               "response_sha256": digest(model)}
    rows = [{**asdict(root), "visit": 0, "schema": SCHEMA, "root_id": digest(asdict(root)),
             "source_revision_sha256": revision, "goals_before": ["test goal"], "proof_before": [],
             "prompt": "test", "model": receipt} for root in ROOTS]

    def write_archive():
        transitions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest.write_text(json.dumps({transitions.name: sha256_file(transitions)}), encoding="utf-8")

    write_archive()
    assert len(_first_turns(tmp_path)) == len(ROOTS)
    rows[0]["model"] = {**receipt, "answer": "exact True.intro"}
    write_archive()
    with pytest.raises(ValueError, match="model request or response checksum"):
        _first_turns(tmp_path)


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


@pytest.mark.skipif(not os.environ.get("DATAGOBLIN_LEAN_412"), reason="requires native Lean 4.12.0")
def test_registered_list_menu_uses_core_lemmas(tmp_path):
    lean = Path(os.environ["DATAGOBLIN_LEAN_412"]).resolve()
    root = next(root for root in generate() if root.split == "ood_test")
    target = rule_candidate(root)
    assert target is not None
    for idx, candidate in enumerate(root.menu):
        receipt = probe(lean, root.statement, [root.prefix, *candidate.splitlines()], tmp_path / f"Test{idx}.lean")
        assert (receipt["status"] == "closed") == (idx == target)
