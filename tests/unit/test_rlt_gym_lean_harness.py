from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from datagoblin.hashing import sha256_file
from datagoblin.training_gym.rlt_gym import lean_harness as harness


def write_rows(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_original_goal_join_and_matched_budget_recovery(tmp_path):
    states = tmp_path / "states.jsonl.gz"
    goal_file = tmp_path / "dev.jsonl.gz"
    autos = [f"auto{i}" for i in range(8)]
    past = ["learned_success", "bad_axiom"]
    receipt = {"ok": True, "goals_after": 0, "cert": {"kernel": True, "axioms": []}}
    results = dict.fromkeys([*autos, *past], {"ok": False})
    results[past[0]] = receipt
    results[past[1]] = receipt | {"cert": {"kernel": True, "axioms": ["sorryAx"]}}
    row = {"file": "Mathlib/Foo.lean", "line": 1, "col": 2, "split": "dev", "decl": "foo",
           "goals": 1, "reached": True, "policies": {"auto": autos, "human": ["reference"],
                                                     "cached": [*past, "reference"]}, "results": results}
    write_rows(states, [row, row | {"line": 2, "results": results | {autos[0]: receipt}}])
    goal = {"file": row["file"], "line": 1, "col": 2, "decl": row["decl"], "checker": "certified",
            "kind": "tactic", "goals_before": ["⊢ True"], "goals_after": [], "tactic": "reference"}
    write_rows(goal_file, [goal | {"terminal": False, "goals_before": ["wrong"]}, goal, goal | {"line": 2}])
    digest = hashlib.sha256()
    for line in (1, 2):
        digest.update(json.dumps([row["file"], line, 2]).encode() + "⊢ True".encode() + b"\0")

    def aleph(packet):
        assert packet["goals"] == ["⊢ True"]
        assert "reference" not in json.dumps(packet)
        return {"content": "[6]"}

    def bonsai(packet):
        if packet["max_choices"] == 2:
            assert packet["failed_choices"] == [{"id": 6, "tactic": "bad_axiom"}]
            assert 6 not in {item["id"] for item in packet["menu"]}
        return {"content": "[5]"}

    kwargs = {"expected_sha256": sha256_file(states), "max_goals": 10,
              "models": {"aleph": aleph, "bonsai": bonsai}, "model_ids": {"aleph": "fixture", "bonsai": "fixture"},
              "context_dir": tmp_path, "encoder_sha256": digest.hexdigest()}
    result = harness.run(states, tmp_path / "out", **kwargs)
    assert result["counts"]["automation8"] == 1
    assert result["counts"]["automation8_miss_with_menu_witness"] == 1
    assert result["counts"]["aleph_close8"] == 1
    assert result["counts"]["bonsai_close8"] == 2
    assert result["counts"]["aleph_then_bonsai_rescues"] == 1
    assert result["counts"]["aleph_then_bonsai_checks"] == 2
    assert result["training_eligible"] is False
    with pytest.raises(ValueError, match="encoder digest"):
        harness.run(states, tmp_path / "bad", **(kwargs | {"encoder_sha256": "bad"}))


@pytest.mark.parametrize("content", ["[1, 1]", "[true]", "[3]", "[1]; sorry", "[1, 2]"])
def test_ranker_rejects_out_of_budget_and_freeform_actions(content):
    assert harness.rank_ids(content, {1, 2}, limit=1) == []
