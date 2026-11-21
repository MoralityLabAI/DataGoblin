"""Source-only checks for native Lean and MCP campaign design and safe root generation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from datagoblin.training_gym.rlt_gym.external_checked_exit import digest
from datagoblin.training_gym.rlt_gym.live_lean_exit_eval import SPLITS as LEAN_SPLITS
from datagoblin.training_gym.rlt_gym.live_lean_exit_eval import _world
from datagoblin.training_gym.rlt_gym.live_mcp03_eval import (
    EXPECTED_HYBRID_COMMIT,
    _source_for_symbol,
    _tree_sha,
    _write_world,
)
from datagoblin.training_gym.rlt_gym.live_mcp03_eval import SPLITS as MCP_SPLITS


def test_live_lean_generator_is_disjoint_and_menu_bound() -> None:
    worlds = [_world(split, i) for split in LEAN_SPLITS for i in range(3)]
    assert len(worlds) == 12
    assert len({x["root_id"] for x in worlds}) == 12
    assert len({digest(x["task"]) for x in worlds}) == 12
    assert all(set(x["menu"]) == {1, 2} for x in worlds)
    assert all("sorry" not in x["menu"][2] for x in worlds)
    assert all(x["task"]["statement"].startswith("∀ (n : Nat)") for x in worlds)
    assert [_world("train", i) for i in range(3)] == worlds[:3]


def test_mcp_source_freeze_and_symbol_source_judge(tmp_path: Path) -> None:
    frozen = tmp_path / "source"
    frozen.mkdir()
    root = _write_world(frozen, split="ordinary_test", index=0)
    assert root["repository_sha"] == _tree_sha(frozen)
    correct = _source_for_symbol(frozen, root["target_uri"])
    assert "return value +" in correct
    assert hashlib.sha256(correct.encode()).hexdigest() == root["target_source_sha256"]
    wrong = _source_for_symbol(frozen, root["menu"][1])
    assert "return value -" in wrong
    assert root["menu"][1] != root["target_uri"]
    assert len(EXPECTED_HYBRID_COMMIT) == 40
    assert len(MCP_SPLITS) == 4

    # A new source file changes the frozen tree identity.
    (frozen / "src" / "extra.py").write_text("def injected():\n    return 1\n")
    assert _tree_sha(frozen) != root["repository_sha"]


def test_mcp_source_escape_and_unrecognized_symbol_rejected(tmp_path: Path) -> None:
    frozen = tmp_path / "source"
    frozen.mkdir()
    root = _write_world(frozen, split="train", index=0)
    with pytest.raises(ValueError, match="escapes"):
        _source_for_symbol(frozen, "../outside.py::malicious")
    with pytest.raises(ValueError, match="not found uniquely"):
        _source_for_symbol(frozen, "src/transform.py::missing")
    assert _write_world(frozen, split="train", index=0)["target_uri"] == root["target_uri"]
