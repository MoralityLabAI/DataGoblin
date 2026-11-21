"""Offline tests for live Lean/MCP source generators and stable certificate IDs.

The native CI jobs separately verify actual Lean compiler and MCP-03 calls.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

from datagoblin.training_gym.rlt_gym.external_checked_exit import digest
from datagoblin.training_gym.rlt_gym.live_lean_kernel import (
    LeanNativeSession,
    _generate,
    _task,
)
from datagoblin.training_gym.rlt_gym.live_mcp03 import _create_root


def test_lean_public_tasks_and_tactic_menu_are_disjoint_by_root():
    roots = _generate(8)
    assert len(roots) == 32
    assert len({digest(task) for task, _, _ in roots}) == 32
    assert {split for _, _, split in roots} == {
        "train", "calibration", "ordinary_test", "ood_test",
    }
    by_case = {task["proof_case"] for task, _, _ in roots}
    assert by_case == {"zero_add", "self_eq", "nat_comm", "and_true"}
    for task, menu, _ in roots:
        assert task["tactic_menu_digest"] == digest(menu)
        assert len(menu) == 3
        assert all("sorry" not in name for name in menu)


def test_multiline_lean_commutativity_proof_preserves_tactic_indent():
    task, tactics = _task(20261122, "train")
    assert task["proof_case"] == "nat_comm"
    assert "intro n\nexact Nat.add_comm n" in tactics[1]
    assert "\n  exact Nat.add_comm" not in tactics[1]


class FixtureLeanSession(LeanNativeSession):
    def _compile(self, source: Path) -> subprocess.CompletedProcess[str]:
        code = source.read_text()
        name = re.search(r"theorem (dgCheck\d+)", code)
        assert name is not None
        return subprocess.CompletedProcess(
            ["fixture"], 0, f"'{name.group(1)}' does not depend on any axioms\n", "",
        )


def test_lean_kernel_receipt_id_is_stable_on_independent_replay(tmp_path: Path):
    session = FixtureLeanSession(Path("/fake/lean"), tmp_path, {1: "7 = 7"})
    first = session.run(1, "rfl")
    certificate = session.kernel(first["state"]["sid"])
    again = session.run(1, "rfl")
    second = session.kernel(again["state"]["sid"])
    assert first == again
    assert certificate == second
    assert certificate["ok"]
    assert certificate["axioms"] == []


def test_mcp03_generated_repository_is_valid_source(tmp_path: Path):
    public, gold = _create_root(tmp_path / "first", 20261122)
    assert gold == f"src/handlers.py::{public['expected_symbol_name']}"
    assert public["hint_module"] == "src/handlers.py"
    for filename in ("src/handlers.py", "src/logging.py", "tests/test_handlers.py"):
        content = (tmp_path / "first" / filename).read_text()
        ast.parse(content)
    other, different = _create_root(tmp_path / "second", 20261123)
    assert different != gold
    assert digest(other) != digest(public)


def test_native_generators_reject_invalid_root_counts():
    with pytest.raises(ValueError, match="per_split"):
        _generate(0)
    with pytest.raises(ValueError, match="per_split"):
        _generate(33)
