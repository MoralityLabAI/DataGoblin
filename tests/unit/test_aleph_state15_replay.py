"""Regression checks for machine-independent archival witnesses and closed theorem audits."""

import hashlib
import json
from pathlib import Path

import pytest

STAGE = Path(__file__).resolve().parents[2] / "training/aleph_next/state01"


def test_machine_independent_archive(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(STAGE))
    import state15_replay
    import state15_state11

    state15_state11.regenerate(tmp_path)
    for name, sha in state15_replay.PINS[11].items():
        raw = (tmp_path / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha
        assert b"\r" not in raw


def test_require_each_axiom_audit_exactly_once(monkeypatch):
    monkeypatch.syspath_prepend(str(STAGE))
    from state15_kernel import proof_axioms

    good = "'p' depends on axioms: [propext, Quot.sound]\n'q' does not depend on any axioms\n"
    assert proof_axioms(good, ["p", "q"]) == (True, {"p": ["Quot.sound", "propext"], "q": []})
    assert not proof_axioms(good, ["q"])[0]
    assert not proof_axioms(good, ["p", "q", "r"])[0]
    assert not proof_axioms(good + good, ["p", "q"])[0]
    assert not proof_axioms(good.replace("propext", "sorryAx"), ["p", "q"])[0]


def test_balanced_compilation_preserves_every_step_and_statement(monkeypatch):
    monkeypatch.syspath_prepend(str(STAGE))
    from state15_balanced import build

    term = (
        "theorem chain (x y z : Nat) : ((x + 0) + 0) = x := by\n"
        "  calc\n"
        "    ((x + 0) + 0) = (x + 0) := Nat.add_zero _\n"
        "    (x + 0) = x := Nat.add_zero _\n"
    )
    source, names, origin = build("chain", term)
    assert origin["original_steps"] == 2
    assert len(names) == 4 and len(set(names)) == 4
    assert source.count(":= Nat.add_zero _") == 2
    assert term.split(" := by")[0] + " := " in source
    assert source.count("Eq.trans") == 1
    assert source.count("#print axioms ") == len(names)
    with pytest.raises(ValueError, match="disconnected"):
        build("chain", term.replace("    (x + 0) = x", "    y = x"))
    with pytest.raises(ValueError, match="different goal"):
        build("chain", term.replace("= x := by", "= y := by"))


def test_shard_audit_recomputes_log_and_detects_a_missing_control(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(STAGE))
    import state15_shards as s

    tasks = [("p", 10, "raw_positive", ["p"], ["p"], "theorem p : True := True.intro\n", {})]
    monkeypatch.setattr(s, "plan", lambda prepared: tasks)
    monkeypatch.setattr(s, "EXPECTED", {10: 1})
    (tmp_path / "PREPARED.json").write_bytes(b"{}")
    s.write_json(
        tmp_path / "shard-0.json",
        {
            "index": 0,
            "count": 1,
            "jobs": ["p"],
            "toolchain": {
                "version": f"Lean (version {s.PIN}, fixture, commit {s.COMMIT}, Release)",
                "executable_sha256": "a" * 64,
            },
        },
    )
    log = b"'p' does not depend on any axioms\n"
    (tmp_path / "p.log").write_bytes(log)
    metadata = {
        "key": "p",
        "stage": 10,
        "kind": "raw_positive",
        "index": 0,
        "roots": ["p"],
        "origin": {},
        "source_sha256": s.digest(tasks[0][5].encode()),
        "log_sha256": s.digest(log),
        "wall_seconds": 1.0,
        "cpu_seconds": 0.75,
        "timed_out": False,
        "exit_code": 0,
        "passed": True,
        "axioms": [],
        "audited_declarations": 1,
    }
    (tmp_path / "p.json").write_text(json.dumps(metadata))
    result = s.collect(tmp_path, tmp_path, 1)
    assert result["positive_declarations_verified"] == 1 and not result["kernel_verified"]
    (tmp_path / "p.log").write_bytes(log.replace(b"does not depend on any axioms", b"depends on axioms: [sorryAx]"))
    with pytest.raises(ValueError, match="hash mismatch"):
        s.collect(tmp_path, tmp_path, 1)
