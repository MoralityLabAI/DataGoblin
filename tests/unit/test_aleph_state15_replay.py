"""Regression checks for machine-independent archival witnesses and closed theorem audits."""

import hashlib
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
