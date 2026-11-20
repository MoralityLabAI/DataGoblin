import pytest

from datagoblin.training_gym.rlt_lean_composition import POOL_LIMIT, expanded_candidates
from datagoblin.training_gym.rlt_lean_pool_audit import pool_repl_class, train_templates
from datagoblin.training_gym.rlt_lean_pool_audit import test_templates as held
from datagoblin.training_gym.rlt_lean_pool_export import checked_rows


def test_composition_uses_shared_lexicon_and_bounded_tactics():
    names = ("Nat.gcd_assoc", "Nat.gcd_comm", "Nat.gcd_dvd_left", "List.reverse_take")
    entries = [{"name": name} for name in names]
    hits, pool = expanded_candidates("⊢ Nat.gcd (Nat.gcd a b) c = Nat.gcd a (Nat.gcd c b)", entries)

    assert hits["wider"] == hits["compose"]
    assert "List.reverse_take" not in hits["compose"]
    assert "rw [Nat.gcd_assoc] <;> congr 2 <;> apply Nat.gcd_comm" in pool["compose"]
    for items in pool.values():
        assert len(items) == len(set(items)) <= POOL_LIMIT
        assert all(len(t.encode("utf-8")) < 96 for t in items)


def test_held_compositions_are_separate_from_training_shapes():
    assert {f for f, _, _ in train_templates(30000)}.isdisjoint(f for f, _, _ in held(31000))


def test_pool_repl_restores_exact_goal_after_a_tactic_timeout():
    class LeanError(RuntimeError):
        def __init__(self, code, message):
            super().__init__(message)
            self.code, self.message = code, message

    class Repl:
        def __init__(self):
            self.sid = 301
            self.initialized = []

        def init(self, stmt, pp):
            self.initialized.append((stmt, pp))
            return {"sid": self.sid}

        def restart(self):
            self.sid += 1

        def request(self, *, op, sid, tactic):
            assert op == "run" and sid == self.sid
            if tactic == "slow":
                raise LeanError("lean_timeout", "server exhausted")
            if tactic == "limit":
                raise LeanError("lean_request_error", "maximum recursion depth has been reached")
            if tactic == "malformed":
                raise LeanError("lean_request_error", "unknown statement ID")
            return {"ok": True}

    lean = pool_repl_class(Repl, LeanError)()
    root = lean.init("True", True)["sid"]
    assert lean.request(op="run", sid=root, tactic="slow") == {"ok": False, "error": "Lean timed out: server exhausted"}
    assert lean.initialized == [("True", True), ("True", True)]
    assert not lean.request(op="run", sid=root, tactic="limit")["ok"]
    assert lean.request(op="run", sid=root, tactic="fast")["ok"]
    with pytest.raises(LeanError, match="unknown statement ID"):
        lean.request(op="run", sid=root, tactic="malformed")


def test_checked_export_separates_proposals_from_unknown_and_verified_outcomes():
    trace = {
        "goal": {"id": "goal", "lineage": "family", "family": "family", "statement": "True"},
        "goal_pp": "⊢ True", "retrieved": {}, "pools": {"compose": ["first", "second"]}, "native": None,
        "retrieval_comparisons": 0, "generation_wall_s": 0.0,
        "outcomes": [
            {"tactic": "first", "closed": True, "kernel_ok": True, "kernel_calls": 1, "axioms": [], "label": 1},
            {"tactic": "second", "closed": False, "kernel_calls": 0, "label": None},
        ],
    }
    public, private = checked_rows(trace, {"lean_sha256": "image"})
    assert public["epoch"] == private["epoch"]
    first, second = public["orders"]["compose"]
    assert "outcomes" not in public and "label" not in public["candidates"][0]
    assert private["outcomes"][first]["label"] == 1
    assert private["outcomes"][second]["label"] is None
    trace["outcomes"][1]["label"] = 1
    with pytest.raises(ValueError, match="inconsistent"):
        checked_rows(trace, {"lean_sha256": "image"})
