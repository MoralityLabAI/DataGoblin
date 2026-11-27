from argparse import Namespace

import pytest

from Agonist.tinker_gym import Journal, Task, canonical, episode, fixture, make_tasks, parse_action
from Agonist.tinker_live import active, advisory_registry, evaluate, train


def test_fixture_deterministic_and_disjoint(tmp_path):
    a = make_tasks(17, 2, "train")
    b = make_tasks(17, 2, "dev")
    assert len(a) == len(b) == 8
    assert a == make_tasks(17, 2, "train")
    assert {x.root for x in a}.isdisjoint(x.root for x in b)
    for task in a:
        assert task.grade(task.gold)["success"]
        assert not task.grade("invalid")["valid"]
        assert parse_action(task.demonstration()["messages"][1]["content"]) == task.gold
    with pytest.raises(ValueError):
        b[0].demonstration()
    result = fixture(tmp_path, 17, 2)
    assert result["passed"] == result["episodes"] == 8
    assert result == fixture(tmp_path, 17, 2)


def test_ledger_replay_stop_cap_and_unknown_outcome(tmp_path):
    j = Journal(tmp_path, "clean")
    calls = []
    def invoke():
        calls.append(1)
        return {"ok": True}
    assert j.call("a", "tinker", ["x"], 5, 10, invoke)["ok"]
    assert j.call("a", "tinker", ["x"], 5, 10, invoke)["ok"]
    assert calls == [1]
    with pytest.raises(ValueError):
        j.call("a", "tinker", ["different"], 5, 10, invoke)
    with pytest.raises(RuntimeError, match="cap"):
        j.call("b", "tinker", [], 6, 10, invoke)
    (tmp_path / "STOP").touch()
    with pytest.raises(RuntimeError, match="STOP"):
        j.call("b", "tinker", [], 1, 10, invoke)
    (tmp_path / "STOP").unlink()
    def broken():
        raise OSError("lost receipt")
    with pytest.raises(OSError):
        j.call("b", "tinker", [], 1, 10, broken)
    with pytest.raises(RuntimeError, match="unknown"):
        j.call("b", "tinker", [], 1, 10, invoke)
    with pytest.raises(RuntimeError, match="unresolved"):
        j.call("c", "tinker", [], 1, 10, invoke)


def test_episode_revision_and_advisory_isolation(tmp_path):
    task = Task("root", "train", "repair", {"state": 1}, {"hold": 0.0, "deploy": 1.0})
    clean = Journal(tmp_path / "clean", "clean")
    advice = Journal(tmp_path / "advice", "advisory")
    def advisor(key, task, history):
        return "Recompute against public evidence."
    def actor(key, task, prompt, history):
        if history:
            assert any("advisory_note" in h for h in history)
        return {"text": canonical({"action": "deploy" if history else "hold"})}
    with pytest.raises(ValueError):
        episode(task, clean, "r", actor, 2, advisor)
    with pytest.raises(ValueError):
        clean.call("x", "fable", {}, 1, 10, lambda: {})
    result = episode(task, advice, "r", actor, 2, advisor)
    assert result["success"] and result["attempts"] == 2
    assert not result["training_ready"]
    assert episode(task, advice, "r", actor, 2, advisor) == result
    with pytest.raises(ValueError):
        Journal(tmp_path / "advice", "clean")


class FakeProvider:
    def __init__(self):
        self.steps = []
        self.tasks = {}
    def train_step(self, key, tasks, state, rank, lr):
        assert all(t.split == "train" for t in tasks)
        self.steps.append((key, state, {t.key for t in tasks}))
        return {"state": key + "/state", "sampler": key + "/sampler"}
    def sample(self, key, prompt, snapshot):
        task = next(t for t in self.tasks.values() if t.prompt() == prompt)
        return {"text": canonical({"action": task.gold if snapshot else "invalid"})}


def test_training_distinct_roots_promotion_and_registry(tmp_path):
    j = Journal(tmp_path / "clean", "clean")
    fake = FakeProvider()
    args = Namespace(groups=4, steps=2, batch_size=2, rank=16, lr=5e-5,
                     curator="fixed", eval_groups=2, min_gain=0.05)
    candidate = train(j, fake, None, args, "route", 1)
    assert len(fake.steps) == 2
    assert fake.steps[0][2].isdisjoint(fake.steps[1][2])
    assert fake.steps[1][1] == fake.steps[0][0] + "/state"
    assert train(j, fake, None, args, "route", 1) == candidate
    fake.tasks = {t.key: t for t in make_tasks(1, 2, "dev") if t.role == "route"}
    promoted = evaluate(j, fake, args, "route", 1)
    assert promoted["promoted"] and promoted["gain"] == 1.0
    assert active(j, "route", 2) == candidate
    assert advisory_registry(tmp_path / "clean", 2)["route"] == candidate
    with pytest.raises(ValueError):
        train(Journal(tmp_path / "advice", "advisory"), fake, None, args, "route", 1)


def test_parse_requires_structured_action():
    assert parse_action('{"action":"STOP"}') == "STOP"
    assert parse_action('{"action":3}') is None
    assert parse_action("not JSON") is None
