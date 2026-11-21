"""Check strategic references, custody and external exchange contracts."""
import json
from hashlib import sha256
from pathlib import Path

import pytest

from datagoblin.training_gym.rlt_gym.research_benchmarks import format_training, import_evaluation
from datagoblin.training_gym.rlt_gym.research_campaign import case, collect, reference, replay
from datagoblin.training_gym.rlt_gym.research_controller import POLICIES, abstract, fit, trial
from datagoblin.training_gym.rlt_gym.research_teacher import score_reply
from datagoblin.training_gym.rlt_gym.research_world import DEPTHS, Game, enumerated_likelihood, message_likelihood


@pytest.mark.parametrize("depth", DEPTHS)
def test_reference_and_public_successors(depth):
    root, sealed = case("ordinary_test", depth, 53, 123, None)
    game = Game(root["public"])
    assert len(root["public"]["world_bits"]) == 1024
    assert game.initial.turns == 4
    exact = game.values(game.initial)
    independent = reference(game, game.initial, game.initial_masses, 4)
    assert exact == {name: float(value) for name, value in independent.items()}
    if depth <= 4:
        assert message_likelihood(game.public) == enumerated_likelihood(game.public)
    state, reward = game.outcome(game.initial, "pilot", sealed)
    assert reward == -1
    assert state.history == (("pilot", sealed["experiment_tape"]["pilot"][0]),)
    assert state.turns == 3


def test_observational_alias_is_split_by_intervention():
    task, _ = case("ordinary_test", 4, 53, 32, None)
    public = task["public"]
    public["origin_p16"] = [7, 7, 7]
    for actor in public["views"]:
        actor["signal_p16"] = [8, 8, 8]
    likelihood = message_likelihood(public)
    assert len(set(likelihood)) == 1
    game = Game(public)
    action = f"intervene:{public['extension']['mode']}"
    targets = game.experiments[action]["p16"]
    assert len(set(targets)) == 2
    assert game.legal(game.initial).count(action) == 1
    assert action not in game.native


def test_replayed_training_ledger_and_formatted_chat(tmp_path):
    key = bytes(range(32))
    archive = tmp_path / "collection"
    collect(archive, 5, (0, 12), 23, key)
    assert replay(archive, key)["verified_rollouts"] == 300
    with pytest.raises(ValueError, match="commitment"):
        replay(archive)
    format_training(archive, tmp_path / "chat")
    root_counts = {}
    for stream in (tmp_path / "chat").glob("*.jsonl"):
        for line in stream.read_bytes().splitlines():
            row = json.loads(line)
            assert row["split"] == "train"
            assert row["messages"][-1]["role"] == "assistant"
            assert "experiment_tape" not in row["messages"][1]["content"]
            root_counts[row["root"]] = root_counts.get(row["root"], 0) + 1
    assert max(root_counts.values()) <= 4
    with (archive / "scored.jsonl").open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(ValueError, match="digest"):
        replay(archive, key)


def test_trial_budgets_and_model_parse():
    task, sealed = case("ordinary_test", 8, 53, 210, None)
    game = Game(task["public"])
    actions, features, heuristic = abstract(game)
    weights, presentations = fit(bytes(range(32)), 1)
    assert presentations <= len(DEPTHS) * 4
    values = game.values(game.initial)
    for name in POLICIES:
        row = trial(name, actions, features, heuristic, game.native, values, weights, task["root"])
        assert row["verifier_calls"] <= 3
        assert row["visits"] <= 4
        assert row["action"] in values
    _, failed = score_reply(game, game.initial, sealed, {"status": "received", "text": '{"action":"irrelevant"}'})
    assert failed["status"] == "parse_error"
    future, passed = score_reply(game, game.initial, sealed, {"status": "received", "text": '{"action":"pilot"}'})
    assert passed["status"] == "valid"
    assert future.history


def test_external_labels_are_separate(tmp_path: Path):
    data = {"id": "example", "domain": "research", "query": "Question?", "prefix_text": "Prior observation",
            "choices": ["measure", "guess"], "answer": "A", "answer_index": 0}
    blob = (json.dumps(data) + "\n").encode()
    path = tmp_path / "data.jsonl"
    path.write_bytes(blob)
    out = tmp_path / "out"
    import_evaluation("tastebench", path, out, "v1.0", sha256(blob).hexdigest())
    labels = [json.loads(line) for line in (out / "eval_labels.jsonl").read_bytes().splitlines()]
    assert [row["target"]["answer"] for row in labels] == ["A", "B"]
    assert b'"target"' not in (out / "eval_inputs.jsonl").read_bytes()
    assert not json.loads((out / "evaluation_manifest.json").read_bytes())["training_eligible"]
