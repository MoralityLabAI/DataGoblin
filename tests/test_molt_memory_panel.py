import json

import pytest

from datagoblin.training_gym.molt_episode import Episode, make_world
from datagoblin.training_gym.molt_memory_panel import (
    MODES,
    ObservationView,
    exact_public_client,
    parse_action,
    roster,
    run_panel,
)


def test_all_history_controls():
    worlds = roster(71, 8, "calibration")
    reports = {}
    for mode in MODES:
        reports[mode] = run_panel(worlds, mode, exact_public_client, max_actions=128,
                                  wall_seconds=10, identity={"role": "control"})
    for mode in ("full", "bank2"):
        assert reports[mode]["summary"]["optimal"] == 16
        assert reports[mode]["summary"]["verifier_calls"] == 8
    for mode in ("current", "recent2", "bank1"):
        assert reports[mode]["summary"]["optimal"] == 0
    assert len({r["roster_digest"] for r in reports.values()}) == 1
    assert reports["bank2"]["summary"]["prompt_bytes"] < reports["full"]["summary"]["prompt_bytes"]


def test_no_hidden_evidence_in_bank():
    world = make_world("calibration", 4, 0, "relevant")
    ep = Episode(world)
    view = ObservationView("bank2")
    view.observe(ep.start())
    for _ in world.distractors:
        view.observe(ep.step({"action": "advance"}))
    change = ep.step({"action": "advance"})
    prompt = view.observe(change)
    packet = json.loads(prompt.split("\nOBSERVATION_PACKET\n")[1].split("\nACTION\n")[0])
    assert {r["source"] for r in packet["retained_receipts"]} == {"s1"}
    assert view.invalidations == 1
    obs = ep.step({"action": "request", "source": "s0"})
    prompt = view.observe(obs)
    assert exact_public_client(prompt, 1)["text"]
    assert view.writes == 3


def test_failed_actions_and_budget_keep_denominator():
    worlds = roster(71, 2, "train")

    def broken(_prompt, _timeout):
        return {"text": '{"action":"advance","action":"submit"}'}

    result = run_panel(worlds, "full", broken, max_actions=1,
                       wall_seconds=10, identity={"role": "control"})
    assert result["summary"]["episodes"] == 4
    assert result["summary"]["optimal"] == 0
    assert result["summary"]["action_attempts"] == 1
    assert result["rows"][0]["stop_reason"] == "invalid_or_failed_action"
    assert all(r["stop_reason"] == "run_budget_exhausted" for r in result["rows"][1:])


@pytest.mark.parametrize("reply", ['{"action":NaN}', '[]', '{"a":1,"a":2}', 'x' * 16385])
def test_strict_action_json(reply):
    with pytest.raises(ValueError):
        parse_action(reply)


def test_no_protected_release():
    with pytest.raises(ValueError):
        roster(71, 2, "ordinary_test")
