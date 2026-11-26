import json

import pytest

from datagoblin.training_gym.molt_episode import Episode, exact_trace, export_train, make_world, replay


def test_paired_lineage_and_exact_replay():
    for i in range(64):
        a = make_world("train", 31, i, "relevant")
        b = make_world("train", 31, i, "unrelated")
        assert a.root_id == b.root_id and a.task_id != b.task_id
        assert a.initial == b.initial and a.candidates == b.candidates
        for world, calls in ((a, 1), (b, 0)):
            ep = exact_trace(world)
            assert ep.result["verified"] and ep.result["optimal"]
            assert ep.result["verifier_calls"] == calls
            again = replay(world, [r["action"] for r in ep.trace if "action" in r])
            assert again.trace == ep.trace
        assert a.root_id != make_world("calibration", 31, i, "relevant").root_id


def test_stale_frame_and_receipts():
    world = make_world("train", 5, 1, "relevant")
    for stale_frame in (True, False):
        ep = Episode(world)
        start = ep.start()
        assert "replacement" not in start
        for _ in world.distractors:
            ep.step({"action": "advance"})
        change = ep.step({"action": "advance"})
        assert "excluded" not in change
        assert change["frame_id"] != start["frame_id"]
        result = ep.step({"action": "submit", "selected": [],
                          "frame_id": start["frame_id"] if stale_frame else change["frame_id"],
                          "receipts": [r["receipt_id"] for r in start["receipts"]]})
        assert result["reason"] == ("stale_frame" if stale_frame else "stale_or_missing_evidence")
        assert not result["verified"]


def test_invalid_and_over_budget():
    world = make_world("ood_test", 6, 1, "relevant")
    ep = Episode(world)
    ep.start()
    for _ in range(len(world.distractors) + 1):
        ep.step({"action": "advance"})
    ep.step({"action": "request", "source": "s0"})
    assert ep.step({"action": "request", "source": "s1"})["reason"] == "verifier_budget_exhausted"
    assert ep.calls == 1
    with pytest.raises(ValueError):
        ep.step({"action": "advance"})


def test_train_export(tmp_path):
    out = tmp_path / "train.jsonl"
    export_train(9, 3, out)
    records = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(records) == 6
    assert len({r["root_id"] for r in records}) == 3
    assert all(r["split"] == "train" and r["result"]["optimal"] for r in records)
    with pytest.raises(FileExistsError):
        export_train(9, 3, out)
