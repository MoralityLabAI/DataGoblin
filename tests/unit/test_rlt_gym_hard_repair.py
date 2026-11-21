import numpy as np

from datagoblin.hashing import sha256_json
from datagoblin.training_gym.rlt_gym.hard_repair import (
    EDIT_WIDTH,
    WIDTH,
    gate_features,
    play,
)
from datagoblin.training_gym.rlt_gym.tasks import generate_tasks, public_task, verify


def test_hard_policy_defers_then_checks_a_repair_outside_the_pool():
    task = next(t for t in generate_tasks(981, 1) if t["family"] == "arithmetic_program")
    good = next(c for c in task["candidates"] if verify(task, c)["passed"])
    negatives = [c for c in task["candidates"] if not verify(task, c)["passed"]]
    item = {"task": {**task, "candidates": negatives}, "condition": "miss"}
    item["task"].pop("task_digest")
    item["task"]["task_digest"] = sha256_json(item["task"])
    bad = negatives[0]
    k, repaired = sha256_json(bad), sha256_json(good)
    step = {"candidate": bad}
    trace = [(step, {k: (bad, "native", 0)}), (step, {repaired: (good, "offset_negative", 0)})]
    world = public_task(item["task"])
    gate = np.zeros(len(gate_features(world, "native", 0, 1, 1, 0)))
    gate[:3] = [-0.3, 0, 2]
    result = play(item, trace, "learned_defer", np.zeros(WIDTH + EDIT_WIDTH), gate, 0)
    assert result["solved"] and result["out_of_pool_solve"]
    assert result["visits"] == 2 and result["deferred"] == 1
    assert len(result["checks"]) == 1 and result["checks"][0]["visit"] == 2
    assert [decision["action"] for decision in result["decisions"]] == ["defer", "check"]
