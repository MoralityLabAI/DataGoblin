import numpy as np

from datagoblin.training_gym.rlt_gym.aleph_proxy import WIDTH, rollout
from datagoblin.training_gym.rlt_gym.tasks import generate_tasks, verify


def test_fixed_endpoint_and_first_valid_spend_only_their_actual_checks():
    task = next(t for t in generate_tasks(seed=310, per_family=1) if t["family"] == "arithmetic_program")
    passes = [verify(task, candidate)["passed"] for candidate in task["candidates"]]
    good, bad = passes.index(True), passes.index(False)
    steps = [{"candidate_index": i} for i in [bad, bad, good, bad, bad, bad, bad, bad]]
    fixed = rollout(task, steps, "fixed_8")
    early = rollout(task, steps, "first_valid")
    assert (fixed["solved"], fixed["visits"], fixed["verifier_calls"]) == (False, 8, 1)
    assert (early["solved"], early["visits"], early["verifier_calls"]) == (True, 3, 2)


def test_learned_policy_checks_an_alternative_after_a_failure():
    task = next(t for t in generate_tasks(seed=310, per_family=1) if t["family"] == "arithmetic_program")
    passes = [verify(task, candidate)["passed"] for candidate in task["candidates"]]
    good, bad = passes.index(True), passes.index(False)
    scores = [2 if i == bad else int(i == good) for i in range(len(passes))]
    step = {"candidate_index": bad, "candidate_scores": scores}
    row = rollout(task, [step] * 8, "learned", weights=np.zeros(WIDTH), threshold=0.0, margin=0.1)
    assert row["solved"]
    assert row["verifier_calls"] == 2
    assert row["repair_checks"] == row["repair_solves"] == 1
    assert [x["receipt"]["passed"] for x in row["checks"]] == [False, True]
