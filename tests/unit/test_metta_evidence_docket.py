from itertools import product
from math import prod

from MeTTa.evidence_docket import Docket, exact_policy, frontier_choice, needs, rollout


def test_joint_teacher_reuses_records_when_individual_goals_prefer_standalone_evidence():
    docket = Docket((((0,), (1, 2)), ((1, 2), (3,))), (80, 85, 85, 90, 50, 50, 50))
    truth = (False, True, True, True, False, False, False)

    def local(public):
        return frontier_choice(public, *needs(docket.claims, public.state), shared=False)

    def joint(public):
        return frontier_choice(public, *needs(docket.claims, public.state), shared=True)

    local_result = rollout(docket, truth, 4, local)
    joint_result = rollout(docket, truth, 4, joint)
    assert joint_result["completed"] and local_result["completed"]
    assert joint_result["checks"] == 2 and local_result["checks"] == 3
    assert [receipt["slot"] for receipt in joint_result["receipts"]] == [1, 2]


def test_exact_policy_matches_exhaustive_world_success_when_budget_is_nonbinding():
    docket = Docket((((0,), (1, 2)), ((2,), (1, 3))), (40, 65, 70, 30, 50, 50, 50))
    policy = exact_policy(docket)
    successes = 0
    for truth in product((False, True), repeat=4):
        completed = rollout(docket, (*truth, False, False, False), 4, policy)["completed"]
        mass = prod(prior if status else 100 - prior for prior, status in zip(docket.priors[:4], truth, strict=True))
        successes += mass * completed
    # True claim A is (a or b and c). True claim B is (c or b and d).
    # Condition on c, so the two disjoint contributions can be computed independently.
    assert successes == 70 * (40 * 100 + 60 * 65) * 100 + 30 * 40 * 65 * 30
