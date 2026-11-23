from itertools import product
from math import prod

import pytest

from MeTTa.evidence_docket import Docket, action_values, exact_policy, frontier_choice, needs, observe, rollout


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


def test_cost_targets_keep_success_ties_and_prefer_testing_riskier_required_record():
    pytest.importorskip("torch")
    pytest.importorskip("research_gym.neural.obligation_pointer")
    pytest.importorskip("verifiers.v1")
    from MeTTa.evidence_docket_cost_targets import supports

    docket = Docket((((0,),), ((1,),)), (75, 30, 50, 50, 50, 50, 50))
    public = observe(docket, (0,) * 7, 4)
    values = action_values(docket)(public)
    assert values == {0: (22_500_000, -175_000_000), 1: (22_500_000, -130_000_000)}
    assert supports(values) == {"success_hard": (0,), "success_ties": (0, 1), "cost_ties": (1,)}
    assert supports({**values, 2: values[1]})["cost_ties"] == (1, 2)


def test_looped_fit_supports_masked_target_distributions_without_changing_hard_target_updates():
    torch = pytest.importorskip("torch")
    pytest.importorskip("research_gym.neural.obligation_pointer")
    pytest.importorskip("verifiers.v1")
    from metta_research_bench.looped.model import fit

    class MaskedPolicy(torch.nn.Linear):
        def forward(self, x, legal):
            return super().forward(x).masked_fill(~legal, -torch.inf)

    initial = MaskedPolicy(1, 3)
    tokens = torch.ones(1, 1)
    mask = torch.tensor([[True, True, False]])
    hard, _ = fit(initial, (tokens, mask, torch.tensor([1])), 17, 2)
    one_hot, _ = fit(initial, (tokens, mask, torch.tensor([[0.0, 1.0, 0.0]])), 17, 2)
    for a, b in zip(hard.parameters(), one_hot.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    _, meta = fit(initial, (tokens, mask, torch.tensor([[0.5, 0.5, 0.0]])), 17, 2)
    assert 0 < meta["final_loss_average"] < 10
    with pytest.raises(ValueError, match="legal actions"):
        fit(initial, (tokens, mask, torch.tensor([[0.5, 0.0, 0.5]])), 17, 2)
