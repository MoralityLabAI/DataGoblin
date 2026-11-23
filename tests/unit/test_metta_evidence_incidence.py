import pytest


def test_public_incidence_inputs_are_equal_except_for_the_routing_bias():
    torch = pytest.importorskip("torch")
    pytest.importorskip("research_gym.neural.obligation_pointer")
    pytest.importorskip("verifiers.v1")
    from MeTTa.evidence_budget import Catalog, observe
    from MeTTa.evidence_incidence_pilot import FIELDS, graph_batch

    routes = ((0, 1), (1, 2), (3, 4))
    public = observe(Catalog(routes, (50,) * 7), (0,) * 7, 3)
    (dense, real, rewired), allowed, changed = graph_batch(public, routes)
    assert allowed[:7].tolist() == [True, True, True, True, True, False, False]
    assert not allowed[7:].any()
    assert torch.equal(dense[:, :FIELDS + 1], real[:, :FIELDS + 1])
    assert torch.equal(dense[:, :FIELDS + 1], rewired[:, :FIELDS + 1])
    assert not dense[:, FIELDS + 1:].any()
    assert real[:, FIELDS + 1:].sum() == rewired[:, FIELDS + 1:].sum()
    assert changed > 0


def test_fresh_graphs_are_non_isomorphic_to_each_other_and_prior_topologies():
    pytest.importorskip("torch")
    pytest.importorskip("research_gym.neural.obligation_pointer")
    pytest.importorskip("verifiers.v1")
    from MeTTa.evidence_incidence_pilot import EVAL, PRIOR_EVAL, TRAIN_FAMILIES, topology

    previous = {topology(routes) for routes in (*TRAIN_FAMILIES.values(), *PRIOR_EVAL.values())}
    fresh = {topology(routes) for routes in EVAL.values()}
    assert len(fresh) == len(EVAL) == 8 and not fresh.intersection(previous)


def test_stage_contrasts_pair_budgets_and_seeds():
    pytest.importorskip("torch")
    pytest.importorskip("research_gym.neural.obligation_pointer")
    pytest.importorskip("verifiers.v1")
    from MeTTa.evidence_incidence_stages import ARMS, PROTOCOL, root_contrasts

    checks = {"neutral": 4, "inference_only": 2, "training_only": 3, "both": 3, "inference_sham": 4}
    results = {(arm, budget, seed): {"checks": checks[arm] + budget - 4, "completed": arm == "both"}
               for arm in ARMS for budget in PROTOCOL["budgets"] for seed in PROTOCOL["seeds"]}
    results["inference_only", 4, 1101]["completed"] = True
    actual = root_contrasts(results)
    assert actual["both_minus_neutral"] == [1, 1]
    assert actual["inference_without_training_bias"] == pytest.approx([2, 1 / 6])
    assert actual["training_inference_interaction"] == pytest.approx([-2, 5 / 6])
