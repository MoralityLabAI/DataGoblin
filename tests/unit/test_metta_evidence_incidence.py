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
