from MeTTa.evidence_budget import ROUTES, Catalog, bayes_check
from MeTTa.evidence_budget_confirmation import FAMILIES, exact_expectations, primary_intervals, relabel, topology


def test_confirmation_topologies_are_distinct_after_slot_permutation():
    fresh = {topology(routes) for routes in FAMILIES.values()}
    assert len(fresh) == len(FAMILIES)
    assert not fresh & {topology(routes) for routes in ROUTES.values()}
    sample = FAMILIES["triple_mix"]
    assert topology(sample) == topology(relabel(sample, (6, 4, 2, 0, 1, 3, 5)))


def test_exact_world_enumeration_confirms_tie_break_on_single_route():
    catalog = Catalog(((0, 1),), (90, 30, 80, 80, 80, 80, 80))
    old = exact_expectations(catalog, bayes_check(catalog), 2)
    new = exact_expectations(catalog, bayes_check(catalog, minimize_work=True), 2)
    assert old == (27 * 100**6, 190 * 100**6)
    assert new == (27 * 100**6, 130 * 100**6)


def test_root_bootstrap_keeps_topology_strata_equally_weighted():
    result = primary_intervals({"short": [[0.0, 0.0]], "long": [[1.0, 0.0]] * 4})
    assert result["calls_saved"] == {"estimate": 0.5, "paired_root_bootstrap_95": [0.5, 0.5]}
    assert result["completion_difference"]["paired_root_bootstrap_95"] == [0.0, 0.0]
