"""Exhaustive small-graph checks of both stronger bounds and tie independence."""
from heapq import heappop, heappush

import pytest
from learned_tie import Graph, replay
from mandatory_bound import MandatoryGraph
from weighted import generate


def all_forward_costs(x):
    """Independent forward-state search, with costs for every attainable fact set."""
    best = {0: 0}
    queue = [(0, 0)]
    while queue:
        cost, mask = heappop(queue)
        if best[mask] != cost:
            continue
        for e in x.edges:
            if mask & (1 << e.head) or any(not (mask & (1 << p)) for p in e.premises):
                continue
            nxt = mask | (1 << e.head)
            if cost + 1 < best.get(nxt, 10**12):
                best[nxt] = cost + 1
                heappush(queue, (cost + 1, nxt))
    return best


@pytest.mark.parametrize('seed', range(12))
def test_all_masks(seed):
    x = generate(seed, 3)
    costs = all_forward_costs(x)
    base = Graph(x)
    mandatory = MandatoryGraph(x)
    packing = MandatoryGraph(x, True)
    for mask in range(1 << x.n):
        optimum = min(cost for got, cost in costs.items() if got & mask == mask)
        assert base.lower(mask) <= mandatory.lower(mask) <= packing.lower(mask) <= optimum
    optimum = min(cost for mask, cost in costs.items() if mask & (1 << x.goal))
    for g in (base, mandatory, packing):
        for mode, estimate in (('deep', None), ('tree', None), ('learned', lambda f: -1000 * f[1])):
            result = g.search(mode, estimate)
            assert result['cost'] == optimum
            assert replay(x, result['selection']) == optimum
    mandatory.lower.cache_clear()
    packing.lower.cache_clear()
