"""Independent forward-state Dijkstra check of backward obligation solver."""
from heapq import heappop, heappush

import pytest
from dag import certificate, exact, verify_dag
from weighted import generate


def forward(x, weighted):
    q = [(0, 0)]
    costs = {0: 0}
    while q:
        cost, mask = heappop(q)
        if cost != costs[mask]:
            continue
        if mask & (1 << x.goal):
            return cost
        for e in x.edges:
            if mask & (1 << e.head) or any(not (mask & (1 << p)) for p in e.premises):
                continue
            nxt = mask | (1 << e.head)
            v = cost + (e.weight if weighted else 1)
            if v < costs.get(nxt, 10**12):
                costs[nxt] = v
                heappush(q, (v, nxt))
    raise ValueError('unreachable')


@pytest.mark.parametrize('seed', range(16))
def test_exhaustive(seed):
    x = generate(seed, 3)
    for weighted in (False, True):
        cost, selection, _ = exact(x, weighted)
        assert cost == forward(x, weighted)
        assert verify_dag(x, certificate(x, selection))[int(weighted)] == cost


def test_bad_reference():
    x = generate(40000, 12)
    _, selection, _ = exact(x)
    rows = certificate(x, selection)
    rid, n, refs, conversions = rows[-1]
    assert refs
    bad = [*rows[:-1], (rid, n, (len(rows) - 1, *refs[1:]), conversions)]
    with pytest.raises(ValueError):
        verify_dag(x, bad)
    with pytest.raises(ValueError):
        verify_dag(x, [*rows, rows[-1]])
