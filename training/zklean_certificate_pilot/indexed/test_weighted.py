"""Independent replay and small exhaustive oracle checks."""
from dataclasses import replace
from itertools import pairwise, product

import pytest

from lean_export import export
from weighted import Z, dp, generate, numeral, offset, one_step, search, trace, verify, weighted


def optimal_proof(seed=10000, limit=12):
    x = generate(seed, limit)
    rows, _ = search(x, dp(x)[1])
    assert rows is not None
    assert weighted(verify(x, rows)) == dp(x)[0][x.goal]
    return x, rows


@pytest.mark.parametrize('seed', range(16))
def test_exhaustive_small(seed):
    x = generate(seed, 3)

    def all_costs(g):
        result = []
        for e in x.edges:
            if e.head == g:
                for children in product(*(all_costs(p) for p in e.premises)):
                    result.append(e.structural + e.iota + 2 * e.reads + sum(children))
        return result

    costs = all_costs(x.goal)
    assert costs and min(costs) == dp(x)[0][x.goal]
    rows, _ = search(x, dp(x)[1])
    assert weighted(verify(x, rows)) == min(costs)


@pytest.mark.parametrize('n', range(12))
def test_normalization(n):
    from weighted import instantiate

    for k in range(6):
        for style in range(4):
            tr = trace(instantiate(offset(k, style), n))
            assert tr[-1] == numeral(n + k)
            assert all(one_step(a, b) for a, b in pairwise(tr))


def test_attacks():
    x, rows = optimal_proof()
    rid, n, refs, tr = rows[-1]
    candidates = [[], rows[:-1], [*rows, rows[-1]],
                  [*rows[:-1], (len(x.rules), n, refs, tr)],
                  [*rows[:-1], (rid, x.limit+1, refs, tr)],
                  [*rows[:-1], (rid, n, (len(rows)-1, *refs[1:]), tr)],
                  [*rows[:-1], (rid, n, refs, ())],
                  [*rows[:-1], (rid, n, refs, ((Z,), *tr[1:]))],
                  [*rows[:-1], (rid, n, refs, ((*tr[0], numeral(x.limit+1)), *tr[1:]))]]
    for bad in candidates:
        with pytest.raises(ValueError):
            verify(x, bad)
    with pytest.raises(ValueError):
        verify(replace(x, goal=x.goal ^ 1), rows)


def test_export():
    x, rows = optimal_proof()
    source = export(x, rows)
    assert 'theorem zklExercise' in source
    assert '∀ n : Nat' in source
    assert 'sorry' not in source
    assert 'axiom' not in source
