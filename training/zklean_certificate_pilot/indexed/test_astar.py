"""Cross-check A* against independent forward search and the state cap."""
import pytest
from astar import solve
from dag import BudgetExceeded, certificate, verify_dag
from test_dag import forward
from weighted import generate


@pytest.mark.parametrize('seed', range(16))
def test_exact(seed):
    x = generate(seed, 3)
    cost, selected, _, _ = solve(x)
    assert cost == forward(x, False)
    assert verify_dag(x, certificate(x, selected))[0] == cost


def test_cap():
    with pytest.raises(BudgetExceeded):
        solve(generate(40000, 12), cap=1)
