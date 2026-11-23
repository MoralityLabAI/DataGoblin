import pytest

from MeTTa.evidence_obligations import ScopedNative, host_frontier


def test_frontier_preserves_satisfied_route_and_removes_blocked_routes():
    full, minimal = host_frontier(((0, 1), (0, 2, 3), (1, 4), (4, 5)), (0, 1, 0, 0, -1, 0))
    assert full == {frozenset((0,)), frozenset((0, 2, 3))}
    assert minimal == {frozenset((0,))}
    assert host_frontier(((0, 1), (0, 2)), (1, 1, 0))[1] == {frozenset()}
    assert host_frontier(((0, 1), (0, 2)), (-1, 1, 0)) == (set(), set())


def test_native_query_scopes_do_not_leak_between_states():
    pytest.importorskip("hyperon")
    native = ScopedNative("(= (Find $claim) (match &self (Fact $claim $id) $id))")
    (first,) = native.evaluate(["(Fact Goal Original)"], ("(Find Goal)",))
    (empty,) = native.evaluate(["(Fact Other Unrelated)"], ("(Find Goal)",))
    (second,) = native.evaluate(["(Fact Goal Revised)"], ("(Find Goal)",))
    assert list(map(str, first)) == ["Original"]
    assert empty == []
    assert list(map(str, second)) == ["Revised"]
