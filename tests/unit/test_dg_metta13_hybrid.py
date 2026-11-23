from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from metta_research_bench.courier_hybrid.controller import Proposal, Repair, run_episode
from metta_research_bench.courier_hybrid.learning import CourierPointer, weighted_set_nll
from metta_research_bench.courier_hybrid.native import (
    ATTEST_FIRST,
    ATTEST_SECOND,
    COLLECT,
    DELIVER,
    ISSUE,
    STOP,
    NativeCourier,
    State,
    World,
    load_storyworld,
    state_digest,
)
from metta_research_bench.courier_hybrid.oracle import continuations, objective_weights, repair_routes, state_graph

WORLD = World(((0, 1, True), (1, 0, True), (1, 2, False)), (0,), (1,), 2, 0, ((0, 1), (2,)))
ROUTE = (ISSUE, 1, COLLECT, ATTEST_FIRST, 2, ATTEST_SECOND, DELIVER)


@pytest.fixture
def native():
    checkout = os.environ.get("DG_METTA13_STORYWORLD")
    if not checkout:
        pytest.skip("Set DG_METTA13_STORYWORLD to the pinned Storyworld checkout")
    upstream, _ = load_storyworld(Path(checkout))
    return NativeCourier(upstream)


def test_native_differential_and_atomic_rollback(native):
    expected, _ = state_graph(WORLD)
    actual, _ = state_graph(WORLD, native.checked_successors)
    assert expected == actual
    session = native.session(WORLD, 4, 12)
    before = session.state
    failed = session.certify((ISSUE, 1, COLLECT, 2, DELIVER), macro=True)
    assert not failed.accepted and not failed.step_sha256
    assert failed.before_sha256 == failed.after_sha256 == state_digest(WORLD, before)
    assert session.state == before and session.actions == () and session.calls == 1
    passed = session.certify(ROUTE, macro=True)
    assert passed.accepted and len(passed.step_sha256) == len(ROUTE)
    assert passed.after_sha256 == state_digest(WORLD, session.state)
    assert session.stop().accepted and session.calls == 3
    replay = native.session(WORLD, 2, 12).certify(ROUTE, macro=True)
    assert passed == replay


def test_source_alias_cannot_supply_independent_evidence(native):
    session = native.session(WORLD, 3, 12)
    assert not session.certify((ATTEST_FIRST, ISSUE, 1, ATTEST_FIRST), macro=True).accepted
    assert session.state == WORLD.initial
    assert not session.stop().accepted
    assert session.calls == 2


def test_stop_repair_same_ceiling_and_call_accounting(native):
    def premature(obs):
        return Proposal(ISSUE if obs.state == WORLD.initial else STOP)

    standard = run_episode(native.session(WORLD, 4, 12), premature)
    repaired = run_episode(native.session(WORLD, 4, 12), premature, Repair("reserve", 6, 2))
    assert not standard["verified_completion"] and standard["calls"] == 2
    assert repaired["verified_completion"] and repaired["stop_repair_completion"]
    assert repaired["calls"] == len(repaired["receipts"]) == 3
    assert repaired["moves"] + 1 > 4
    assert repaired["unused_calls"] == 1
    exhausted = run_episode(native.session(WORLD, 2, 12), premature, Repair("reserve", 6, 2))
    assert exhausted["calls"] == 2 and exhausted["macro_attempts"] == 0


def test_budget_equivalence_separates_exact_ties_and_cost():
    graph, distances = state_graph(WORLD)
    outcomes = continuations(WORLD.initial, graph, distances, 12, 13)
    weights = objective_weights(WORLD, WORLD.initial, outcomes)
    assert set(weights["shortest"]) == {ISSUE, ATTEST_FIRST}
    assert len(weights["hashed"]) == 1
    assert not continuations(WORLD.initial, graph, distances, 12, 7)
    assert repair_routes(WORLD, WORLD.initial, 6, 2) == ()
    assert repair_routes(WORLD, WORLD.initial, 7, 2)[0] == ROUTE
    reviewed = State(1, True, 0, 1)
    outcomes = continuations(reviewed, graph, distances, 12, 13)
    weights = objective_weights(WORLD, reviewed, outcomes)
    assert set(weights["shortest"]) == {COLLECT}
    assert set(weights["budget"]) == {0, COLLECT}
    assert 0 < weights["soft"][0] < 1


def test_weighted_mass_preserves_ties_and_masks():
    logits = torch.tensor([[2.0, 0.0, 5.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    full = weighted_set_nll(logits, mask, torch.tensor([[1.0, 1.0, 0.0]]))
    weighted = weighted_set_nll(logits, mask, torch.tensor([[1.0, 0.5, 0.0]]))
    assert full.item() == 0 and weighted.item() > 0
    weighted.backward()
    assert logits.grad.isfinite().all() and logits.grad[0, 2] == 0
    assert all(sum(p.numel() for p in CourierPointer(mode).parameters()) == 44417 for mode in (False, True))
