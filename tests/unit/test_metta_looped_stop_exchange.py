from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest


def exchange_module():
    pytest.importorskip("torch")
    pytest.importorskip("research_gym.neural.looped_decoder")
    pytest.importorskip("verifiers.v1")
    from metta_research_bench.looped import stop_exchange

    return stop_exchange


def test_exchange_prefilter_agrees_with_the_directed_gate_for_all_curation_families():
    module = exchange_module()
    from MeTTa.ram_trajectory import RepairVerifier, typed_proposal
    from MeTTa.sft_oracle import FAMILIES, evaluate_subset, first_state, make_task

    for family, seed in product(FAMILIES, range(4)):
        world = make_task(family, seed)
        verifier = RepairVerifier(world)
        selected = first_state(world, seed)
        candidates, _, _, _ = module.exchange_candidates(world, selected, {tuple(selected)}, set())
        observed = {(drop, add) for _, drop, add in candidates}
        ids = {item["id"] for item in world["items"]}
        start_rank = evaluate_subset(world, selected)["rank"]
        exact = set()
        for drop, add in product(selected, ids - set(selected)):
            remove = verifier.refine(typed_proposal({"op": "remove", "id": drop}), selected)
            if remove["accepted"]:
                append = verifier.refine(typed_proposal({"op": "add", "id": add}), remove["selected_after"])
                if append["accepted"]:
                    final = append["selected_after"]
                    rank = evaluate_subset(world, final)["rank"]
                    if rank > start_rank or (rank == start_rank and final < selected):
                        exact.add((drop, add))
        assert observed == exact, (family, seed)


def trial_world():
    from MeTTa.sft_oracle import make_task, render_metta

    world = make_task("budget", 0)
    world.update(goals=["skill0"], token_budget=10, max_items=1)
    world["items"] = [
        {"id": f"r{i}", "cost": 10 - i, "quality": 2 + i, "eligible": True, "epoch": 1,
         "topics": ["skill0"], "requires": [], "lineage": f"lin{i}", "key": f"k{i}"}
        for i in range(2)
    ]
    world["program"] = render_metta(world)
    return world


def native_exchange(module):
    directory = Path(module.__file__).parent
    rules = (directory / "stop_witness.metta").read_text(encoding="utf-8")
    return module.StopExchange(rules + "\n" + (directory / "stop_exchange.metta").read_text(encoding="utf-8"))


def test_exchange_does_not_commit_a_removal_when_the_second_edit_is_vetoed():
    pytest.importorskip("hyperon")
    module = exchange_module()
    from MeTTa.ram_trajectory import RepairVerifier

    world = trial_world()
    witness = native_exchange(module)
    verifier = RepairVerifier(world)

    def reject_add(proposal, selected):
        receipt = verifier.refine(proposal, selected)
        if proposal["domain_proposed"] == ["included"]:
            return {**receipt, "accepted": False, "result_violations": ["independent_gate_veto"],
                    "selected_after": selected}
        return receipt

    initial = ["r0"]
    blocked = witness(world, initial, {("r0",)}, set(), 1, 0, reject_add)
    assert not blocked["checks"]
    result = witness(world, initial, {("r0",)}, set(), 2, 0, reject_add)
    assert len(result["checks"]) == verifier.checks == 2 and result["chosen"] is None
    assert result["checks"][0]["receipt"]["accepted"] and initial == ["r0"]
    assert result["native"]["offers"] == []


def test_exchange_consumes_two_action_slots_and_reserves_final_stop(monkeypatch):
    pytest.importorskip("hyperon")
    torch = pytest.importorskip("torch")
    module = exchange_module()
    from metta_research_bench.looped import model

    class AlwaysStop:
        def __call__(self, codes, mask):
            logits = torch.full_like(mask, -1.0, dtype=torch.float32)
            logits[:, -1] = 0
            return logits

    world = trial_world()
    monkeypatch.setattr(model, "make_task", lambda family, seed: world)
    monkeypatch.setattr(model, "first_state", lambda task, seed: ["r0"])
    witness = native_exchange(module)
    data = SimpleNamespace(family="budget", source_seed=0)

    monkeypatch.setattr(model, "MAX_DECISIONS", 3)
    actions, costs = model.solve(AlwaysStop(), data, exchange_guard=witness)
    assert actions == [{"op": "remove", "id": "r0"}, {"op": "add", "id": "r1"}, {"op": "stop", "id": "none"}]
    assert costs["policy_verifier_calls"] == costs["witness_verifier_calls"] == 2
    assert costs["witness_interventions"] == 1 and costs["passes"] == 2 * model.LOOPS

    monkeypatch.setattr(model, "MAX_DECISIONS", 2)
    actions, costs = model.solve(AlwaysStop(), data, exchange_guard=witness)
    assert actions == [{"op": "stop", "id": "none"}]
    assert costs["policy_verifier_calls"] == 0


def test_episode_call_cap_leaves_an_explicit_stop_available_without_an_extra_check(monkeypatch):
    pytest.importorskip("hyperon")
    torch = pytest.importorskip("torch")
    module = exchange_module()
    from metta_research_bench.looped import model

    class AlwaysStop:
        def __call__(self, codes, mask):
            logits = torch.full_like(mask, -1.0, dtype=torch.float32)
            logits[:, -1] = 0
            return logits

    world = trial_world()
    monkeypatch.setattr(model, "make_task", lambda family, seed: world)
    monkeypatch.setattr(model, "first_state", lambda task, seed: ["r0"])
    data = SimpleNamespace(family="budget", source_seed=0)
    witness = native_exchange(module)
    events = []
    actions, costs = model.solve(AlwaysStop(), data, exchange_guard=witness, episode_call_cap=2, audit=events)
    assert actions == [{"op": "remove", "id": "r0"}, {"op": "add", "id": "r1"}, {"op": "stop", "id": "none"}]
    assert costs["policy_verifier_calls"] == 2 and not costs["blocked_by_call_cap"]

    facts = events[0]["witness"]["native"]["facts"]
    assert "(StopExchangeAllowance CurrentState 2 0 2)" in facts
    for allowance in ("(StopExchangeAllowance CurrentState 1 0 2)", "(StopExchangeAllowance CurrentState 2 1 2)"):
        revised = [allowance if fact.startswith("(StopExchangeAllowance ") else fact for fact in facts]
        (answers,) = witness.native.evaluate(revised, (module.QUERY,))
        assert not answers

    actions, costs = model.solve(AlwaysStop(), data, exchange_guard=witness, episode_call_cap=1)
    assert actions == [{"op": "stop", "id": "none"}] and costs["policy_verifier_calls"] == 0


def test_exhaustion_blocks_an_edit_without_inserting_a_stop(monkeypatch):
    torch = pytest.importorskip("torch")
    exchange_module()
    from metta_research_bench.looped import model

    class AlwaysEdit:
        def __call__(self, codes, mask):
            logits = torch.full_like(mask, -2.0, dtype=torch.float32)
            logits[:, 0] = 0
            logits[:, 1] = -1
            return logits

    world = trial_world()
    monkeypatch.setattr(model, "make_task", lambda family, seed: world)
    monkeypatch.setattr(model, "first_state", lambda task, seed: ["r0"])
    data = SimpleNamespace(family="budget", source_seed=0)
    events = []
    actions, costs = model.solve(AlwaysEdit(), data, episode_call_cap=1, audit=events)
    assert actions == [{"op": "remove", "id": "r0"}]
    assert costs["policy_verifier_calls"] == 1 and costs["blocked_by_call_cap"] == 1
    assert events[-1]["budget_blocked"] == {"op": "add", "id": "r1"}


def test_last_two_reservation_can_preserve_the_models_preferred_removal(monkeypatch):
    pytest.importorskip("hyperon")
    torch = pytest.importorskip("torch")
    module = exchange_module()
    from metta_research_bench.looped import model

    class RemoveThenStop:
        seen = 0

        def __call__(self, codes, mask):
            logits = torch.full_like(mask, -1.0, dtype=torch.float32)
            logits[:, 0 if self.seen == 0 else -1] = 0
            self.seen += 1
            return logits

    world = trial_world()
    monkeypatch.setattr(model, "make_task", lambda family, seed: world)
    monkeypatch.setattr(model, "first_state", lambda task, seed: ["r0"])
    witness = native_exchange(module)
    data = SimpleNamespace(family="budget", source_seed=0)
    expected = [{"op": "remove", "id": "r0"}, {"op": "add", "id": "r1"}, {"op": "stop", "id": "none"}]

    baseline, _ = model.solve(RemoveThenStop(), data, exchange_guard=witness, episode_call_cap=2)
    assert baseline == [expected[0], expected[-1]]
    for reservation in ("last_two", "agreed_removal"):
        events = []
        actions, costs = model.solve(RemoveThenStop(), data, exchange_guard=witness, episode_call_cap=2,
                                     exchange_reservation=reservation, audit=events)
        assert actions == expected and costs["policy_verifier_calls"] == 2 and costs["reserved_exchanges"] == 1
        assert events[0]["preferred_action"] == expected[0]
        assert events[0]["witness"]["native"]["query"] == module.RESERVED_QUERY
        facts = events[0]["witness"]["native"]["facts"]
        conflicting = ["(ExchangeReservation CurrentState 2 (ModelRemoval r1))"
                       if fact.startswith("(ExchangeReservation ") else fact for fact in facts]
        (answers,) = witness.native.evaluate(conflicting, (module.RESERVED_QUERY,))
        assert not answers

    actions, costs = model.solve(RemoveThenStop(), data, exchange_guard=witness, episode_call_cap=3,
                                 exchange_reservation="last_two")
    assert actions == baseline and costs["reserved_exchanges"] == 0
