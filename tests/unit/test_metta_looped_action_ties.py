from collections import deque
from pathlib import Path

import pytest


def test_verified_action_set_matches_shortest_paths_through_the_repair_gate():
    pytest.importorskip("torch")
    pytest.importorskip("research_gym.neural.looped_decoder")
    pytest.importorskip("verifiers.v1")
    from MeTTa.ram_trajectory import STOP, RepairVerifier, typed_proposal
    from MeTTa.sft_oracle import first_state, make_task
    from metta_research_bench.looped.action_ties import safe_shortest

    world = make_task("dependency", 8)
    initial = first_state(world, 8)
    verifier = RepairVerifier(world)
    goal = tuple(verifier.optimum["selected"])
    possible, _ = safe_shortest(verifier, initial)
    assert initial != list(goal)
    queue = deque([(tuple(initial), 0, None)])
    seen = set()
    depth = None
    winners = set()
    while queue:
        state, edits, first = queue.popleft()
        if depth is not None and edits > depth:
            break
        if (state, first) in seen:
            continue
        seen.add((state, first))
        if state == goal:
            winners.add(first)
            depth = edits
            continue
        for item in world["items"]:
            key = item["id"]
            action = {"op": "remove" if key in state else "add", "id": key}
            receipt = verifier.refine(typed_proposal(action), list(state))
            if receipt["accepted"]:
                queue.append((tuple(receipt["selected_after"]), edits + 1, first or (action["op"], key)))
    assert depth == len(set(initial) ^ set(goal))
    assert winners == {(action["op"], action["id"]) for action in possible}
    assert safe_shortest(verifier, list(goal))[0] == [STOP]


def test_native_shortest_choice_requires_a_checked_progress_edit_or_terminal_stop():
    pytest.importorskip("hyperon")
    from MeTTa.evidence_obligations import ScopedNative

    path = Path(__file__).parents[2] / "metta_research_bench" / "looped" / "action_ties.metta"
    native = ScopedNative(path.read_text(encoding="utf-8"))
    facts = """
    (RepairDistance Current 2)
    (RepairDistance Nearer 1)
    (RepairDistance Detour 3)
    (RepairDistance Goal 0)
    (CheckedRepairEdit Current remove r0 Accepted Nearer)
    (CheckedRepairEdit Current add r1 Vetoed Nearer)
    (CheckedRepairEdit Current remove r2 Accepted Detour)
    (CheckedRepairStop Current Accepted)
    (CheckedRepairStop Goal Accepted)
    """.splitlines()
    edit, stop = native.evaluate(facts, ("(VerifiedShortestActions Current)", "(VerifiedShortestActions Goal)"))
    assert {str(value) for value in edit} == {"(RepairChoice remove r0)"}
    assert {str(value) for value in stop} == {"(RepairChoice stop none)"}
