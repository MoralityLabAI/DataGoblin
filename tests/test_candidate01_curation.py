"""Adversarial public checks for six-skill dataset admission."""

import copy
import itertools
import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "aleph_next" / "candidate01"))
import curation
import curation_check as check
import curation_tasks as task
import tournament_admission as supplement


@pytest.mark.parametrize("kind", ["state", "operator", "composition", "repair", "allocation"])
def test_independent_task_solvers_and_anti_admission(kind):
    for i in range(16):
        rec = getattr(task, kind)(random.Random(11 + i), i)
        if rec is None:
            continue
        root, difficulty = curation.inspect_candidate(rec, {}, {})
        assert len(root) == 64 and difficulty
        for diff in (dict(split="held"), dict(answer="malformed"), dict(family="mixed_nested_product")):
            with pytest.raises(ValueError):
                curation.inspect_candidate(dict(rec, **diff), {}, {})


def test_operator_certificate_is_identified_on_whole_domain():
    rec = task.operator(random.Random(15), 1)
    pub = rec["public"]
    pub["p"] = 5
    pub["basis"] = ["1", "x", "y", "xy"]
    pub["observations"] = [[0, 0, 1], [0, 1, 4], [1, 0, 3], [1, 1, 0]]
    pub["query"] = [1, 2]
    solutions = [
        a
        for a in itertools.product(range(5), repeat=4)
        if all((a[0] + a[1] * x + a[2] * y + a[3] * x * y) % 5 == z for x, y, z in pub["observations"])
    ]
    answer = check.operator(pub)[0]
    assert [tuple(answer["coefficients"])] == solutions
    bad = copy.deepcopy(pub)
    bad["observations"][-1] = bad["observations"][0]
    with pytest.raises(ValueError, match="unidentifiable"):
        check.operator(bad)


def test_machine_quotient_and_shortlex_witness():
    a = {"start": 0, "next": [[1, 0], [1, 2], [2, 2]], "accept": [2]}
    b = {**a, "accept": []}
    assert check.machine_witness(a, b) == ("01", 2)
    # Duplicate an equivalent accepting sink and append an arbitrary unreachable state.
    equivalent = {"start": 0, "next": [[1, 0], [1, 3], [2, 2], [2, 3], [4, 4]], "accept": [2, 3, 4]}
    assert check.machine_form(a) == check.machine_form(equivalent)
    assert check.machine_witness(a, equivalent) == (None, None)


def test_graph_allocation_renaming_and_nontrivial_targets():
    counter = set()
    for i in range(80):
        rec = task.allocation(random.Random(i), i)
        q = rec["public"]
        old = check.allocation(q)

        def relabel(v):
            return "node" + str(1000 - v)

        new = check.allocation(
            dict(
                start=relabel(q["start"]),
                goal=relabel(q["goal"]),
                edges=[[relabel(a), relabel(b)] for a, b in reversed(q["edges"])],
            )
        )
        assert old == new
        counter.add(old[0]["solver"])
    assert counter == {"tie", "forward", "reverse"}


def test_lean_template_revokes_literal_substitution_and_unchecked_proof():
    row = task.lean(random.Random(0), 0)
    base = check.lean_template(row["public"]["statement"])
    variant = re.sub(r"\b\d+\b", "170", row["public"]["statement"], count=1)
    assert variant != row["public"]["statement"] and base == check.lean_template(variant)
    with pytest.raises(ValueError):
        curation.inspect_candidate(dict(row, answer="by\n  sorry"), {}, {})


def test_certificate_reverse_step_supplies_erased_term_and_preserves_context():
    rules = [("plus_zero", "id", ("+", "?a", "0"), "?a")]
    start, end = ("*", "x", "y"), ("*", ("+", "x", "0"), "y")
    assert supplement.compact_path("nat", start, end, [(end, 0, (1,), True)], rules) == [[0, [1], ("+", "x", "0")]]
    wrong = ("*", ("+", "x", "0"), "z")
    with pytest.raises(ValueError, match="context changed"):
        supplement.compact_path("nat", start, wrong, [(wrong, 0, (1,), True)], rules)


def test_tournament_allocation_prices_prerequisites_and_rejects_single_option():
    options = {
        "feedback": {"accepted": True, "stages": [[2, 3, 1], [1, 4, 1]]},
        "normal": {"accepted": True, "stages": [[4, 3, 1]]},
        "frontiers": {"accepted": False, "stages": [[1, 0, 0]]},
    }
    q = {"options": options, "allowance_ns": 12, "call_cap": 2, "gate_ns": 2}
    assert supplement.choose(q) == {"action": "normal", "elapsed_ns": 9}
    assert supplement.choose(dict(q, call_cap=1)) is None
