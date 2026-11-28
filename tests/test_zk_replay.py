import copy
from dataclasses import replace

import pytest

from datagoblin.zk_replay import Job, Plan, canonical, execute, verify


def graph(n: int = 4, depth: int = 4) -> Plan:
    jobs = []
    for i in range(n):
        parents = tuple(f"j{p}" for p in range(max(0, i - 2), i))
        jobs.append(Job(f"j{i}", parents, (i % 7, -2), ((1, 2), (-1, 1)), (0, 1), depth))
    return Plan("tenant-a", "run-001", "policy-v1", tuple(jobs))


@pytest.mark.parametrize("n", [1, 8, 32, 128])
@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16, 32])
def test_round_trip(n, depth):
    plan = graph(n, depth)
    receipts = execute(plan)
    assert verify(plan, receipts) == verify(plan, execute(plan))
    assert len(receipts) == n
    assert len(receipts[-1]["trace"]) == depth + 1


def test_known_arithmetic():
    job = Job("root", (), (1, -2), ((1, 2), (-1, 1)), (0, 1), 2)
    receipt = execute(Plan("a", "r", "p", (job,)))[0]
    assert receipt["trace"] == [(1, -2), (-2, -4), (-9, -3)]


@pytest.mark.parametrize("field", ["backend", "context", "job", "parents", "model", "input", "output", "steps"])
def test_statement_tamper(field):
    plan = graph()
    receipts = execute(plan)
    receipts[-1]["statement"][field] = "tampered"
    with pytest.raises(ValueError):
        verify(plan, receipts)


@pytest.mark.parametrize("field", ["tenant", "run", "policy"])
def test_context_replay(field):
    plan = graph()
    with pytest.raises(ValueError):
        verify(replace(plan, **{field: "different"}), execute(plan))


def test_trace_tamper_and_missing_node():
    plan = graph()
    receipts = execute(plan)
    bad = copy.deepcopy(receipts)
    bad[0]["trace"][1] = (0, 0)
    with pytest.raises(ValueError):
        verify(plan, bad)
    with pytest.raises(ValueError):
        verify(plan, receipts[1:])
    with pytest.raises(ValueError):
        verify(plan, list(reversed(receipts)))


def test_invalid_plan():
    plan = graph()
    for job in (replace(plan.jobs[-1], parents=("j0", "j0")), replace(plan.jobs[-1], parents=("missing",)),
                replace(plan.jobs[-1], parents=()), replace(plan.jobs[-1], steps=True)):
        with pytest.raises(ValueError):
            execute(replace(plan, jobs=(*plan.jobs[:-1], job)))
    assert canonical(True) != canonical(1)
