import itertools
import random
from dataclasses import replace

import pytest

from datagoblin import zk_frontier as f
from datagoblin import zk_schedule as dense
from datagoblin.zk_replay import Job, Plan, execute
from datagoblin.zk_sigma import r1cs_satisfied


def fixture(n, w, mode, seed=7):
    rng = random.Random(seed)
    jobs = []

    def vec():
        return tuple(rng.randint(-127, 127) for _ in range(w))

    for i in range(n):
        if mode == "chain":
            parents = [i-1] if i else []
        elif mode == "window":
            parents = list(range(max(0, i-2), i))
        elif mode == "star":
            parents = list(range(i)) if i == n-1 else []
        else:
            parents = [p for p in range(i) if p == i-1 or rng.random() < .3]
        jobs.append(Job(f"j{i}", tuple(f"j{p}" for p in parents), vec(),
                        tuple(vec() for _ in range(w)), vec(), rng.randint(1, 4)))
    return Plan("tenant", f"{mode}-{n}-{seed}", "frontier", tuple(jobs))


@pytest.mark.parametrize("n", [1, 8, 32])
@pytest.mark.parametrize("w", [1, 2])
@pytest.mark.parametrize("mode", ["chain", "window", "star", "random"])
def test_oracle(n, w, mode):
    p = fixture(n, w, mode)
    alloc = f.layout(p)
    rows = f.schedule(p)
    expected = execute(p)
    for i, row in enumerate(rows):
        assert row["constraints"] == rows[0]["constraints"]
        assert row["state_in"] == rows[0]["state_in"]
        assert row["state_out"] == rows[0]["state_out"]
        assert r1cs_satisfied(row, dense.PALLAS_BASE)
        assert r1cs_satisfied(row, dense.PALLAS_SCALAR)
        state = dense.next_state(row)
        for slot, owner in enumerate(alloc.after[i]):
            assert state[3+slot*w:3+(slot+1)*w] == (
                expected[owner]["trace"][-1] if owner >= 0 else (0,)*w
            )
        if i:
            assert tuple(row["witness"][j] for j in row["state_in"]) == dense.next_state(rows[i-1])
    assert f.terminal(p, dense.next_state(rows[-1])) == expected[-1]["trace"][-1]
    if mode == "chain":
        assert alloc.slots == 1
    elif mode == "window":
        assert alloc.slots == max(1, min(n-1, 2))


@pytest.mark.parametrize("mode", ["chain", "window", "star", "random"])
def test_constraint_count(mode):
    p = fixture(8, 2, mode)
    old = dense.schedule(p)[0]
    new = f.schedule(p)[0]
    k = f.layout(p).slots
    assert len(new["state_in"]) == 3 + 2*k
    assert len(old["constraints"]) - len(new["constraints"]) == 20*8*2 - 21*k*2


@pytest.mark.parametrize("kind", ["counter", "context", "register", "output", "selector"])
def test_tamper(kind):
    p = fixture(8, 2, "window")
    row = f.schedule(p)[3]
    count = len(row["state_in"])
    wire = {"counter": row["state_out"][0], "context": row["state_in"][1],
            "register": row["state_in"][3], "output": row["state_out"][-1],
            "selector": count+1+16*(count-3)}[kind]
    row["witness"][wire] += 1
    assert not r1cs_satisfied(row, dense.PALLAS_SCALAR)


def test_terminal_and_context():
    p = fixture(8, 2, "window")
    end = dense.next_state(f.schedule(p)[-1])
    with pytest.raises(ValueError):
        f.compile_step(p, end)
    with pytest.raises(ValueError):
        f.compile_step(replace(p, run="different"), f.initial(p))
    dest = f.layout(p).destination[-1]
    other = 1-dest
    altered = list(end)
    altered[3+2*other] = 1
    with pytest.raises(ValueError):
        f.terminal(p, tuple(altered))


def test_native_bundle():
    p = fixture(8, 2, "window")
    statement, advice = f.native_bundle(p)
    assert len(advice) == statement["steps"] == 8
    assert statement["z0"] == [advice[0][i] for i in statement["state_in"]]
    assert statement["zn"] == [advice[-1][i] for i in statement["state_out"]]
    assert all(len(a) == statement["wires"] for a in advice)


def scheduling_fixture():
    # Breadth-first order wastes one register across two separate joins.
    names = ("a", "b", "c", "d", "x", "y", "sink")
    parents = ((), (), (), (), ("a", "b"), ("c", "d"), ("x", "y"))
    jobs = tuple(Job(name, ps, (1,), ((2,),), (0,), 1) for name, ps in zip(names, parents, strict=True))
    return Plan("tenant", "branches", "policy", jobs)


def test_exact_schedule_against_exhaustive_orders():
    p = scheduling_fixture()
    proposed = f.optimize_order(p)
    best = len(p.jobs)
    for ordering in itertools.permutations(p.jobs):
        seen = set()
        for j in ordering:
            if not set(j.parents) <= seen:
                break
            seen.add(j.name)
        else:
            best = min(best, f.layout(replace(p, jobs=ordering)).slots)
    assert f.layout(p).slots == 4
    assert f.layout(proposed).slots == best == 3
    assert execute(proposed)[-1]["trace"][-1] == execute(p)[-1]["trace"][-1]


def test_greedy_never_regresses():
    p = scheduling_fixture()
    assert f.layout(f.optimize_order(p, exact_limit=0)).slots <= f.layout(p).slots
