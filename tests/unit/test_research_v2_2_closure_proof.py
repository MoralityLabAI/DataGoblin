"""Independent certificate regression and deliberate proof corruption."""
import copy
from random import Random

import pytest

from datagoblin.research_v2_2 import closure_proof as m


def small(seed):
    rng = Random(seed)
    rows = []
    for i in range(4+seed%5):
        deps = [f"e{rng.randrange(i)}"] if i and rng.random()<.45 else []
        rows.append(dict(id=f"e{i}", cost=rng.randint(1,5), requires=deps,
                         predictions=[rng.randrange(2) for _ in range(3+seed%3)]))
    return dict(mechanisms=[f"h{i}" for i in range(3+seed%3)],experiments=rows,
                error_bits=seed%2,budget=4+seed%15)


def test_full_certificate_and_term_fallback():
    for seed in range(96):
        d = small(seed)
        optimum = m.exhaustive(d)
        for arm in m.ARMS:
            for term_cap in (0, m.MAX_TERMS) if arm.startswith("closure") else (0,):
                a,s,p = m.solve(d,arm,1 << len(d["experiments"]),term_cap)
                assert s["certified"] and s["closure_terms"] <= term_cap
                assert a.get("cost") == optimum
                assert (a["status"] == "infeasible") == (optimum is None)
                assert m.verify(d,p)["verified"]
                if optimum is not None:
                    assert m.verify_plan(d,a) == optimum
        a,s,p = m.solve(d,"closure_all",1)
        assert s["inspected"] == 1
        assert bool(p) == bool(s["certified"])


def test_dependency_union_and_tampering():
    d = dict(mechanisms=["a","b"],error_bits=1,budget=9,experiments=[
        dict(id="c",cost=3,requires=[],predictions=[0,0]),
        *[dict(id=f"e{i}",cost=2,requires=["c"],predictions=[0,1]) for i in range(3)]])
    a,s,p = m.solve(d,"closure_all",16)
    assert a["cost"] == 9 and s["certified"] and m.verify(d,p)["verified"]
    d["budget"] = 8
    a,s,p = m.solve(d,"closure_all",16)
    assert a == {"status":"infeasible"} and s["certified"]
    assert m.verify(d,p)["verified"]
    assert any(r[1] == "closure" for r in p["nodes"])
    for mutate in ("root","answer","missing","bound","orphan"):
        q = copy.deepcopy(p)
        if mutate == "root":
            q["root_id"] = "0"*64
        elif mutate == "answer":
            q["answer"] = {"status":"plan","experiments":[],"cost":0}
        elif mutate == "missing":
            q["nodes"] = []
        elif mutate == "bound":
            q["nodes"][0] = [0,"cost"]
        else:
            q["nodes"].append([1,"cost"])
        with pytest.raises(ValueError):
            m.verify(d,q)
    # Genuine branching proof with an omitted child must fail.
    d["budget"] = 9
    _,_,p = m.solve(d,"cover",16)
    assert len(p["nodes"]) > 1
    p["nodes"].pop()
    with pytest.raises(ValueError):
        m.verify(d,p)
