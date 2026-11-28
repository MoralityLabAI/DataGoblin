"""RV23-D: exact union equivalence, bound safety, proof replay and limits."""
import copy
from itertools import combinations
from random import Random

import pytest

from datagoblin.research_v2_2 import dependency_compact as m


def small(seed):
    rng = Random(seed)
    n = 4+seed%5
    rows = [dict(id=f"e{i}",cost=rng.randint(1,5),
                 requires=[f"e{rng.randrange(n)}"] if rng.random()<.45 else [],
                 predictions=[rng.randrange(2) for _ in range(3)]) for i in range(n)]
    return dict(mechanisms=["a","b","c"],experiments=rows,error_bits=seed%2,budget=3+seed%20)


def test_exact_relaxations():
    comparisons = 0
    for seed in range(40):
        d = small(seed)
        for mask in range(1 << len(d["experiments"])):
            p = m.CompactSearch(d,"quotient_cached",65536)
            if any(p.closures[i]&~mask for i in p.indices(mask)):
                continue
            for j,sep in enumerate(p.seps):
                need = max(0,p.need-(mask&p.smasks[j]).bit_count())
                left = [i for i in sep if not mask>>i&1]
                if not need or len(left)<need:
                    continue
                union_prices = []
                for combo in combinations(left,need):
                    u = 0
                    for i in combo:
                        u |= p.closures[i]
                    union_prices.append(p.cost(u&~mask))
                expected = min(union_prices)
                assert p.control(mask,left,need) <= expected
                assert p.quotient(mask,j,need,True) == expected
                assert p.quotient(mask,j,need,True) == expected
                assert p.stats["residual_hits"] == 1
                assert p.atlas_bound(mask,j) == expected
                comparisons += 1
    assert comparisons > 100


def test_proof_and_fallback():
    for seed in range(80):
        d = small(seed)
        expected = m.exhaustive(d)
        for mode in m.ARMS:
            for cap in (0,m.MAX_TERMS) if mode in m.ARMS[4:] else (0,):
                a,s,p = m.solve(d,mode,65536,cap)
                assert a.get("cost")==expected and s["certified"]
                assert s["relax_terms"]<=cap and s["closure_terms"]<=cap
                assert m.verify(d,p)["verified"]
                assert (a["status"]=="infeasible")== (expected is None)
        _,s,p = m.solve(d,"quotient_cached",1,0)
        assert s["inspected"]==1 and bool(p)==bool(s["certified"])


def test_control_shared_cost_and_proof_tamper():
    d = dict(mechanisms=["a","b"],error_bits=1,budget=8,experiments=[
        dict(id="c",cost=3,requires=[],predictions=[0,0]),
        *[dict(id=f"e{i}",cost=2,requires=["c"],predictions=[0,1]) for i in range(3)]])
    a,s,p = m.solve(d,"control",16)
    assert a=={"status":"infeasible"} and s["inspected"]==1
    assert p["nodes"]==[[0,"control",0]]
    assert m.verify(d,p)["verified"]
    bad = copy.deepcopy(p)
    bad["schema"] = "rv23c.cover_proof.v1"
    with pytest.raises(ValueError,match="version"):
        m.verify(d,bad)
    bad = copy.deepcopy(p)
    bad["nodes"][0] = [0,"cost"]
    with pytest.raises(ValueError,match="invalid pruning"):
        m.verify(d,bad)
    d["budget"] = 9
    for mode in m.ARMS:
        a,s,p = m.solve(d,mode,16)
        assert a["cost"]==9 and s["certified"] and m.verify(d,p)["verified"]
