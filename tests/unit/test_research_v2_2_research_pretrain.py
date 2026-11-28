"""Research corpus checks: deterministic controls, corruptions and split custody."""
from datagoblin.research_v2_2 import research_pretrain as m


def test_memory_ledgers():
    for i in range(128):
        d = m.memory_world(3,i)
        rows = m.memory_views(d)
        assert len(rows)==3 and {r[0] for r in rows}=={"decision","repair","matched_control"}
        assert rows[1][2]["verdict"]=="reject"


def test_native_target_contracts():
    d = dict(hypotheses=[{"class":c,"weight":1} for c in "ABBA"],
             tests=[dict(id="x",cost=1,bits=[0,0,1,1]),dict(id="y",cost=1,bits=[0,1,0,1])],
             budget=2,correct_reward=12,wrong_reward=-16)
    a,_ = m.solve_native("adaptive_evidence",d)
    assert a["value"]=="10/1"
    bad,_ = m.corrupt("adaptive_evidence",a)
    assert bad["value"]=="11/1"
    assert m.baseline("adaptive_evidence",d)[0]["value"]=="0/1"
    d = dict(mechanisms=["a","b"],error_bits=0,budget=3,experiments=[
        dict(id="c",cost=1,requires=[],predictions=[0,0]),
        dict(id="e",cost=2,requires=["c"],predictions=[0,1])])
    a,w = m.solve_native("robust_discrimination",d)
    assert a["cost"]==3 and w["proof_nodes"]>0
    assert m.corrupt("robust_discrimination",a)[0]["cost"]==4
