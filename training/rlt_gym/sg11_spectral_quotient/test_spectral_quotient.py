from itertools import combinations
from random import Random

import pytest
import sg07
from datagoblin.training_gym.spectral_quotient import SparseSession, coordinate, dual, quotient, quotient_count, replay, score
from datagoblin.training_gym.spectral_dual import DualSearchSession, coordinate_minimizer, dual_bound, verify_dual_trace
from datagoblin.training_gym.spectral_routing import TaskError, oracle


def test_quotient_exhaustive():
    checks = 0
    for seed in range(12):
        rng = Random(seed)
        d = 7
        row = [rng.randrange(-2,3) for _ in range(d)]
        for k in range(d+1):
            for s in range(1 << d):
                ref = {}
                for js in combinations(range(d),k):
                    m,u = sum(1 << j for j in js),sum(row[j] for j in js)
                    key = s | m
                    if key not in ref or u > ref[key][1]:
                        ref[key] = (m,u)
                out = {key:(m,u) for key,m,u in quotient(row,k,s)}
                assert out == ref and len(out) == quotient_count(d,k,s)
                checks += len(ref)
    assert checks == 87480


def test_interleaved_replay():
    for seed in range(40):
        q = sg07.task(981000+seed,'mixed',5,4)
        if seed % 3 == 0:
            q['k'][0]=0
        if seed % 5 == 0:
            q['k'][1]=5
        if seed % 7 == 0:
            q['lambda']=0
        init=sg07.initial(q)
        old=DualSearchSession(q,init,transition_cap=10000,proposal_cap=10000)
        new=SparseSession(q,init,class_cap=10000,raw_cap=10000,proposal_cap=10000)
        rng=Random(seed)
        for t in range(1,q['T']+1):
            for s in range(1<<q['d']):
                e=[rng.randrange(-10,11) for _ in range(t)]
                assert dual(q,t,s,e)==dual_bound(q,t,s,e)
                assert new.bound(t,s)==old.remaining_bound(t,s)
                i=rng.randrange(t)
                assert coordinate(q,t,s,e,i)==coordinate_minimizer(q,t,s,e,i)
        while old.live:
            t,s,_=rng.choice(old.frontier())
            e=[rng.randrange(-20,21) for _ in range(t)]
            old.propose(t,s,e);new.propose(t,s,e)
            assert new.live==old.live and new.seen==old.seen
            if (t,s) in old.live:
                old.advance(t,s);new.advance(t,s)
                assert new.live==old.live and new.seen==old.seen
            assert {k:v for k,v in new.receipt().items() if k!='classes'}==old.receipt()
        r=new.receipt()
        assert r['gap']==0 and r['objective']==oracle(q)['objective']
        assert verify_dual_trace(q,init,r['events'],transition_cap=10000,proposal_cap=10000)==old.receipt()
        assert replay(q,init,r['events'],class_cap=10000,raw_cap=10000,proposal_cap=10000)==r


def test_high_dimension_and_atomic_cap():
    assert quotient_count(32,8,(1<<28)-1)==16
    assert len(list(quotient(list(range(32)),8,(1<<28)-1)))==16
    q=dict(d=32,T=2,bits=[1]*32,x=[0,1,2],k=[8,8],u=[list(range(32)),list(range(32))],**{'lambda':0})
    s=SparseSession(q,[255,255],class_cap=16)
    before=s.receipt()
    with pytest.raises(TaskError):
        s.advance(2,0)
    assert s.receipt()==before
    assert not hasattr(s,'actions') and not hasattr(s,'weights')
    assert score(q,[255<<24,255<<24])['objective']==2*sum(range(24,32))
    # Single remaining visit is optimized directly without enumeration.
    q['T']=1;q['x']=[0,1];q['k']=[8];q['u']=q['u'][:1]
    s=SparseSession(q,[255],class_cap=1)
    s.advance(1,0)
    assert s.receipt()['gap']==0 and s.schedule==[255<<24] and s.classes==1
    assert s.transitions==10518300


def test_bad_inputs():
    q=sg07.task(1,'mixed',5,4); init=sg07.initial(q)
    for cap in (True,0,1000001):
        with pytest.raises(TaskError):
            SparseSession(q,init,class_cap=cap)
    with pytest.raises(TaskError):
        replay(q,init,[{'kind':'expand','state':[0,0],'gap':0}])
