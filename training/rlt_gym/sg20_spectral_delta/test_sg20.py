from itertools import combinations, product
from random import Random

import pytest
from sg20_delta import State, Task
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_primal import dual_round
from datagoblin.training_gym.spectral_quotient import score
from datagoblin.training_gym.spectral_routing import TaskError


def fixture(seed):
    r=Random(seed); d,t=5,4
    q=dict(d=d,T=t,bits=[r.randrange(1,4) for _ in range(d)],x=[0,1,3,4,6],
           k=[r.randrange(d+1) for _ in range(t)],u=[[r.randrange(-8,13) for _ in range(d)] for _ in range(t)],
           **{'lambda':r.randrange(4)})
    return q,[sum(1<<j for j in r.sample(range(d),k)) for k in q['k']],[r.randrange(-10,11) for _ in range(t)]


def check(s):
    q=s.q.thaw()
    assert s.certificate()==verify_certificate(q,list(s.schedule),list(s.eta),s.den)
    scaled={**q,'u':[[v*s.den for v in row] for row in q['u']], 'lambda':q['lambda']*s.den}
    assert s.rounded()==dual_round(scaled,list(s.eta))
    assert s.with_schedule(s.rounded()).certificate()==verify_certificate(q,s.rounded(),list(s.eta),s.den)


def test_differential_full_and_snapshot():
    n=0
    for seed in range(96):
        q,a,e=fixture(2001000+seed); s=State.build(Task.from_task(q),e,1+seed%5,a)
        check(s)
        for patch in ({'u':[[0,1,7],[3,1,-9]]},{'bits':[[3,5]]},{'k':[[2,seed%6]]},
                      {'x':[[0,-1],[2,2]]},{'lambda':(seed+1)%5},{'u':[[1,2,12]],'k':[[0,4]],'bits':[[0,8]]},{}):
            old=s; s,nc,np=s.apply(patch); full,_,_=old.apply(patch,full=True)
            assert s==full==old.replace(s.q.thaw())[0]
            check(s);check(old);n+=1
            assert 0<=nc<=np<=q['d']
    assert n==672


def test_exact_row_budget_repair():
    for seed in range(64):
        q,a,e=fixture(2002000+seed); old=State.build(Task.from_task(q),e,3,a)
        i=seed%q['T'];k=(q['k'][i]+1)%(q['d']+1)
        s,_,_=old.apply({'k':[[i,k]]}); nq=s.q.thaw()
        best=max(score(nq,a[:i]+[sum(1<<j for j in js)]+a[i+1:])['objective']
                 for js in combinations(range(q['d']),k))
        assert s.objective==best
        assert all(a[j]==s.schedule[j] for j in range(q['T']) if j!=i)
        check(s)


def test_invalid_atomic_and_price_invariance():
    q,a,e=fixture(2003000);s=State.build(Task.from_task(q),e,2,a)
    before=s.certificate();snap=s.q.thaw()
    for p in ({'u':[[0,0,True]]},{'u':[[0,0,1],[0,0,2]]},{'bits':[[5,1]]},
              {'k':[[0,6]]},{'x':[[1,9]]},{'lambda':-1},{'d':7},{'u':[[0,0]]}):
        with pytest.raises(TaskError):s.apply(p)
        assert s.certificate()==before and s.q.thaw()==snap
    shifted,n,m=s.apply({'x':[[i,v+10] for i,v in enumerate(q['x'])]})
    assert shifted.certificate()==before and n==m==0;check(shifted)
    zero,_,_=s.apply({'lambda':0}); after,n,m=zero.apply({'bits':[[0,99]],'x':[[1,2]]})
    assert n==m==0 and after.certificate()==zero.certificate();check(after)


def test_fractional_and_negative_floor():
    q=dict(d=3,T=3,bits=[1,1,1],x=[0,1,1,1],k=[1,1,1],
           u=[[10,10,-100],[-100,10,10],[10,-100,10]],**{'lambda':1})
    s=State.build(Task.from_task(q),[19]*3,2,[1,2,1]);assert s.certificate()['certified'] and s.bound==57
    s,n,m=s.apply({'u':[[i,j,v-20] for i,row in enumerate(q['u']) for j,v in enumerate(row)]})
    s=State.build(s.q,[-21]*3,2,list(s.schedule));assert s.bound==-63 and s.objective==-32
    assert s.certificate()['integer_upper']==-32 and s.certificate()['certified'];check(s)
