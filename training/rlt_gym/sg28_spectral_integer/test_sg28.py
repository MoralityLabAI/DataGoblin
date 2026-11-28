from itertools import combinations, product
from random import Random
from copy import deepcopy

import pytest
from sg28_integer import conditioned, domains, split, infeasible, check_witness, replay
from sg28 import solve
from sg17_lp import chain_lp
from sg11 import initial
from datagoblin.training_gym.spectral_quotient import score
from datagoblin.training_gym.spectral_routing import TaskError


def fixture(seed):
    r=Random(seed);d,t=4,3
    return dict(d=d,T=t,bits=[r.randrange(1,4) for _ in range(d)],x=sorted([0]+[r.randrange(5) for _ in range(t)]),k=[r.randrange(d+1) for _ in range(t)],u=[[r.randrange(-5,10) for _ in range(d)] for _ in range(t)],**{'lambda':r.randrange(4)})


def schedules(q):
    return [list(a) for a in product(*[[sum(1<<j for j in js) for js in combinations(range(q['d']),k)] for k in q['k']])]


def triangle(g):
    d=t=3*g;u=[[-100]*d for _ in range(t)]
    for n in range(g):
        for i in range(3):
            for j in (i,(i+1)%3):u[n*3+i][n*3+j]=10
    return dict(d=d,T=t,bits=[1]*d,x=[0]+[1]*t,k=[1]*t,u=u,**{'lambda':1})


def test_conditioned_exhaustive_and_witness():
    checks=feas=0
    for seed in range(64):
        q=fixture(2800000+seed);r=Random(seed);acts=schedules(q)
        for _ in range(12):
            box=tuple(tuple(sorted((r.randrange(4),r.randrange(4)))) for _ in range(4))
            den=r.randrange(1,4);eta=[r.randrange(-16,17) for _ in range(3)]
            total=sum(e*k for e,k in zip(eta,q['k'],strict=True))
            for j,(a,b) in enumerate(box):
                vals=[sum(den*q['u'][i][j]-eta[i] for i in range(3) if m&(1<<i))-den*q['lambda']*q['bits'][j]*(q['x'][m.bit_length()]-q['x'][0]) for m in range(8) if a<=m.bit_length()<=b]
                total+=max(vals);checks+=len(vals)
            assert conditioned(q,box,eta,den)==total//den
            legal=[a for a in acts if all(lo<=max([0,*[i+1 for i,m in enumerate(a) if m&(1<<j)]])<=hi for j,(lo,hi) in enumerate(box))]
            w=infeasible(q,box)
            assert bool(legal)==(w is None)
            if w is not None:assert check_witness(q,box,w)
            else:assert max(score(q,a)['objective'] for a in legal)<=total//den
            feas+=1
    print('mask evaluations',checks,'conditioned feasibility boxes',feas)


def test_replay_oracle_and_caps():
    n=0
    for seed in range(48):
        q=fixture(2801000+seed);opt=max(score(q,a)['objective'] for a in schedules(q));a=initial(q)
        p,_=chain_lp(q)
        for cap in (1,7,63):
            r=solve(q,a,cap=cap,use_milp=False,root=p)
            assert r['objective']<=opt<=r['upper'] and r['nodes']<=cap
            assert replay(q,r,cap)['gap']==r['gap'];n+=1
            if cap==63:assert r['gap']==0
    print('oracle receipts',n)


def test_disjunction_genuine_gap():
    for g in range(1,6):
        q=triangle(g);p,_=chain_lp(q);a=initial(q)
        assert conditioned(q,domains(q),*p)==(57*g)//2
        r=solve(q,a,cap=63,use_milp=False,root=p)
        assert r['objective']==r['upper']==28*g
        if g>1:assert r['nodes']>1
        assert replay(q,r)['gap']==0


def test_tampering_and_negative_floor():
    q=triangle(2);p,_=chain_lp(q);r=solve(q,initial(q),cap=63,root=p,use_milp=False)
    for mutation in ('omit','wrong_bound','split','gap','cap'):
        a=deepcopy(r);cap=127
        if mutation=='omit':a['tree']['children'].pop()
        elif mutation=='wrong_bound':a['tree']['children'][0]['upper']+=1
        elif mutation=='split':a['tree']['cut']=q['T']
        elif mutation=='gap':a['gap']=1
        else:cap=1
        with pytest.raises(TaskError):replay(q,a,cap)
    q=dict(d=3,T=3,bits=[1]*3,x=[0,1,1,1],k=[1]*3,u=[[-10,-10,-120],[-120,-10,-10],[-10,-120,-10]],**{'lambda':1})
    p=([-21]*3,2);r=solve(q,[1,2,1],root=p,cap=1,use_milp=False)
    assert r['objective']==r['upper']==-32 and replay(q,r)['gap']==0
