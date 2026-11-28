from copy import deepcopy
from random import Random

import pytest
from sg30 import best, solve, replay
from sg28_integer import domains, split
from sg29_factor import column, value
from sg17_lp import chain_lp
from test_sg28 import fixture, schedules
from test_sg29 import bridge_task, triangle_schedule
from datagoblin.training_gym.spectral_quotient import score
from datagoblin.training_gym.spectral_routing import TaskError


def test_conditional_screen():
    for seed in range(64):
        q=fixture(3000000+seed);r=Random(seed);e=[r.randrange(-8,9) for _ in q['k']];d=1+seed%3
        for j in range(q['d']):
            z,p=column(q,j,e,d)
            for lo in range(q['T']+1):
                for hi in range(lo,q['T']+1):
                    for i in range(q['T']):
                        for on in (False,True):
                            expected=max((value(z,p,m) for m in range(1<<q['T']) if lo<=m.bit_length()<=hi and bool(m&(1<<i))==on),default=None)
                            assert best(z,p,lo,hi,i,on)==best(z,p,lo,hi,i,on,True)==expected


def test_caps_and_exact_oracle():
    for seed in range(48):
        q=fixture(3001000+seed);acts=schedules(q);a=acts[0];opt=max(score(q,x)['objective'] for x in acts);p,_=chain_lp(q)
        for cap,work in ((1,1),(7,64),(63,100000)):
            for memo in (False,True):
                r=solve(q,a,p,memo=memo,cap=cap,work_cap=work);c=replay(q,r,cap,work)
                assert c['objective']<=opt<=c['upper'] and c['nodes']<=cap and c['work']<=work
                if cap==63:assert c['gap']==0


def test_cross_branch_reuse_and_tamper():
    for g in (2,3,4,5):
        q=bridge_task(g,3,0);a=triangle_schedule(g,0);p,_=chain_lp(q)
        x=solve(q,a,p,force=(0,0));y=solve(q,a,p,memo=True,force=(0,0))
        assert x['gap']==y['gap']==0 and x['objective']==y['objective']==24*g
        assert x['work']==y['work'] and y['hits']==g-1 and y['computed']<x['computed']
        assert len(x['bank'])-len(y['bank'])==g-1
        assert replay(q,x)['gap']==replay(q,y)['gap']==0
    for mutation in ('bank','ref','cutoff','cap','omit'):
        z=deepcopy(y);cap=63
        if mutation=='bank':z['bank'][0].append([[0],-1])
        elif mutation=='ref':z['tree']['children'][0]['parts'][0][2]=len(z['bank'])
        elif mutation=='cutoff':z['tree']['children'][0]['cutoff']=z['objective']+1
        elif mutation=='cap':cap=1
        else:z['tree']['children'].pop()
        with pytest.raises(TaskError):replay(q,z,cap)


def test_partial_and_negative():
    q=bridge_task(5,3,1);a=triangle_schedule(5,1);p,_=chain_lp(q)
    for work in (1,64,400,1000,100000):
        z=solve(q,a,p,cap=3,work_cap=work,memo=True)
        assert z['objective']<=130<=z['upper'];replay(q,z,3,work)
    q=dict(d=3,T=3,bits=[1]*3,x=[0,1,1,1],k=[1]*3,u=[[-10,-10,-120],[-120,-10,-10],[-10,-120,-10]],**{'lambda':1})
    z=solve(q,[1,2,1],([-21]*3,2));assert replay(q,z)['objective']==-32 and z['gap']==0
