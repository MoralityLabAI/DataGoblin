from copy import deepcopy
from itertools import combinations, product
from random import Random

import pytest
from datagoblin.training_gym.spectral_quotient import score
from datagoblin.training_gym.spectral_routing import TaskError
from sg17_lp import chain_lp
from sg29_factor import best, column, replay, solve, value
from test_sg28 import fixture, schedules, triangle


def bridge_task(g, cost, bridges):
    q=triangle(g);q['lambda']=cost
    for i in range(bridges):
        q['T']+=1;q['u'].append([10]*q['d']);q['k'].append(1+i);q['x'].append(1)
    return q


def triangle_schedule(g, bridges):
    a=[]
    for i in range(g):
        a.extend([1<<(3*i+1),1<<(3*i+1),1<<(3*i+2)])
    for i in range(bridges):
        a.append(sum(1<<j for j in (1,2)[:i+1]))
    return a


def test_screen_against_exhaustive_masks():
    n=0
    for seed in range(64):
        q=fixture(2900000+seed);rng=Random(seed);den=1+seed%3;eta=[rng.randrange(-20,21) for _ in q['k']]
        for j in range(q['d']):
            z,p=column(q,j,eta,den)
            for i in range(q['T']):
                for inc in (False,True):
                    assert best(z,p,i,inc)==max(value(z,p,m) for m in range(1<<q['T']) if bool(m&(1<<i))==inc)
                    n+=1
    assert n==1536


def test_random_exact_and_caps():
    n=0
    for seed in range(96):
        q=fixture(2901000+seed);acts=schedules(q);rng=Random(seed)
        a=rng.choice(acts);eta=[rng.randrange(-20,21) for _ in q['k']];den=1+seed%3
        opt=max(score(q,x)['objective'] for x in acts)
        for cap in (1,64,100000):
            for k in (0,1,2):
                r=solve(q,a,eta,den,cap=cap,max_sep=k)
                c=replay(q,r,cap=cap,max_sep=k)
                assert r['objective']<=opt<=r['upper'] and c['gap']==r['gap'] and c['work']<=cap
                if cap==100000:assert r['objective']==r['upper']==opt
                n+=1
    assert n==864


def test_planted_separators():
    for g in (2,3,4,5):
        for bridges in (0,1,2):
            if 3*g+bridges>16:continue
            q=bridge_task(g,2,bridges);p,_=chain_lp(q);a=triangle_schedule(g,bridges)
            r=solve(q,a,*p,max_sep=bridges)
            assert r['objective']==r['upper']==g*26+10*sum(range(1,bridges+1))
            assert replay(q,r,max_sep=bridges)['gap']==0
            assert len(r['proof']['parts'])==g
            if bridges:assert r['proof']['separator']==list(range(3*g,3*g+bridges))


def test_tamper_and_negative():
    q=bridge_task(4,2,1);p,_=chain_lp(q);r=solve(q,triangle_schedule(4,1),*p)
    for kind in ('omit','overlap','screen','coupling','table','upper','cap'):
        a=deepcopy(r);cap=100000
        if kind=='omit':a['proof']['parts'].pop()
        elif kind=='overlap':a['proof']['parts'][0]['labels'].append(a['proof']['parts'][1]['labels'][0])
        elif kind=='screen':a['proof']['may'][0]^=1
        elif kind=='coupling':a['proof']['separator']=[]
        elif kind=='table':a['proof']['parts'][0]['table']=[] if a['proof']['parts'][0]['table'] else [[[],0]]
        elif kind=='upper':a['upper']-=1
        else:cap=1
        with pytest.raises(TaskError):replay(q,a,cap=cap)
    q=dict(d=3,T=3,bits=[1]*3,x=[0,1,1,1],k=[1]*3,u=[[-10,-10,-120],[-120,-10,-10],[-10,-120,-10]],**{'lambda':1})
    r=solve(q,[1,2,1],[-21]*3,2)
    assert r['objective']==r['upper']==-32 and replay(q,r)['gap']==0
