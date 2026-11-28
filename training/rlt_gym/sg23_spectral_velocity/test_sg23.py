from fractions import Fraction
from itertools import combinations, product
from random import Random

import pytest

from sg17_lp import chain_lp
from sg20 import finish
from sg20_delta import State, Task
from sg21_ray import Ray
from sg23_cone import Cone, endpoint
from sg23_fast import FastCone
from sg23 import ARMS, run
from sg23b import run as run_fast
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_quotient import score
from datagoblin.training_gym.spectral_routing import TaskError


def fixture(seed):
    r=Random(seed);d,t=4,3
    return dict(d=d,T=t,bits=[r.randrange(1,4) for _ in range(d)],x=[0,1,2,4],
                k=[r.randrange(d+1) for _ in range(t)],u=[[r.randrange(-4,9) for _ in range(d)] for _ in range(t)],
                **{'lambda':r.randrange(3)})


def schedules(q):
    return [list(a) for a in product(*[[sum(1<<j for j in js) for js in combinations(range(q['d']),k)] for k in q['k']])]


def mask_value(q,j,m,e,den):
    return sum(den*q['u'][i][j]-e[i] for i in range(q['T']) if m&(1<<i))-den*q['lambda']*q['bits'][j]*(q['x'][m.bit_length()]-q['x'][0])


def exhaustive_gap(anchor,target,v,den):
    q,r=anchor.q.thaw(),target.thaw();total=0
    for j,m in enumerate(anchor.visits):
        base=[mask_value(q,j,z,anchor.eta,anchor.den) for z in range(1<<q['T'])]
        delta=[mask_value(r,j,z,v,den)-mask_value(q,j,z,[0]*q['T'],den) for z in range(1<<q['T'])]
        total+=max(d for b,d in zip(base,delta,strict=True) if b==max(base))-delta[m]
    return total


def linear_ok(c,v,den):
    if c.impossible:return False
    if any((a is not None and w<a*den) or (b is not None and w>b*den)
           for w,a,b in zip(v,c.lower,c.upper,strict=True)):return False
    values=list(v)+[0]*len(c.aux)
    for (j,i),h in c.aux.items():values[h]=max(0,den*c.du[j][i]-v[i])
    return all(sum(values[i]*a for i,a in row)<=rhs*den for row,rhs in c.rows.items())


def test_active_box_epigraph_and_sweep():
    checks=0; coordinates=0; roots=0
    for seed in range(64):
        q=fixture(2300000+seed);p,_=chain_lp(q);e,den=p
        a=max(schedules(q),key=lambda z:score(q,z)['objective']);s=State.build(Task.from_task(q),e,den,a)
        if not s.certificate()['exact_face']:continue
        r=Random(seed);target=s.q.thaw();target['lambda']+=1
        target['u'][seed%q['T']][seed%q['d']]+=r.choice((-1,1))
        t=Task.from_task(target);c=Cone(s,t);fast=FastCone(s,t);roots+=1
        for n in range(12):
            v=[r.randrange(-12,13) for _ in e];d=1+n%3
            g=exhaustive_gap(s,t,v,d)
            assert g==c.gap(v,d)>=0 and linear_ok(c,v,d)==(g==0)
            checks+=1
        v=[r.randrange(-12,13) for _ in e];w=c.sweep(v)
        assert w==fast.sweep(v) and c.gap(w,1)<=c.gap(v,1)
        # Reverse order: later coordinates already use w, earlier ones use v.
        for i in range(q['T']-1,-1,-1):
            z=v[:i]+w[i:];g=c.gap(z,1)
            for h in range(-20,21):
                z[i]=h;assert g<=c.gap(z,1);coordinates+=1
        for mode in ('clamp','sweep','lp'):
            p,info=c.propose(v,mode)
            if p is not None:
                assert exhaustive_gap(s,t,*p)==0
                x=endpoint(s,t,*p)
                assert verify_certificate(target,list(x.schedule),list(x.eta),x.den)==x.certificate()
    print('exact-face roots',roots,'direction checks',checks,'coordinate comparisons',coordinates)
    assert roots>=60


def test_local_direction_is_not_finite_certificate():
    q=dict(d=2,T=1,bits=[1,1],x=[0,1],k=[1],u=[[2,0]],**{'lambda':0})
    s=State.build(Task.from_task(q),[1],1,[1]);t=s.q.thaw();t['u'][0][1]=4;t=Task.from_task(t)
    c=Cone(s,t);assert c.gap([0],1)==0
    x=endpoint(s,t,[0],1);assert x.certificate()['gap']>0
    assert verify_certificate(t.thaw(),[1],list(x.eta),x.den)==x.certificate()


def test_infeasible_direction_and_scope():
    q=dict(d=2,T=1,bits=[1,1],x=[0,1],k=[1],u=[[0,0]],**{'lambda':0})
    s=State.build(Task.from_task(q),[0],1,[1]);t=s.q.thaw();t['u'][0][1]=1
    c=Cone(s,Task.from_task(t));assert c.impossible
    assert all(c.propose([0],mode)[0] is None for mode in ('clamp','sweep','lp'))
    t['k'][0]=2
    with pytest.raises(TaskError):Cone(s,Task.from_task(t))
    q=dict(d=3,T=3,bits=[1,1,1],x=[0,1,1,1],k=[1,1,1],u=[[10,10,-100],[-100,10,10],[10,-100,10]],**{'lambda':1})
    s=State.build(Task.from_task(q),[19]*3,2,[1,2,1]);assert s.certificate()['certified'] and not s.certificate()['exact_face']
    with pytest.raises(TaskError):Cone(s,s.q)


def test_all_policies_exact_oracle():
    count=0
    for seed in range(12):
        q=fixture(2301000+seed);end=Task.from_task(q).thaw();end['lambda']+=4;ray=Ray.build(q,end,4)
        initial,_=finish(State.build(ray.base,[0]*q['T'],1,schedules(q)[0]));tasks=[ray.at(i).thaw() for i in range(1,5)]
        ref=None
        for arm in (*ARMS,'fast_sweep'):
            rows=run_fast(ray,tasks,initial,arm) if arm=='fast_sweep' else run(ray,tasks,initial,arm)
            if arm=='sweep':ref=rows
            if arm=='fast_sweep':assert [{k:v for k,v in z.items() if k!='cpu_ns'} for z in ref]==[{k:v for k,v in z.items() if k!='cpu_ns'} for z in rows]
            for task,z in zip(tasks,rows,strict=True):
                assert z['objective']==z['upper']==max(score(task,a)['objective'] for a in schedules(task));count+=1
    print('oracle policy runs',count)
