from itertools import combinations, product
from random import Random
from fractions import Fraction

import pytest
from sg20_delta import State, Task
from sg20 import finish
from sg21_ray import Ray
from sg22_move import Line, correct, region, value, verify_region
from sg22 import ARMS, run
from sg17_lp import chain_lp
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_quotient import score
from datagoblin.training_gym.spectral_routing import TaskError


def fixture(seed):
    r=Random(seed);d,t=4,3
    return dict(d=d,T=t,bits=[r.randrange(1,4) for _ in range(d)],x=[0,1,2,4],
                k=[r.randrange(d+1) for _ in range(t)],u=[[r.randrange(-4,9) for _ in range(d)] for _ in range(t)],**{'lambda':r.randrange(3)})


def schedules(q):
    return [list(a) for a in product(*[[sum(1<<j for j in js) for js in combinations(range(q['d']),k)] for k in q['k']])]


def test_rational_secant_and_caps():
    a=(1,[1,-3],2);b=(4,[3,5],7);line=Line.secant(a,b)
    for s in range(-5,12):
        e,den=line.at(s)
        for x,y,v in zip(a[1],b[1],e,strict=True):
            expected=Fraction(x,2)+(s-1)*(Fraction(y,7)-Fraction(x,2))/3
            assert Fraction(v,den)==expected
    with pytest.raises(TaskError):Line.secant(a,a)
    with pytest.raises(TaskError):Line.secant((0,[1],999983),(1,[1],999979))
    with pytest.raises(TaskError):Line.secant((0,[True],1),(1,[1],1))
    with pytest.raises(TaskError):line.at(True)


def test_moving_region_exact_maximality():
    count=0;covered=0;regions=0
    for seed in range(64):
        q=fixture(2200000+seed);e,den=chain_lp(q)[0];acts=schedules(q)
        a=max(acts,key=lambda x:score(q,x)['objective']);state=State.build(Task.from_task(q),e,den,a)
        if not state.certificate()['certified']:continue
        H=8;end=state.q.thaw();end['lambda']+=H
        if seed%2:end['u'][seed%q['T']][seed%q['d']]+=H
        ray=Ray.build(q,end,H)
        # Prior rational vector gives an arbitrary slope; most paths will end early.
        line=Line.secant((-1,[v+(seed+i)%7-3 for i,v in enumerate(e)],den),(0,e,den))
        p=region(ray,state,0,line);assert verify_region(ray,p);regions+=1
        actual=[]
        for s in range(H+1):
            task=ray.at(s).thaw();v,d=line.at(s);c=verify_certificate(task,a,v,d)
            opt=max(score(task,x)['objective'] for x in acts)
            assert c['objective']<=opt<=c['integer_upper'];count+=1
            if c['certified']:
                actual.append(s);z=value(p,s);assert z['objective']==z['upper']==opt;covered+=1
        assert actual==list(range(p['end']+1))
    print('positions',count,'covered',covered,'regions',regions)
    assert count>=500


def test_negative_fractional_moving_chord_and_tamper():
    q=dict(d=3,T=3,bits=[1,1,1],x=[0,1,1,1],k=[1,1,1],u=[[-10,-10,-120],[-120,-10,-10],[-10,-120,-10]],**{'lambda':1})
    # Every read utility decreases by one per step; move each multiplier by -1.
    state=State.build(Task.from_task(q),[-21]*3,2,[1,2,1]);assert state.certificate()['certified']
    end=state.q.thaw();end['u']=[[v-8 for v in row] for row in q['u']]
    ray=Ray.build(q,end,8);line=Line.secant((-1,[-19]*3,2),(0,[-21]*3,2))
    p=region(ray,state,0,line);assert p['end']==8 and p['bounds']==[-63,-111]
    assert verify_region(ray,p)
    for s in range(9):assert value(p,s)['objective']==-32-3*s
    with pytest.raises(TaskError):verify_region(ray,{**p,'bounds':[-63,-112]})
    with pytest.raises(TaskError):verify_region(ray,{**p,'slope':[0,0,0]})
    with pytest.raises(TaskError):value(p,9)


def test_scaled_coordinate_correction():
    for seed in range(64):
        q=fixture(2201000+seed);task=Task.from_task(q);a=schedules(q)[0];den=1+seed%3;e=[seed%11-i for i in range(q['T'])]
        old=verify_certificate(q,a,e,den);v,d=correct(task,e,den);new=verify_certificate(q,a,v,d)
        assert new['bound_num']<=old['bound_num'] and d==den


def test_all_policies_exact_oracle():
    n=0
    for seed in range(12):
        q=fixture(2202000+seed);end=Task.from_task(q).thaw();end['lambda']+=4;ray=Ray.build(q,end,4)
        a=schedules(q)[0];initial,z=finish(State.build(ray.base,[0]*q['T'],1,a));tasks=[ray.at(s).thaw() for s in range(1,5)]
        for arm in ARMS:
            out=run(ray,tasks,initial,arm)
            for task,r in zip(tasks,out['rows'],strict=True):
                opt=max(score(task,x)['objective'] for x in schedules(task))
                assert r['gap']==0 and r['objective']==r['upper']==opt;n+=1
    print('oracle runs',n)
