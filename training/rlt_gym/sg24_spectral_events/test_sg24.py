from fractions import Fraction
from itertools import combinations, product
from random import Random

import pytest
from sg17_lp import chain_lp
from sg20 import finish
from sg20_delta import State, Task
from sg21_ray import Ray
from sg22_move import Line, region as binary, value, verify_region
from sg24_events import lines, envelope_cross, region
from sg24 import ARMS, run
from datagoblin.training_gym.spectral_quotient import score
from datagoblin.training_gym.spectral_routing import TaskError


def fixture(seed):
    r=Random(seed);d,t=4,3
    return dict(d=d,T=t,bits=[r.randrange(1,4) for _ in range(d)],x=[0,1,2,4],
                k=[r.randrange(d+1) for _ in range(t)],u=[[r.randrange(-4,9) for _ in range(d)] for _ in range(t)],**{'lambda':r.randrange(3)})


def schedules(q):
    return [list(a) for a in product(*[[sum(1<<j for j in js) for js in combinations(range(q['d']),k)] for k in q['k']])]


def exhaustive(z,dz,p,dp):
    return {(sum(z[i] for i in range(len(z)) if m&(1<<i))-p[m.bit_length()],
             sum(dz[i] for i in range(len(z)) if m&(1<<i))-dp[m.bit_length()]) for m in range(1<<len(z))}


def test_last_use_envelopes():
    checks=0
    for seed in range(160):
        r=Random(seed);t=1+seed%8;H=7
        z=[r.randrange(-10,11) for _ in range(t)];dz=[r.randrange(-8,9) for _ in range(t)]
        p=[0]+[r.randrange(20) for _ in range(t)];dp=[0]+[r.randrange(-7,8) for _ in range(t)]
        full=exhaustive(z,dz,p,dp);reduced=lines(z,dz,p,dp,H)
        assert reduced<=full and len(reduced)<=1+t*(t+1)
        for s in [Fraction(i,4) for i in range(4*H+1)]:
            assert max(a+b*s for a,b in full)==max(a+b*s for a,b in reduced);checks+=1
    print('envelope checks',checks)


def test_first_crossings():
    checks=0
    for seed in range(160):
        r=Random(seed);H=8;terms=[];all_lines=[]
        for _ in range(3):
            t=1+seed%4;z=[r.randrange(-8,9) for _ in range(t)];dz=[r.randrange(-5,6) for _ in range(t)]
            p=[0]+[r.randrange(15) for _ in range(t)];dp=[0]+[r.randrange(-4,5) for _ in range(t)]
            full=exhaustive(z,dz,p,dp);a,b=max(full)
            terms.append({(x-a,y-b) for x,y in lines(z,dz,p,dp,H)})
            all_lines.append({(x-a,y-b) for x,y in full})
        # Exhaustive product of genuine masks: aggregate envelope has these lines.
        brute=[(sum(a for a,b in c),sum(b for a,b in c)) for c in product(*all_lines)]
        for threshold in (0,1,3,7):
            roots=[Fraction(threshold-a,b) for a,b in brute if b>0 and (a+b*H>threshold if threshold==0 else a+b*H>=threshold)]
            got,_,_=envelope_cross(terms,H,threshold)
            assert got==(min(roots) if roots else None);checks+=1
    print('crossing checks',checks)


def test_regions_against_binary_and_oracle():
    count=0;crossings=0
    for seed in range(80):
        q=fixture(2400000+seed);e,den=chain_lp(q)[0];acts=schedules(q)
        a=max(acts,key=lambda z:score(q,z)['objective']);s=State.build(Task.from_task(q),e,den,a)
        if not s.certificate()['certified']:continue
        H=8;end=s.q.thaw();end['lambda']+=H
        if seed%2:end['u'][seed%q['T']][seed%q['d']]+=H
        ray=Ray.build(q,end,H)
        line=Line.secant((-1,[v+(seed+i)%7-3 for i,v in enumerate(e)],den),(0,e,den))
        p=region(ray,s,0,line);b=binary(ray,s,0,line)
        assert all(p[k]==v for k,v in b.items() if k!='probes') and verify_region(ray,p)
        crossings+=p['crossing'] is not None
        for j in range(H+1):
            task=ray.at(j).thaw();v,D=line.at(j);st=State.build(ray.at(j),v,D,a)
            opt=max(score(task,x)['objective'] for x in acts)
            assert st.objective<=opt<=st.certificate()['integer_upper']
            assert st.certificate()['certified']==(j<=p['end'])
            if j<=p['end']:assert value(p,j)['objective']==opt
            count+=1
    print('region positions',count,'finite crossings',crossings)
    assert count>=650 and crossings>0


def test_exact_integer_crossing_and_fractional_negative():
    q=dict(d=2,T=1,bits=[1,1],x=[0,1],k=[1],u=[[2,0]],**{'lambda':0})
    s=State.build(Task.from_task(q),[1],1,[1]);end=s.q.thaw();end['u'][0][1]=4
    ray=Ray.build(q,end,4);line=Line(0,(1,),(0,),1)
    p=region(ray,s,0,line);assert p['crossing']==[2,1] and p['end']==1
    q=dict(d=3,T=3,bits=[1]*3,x=[0,1,1,1],k=[1]*3,u=[[-10,-10,-120],[-120,-10,-10],[-10,-120,-10]],**{'lambda':1})
    s=State.build(Task.from_task(q),[-21]*3,2,[1,2,1]);end=s.q.thaw();end['u']=[[u-8 for u in row] for row in q['u']]
    ray=Ray.build(q,end,8);line=Line(0,(-21,)*3,(-2,)*3,2)
    p=region(ray,s,0,line);assert p['end']==8 and value(p,8)['objective']==-56
    assert verify_region(ray,p)
    with pytest.raises(TaskError):verify_region(ray,{**p,'bounds':[-63,-112]})
    with pytest.raises(TaskError):region(ray,s,0,Line(0,(0,)*3,(0,)*3,1))


def test_complete_policies():
    count=0
    for seed in range(12):
        q=fixture(2401000+seed);end=Task.from_task(q).thaw();end['lambda']+=4;ray=Ray.build(q,end,4)
        s,_=finish(State.build(ray.base,[0]*q['T'],1,schedules(q)[0]));tasks=[ray.at(i).thaw() for i in range(1,5)]
        for arm in ARMS:
            z=run(ray,tasks,s,arm)
            for task,r in zip(tasks,z['rows'],strict=True):
                assert r['objective']==r['upper']==max(score(task,a)['objective'] for a in schedules(task));count+=1
    print('policy oracle checks',count)
