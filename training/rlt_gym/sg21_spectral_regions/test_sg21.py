from itertools import combinations, product
from random import Random

import pytest
from sg20_delta import State, Task
from sg21_ray import Ray, interval_value, region, verify_region
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_quotient import score
from datagoblin.training_gym.spectral_routing import TaskError


def fixture(seed):
    r=Random(seed); d,t=4,3
    q=dict(d=d,T=t,bits=[r.randrange(1,4) for _ in range(d)],x=[0,1,3,4],
           k=[r.randrange(d+1) for _ in range(t)],u=[[r.randrange(-4,9) for _ in range(d)] for _ in range(t)],**{'lambda':r.randrange(3)})
    return q


def test_radius_against_full_recomputation():
    from sg17_lp import chain_lp
    from datagoblin.training_gym.spectral_face import recover_face
    count=0
    for seed in range(60):
        q=fixture(2100000+seed); p,_=chain_lp(q);e,den=p
        actions=[[sum(1<<j for j in a) for a in combinations(range(q['d']),k)] for k in q['k']]
        a=list(max(product(*actions),key=lambda z:score(q,list(z))['objective']))
        state=State.build(Task.from_task(q),e,den,a)
        if not state.certificate()['certified']:continue
        H=8
        end=state.q.thaw();j=seed%q['d']
        if seed%4==0:end['bits'][j]+=H
        elif seed%4==1:end['x']=[x+H for x in q['x']]
        elif seed%4==2:end['lambda']+=H
        else:end['u'][seed%q['T']][j]+=H
        ray=Ray.build(q,end,H);p=region(ray,state,0);assert verify_region(ray,p)
        expected=[]
        for s in range(H+1):
            task=ray.at(s).thaw();c=verify_certificate(task,a,e,den)
            if c['certified']:
                expected.append(s);assert interval_value(p,s)==(c['objective'],c['objective'])
                assert c['objective']==max(score(task,list(z))['objective'] for z in product(*actions))
            count+=1
        assert expected==list(range(p['end']+1))
        # Re-anchor inside the same interval and recompute its maximal suffix.
        a0=p['end']//2;s=State.build(ray.at(a0),e,den,a)
        assert region(ray,s,a0)['end']==p['end']
    print('ray positions',count)
    assert count>=450


def test_fractional_negative_and_tampering():
    q=dict(d=3,T=3,bits=[1,1,1],x=[0,1,1,1],k=[1,1,1],u=[[-10,-10,-120],[-120,-10,-10],[-10,-120,-10]],**{'lambda':1})
    s=State.build(Task.from_task(q),[-21]*3,2,[1,2,1]);assert s.certificate()['certified'] and s.objective==-32
    end=s.q.thaw();end['x']=[x+10 for x in q['x']]
    ray=Ray.build(q,end,10);p=region(ray,s,0);assert p['end']==10 and p['bounds']==[-63,-63]
    assert interval_value(p,7)==(-32,-32) and verify_region(ray,p)
    with pytest.raises(TaskError):verify_region(ray,{**p,'bounds':[-63,-64]})
    with pytest.raises(TaskError):interval_value(p,11)


def test_nonaffine_and_invalid():
    q=fixture(13);a=Task.from_task(q);end=a.thaw();end['bits'][0]+=4;end['lambda']+=4
    with pytest.raises(TaskError):Ray.build(q,end,4)
    end=a.thaw();end['u'][0][0]+=1
    with pytest.raises(TaskError):Ray.build(q,end,4)
    end=a.thaw();end['k'][0]=(end['k'][0]+1)%5
    with pytest.raises(TaskError):Ray.build(q,end,4)
    with pytest.raises(TaskError):Ray.build(q,q,True)


def test_gated_reuses_interval_and_matches_baseline():
    from sg21 import run
    from sg21_gate import gated
    q=dict(d=3,T=3,bits=[1,1,1],x=[0,1,1,1],k=[1,1,1],u=[[-10,-10,-120],[-120,-10,-10],[-10,-120,-10]],**{'lambda':1})
    initial=State.build(Task.from_task(q),[-21]*3,2,[1,2,1])
    end=initial.q.thaw();end['x']=[x+6 for x in q['x']]
    ray=Ray.build(q,end,6);tasks=[ray.at(s).thaw() for s in range(1,7)]
    a=run(ray,tasks,initial,'delta');b=gated(ray,tasks,initial)
    assert sum(z['reused'] for z in b['rows'])==5 and len(b['proofs'])==1
    for x,y in zip(a['rows'],b['rows'],strict=True):
        assert all(x[k]==y[k] for k in ('schedule','eta','den','objective','upper','gap'))
