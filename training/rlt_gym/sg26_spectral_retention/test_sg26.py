from itertools import combinations, product
from random import Random

import pytest
from sg25_regions import Plane, Vertex
from sg26_memory import AcceptedMemory, eviction_key, measure, payload
from sg26 import run, ARMS
from sg20 import finish
from sg20_delta import State
from datagoblin.training_gym.spectral_quotient import score
from datagoblin.training_gym.spectral_routing import TaskError


def stable_plane():
    q=dict(d=3,T=3,bits=[1]*3,x=[0,1,1,1],k=[1]*3,u=[[-10,-10,-120],[-120,-10,-10],[-10,-120,-10]],**{'lambda':1})
    dirs=[[(i,j,s) for i in range(3) for j in range(3)] for s in (-1,2)]
    return Plane.build(q,4,dirs,[0,0])


def result(point):
    x,y=point;v=-32+3*(-x+2*y)
    return dict(schedule=[1,2,1],eta=[-21+2*(-x+2*y)]*3,den=2,objective=v,upper=v,gap=0)


def test_geometry_eviction_and_payload():
    p=stable_plane();size=len(payload(Vertex.checked(p,0,(0,0),[1,2,1],[-21]*3,2),1))
    assert size==72
    for kind in ('lru','area','weighted'):
        m=AcceptedMemory(p,3*size,kind)
        for i,point in enumerate(((0,0),(4,0),(0,4),(1,1))):
            _,drop=m.admit(point,i,result(point),1)
        assert m.used==m.peak==3*size
        if kind=='lru':assert drop==[0] and m.lookup((0,0)) is None
        else:assert drop==[3] and m.lookup((1,1)) is not None and m.lookup((0,0)) is not None
    m=AcceptedMemory(p,2*size,'area')
    for i,point in enumerate(((0,0),(4,0),(2,0))):
        _,drop=m.admit(point,i,result(point),1)
    assert drop==[2] and m.lookup((3,0)) is not None
    m=AcceptedMemory(p,size-1,'area');m.admit((0,0),0,result((0,0)),1)
    assert not m.entries and m.used==0


def test_acceptance_and_lru_touch():
    p=stable_plane();size=len(payload(Vertex.checked(p,0,(0,0),[1,2,1],[-21]*3,2),1))
    m=AcceptedMemory(p,3*size,'lru')
    for i,point in enumerate(((0,0),(4,0),(0,4))):m.admit(point,i,result(point),1)
    z=m.lookup((1,2));assert m.accept((1,2),z,4) and m.accept((1,2),z,4,full=True)
    for bad in ({**z,'upper':z['upper']+1},{**z,'eta':[0]*3},{**z,'eta':[]},{**z,'den':True},
                {**z,'schedule':[2,2,2]},{**z,'proof':{**z['proof'],'weights':[-1,2,3]}},
                {**z,'proof':{**z['proof'],'vertices':[0,1,99]}}):
        before=dict(m.clock)
        with pytest.raises(TaskError):m.accept((1,2),bad,5)
        assert before==m.clock
    with pytest.raises(TaskError):m.accept((4,4),z,5)
    with pytest.raises(TaskError):m.admit((1,1),3,{**result((1,1)),'objective':1},1)
    assert list(m.entries)==[0,1,2]
    # A singleton hit updates that vertex's recency before the next admission.
    a=m.lookup((0,0));assert a is not None
    # The hull fan may reference zero-weight vertices too; use an explicit singleton proof.
    a=dict(**result((0,0)),proof=dict(vertices=[0],weights=[1],den=1));assert m.accept((0,0),a,6)
    _,drop=m.admit((4,4),7,result((4,4)),1);assert drop==[1] and 0 in m.entries
    with pytest.raises(TaskError):m.accept((1,2),z,8)


def test_geometry_loss_and_cost_weight():
    p=stable_plane();vs=[Vertex.checked(p,i,xy,[1,2,1],result(xy)['eta'],2) for i,xy in enumerate(((0,0),(4,0),(0,4),(1,1)))]
    assert eviction_key(vs[-1],vs,100)==(0,0,0)
    assert eviction_key(vs[0],vs,5)[0]==5*eviction_key(vs[0],vs,1)[0]>0
    assert measure(vs[:1])==(0,0,1)


def test_small_oracle_policy_runs():
    count=0
    for seed in range(12):
        r=Random(2600000+seed);d,t=4,3
        q=dict(d=d,T=t,bits=[r.randrange(1,4) for _ in range(d)],x=[0,1,2,4],k=[r.randrange(d+1) for _ in range(t)],u=[[r.randrange(-4,9) for _ in range(d)] for _ in range(t)],**{'lambda':r.randrange(3)})
        p=Plane.build(q,2,[[(0,0,1)],[(1,1,-1)]],[seed%2,0]);points=[(2,0),(0,2),(2,2),(1,1),(1,0)]
        acts=[list(a) for a in product(*[[sum(1<<j for j in js) for js in combinations(range(d),k)] for k in q['k']])]
        _,a=finish(State.build(p.base,[0]*t,1,acts[0]));anchor=dict(**a,objective=a['certificate']['objective'])
        tasks=[p.at(pt) for pt in points]
        for arm in ARMS:
            rows=run(p,points,tasks,anchor,arm)
            for task,z in zip(tasks,rows,strict=True):
                opt=max(score(task.thaw(),a)['objective'] for a in acts)
                assert z['objective']==z['upper']==opt and z['bytes']<=900;count+=1
    assert count==300
