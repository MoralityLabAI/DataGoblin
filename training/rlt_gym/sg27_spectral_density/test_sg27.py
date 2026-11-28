from itertools import combinations, product
from random import Random

from sg25_regions import Plane, Vertex, hull, weights
from sg26_memory import payload
from sg27 import ARMS, CAP, DensityMemory, contains, run, samples, trajectory
from test_sg26 import stable_plane, result
from sg20 import finish
from sg20_delta import State
from datagoblin.training_gym.spectral_quotient import score


def test_contains_and_density():
    pts=list(product(range(5),repeat=2))
    for seed in range(48):
        v=[Vertex(i,p,(1,),(0,),1,0,0) for i,p in enumerate(Random(seed).sample(pts,5))]
        h=hull(v)
        for p in pts:
            exact=any((w:=weights([x.point for x in a],p)) is not None and min(w[0])>=0 for k in (1,2,3) for a in combinations(v,k))
            assert contains(h,p)==exact
    assert samples([(0,0)],4)=={(0,0):2,(1,0):1,(0,1):1}


def test_eviction_and_budget():
    p=stable_plane();size=len(payload(Vertex.checked(p,0,(0,0),[1,2,1],[-21]*3,2),1))
    m=DensityMemory(p,64+size*3)
    for i,point in enumerate(((0,0),(4,0),(0,4))):m.admit(point,i,result(point),1)
    for i in range(20):m.lookup((1,1))
    assert len(m.history)==8
    _,drop=m.admit((1,1),3,result((1,1)),1)
    assert drop==[3] and m.used==m.peak==size*3
    z=m.lookup((1,2));assert m.accept((1,2),z,4)


def test_oracle_runs():
    count=0
    for seed in range(8):
        r=Random(2700000+seed);d,t=4,3
        q=dict(d=d,T=t,bits=[r.randrange(1,4) for _ in range(d)],x=[0,1,2,4],k=[r.randrange(d+1) for _ in range(t)],u=[[r.randrange(-4,9) for _ in range(d)] for _ in range(t)],**{'lambda':r.randrange(3)})
        p=Plane.build(q,2,[[(0,0,1)],[(1,1,-1)]],[seed%2,0]);points=[(2,0),(0,2),(2,2),(1,1),(1,0)]
        acts=[list(a) for a in product(*[[sum(1<<j for j in js) for js in combinations(range(d),k)] for k in q['k']])]
        _,a=finish(State.build(p.base,[0]*t,1,acts[0]));anchor=dict(**a,objective=a['certificate']['objective'])
        tasks=[p.at(pt) for pt in points]
        for arm in ARMS:
            rows=run(p,points,tasks,anchor,arm)
            for task,z in zip(tasks,rows,strict=True):
                assert z['objective']==z['upper']==max(score(task.thaw(),a)['objective'] for a in acts)
                assert z['charged_bytes']<=CAP;count+=1
    assert count==200
    for i in range(12):
        _,_,_,points=trajectory(2719000+i,'mixed',bool(i%2),bool(i//2%2))
        assert len(points)==len(set(points))==32
