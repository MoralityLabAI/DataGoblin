from fractions import Fraction
from itertools import combinations, product
from random import Random

import pytest
from audit_sg25 import chord
from sg25_regions import Memory, Plane, Vertex, certificate, mix_eta, verify_mix, weights
from sg17_lp import chain_lp
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_quotient import score
from datagoblin.training_gym.spectral_routing import TaskError


def fixture(seed):
    r=Random(seed);d,t=4,3
    return dict(d=d,T=t,bits=[r.randrange(1,4) for _ in range(d)],x=[0,1,2,4],
                k=[r.randrange(d+1) for _ in range(t)],u=[[r.randrange(-4,9) for _ in range(d)] for _ in range(t)],**{'lambda':r.randrange(3)})


def test_weights_and_hull():
    points=[(x,y) for x in range(5) for y in range(5)]
    checks=0
    for seed in range(64):
        r=Random(seed);pts=r.sample(points,6);m=Memory(6)
        for i,p in enumerate(pts):m.put(Vertex(i,p,(1,),(p[0]+2*p[1],),1,0,0))
        for p in points:
            valid=any((w:=weights(vs,p)) is not None and min(w[0])>=0 for k in (1,2,3) for vs in combinations(pts,k))
            z=m.lookup(p);assert (z is not None)==valid
            if z is not None:assert Fraction(z['eta'][0],z['den'])==p[0]+2*p[1]
            checks+=1
    assert checks==1600
    a,b,c=(4,0),(0,0),(0,4);v=weights([a,b,c],(1,2));assert v and min(v[0])>=0
    assert weights([(0,0),(4,0)],(2,1)) is None
    assert weights([(0,0),(1,0)],(2,0))==((-1,2),1)


def test_exhaustive_convex_certificates():
    checked=0;hits=0;triangles=0
    for seed in range(48):
        q=fixture(2500000+seed);d,t=q['d'],q['T'];r=Random(seed)
        # Arbitrary sparse objective directions, half include global price.
        dirs=[[(r.randrange(t),j,r.choice((-1,1)))] for j in (0,1)]
        plane=Plane.build(q,4,dirs,[seed%2,0]);mem=Memory(4);records={}
        acts=[list(a) for a in product(*[[sum(1<<j for j in js) for js in combinations(range(d),k)] for k in q['k']])]
        opt=lambda task:max(score(task,a)['objective'] for a in acts)
        for i,p in enumerate(((0,0),(4,0),(0,4),(4,4))):
            task=plane.at(p).thaw();e,den=chain_lp(task)[0];best=max(acts,key=lambda a:score(task,a)['objective']);c=verify_certificate(task,best,e,den)
            if c['certified']:
                v=Vertex.checked(plane,i,p,best,e,den);mem.put(v);records[i]=v
        for p in product(range(5),repeat=2):
            z=mem.lookup(p);task=plane.at(p).thaw();checked+=1
            if z is None:continue
            assert verify_mix(plane,p,z,records) and chord(plane,p,z,records)
            assert z['objective']==z['upper']==opt(task);hits+=1
            triangles+=len(z['proof']['vertices'])==3 and min(z['proof']['weights'])>0
    print('positions',checked,'covered',hits,'strict triangles',triangles)
    assert checked==1200 and triangles>0


def test_fractional_negative_and_tampering():
    q=dict(d=3,T=3,bits=[1]*3,x=[0,1,1,1],k=[1]*3,u=[[-10,-10,-120],[-120,-10,-10],[-10,-120,-10]],**{'lambda':1})
    dirs=[[(i,j,s) for i in range(3) for j in range(3)] for s in (-1,2)]
    plane=Plane.build(q,4,dirs,[0,0]);m=Memory(3);records={}
    for i,(x,y) in enumerate(((0,0),(4,0),(0,4))):
        v=Vertex.checked(plane,i,(x,y),[1,2,1],[-21+2*(-x+2*y)]*3,2);records[i]=v;m.put(v)
    z=m.lookup((1,2));assert z['objective']==z['upper']==-23 and z['den']==2
    assert verify_mix(plane,(1,2),z,records) and chord(plane,(1,2),z,records)
    for bad in ({**z,'upper':-22},{**z,'eta':[0,0,0]},{**z,'proof':{**z['proof'],'weights':[-1,2,3]}},{**z,'schedule':[2,2,2]}):
        with pytest.raises(TaskError):chord(plane,(1,2),bad,records)
    with pytest.raises(TaskError):chord(plane,(4,4),z,records)
    with pytest.raises(TaskError):chord(plane,(1,2),z,{0:records[0]})


def test_memory_capacity_and_validation():
    q=fixture(13);p=Plane.build(q,3,[[],[]],[0,0]);m=Memory(2)
    for i,point in enumerate(((0,0),(3,0),(0,3))):m.put(Vertex(i,point,(1,),(0,),1,0,0))
    assert [v.ident for v in m.vertices]==[2,1] and m.lookup((0,0)) is None
    with pytest.raises(TaskError):Plane.build(q,3,[[(0,0,1),(0,0,2)],[]],[0,0])
    with pytest.raises(TaskError):Plane.build(q,3,[[],[]],[100,0])
    with pytest.raises(TaskError):p.at((-1,0))
    with pytest.raises(TaskError):p.at((True,0))
    a=Vertex(0,(0,0),(1,),(1,),999983,0,0);b=Vertex(1,(2,0),(1,),(1,),999979,0,0)
    with pytest.raises(TaskError):mix_eta([a,b],(1,1),2)
    with pytest.raises(TaskError):certificate([a,Vertex(2,(2,0),(2,),(0,),1,0,0)],(1,0),(1,1),2)
