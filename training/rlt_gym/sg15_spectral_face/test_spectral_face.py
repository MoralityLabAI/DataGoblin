from itertools import combinations, product
from random import Random

import pytest

from datagoblin.training_gym.spectral_face import face, recover_face, verify_certificate
from datagoblin.training_gym.spectral_quotient import score
from datagoblin.training_gym.spectral_routing import TaskError
from sg15_lp import propose


def fixture(seed):
    r=Random(seed);d,T=4,3
    return dict(d=d,T=T,bits=[r.randrange(1,4) for _ in range(d)],x=sorted([0]+[r.randrange(5) for _ in range(T)]),
                k=[r.randrange(d+1) for _ in range(T)],u=[[r.randrange(-4,9) for _ in range(d)] for _ in range(T)],
                **{'lambda':r.randrange(4)})


def exhaustive(q):
    return max(score(q,list(a))['objective'] for a in product(*[
        [sum(1<<j for j in js) for js in combinations(range(q['d']),k)] for k in q['k']]))


def test_boxes_and_face_exhaustive():
    comparisons=0;exact=infeasible=0
    for seed in range(128):
        q=fixture(1500000+seed);r=Random(seed);den=1+seed%3
        eta=[r.randrange(-12,13) for _ in range(q['T'])]
        bound,boxes=face(q,eta,den)
        total=sum(x*k for x,k in zip(eta,q['k'],strict=True))
        for j,domain in enumerate(boxes):
            vals={}
            for m in range(1<<q['T']):
                u=sum(den*q['u'][i][j]-eta[i] for i in range(q['T']) if m&(1<<i))
                c=den*q['lambda']*q['bits'][j]*(q['x'][m.bit_length()]-q['x'][0])
                vals[m]=u-c
            want={m for m,v in vals.items() if v==max(vals.values())}
            got={m for m in vals if any(m&a==a and m&~b==0 for a,b in domain)}
            assert want==got
            total+=max(vals.values());comparisons+=len(vals)
        opt=exhaustive(q);assert total==bound>=den*opt
        z=recover_face(q,eta,den,node_cap=10000)
        assert z['status']==('exact' if bound==den*opt else 'infeasible_face')
        exact+=z['status']=='exact';infeasible+=z['status']=='infeasible_face'
        if z['schedule'] is not None:
            assert verify_certificate(q,z['schedule'],eta,den)['exact_face']
    print('mask comparisons',comparisons,'exact',exact,'infeasible',infeasible)


def test_lp_rational_bound_and_face():
    counts={'exact':0,'infeasible_face':0}
    for seed in range(48):
        q=fixture(1510000+seed);v,info=propose(q);assert v is not None
        eta,den=v;opt=exhaustive(q)
        z=recover_face(q,eta,den,node_cap=10000)
        assert z['bound_num']>=den*opt
        assert z['status']==('exact' if z['bound_num']==den*opt else 'infeasible_face')
        counts[z['status']]+=1
        if z['schedule'] is not None:
            assert verify_certificate(q,z['schedule'],eta,den)['objective']==opt
    print('LP small roots',counts)


def test_cap_and_invalid():
    q=dict(d=3,T=2,bits=[1,1,1],x=[0,0,0],k=[1,1],u=[[0]*3,[0]*3],**{'lambda':0})
    z=recover_face(q,[0,0],node_cap=1)
    assert z['status']=='unknown'
    z=recover_face(q,[0,0],node_cap=100)
    assert z['status']=='exact' and z['objective']==0
    for eta,den in (([0],1),([0,True],1),([0,0],0),([0,0],True)):
        with pytest.raises(TaskError):recover_face(q,eta,den)
    with pytest.raises(TaskError):verify_certificate(q,[0,0],[0,0])


def test_fractional_triangle_cover():
    # Three labels each cover two visits. Each activated label costs one.
    # The relaxed cover costs 3/2; an integer schedule needs two labels.
    q=dict(d=3,T=3,bits=[1,1,1],x=[0,1,1,1],k=[1,1,1],
           u=[[10,10,-100],[-100,10,10],[10,-100,10]],**{'lambda':1})
    eta=[19,19,19];den=2
    assert exhaustive(q)==28
    z=recover_face(q,eta,den,node_cap=1000)
    assert z['bound_num']==57 and z['status']=='infeasible_face'
    cert=verify_certificate(q,[1,2,1],eta,den)
    assert cert['certified'] and not cert['exact_face'] and cert['gap']==0
    v,_=propose(q);assert v is not None
    assert verify_certificate(q,[1,2,1],*v)['certified']
