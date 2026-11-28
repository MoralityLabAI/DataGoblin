from itertools import combinations, product
from random import Random

import pytest

from sg15_lp import propose
from sg17_lp import best_mask, chain_lp, cut_lp
from sg17 import run
from datagoblin.training_gym.spectral_face import exact_bound
from datagoblin.training_gym.spectral_quotient import score


def fixture(seed):
    r=Random(seed);d,t=4,3
    return dict(d=d,T=t,bits=[r.randrange(1,4) for _ in range(d)],x=sorted([0]+[r.randrange(5) for _ in range(t)]),
                k=[r.randrange(d+1) for _ in range(t)],u=[[r.randrange(-4,9) for _ in range(d)] for _ in range(t)],
                **{'lambda':r.randrange(4)})


def optimum(q):
    a=[[sum(1<<j for j in c) for c in combinations(range(q['d']),k)] for k in q['k']]
    return max(score(q,list(s))['objective'] for s in product(*a))


def test_separation():
    for seed in range(64):
        q=fixture(1700000+seed);r=Random(seed);den=1+seed%3;e=[r.randrange(-20,21) for _ in q['k']]
        for j in range(q['d']):
            values=[sum(den*q['u'][i][j]-e[i] for i in range(q['T']) if mask&(1<<i))
                    -den*q['lambda']*q['bits'][j]*(q['x'][mask.bit_length()]-q['x'][0])
                    for mask in range(1<<q['T'])]
            m,v=best_mask(q,j,e,den)
            assert v==values[m]==max(values)


def test_chain_dual_parity():
    for seed in range(64):
        q=fixture(1701000+seed);opt=optimum(q)
        a,ia=propose(q);b,ib=chain_lp(q)
        assert a is not None and b is not None
        av,ad=a;bv,bd=b
        x,y=exact_bound(q,av,ad),exact_bound(q,bv,bd)
        assert x>=ad*opt and y>=bd*opt and x*bd==y*ad
        assert ia['objective']==pytest.approx(ib['objective'],abs=1e-7)


def test_cuts_capped_and_certificate_fallback():
    for seed in range(32):
        q=fixture(1702000+seed);opt=optimum(q)
        initial=[(1<<k)-1 for k in q['k']]
        for cap in (1,8):
            (e,den),info=cut_lp(q,initial,[0]*q['T'],round_cap=cap)
            assert exact_bound(q,e,den)>=den*opt and info['rounds']<=cap
        for arm in ('direct','chain','cuts8','cuts2_chain'):
            z=run(q,arm)
            assert z['objective']==z['upper']==opt


def test_fractional_cover():
    q=dict(d=3,T=3,bits=[1,1,1],x=[0,1,1,1],k=[1,1,1],u=[[10,10,-100],[-100,10,10],[10,-100,10]],**{'lambda':1})
    p,_=chain_lp(q)
    assert exact_bound(q,*p)*2==57*p[1]
    assert run(q,'chain')['objective']==28
