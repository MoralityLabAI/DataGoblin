from fractions import Fraction
from itertools import product
from random import Random

from sg24_events import envelope_cross, lines
from sg24_support import cross_support, region
from sg22_move import Line, region as binary, verify_region
from sg20_delta import Task, State
from sg21_ray import Ray
from sg17_lp import chain_lp
from test_sg24 import exhaustive, fixture, schedules
from datagoblin.training_gym.spectral_quotient import score


def test_support_crossing_exhaustive():
    checks=fallbacks=0
    for seed in range(160):
        r=Random(seed+2420000);H=8;terms=[];all_lines=[]
        for _ in range(3):
            t=1+seed%4;z=[r.randrange(-8,9) for _ in range(t)];dz=[r.randrange(-5,6) for _ in range(t)]
            p=[0]+[r.randrange(15) for _ in range(t)];dp=[0]+[r.randrange(-4,5) for _ in range(t)]
            full=exhaustive(z,dz,p,dp);a,b=max(full)
            terms.append((z,dz,p,dp,a,b));all_lines.append({(x-a,y-b) for x,y in full})
        brute=[(sum(a for a,b in c),sum(b for a,b in c)) for c in product(*all_lines)]
        for threshold in (0,1,3,7):
            roots=[Fraction(threshold-a,b) for a,b in brute if b>0 and (a+b*H>threshold if threshold==0 else a+b*H>=threshold)]
            expected=min(roots) if roots else None
            for cap in (1,128):
                got,n,fallback,_=cross_support(terms,H,threshold,cap)
                assert got==expected and n<=cap;checks+=1;fallbacks+=fallback
    print('support crossing checks',checks,'forced fallbacks',fallbacks)


def test_support_regions():
    checks=0
    for seed in range(80):
        q=fixture(2402000+seed);e,den=chain_lp(q)[0];a=max(schedules(q),key=lambda x:score(q,x)['objective'])
        s=State.build(Task.from_task(q),e,den,a)
        if not s.certificate()['certified']:continue
        H=8;end=s.q.thaw();end['lambda']+=H;ray=Ray.build(q,end,H)
        line=Line.secant((-1,[v+(seed+i)%7-3 for i,v in enumerate(e)],den),(0,e,den))
        p=region(ray,s,0,line);b=binary(ray,s,0,line)
        assert all(p[k]==v for k,v in b.items() if k!='probes') and verify_region(ray,p)
        checks+=1
    print('support region checks',checks)
