"""Independent composition, exchange-rule and saved-receipt audit."""
import itertools
import json
import random
from fractions import Fraction
from pathlib import Path
from statistics import mean
try:
    from .alg_descent_v4 import lower, mandatory, K
except ImportError:
    from alg_descent_v4 import lower, mandatory, K

def audit(path):
    independent={1:0}; partitions=0
    for d in range(2,15):
        costs=[]
        for k in K:
            if k>d: continue
            for cuts in itertools.combinations(range(1,d),k-1):
                ds=[b-a for a,b in zip((0,*cuts),(*cuts,d))]
                prefix=0; c=0
                for x in ds:
                    c+=(prefix+1)*(x+1); prefix+=x
                costs.append(c+sum(independent[x]+mandatory(x) for x in ds)-mandatory(d)); partitions+=1
        independent[d]=min(costs); assert independent[d]==lower(d)
    rng=random.Random(261104); orders=0
    for _ in range(300):
        k=rng.randint(2,5)
        cs=[rng.randint(1,100) for _ in range(k)]
        ps=[Fraction(rng.randint(1,10),10) for _ in range(k)]
        vs=[rng.randint(0,200) for _ in range(k)]
        fallback=rng.randint(0,300)
        def expected(order):
            value=Fraction(fallback)
            for i in reversed(order): value=cs[i]+ps[i]*vs[i]+(1-ps[i])*value
            return value
        order=sorted(range(k),key=lambda i:cs[i]/ps[i]+vs[i]); best=expected(order)
        for perm in itertools.permutations(range(k)):
            assert best<=expected(perm); orders+=1
    r=json.loads(Path(path).read_text()); checked=closed_uncached=0; failed=[]
    for a in r['runs']:
        assert a['work']<=3*a['n']**2 and a['calls']<=8*a['n'] and a['peak']<=a['cap']
        if a['closed'] and a['cap']==0:
            assert a['work']==mandatory(a['n'])+a['accepted_overhead']+a['rejected_work']; closed_uncached+=1
        if not a['closed']: failed.append((a['split'],a['root'],a['arm'],a['cap']))
        checked+=1
    for a in r['summary']:
        group=[b for b in r['runs'] if [b['split'],b['arm'],b['cap'],b['eviction']]==a['key']]
        assert len(group)==a['n'] and sum(b['closed'] for b in group)==a['closed']
        assert mean(b['work'] for b in group)==a['work']
    return dict(partitions=partitions,degree_bounds=independent,priority_instances=300,orders=orders,
                saved_rollouts=checked,closed_uncached=closed_uncached,summary_cells=len(r['summary']),failed=failed)
if __name__=='__main__':
    import sys
    print(json.dumps(audit(sys.argv[1] if len(sys.argv)>1 else 'receipt_v4.json'),sort_keys=True))
