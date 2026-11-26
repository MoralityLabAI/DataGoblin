"""Independent arithmetic and certificate-mask audit; no model execution."""
import json, random
from statistics import mean
try:
    from .alg_descent_v2 import Root, verify, action_cost
    from .alg_descent_v3 import eligible
except ImportError:
    from alg_descent_v2 import Root, verify, action_cost
    from alg_descent_v3 import eligible


def overhead(ds):
    return sum(ds)+len(ds)+sum(sum(ds[:j]) for j in range(len(ds)))


def main():
    rng=random.Random(92103)
    local=tree=checked=invalid=masked=hidden=0
    def rec(d):
        nonlocal local
        if d==1: return 0,0
        cuts=sorted(rng.sample(range(1,d),rng.randint(1,min(7,d-1))))
        ds=[b-a for a,b in zip([0]+cuts,cuts+[d])]
        c=action_cost(ds); o=overhead(ds)
        assert 2*c==d*d-sum(x*x for x in ds)+2*o
        local+=1
        kids=[rec(x) for x in ds]
        return c+sum(t[0] for t in kids),o+sum(t[1] for t in kids)
    for _ in range(1000):
        d=rng.randint(2,128); c,o=rec(d)
        assert 2*c==d*d-d+2*o; tree+=1
    for seed in range(91000,91010):
        r=Root(seed,16); r.oracle((0,16))
        for s,(_,_,vals) in r.dp.items():
            parent=r.poly(s)
            for j,(spans,cs,cost,features) in enumerate(r.bank(s)):
                q=[1]
                for poly in cs:
                    q=[sum(q[t]*poly[k-t] for t in range(len(q)) if 0<=k-t<len(poly))%r.p for k in range(len(q)+len(poly)-1)]
                ok=tuple(q)==parent; checked+=1
                assert ok==(vals[j] is not None)
                fp=True
                for z in (0,1):
                    ev=lambda pol: sum(a*pow(z,i,r.p) for i,a in enumerate(pol))%r.p
                    v=1
                    for child in cs: v=v*ev(child)%r.p
                    fp &= v==ev(parent)
                assert fp==eligible((spans,cs,cost,features))
                assert fp or not ok
                if not ok:
                    invalid+=1; masked+=not fp; hidden+=fp
    # Retrospective teacher gap; offline privilege, no policy tuning.
    vals=[]
    for i in range(24):
        d=(24,32,40,48)[i%4]; r=Root(71000+i,d); v=r.oracle((0,d))[0]
        assert v>=d*(d-1)//2
        vals.append(dict(root=r.seed,n=d,oracle=v,mandatory_quadratic=d*(d-1)//2,overhead=v-d*(d-1)//2))
    print(json.dumps(dict(local_identities=local,tree_identities=tree,candidates=checked,invalid=invalid,masked=masked,hidden=hidden,oracle_mean=mean(v['oracle'] for v in vals),oracle_roots=vals),sort_keys=True))
if __name__=='__main__': main()
