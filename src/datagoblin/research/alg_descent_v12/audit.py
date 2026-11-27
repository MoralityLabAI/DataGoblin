"""Independent set-based trace, bound and incumbent replay; no learned ranking replay."""
import hashlib,json,math,random
from functools import lru_cache
from statistics import mean
from pathlib import Path
from anytime import roots,BUDGETS
HERE=Path(__file__).resolve().parent


def audit_graph(g,rows):
    table=[(r.id,r.head,tuple(r.premises),r.cost) for r in g.rules]
    initial=frozenset(range(3));goal=g.goal
    scope=hashlib.sha256(json.dumps([g.n,sum(2**x for x in initial),goal,table],separators=(',',':')).encode()).hexdigest()
    def decode(mask):return frozenset(i for i in range(g.n) if mask//(2**i)%2)
    def proof_cost(proof,want):
        known=set(initial);c=0
        for i in proof:
            assert type(i) is int and 0<=i<len(table)
            rid,h,ps,w=table[i];assert rid==i and h not in known and set(ps)<=known
            known.add(h);c+=w
        assert frozenset(known)==want if want is not None else goal in known
        return c
    @lru_cache(None)
    def bounds(s):
        @lru_cache(None)
        def maxval(h):
            if h in s:return 0
            return min(w+max((maxval(p) for p in ps),default=0) for _,q,ps,w in table if q==h)
        @lru_cache(None)
        def additive(h):
            if h in s:return 0,None
            return min((w+sum(additive(p)[0] for p in ps),i) for i,q,ps,w in table if q==h)
        ids=set()
        def collect(h):
            _,i=additive(h)
            if i is None or i in ids:return
            ids.add(i)
            for p in table[i][2]:collect(p)
        collect(goal);p=tuple(sorted(ids,key=lambda i:table[i][1]));return maxval(goal),sum(table[i][3] for i in p),p
    @lru_cache(None)
    def transitions(s):
        out={}
        for i,h,ps,c in table:
            if h in s or not set(ps)<=s:continue
            t=s|{h}
            if t not in out or (c,i)<out[t]:out[t]=(c,i)
        return tuple((t,c,i) for t,(c,i) in out.items())
    @lru_cache(None)
    def optimal(s):
        return 0 if goal in s else min(c+optimal(t) for t,c,_ in transitions(s))
    opt=optimal(initial);fields=expansions=proof_steps=snapshots=0
    for row in rows:
        assert scope==row['scope'] and opt==row['optimum']
        b,U,inc=bounds(initial);assert proof_cost(inc,None)==U
        best={initial:0};opened={initial:(0,())} if b<U else {}
        at=0
        def improve(c,p):
            nonlocal U,inc
            if c<U:
                assert proof_cost(p,None)==c;U,inc=c,p
                for t,(gc,_) in list(opened.items()):
                    if gc+bounds(t)[0]>=U:del opened[t]
        for snap,cap in zip(row['snapshots'],BUDGETS):
            assert snap['budget']==cap
            while at<snap['expansions']:
                e=row['trace'][at];s=decode(e['state']);gc,path=opened.pop(s);assert gc==e['g'] and path==tuple(e['path']) and proof_cost(path,s)==gc and gc+bounds(s)[0]<U
                at+=1;expansions+=1
                _,f,p=bounds(s);improve(gc+f,path+p)
                for t,c,i in transitions(s):
                    new=gc+c
                    if new>=best.get(t,math.inf):continue
                    best[t]=new;opened.pop(t,None);p=path+(i,)
                    if goal in t:improve(new,p)
                    elif new+bounds(t)[0]<U:opened[t]=(new,p)
            L=min(U,min((gc+bounds(s)[0] for s,(gc,_) in opened.items()),default=U));assert L<=opt<=U
            want=dict(L=L,U=U,proof=list(inc),open=len(opened),certified=L==U,expansions=at,probes=at*len(table))
            for k,v in want.items():assert snap[k]==v,(row['root'],row['arm'],k,snap[k],v)
            assert at<=cap and (at==cap or not opened);assert proof_cost(snap['proof'],None)==U
            fields+=len(want);proof_steps+=len(inc);snapshots+=1
        assert at==len(row['trace'])
    return dict(root=g.seed,optimum=opt,oracle_states=optimal.cache_info().currsize,fields=fields,expansions=expansions,snapshots=snapshots,proof_steps=proof_steps)


def main():
    data=json.loads((HERE/'receipt.json').read_text());runs=data['runs'];audits=[]
    for split in ('iid','ood'):
        for g in roots(split):audits.append(audit_graph(g,[r for r in runs if r['root']==g.seed]))
        print(split,len(audits),flush=True)
    def root_metric(arm,cap,fn):
        out={}
        for r in runs:
            if r['split']=='ood' and r['arm']==arm:
                v=next(v for v in r['snapshots'] if v['budget']==cap);out.setdefault(r['root'],[]).append(fn(r,v))
        return {k:mean(v) for k,v in out.items()}
    fn=lambda r,v:(v['U']-v['L'])/r['optimum'];paired=[]
    for a in ('mlp','tied'):
        for cap in (4,16,32,64,128):
            x,y=root_metric(a,cap,fn),root_metric('hmax',cap,fn);d=[x[k]-y[k] for k in sorted(x)];rng=random.Random(12);boot=sorted(mean(rng.choices(d,k=len(d))) for _ in range(5000));paired.append(dict(arm=a,budget=cap,mean=mean(d),wins=sum(v<0 for v in d),ties=sum(v==0 for v in d),losses=sum(v>0 for v in d),interval=[boot[125],boot[4874]]))
    out=dict(roots=len(audits),**{k:sum(r[k] for r in audits) for k in ('oracle_states','fields','expansions','snapshots','proof_steps')},paired=paired,root_audits=audits,scope='independent set-based trace, bounds, proof and exact optimum; no priority/inference replay')
    (HERE/'audit.json').write_text(json.dumps(out,sort_keys=True));print({k:v for k,v in out.items() if k!='root_audits'})
if __name__=='__main__':main()
