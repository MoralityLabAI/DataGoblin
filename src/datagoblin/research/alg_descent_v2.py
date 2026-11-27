"""Exact synthetic certificate routing, independent of discrete-log solving."""
import json
import random
from collections import OrderedDict
from statistics import mean
import torch
from torch import nn
try:
    from .alg_descent import product, verify, action_cost
except ImportError:
    from alg_descent import product, verify, action_cost

K = (2, 3, 4, 8)
W = 10


class Root:
    def __init__(self, seed, n):
        self.seed, self.n = seed, n
        rng = random.Random(seed)
        self.p = rng.choice((2, 3, 5, 7, 11))
        motif = [rng.randrange(self.p) for _ in range(rng.choice((1, 2, 4)))]
        self.fs = tuple((motif[i % len(motif)] if rng.random() < .9 else rng.randrange(self.p), 1) for i in range(n))
        self.rate = rng.choice((.15, .45, .75))
        self.polys, self.banks, self.dp = {}, {}, {}

    def poly(self, s):
        if s not in self.polys:
            self.polys[s] = product(self.fs[s[0]:s[1]], self.p)[0]
        return self.polys[s]

    def bank(self, s):
        if s in self.banks:
            return self.banks[s]
        l, r = s
        n, p = r-l, self.p
        rng = random.Random(self.seed*1000003+l*1009+r*17)
        parent, bank = self.poly(s), []
        for k in K:
            if k > n:
                continue
            for variant in range(2):
                cuts = sorted(rng.sample(range(l+1,r),k-1)) if variant else [l+n*i//k for i in range(1,k)]
                spans = tuple(zip((l,*cuts),(*cuts,r)))
                cs = [self.poly(t) for t in spans]
                if (k != 2 or variant) and rng.random() < self.rate:
                    i = rng.randrange(k)
                    c = list(cs[i])
                    if len(c)>=4 and rng.random()<.6:
                        c[1],c[2]=(c[1]-1)%p,(c[2]+1)%p
                    else:
                        c[0]=(c[0]+1)%p
                    cs[i]=tuple(c)
                cs = tuple(cs)
                ds = [len(c)-1 for c in cs]
                cost = action_cost(ds)
                z0,z1=1,1
                for c in cs:
                    z0,z1=z0*c[0]%p,z1*sum(c)%p
                f=(n/48,k/8,max(ds)/n,min(ds)/n,sum(d*d for d in ds)/n**2,cost/n**2,
                   float(z0==parent[0]),float(z1==sum(parent)%p),p/11,
                   sum(x==0 for c in cs for x in c)/sum(map(len,cs)))
                bank.append((spans,cs,cost,f))
        rng.shuffle(bank)
        self.banks[s]=bank
        return bank

    def oracle(self,s):
        if s[1]-s[0]==1:
            return 0,-1,()
        if s not in self.dp:
            vals=[]
            for spans,cs,cost,_ in self.bank(s):
                ok,paid=verify(self.poly(s),cs,self.p)
                assert paid==cost
                vals.append(cost+sum(self.oracle(t)[0] for t in spans) if ok else None)
            v,i=min((v,i) for i,v in enumerate(vals) if v is not None)
            self.dp[s]=(v,i,vals)
        return self.dp[s]


class Router(nn.Module):
    def __init__(self,recurrent):
        super().__init__()
        self.gru=nn.GRUCell(W,8) if recurrent else None
        h=20 if recurrent else 72
        self.score=nn.Sequential(nn.Linear(W+(8 if recurrent else 0),h),nn.Tanh(),nn.Linear(h,1))

    def forward(self,x,h):
        if self.gru is not None:
            x=torch.cat((x,h.unsqueeze(1).expand(-1,x.shape[1],-1)),dim=-1)
        return self.score(x).squeeze(-1)

    def advance(self,h,x):
        return self.gru(x,h) if self.gru is not None else h


def examples():
    rows=[]
    for i in range(40):
        r=Root(1000+i,8+i%9)
        r.oracle((0,r.n))
        rng=random.Random(7000+i)
        for s,(_,idx,vals) in sorted(r.dp.items()):
            fs=[a[3] for a in r.bank(s)]
            bad=[j for j,v in enumerate(vals) if v is None]
            rows.append((fs,idx,-1))
            if bad:
                rows.append((fs,idx,rng.choice(bad)))
    return rows


def fit(rows,recurrent,seed):
    torch.manual_seed(seed)
    rng=random.Random(seed)
    m=Router(recurrent)
    opt=torch.optim.Adam(m.parameters(),lr=.004)
    for _ in range(6):
        rng.shuffle(rows)
        for start in range(0,len(rows),128):
            batch=rows[start:start+128]
            x=torch.zeros(len(batch),8,W)
            mask=torch.zeros(len(batch),8,dtype=torch.bool)
            h=torch.zeros(len(batch),8)
            prev=torch.zeros(len(batch),W)
            used=torch.zeros(len(batch),1)
            for i,(fs,_,bad) in enumerate(batch):
                x[i,:len(fs)]=torch.tensor(fs)
                mask[i,:len(fs)]=True
                if bad>=0:
                    mask[i,bad]=False
                    prev[i],used[i]=x[i,bad],1
            logits=m(x,m.advance(h,prev)*used).masked_fill(~mask,-1e9)
            loss=nn.functional.cross_entropy(logits,torch.tensor([a[1] for a in batch]))
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def run(r,policy,m=None,cap=0,eviction='lru'):
    memo=OrderedDict()
    work=calls=bad=hits=peak=used=0
    budget,callcap=3*r.n**2,8*r.n
    def visit(s):
        nonlocal work,calls,bad,hits,peak,used
        p=r.poly(s)
        if s[1]-s[0]==1:
            return len(p)==2 and p[-1]==1
        if cap and p in memo and work<budget:
            memo.move_to_end(p); hits+=1; work+=1
            return True
        begin=work
        bank=r.bank(s)
        h=torch.zeros(1,8)
        x=torch.tensor([[a[3] for a in bank]]) if m is not None else None
        remain=set(range(len(bank)))
        while remain:
            if m is not None:
                scores=m(x,h)[0].tolist()
                i=max(remain,key=lambda j:(scores[j],-j))
            elif policy=='oracle':
                i=r.oracle(s)[1]
            elif policy=='wide':
                i=min(remain,key=lambda j:(-len(bank[j][0]),bank[j][2],j))
            elif policy=='fingerprint':
                i=min(remain,key=lambda j:(-sum(bank[j][3][6:8]),bank[j][2]+sum((t[1]-t[0])**2 for t in bank[j][0])/2,j))
            else:
                i=min(remain,key=lambda j:(len(bank[j][0]),bank[j][2],j))
            remain.remove(i)
            spans,cs,cost,_=bank[i]
            if calls>=callcap or work+cost>budget:
                return False
            ok,paid=verify(p,cs,r.p)
            assert paid==cost
            work+=paid; calls+=1
            if not ok:
                bad+=1
                if m is not None:
                    h=m.advance(h,x[:,i,:])
                continue
            assert all(c==r.poly(t) for c,t in zip(cs,spans))
            if not all(visit(t) for t in spans):
                return False
            size=len(p)+16
            if size<=cap and p not in memo:
                while used+size>cap:
                    v=next(iter(memo)) if eviction=='lru' else min(memo,key=lambda k:memo[k][1]/memo[k][0])
                    used-=memo.pop(v)[0]
                memo[p]=(size,work-begin); used+=size; peak=max(peak,used)
            return True
        return False
    closed=visit((0,r.n))
    assert work<=budget and calls<=callcap and peak<=cap
    return dict(closed=closed,work=work,calls=calls,rejects=bad,hits=hits,peak=peak)


def main():
    torch.set_num_threads(1)
    rows=examples()
    models={(k,s):fit(list(rows),k=='gru',s) for k in ('mlp','gru') for s in (11,23,37)}
    out=[]
    for split,base,sizes in [('iid',20000,(8,12,16)),('ood',30000,(24,32,48))]:
        for i in range(18):
            r=Root(base+i,sizes[i%3])
            for (k,s),m in list(models.items())+[((a,0),None) for a in ('binary','wide','fingerprint','oracle')]:
                for cap,e in ((0,'lru'),(128,'lru'),(128,'utility'),(512,'utility')) if k!='oracle' else ((0,'lru'),):
                    out.append(dict(split=split,root=r.seed,n=r.n,arm=k,seed=s,cap=cap,eviction=e,**run(r,k,m,cap,e)))
    summary=[]
    for key in sorted({(r['split'],r['arm'],r['cap'],r['eviction']) for r in out}):
        a=[r for r in out if (r['split'],r['arm'],r['cap'],r['eviction'])==key]
        b=[r for r in a if r['closed']]
        summary.append(dict(key=key,closed=len(b),runs=len(a),work=round(mean(r['work'] for r in a),2),solved_work=round(mean(r['work'] for r in b),2) if b else None,rejects=sum(r['rejects'] for r in a),hits=sum(r['hits'] for r in a)))
    print(json.dumps(dict(rows=len(rows),params={k:sum(p.numel() for p in models[k,11].parameters()) for k in ('mlp','gru')},summary=summary,runs=out),sort_keys=True))


if __name__=='__main__':
    main()
