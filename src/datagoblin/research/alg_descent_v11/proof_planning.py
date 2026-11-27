"""ALG-DESCENT-11: public Horn planning; no solution witness in planner inputs."""
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import heapq, json, math, random, time, hashlib
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
SEEDS=(19,41,67); CAP=64; W=16; INIT=7

@dataclass(frozen=True)
class Rule:
    id:int
    head:int
    premises:tuple
    cost:int
    @property
    def mask(self):return sum(1<<p for p in self.premises)

class Graph:
    def __init__(self,seed,n,family):
        self.seed,self.n,self.family=seed,n,family
        self.goal=n-1;self.goalbit=1<<self.goal;self.initial=INIT
        rng=random.Random(seed);rules=[];groups=[[] for _ in range(n)];hub=max(3,n//3)
        for head in range(3,n):
            for j in range(rng.choice((2,3))):
                k=rng.randint(1,min(3,head));pool=list(range(head))
                if head==n-1:
                    pool=list(range(max(3,head//2),head));k=min(len(pool),rng.randint(2,3))
                elif family=='clustered':pool=list(range(max(0,head-5),head))
                if family=='shared' and head>hub+1 and head!=n-1 and j==0:
                    ps=tuple(sorted({hub,*rng.sample([p for p in pool if p!=hub],max(0,k-1))}))
                else:ps=tuple(sorted(rng.sample(pool,min(k,len(pool)))))
                r=Rule(len(rules),head,ps,rng.randint(1,7));rules.append(r);groups[head].append(r)
        self.rules=tuple(rules);self.groups=tuple(tuple(x) for x in groups)
        self.compact=tuple((r.id,1<<r.head,r.mask,r.cost) for r in rules)
        relevant={self.goal}
        for head in range(self.goal,2,-1):
            if head in relevant:
                for r in self.groups[head]:relevant.update(r.premises)
        self.relevant=sum(1<<p for p in relevant)
        self.registry=hashlib.sha256(json.dumps([(r.id,r.head,r.premises,r.cost) for r in rules],separators=(',',':')).encode()).hexdigest()
    def ready(self,s):
        best={}
        for i,b,m,c in self.compact:
            if s&b or s&m!=m:continue
            t=s|b
            if t not in best or (c,i)<best[t]:best[t]=(c,i)
        return tuple((t,c,i) for t,(c,i) in best.items())
    def public(self,s):
        mx=[0]*self.n;ad=[0]*self.n;sel=[None]*self.n
        for h in range(3,self.n):
            if s>>h&1:continue
            grp=self.groups[h]
            mx[h]=min(r.cost+max((mx[p] for p in r.premises),default=0) for r in grp)
            a,r=min(((r.cost+sum(ad[p] for p in r.premises),r) for r in grp),key=lambda z:(z[0],z[1].id))
            ad[h]=a;sel[h]=r
        chosen=set();todo=[self.goal];seen=set()
        while todo:
            h=todo.pop()
            if s>>h&1 or h in seen:continue
            seen.add(h);r=sel[h];assert r is not None
            chosen.add(r.id);todo.extend(r.premises)
        ff=sum(self.rules[i].cost for i in chosen)
        ready=[(i,b,m,c) for i,b,m,c in self.compact if not s&b and s&m==m]
        remaining=[(i,b,m,c) for i,b,m,c in self.compact if not s&b]
        goal_rules=self.groups[self.goal];coverage=[(r.mask&s).bit_count()/len(r.premises) for r in goal_rules]
        rel_total=max(1,(self.relevant&~INIT).bit_count());rel_missing=(self.relevant&~s).bit_count()
        nn=max(1,self.n-3);norm=7*nn
        feat=((s&~INIT).bit_count()/nn,rel_missing/rel_total,len(ready)/max(1,len(self.rules)),
              mx[self.goal]/norm,ad[self.goal]/norm,ff/norm,len(chosen)/nn,max(coverage),mean(coverage),min(r.cost for r in goal_rules)/7,
              mean([c for _,_,_,c in ready] or [0])/7,sum(bool(b&self.relevant) for _,b,_,_ in ready)/max(1,len(ready)),
              len(remaining)/len(self.rules),self.n/18,(ad[self.goal]-ff)/max(1,ad[self.goal]),
              sum((m&~s).bit_count() for _,_,m,_ in remaining)/max(1,3*len(remaining)))
        assert len(feat)==W and mx[self.goal]<=ff<=ad[self.goal]
        return mx[self.goal],ad[self.goal],ff,feat

class Oracle:
    def __init__(self,graph):self.graph=graph;self.values={}
    def value(self,s):
        if s in self.values:return self.values[s]
        if s&self.graph.goalbit:v=0
        else:
            vals=[c+self.value(t) for t,c,_ in self.graph.ready(s)];assert vals;v=min(vals)
        self.values[s]=v;return v
    def audit(self):
        g=self.graph;checked=0
        for s,v in self.values.items():
            if s&g.goalbit:assert v==0
            else:
                known={i for i in range(g.n) if s>>i&1}
                vals=[r.cost+self.values[s|(1<<r.head)] for r in g.rules if r.head not in known and set(r.premises)<=known]
                assert vals and min(vals)==v
            checked+=1
        return checked

def roots(split):
    base,count,sizes,families={'train':(610000,48,(10,11,12,13),('uniform','clustered')),'iid':(620000,16,(10,11,12,13),('uniform','clustered')),'ood':(630000,24,(16,17,18),('shared','uniform','clustered'))}[split]
    return [Graph(base+i,sizes[i%len(sizes)],families[(i+i//len(sizes))%len(families)] if split=='ood' else families[i%2]) for i in range(count)]

class Router(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='mlp':self.net=nn.Sequential(nn.Linear(W,24),nn.Tanh(),nn.Linear(24,1))
        else:self.enc=nn.Linear(W,12);self.mat=nn.Linear(12,12,bias=False);self.out=nn.Linear(12,1)
    def forward(self,x):
        if self.kind=='mlp':return self.net(x).squeeze(-1)
        b=self.enc(x);h=b.tanh()
        for _ in range(3):h=(b+self.mat(h)).tanh()
        return self.out(h).squeeze(-1)

def fit(x,y,kind,seed):
    torch.manual_seed(seed);m=Router(kind);opt=torch.optim.AdamW(m.parameters(),lr=.003)
    for _ in range(30):
        for i in torch.randperm(len(x)).split(128):
            loss=nn.functional.smooth_l1_loss(m(x[i]),y[i],beta=.15)
            opt.zero_grad();loss.backward();opt.step()
    return m.eval()

@torch.no_grad()
def search(g,arm,model=None,cap=CAP):
    """Search-only API: immutable public graph; no oracle parameter or labels."""
    start=time.perf_counter_ns();root=g.initial;heap=[];best={root:0};parent={};counter=0
    expansions=probes=duplicates=generated=features=infer_ns=0;feature_cache={}
    def estimates(ss):
        nonlocal features,infer_ns
        if not ss:return []
        vals=[]
        for s in ss:
            if arm=='ucs':vals.append((0,0,0,()));continue
            if s not in feature_cache:feature_cache[s]=g.public(s);features+=1
            vals.append(feature_cache[s])
        if model is not None and vals:
            t=time.perf_counter_ns();pred=model(torch.tensor([v[3] for v in vals])).tolist();infer_ns+=time.perf_counter_ns()-t
            return [0. if s&g.goalbit else max(float(v[0]),v[2]+p*7*(g.n-3)) for s,v,p in zip(ss,vals,pred)]
        ix={'ucs':0,'hmax':0,'hadd':1,'ff':2,'wff':2}[arm]
        return [v[ix]*(2 if arm=='wff' else 1) for v in vals]
    h=estimates([root])[0];heapq.heappush(heap,(h,h,0,counter,root));peak=1;goal=None
    while heap:
        _,_,gc,_,s=heapq.heappop(heap)
        if gc!=best[s]:continue
        if s&g.goalbit:goal=s;break
        if expansions>=cap:break
        expansions+=1;probes+=len(g.rules);fresh=[]
        for t,c,i in g.ready(s):
            generated+=1;new=gc+c
            if new>=best.get(t,math.inf):duplicates+=1;continue
            best[t]=new;parent[t]=(s,i);fresh.append((t,new))
        hs=estimates([t for t,_ in fresh])
        for (t,new),h in zip(fresh,hs):
            counter+=1;heapq.heappush(heap,(new+h,h,new,counter,t))
        peak=max(peak,len(heap))
    proof=[]
    if goal is not None:
        s=goal
        while s!=root:s,i=parent[s];proof.append(i)
        proof.reverse();cost=sum(g.rules[i].cost for i in proof);assert cost<=best[goal]
    else:cost=None
    return dict(closed=goal is not None,proof=proof,cost=cost,expansions=expansions,probes=probes,generated=generated,duplicates=duplicates,unique_states=len(best),peak_queue=peak,feature_states=features,infer_ns=infer_ns,elapsed_ns=time.perf_counter_ns()-start,registry=g.registry)

def verify_proof(g,row):
    assert row['registry']==g.registry
    known=set(range(3));cost=0
    if not row['closed']:assert not row['proof'] and row['cost'] is None;return 0
    for i in row['proof']:
        assert type(i) is int and 0<=i<len(g.rules)
        r=g.rules[i];assert r.id==i and r.head not in known and set(r.premises)<=known
        known.add(r.head);cost+=r.cost
    assert g.goal in known and cost==row['cost']
    return len(row['proof'])

def main():
    torch.set_num_threads(1);x=[];y=[];train_stats=[];t0=time.perf_counter_ns()
    for g in roots('train'):
        o=Oracle(g);opt=o.value(g.initial);checked=o.audit()
        states=sorted(s for s in o.values if not s&g.goalbit);random.Random(g.seed+77).shuffle(states)
        for s in states[:96]:
            mx,ad,ff,feat=g.public(s);v=o.values[s];assert mx<=v<=ff
            x.append(feat);y.append((v-ff)/(7*(g.n-3)))
        train_stats.append(dict(root=g.seed,n=g.n,family=g.family,states=checked,optimum=opt,examples=min(96,len(states))))
    print('teacher',len(x),sum(v['states'] for v in train_stats),flush=True)
    x=torch.tensor(x,dtype=torch.float32);y=torch.tensor(y,dtype=torch.float32)
    models={(kind,seed):fit(x,y,kind,seed) for kind in ('mlp','tied') for seed in SEEDS}
    torch.save({f'{kind}_{seed}':m.state_dict() for (kind,seed),m in models.items()},HERE/'models.pt');print('trained',flush=True)
    rows=[];reference=[];checks=0
    for split in ('iid','ood'):
        for g in roots(split):
            oracle=Oracle(g);opt=oracle.value(g.initial);n=oracle.audit();reference.append(dict(split=split,root=g.seed,n=g.n,family=g.family,optimum=opt,states=n,registry=g.registry));del oracle
            arms=[(a,0,None) for a in ('ucs','hmax','hadd','ff','wff')]+[(a,s,m) for (a,s),m in models.items()]
            random.Random(g.seed+332).shuffle(arms)
            for arm,seed,model in arms:
                r=search(g,arm,model);checks+=verify_proof(g,r)
                assert r['expansions']<=CAP and r['probes']==r['expansions']*len(g.rules)
                assert not r['closed'] or r['cost']>=opt
                if arm in ('ucs','hmax') and r['closed']:assert r['cost']==opt
                rows.append(dict(split=split,root=g.seed,n=g.n,family=g.family,arm=arm,seed=seed,optimum=opt,**r))
        print(split,len(rows),flush=True)
    result=dict(protocol='8af9d2066df289daf720ff2ab541ac94497f4fe0',train_examples=len(x),train=train_stats,reference=reference,runs=rows,proof_steps_checked=checks,params={a:sum(p.numel() for p in models[a,SEEDS[0]].parameters()) for a in ('mlp','tied')},total_ns=time.perf_counter_ns()-t0)
    (HERE/'receipt.json').write_text(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
