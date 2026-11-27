"""Certified anytime search: learned priority never controls pruning."""
import hashlib, heapq, json, math, random, sys, time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
try:
    from proof_planning import Graph, Oracle, Router, SEEDS
except ImportError:
    sys.path.insert(0,str(HERE.parent/'alg_descent_v11'))
    from proof_planning import Graph, Oracle, Router, SEEDS
BUDGETS=(0,4,16,32,64,128)


def roots(split):
    base,count,ns,fs=(710000,12,(10,11,12,13),('uniform','clustered')) if split=='iid' else (720000,24,(16,17,18),('shared','uniform','clustered'))
    return [Graph(base+i,ns[i%len(ns)],fs[(i+i//len(ns))%len(fs)] if split=='ood' else fs[(i//len(ns)+i)%2]) for i in range(count)]


def scope(g):
    return hashlib.sha256(json.dumps([g.n,g.initial,g.goal,[(r.id,r.head,r.premises,r.cost) for r in g.rules]],separators=(',',':')).encode()).hexdigest()


def check(g,proof):
    s=g.initial;cost=0
    for i in proof:
        assert type(i) is int and 0<=i<len(g.rules)
        r=g.rules[i];assert r.id==i and not s&(1<<r.head) and s&r.mask==r.mask
        s|=1<<r.head;cost+=r.cost
    assert s&g.goalbit
    return cost


def hmax(g,s):
    v=[0]*g.n
    for h in range(3,g.n):
        if not s>>h&1:
            v[h]=min(r.cost+max((v[p] for p in r.premises),default=0) for r in g.groups[h])
    return v[g.goal]


def completion(g,s):
    v=[0]*g.n;chosen={}
    for h in range(3,g.n):
        if not s>>h&1:
            v[h],i=min((r.cost+sum(v[p] for p in r.premises),r.id) for r in g.groups[h]);chosen[h]=i
    seen=set();ids=[]
    def visit(h):
        if s>>h&1 or h in seen:return
        seen.add(h);r=g.rules[chosen[h]]
        for p in r.premises:visit(p)
        ids.append(r.id)
    visit(g.goal)
    return sum(g.rules[i].cost for i in ids),tuple(sorted(ids,key=lambda i:g.rules[i].head))


@torch.no_grad()
def search(g,arm,model=None):
    start=time.perf_counter_ns();initial=g.initial
    count=dict(expansions=0,probes=0,hmax_calls=0,ff_calls=0,feature_calls=0,verify_calls=0,verify_steps=0,duplicates=0,pruned=0,reopened=0,infer_ns=0,hmax_ns=0,ff_ns=0,feature_ns=0,peak_open=0,peak_best=0,peak_queue=0)
    hc={};fc={};sc={};best={initial:0};opened={};pq=[];serial=0;trace=[];snap=[]
    def low(s):
        if s not in hc:
            t=time.perf_counter_ns();hc[s]=hmax(g,s);count['hmax_ns']+=time.perf_counter_ns()-t;count['hmax_calls']+=1
        return hc[s]
    def ff(s):
        if s not in fc:
            t=time.perf_counter_ns();fc[s]=completion(g,s);count['ff_ns']+=time.perf_counter_ns()-t;count['ff_calls']+=1
        return fc[s]
    def proofcheck(p,c):
        count['verify_calls']+=1;count['verify_steps']+=len(p);assert check(g,p)==c
    U,inc=ff(initial);proofcheck(inc,U)
    def admit(s,c,p):
        nonlocal serial
        b=low(s)
        if c+b>=U:count['pruned']+=1;return
        if s not in sc:
            if arm=='hmax':score,tie=b,b
            elif arm=='hmax_ff':score,tie=b,ff(s)[0]
            elif arm in ('ff','wff'):score=ff(s)[0]*(2 if arm=='wff' else 1);tie=score
            else:
                t=time.perf_counter_ns();v=g.public(s);count['feature_calls']+=1;count['feature_ns']+=time.perf_counter_ns()-t
                assert v[0]==b
                t=time.perf_counter_ns();pred=float(model(torch.tensor([v[3]],dtype=torch.float32))[0]);count['infer_ns']+=time.perf_counter_ns()-t
                score=max(float(b),v[2]+pred*7*(g.n-3));tie=score
            sc[s]=(score,tie)
        score,tie=sc[s];serial+=1;opened[s]=(c,p,b,serial);heapq.heappush(pq,(c+score,tie,c,serial,s))
        count['peak_open']=max(count['peak_open'],len(opened));count['peak_queue']=max(count['peak_queue'],len(pq))
    def update(c,p):
        nonlocal U,inc
        if c<U:
            proofcheck(p,c);U,inc=c,p
            for s,(gc,_,b,_) in list(opened.items()):
                if gc+b>=U:del opened[s];count['pruned']+=1
    admit(initial,0,());count['peak_best']=1
    for cap in BUDGETS:
        while count['expansions']<cap and opened:
            while pq:
                _,_,gc,k,s=heapq.heappop(pq)
                if s in opened and opened[s][3]==k:break
            else:raise AssertionError('frontier lost')
            gc,path,b,_=opened.pop(s);assert gc+b<U
            trace.append(dict(state=s,g=gc,path=path));count['expansions']+=1
            f,p=ff(s);update(gc+f,path+p)
            count['probes']+=len(g.rules)
            for t,c,i in g.ready(s):
                new=gc+c;old=best.get(t,math.inf)
                if new>=old:count['duplicates']+=1;continue
                if old<math.inf and t not in opened:count['reopened']+=1
                best[t]=new;opened.pop(t,None);count['peak_best']=max(count['peak_best'],len(best))
                if t&g.goalbit:update(new,path+(i,));continue
                admit(t,new,path+(i,))
        L=min(U,min((gc+b for gc,_,b,_ in opened.values()),default=U))
        assert L<=U
        snap.append(dict(budget=cap,L=L,U=U,proof=inc,open=len(opened),certified=L==U,elapsed_ns=time.perf_counter_ns()-start,**count))
    return dict(scope=scope(g),trace=trace,snapshots=snap)


def main():
    torch.set_num_threads(1);ck=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'models.pt';pack=torch.load(ck,weights_only=True,map_location='cpu');models={}
    for a in ('mlp','tied'):
        for seed in SEEDS:
            m=Router(a);m.load_state_dict(pack[f'{a}_{seed}']);models[a,seed]=m.eval()
    out=[];ref=[]
    for split in ('iid','ood'):
        for g in roots(split):
            arms=[(a,0,None) for a in ('hmax','hmax_ff','ff','wff')]+[(a,s,m) for (a,s),m in models.items()]
            random.Random(g.seed+12).shuffle(arms);rows=[]
            for a,s,m in arms:rows.append(dict(split=split,root=g.seed,n=g.n,family=g.family,arm=a,seed=s,**search(g,a,m)))
            oracle=Oracle(g);opt=oracle.value(g.initial);states=oracle.audit();ref.append(dict(split=split,root=g.seed,optimum=opt,states=states))
            for r in rows:
                for v in r['snapshots']:assert v['L']<=opt<=v['U']
                out.append(dict(optimum=opt,**r))
        print(split,len(out),flush=True)
    (HERE/'receipt.json').write_text(json.dumps(dict(protocol='91f863e0a0cda3bf7a76a947fbef939818de6cf0',checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),runs=out,reference=ref),sort_keys=True))

if __name__=='__main__':main()
