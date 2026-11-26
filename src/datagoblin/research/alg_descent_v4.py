"""Potential-normalized public-witness certificate routing and cache sweep."""
import json, math, multiprocessing as mp, os, random, resource, sys, time
from collections import Counter, OrderedDict, defaultdict
from functools import lru_cache
from statistics import mean
import torch
from torch import nn
try:
    from .alg_descent_v2 import Root, verify
    from .alg_descent_v3 import eligible, heuristic
except ImportError:
    from alg_descent_v2 import Root, verify
    from alg_descent_v3 import eligible, heuristic
K=(2,3,4,8)
SEEDS=(17,31,53)
W,H,HIST=12,8,2

def mandatory(d):
    return d*(d-1)//2

def degrees(a):
    return tuple(r-l for l,r in a[0])

def overhead(ds):
    return sum(ds)+len(ds)+sum((len(ds)-1-i)*d for i,d in enumerate(ds))

@lru_cache(None)
def lower(d):
    if d==1:
        return 0
    best=math.inf
    for k in K:
        if k>d:
            continue
        dp={0:0}
        for i in range(k):
            nxt={}
            for used,val in dp.items():
                for x in range(1,d-used-(k-i-1)+1):
                    v=val+lower(x)+(k-1-i)*x
                    s=used+x
                    nxt[s]=min(nxt.get(s,math.inf),v)
            dp=nxt
        best=min(best,d+k+dp[d])
    return best

def structural(a):
    ds=degrees(a)
    return overhead(ds)+sum(lower(d) for d in ds)

def bucket(r,a):
    ds=degrees(a); d,k=sum(ds),len(ds)
    balanced=ds==tuple(d*(i+1)//k-d*i//k for i in range(k))
    return r.p,k,int(balanced)

def probability(prior,r,a):
    yes,total=prior.get(bucket(r,a),(0,0))
    return min(.995,max(.05,(yes+1)/(total+2)))

def features(r,a,scale):
    ds=degrees(a); d=sum(ds); f=a[3]
    return (f[1],f[2],f[3],f[4],f[5],f[8],f[9],bucket(r,a)[2],
            overhead(ds)/d/8,structural(a)/d/8,
            sum(x*math.log2(x) for x in ds)/d/math.log2(d),math.log2(d)/6 if scale else 0.)

def training_data():
    prior=defaultdict(lambda:[0,0]); rows=[]
    for i in range(80):
        r=Root(110000+i,8+i%13); r.oracle((0,r.n)); rng=random.Random(140000+i)
        for s,(_,_,vals) in sorted(r.dp.items()):
            bank=r.bank(s); d=s[1]-s[0]
            indices=[j for j,a in enumerate(bank) if eligible(a)]
            invalid=[j for j in indices if vals[j] is None]; rng.shuffle(invalid)
            for j in indices:
                a=bank[j]; valid=vals[j] is not None
                assert a[2]==mandatory(d)-sum(mandatory(x) for x in degrees(a))+overhead(degrees(a))
                excess=(vals[j]-mandatory(d)-structural(a))/d if valid else 0.
                assert excess>=0
                key=bucket(r,a); prior[key][0]+=int(valid); prior[key][1]+=1
                for h in range(min(HIST,len(invalid))+1):
                    if j not in invalid[:h]:
                        rows.append((r,a,tuple(bank[t] for t in invalid[:h]),float(valid),excess))
    return dict(prior),rows

class Model(nn.Module):
    def __init__(self,recurrent):
        super().__init__(); self.cell=nn.GRUCell(W,H) if recurrent else None
        width=24 if recurrent else 84
        self.head=nn.Sequential(nn.Linear(W+(H if recurrent else 0),width),nn.Tanh(),nn.Linear(width,2))
    def forward(self,x,history,mask):
        if self.cell is not None:
            h=x.new_zeros((len(x),H))
            for i in range(HIST):
                h=torch.where(mask[:,i:i+1],self.cell(history[:,i],h),h)
            x=torch.cat((x,h),-1)
        y=self.head(x)
        return nn.functional.softplus(y[:,0]),y[:,1]

def tensorize(prior,rows,scale):
    x=torch.tensor([features(r,a,scale) for r,a,*_ in rows]); h=torch.zeros(len(rows),HIST,W)
    mask=torch.zeros(len(rows),HIST,dtype=torch.bool); ys=torch.tensor([[v,e] for _,_,_,v,e in rows])
    ps=torch.tensor([probability(prior,r,a) for r,a,*_ in rows]).logit()
    for i,(r,_,hist,_,_) in enumerate(rows):
        if hist:
            h[i,:len(hist)]=torch.tensor([features(r,a,scale) for a in hist]); mask[i,:len(hist)]=True
    return x,h,mask,ys,ps

def fit(data,recurrent,seed):
    torch.manual_seed(seed); m=Model(recurrent); opt=torch.optim.AdamW(m.parameters(),lr=.003)
    x,h,mask,y,p=data
    for _ in range(12):
        for ix in torch.randperm(len(x)).split(256):
            ex,delta=m(x[ix],h[ix],mask[ix]); valid=y[ix,0].bool()
            loss=nn.functional.binary_cross_entropy_with_logits(p[ix]+delta,y[ix,0])
            if valid.any():
                loss=loss+nn.functional.smooth_l1_loss(ex[valid],y[ix,1][valid])
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()

def scores(r,bank,prior,arm,model,history):
    if arm=='heuristic':
        return [heuristic(a) for a in bank]
    st=[structural(a) for a in bank]
    if arm=='structural':
        return st
    p=[probability(prior,r,a) for a in bank]
    if model is None:
        return [s+a[2]*(1/q-1) for s,a,q in zip(st,bank,p)]
    scale=arm!='scale_free'; hs=[] if arm=='reset' else history[-HIST:]
    x=torch.tensor([features(r,a,scale) for a in bank]); h=torch.zeros(len(bank),HIST,W)
    mask=torch.zeros(len(bank),HIST,dtype=torch.bool)
    if hs:
        h[:,:len(hs)]=torch.tensor([features(r,a,scale) for a in hs]); mask[:,:len(hs)]=True
    ex,delta=model(x,h,mask); prob=(torch.tensor(p).logit()+delta).sigmoid().clamp(.05,.995).tolist()
    return [s+e*sum(degrees(a))+a[2]*(1/q-1) for s,e,a,q in zip(st,ex.tolist(),bank,prob)]

def retained_bytes(memo):
    # Conservative object-size estimate; shared key/int objects can be overcounted.
    return sys.getsizeof(memo)+sum(sys.getsizeof(k)+sum(map(sys.getsizeof,k))+sys.getsizeof(v)+sum(map(sys.getsizeof,v)) for k,v in memo.items())

@torch.no_grad()
def rollout(r,prior,arm,model=None,cap=0,eviction='lru'):
    memo,pending=OrderedDict(),Counter()
    work=calls=rejected_work=reject=hits=used=peak=scans=pending_peak=infer_ns=accepted_overhead=0
    object_peak=retained_bytes(memo); budget,callcap=3*r.n**2,8*r.n
    def insert(p,paid):
        nonlocal used,peak,object_peak,scans
        size=len(p)+16
        if not cap or size>cap or p in memo:
            return
        if eviction=='lru':
            while used+size>cap:
                _,old=memo.popitem(last=False); used-=old[0]
        elif used+size>cap:
            def value(k,e): return pending[k]*max(0,e[1]-1)
            scans+=len(memo)+1; victims=[]; free=cap-used
            for k in sorted(memo,key=lambda k:value(k,memo[k])/memo[k][0]):
                if free>=size: break
                victims.append(k); free+=memo[k][0]
            if value(p,(size,paid))<sum(value(k,memo[k]) for k in victims): return
            for k in victims: used-=memo.pop(k)[0]
        memo[p]=(size,paid); used+=size; peak=max(peak,used); object_peak=max(object_peak,retained_bytes(memo))
    def drop(c):
        if cap and len(c)>2:
            pending[c]-=1
            if pending[c]==0: del pending[c]
    def visit(s):
        nonlocal work,calls,rejected_work,reject,hits,pending_peak,infer_ns,accepted_overhead
        p=r.poly(s)
        if s[1]-s[0]==1: return len(p)==2 and p[-1]==1
        if cap and p in memo and work<budget:
            memo.move_to_end(p); hits+=1; work+=1; return True
        begin=work; bank=r.bank(s); remain={i for i,a in enumerate(bank) if eligible(a)}; history=[]
        while remain:
            start=time.perf_counter_ns(); sc=scores(r,bank,prior,arm,model,history); infer_ns+=time.perf_counter_ns()-start
            i=min(remain,key=lambda j:(sc[j],j)); remain.remove(i); spans,cs,cost,_=bank[i]
            if work+cost>budget or calls>=callcap: return False
            ok,paid=verify(p,cs,r.p); assert paid==cost; work+=paid; calls+=1
            if not ok:
                reject+=1; rejected_work+=paid; history.append(bank[i]); continue
            accepted_overhead+=overhead(degrees(bank[i])); assert all(c==r.poly(t) for c,t in zip(cs,spans))
            if cap:
                pending.update(c for c in cs if len(c)>2); pending_peak=max(pending_peak,len(pending))
            for j,(t,c) in enumerate(zip(spans,cs)):
                drop(c)
                if not visit(t):
                    for rem in cs[j+1:]: drop(rem)
                    return False
            insert(p,work-begin); return True
        return False
    start=time.perf_counter_ns(); closed=visit((0,r.n)); elapsed=time.perf_counter_ns()-start
    assert work<=budget and calls<=callcap and peak<=cap and not pending
    if closed and not cap: assert work==mandatory(r.n)+accepted_overhead+rejected_work
    return dict(closed=closed,work=work,calls=calls,rejects=reject,rejected_work=rejected_work,
                hits=hits,peak=peak,object_peak=object_peak,pending_peak=pending_peak,scans=scans,score_ns=infer_ns,elapsed_ns=elapsed,accepted_overhead=accepted_overhead)

def rss():
    with open('/proc/self/statm') as f: return int(f.read().split()[1])*os.sysconf('SC_PAGE_SIZE')

def cache_worker(conn,roots,prior,cap,eviction):
    start=rss(); sampled=start; out=[]
    for split,r in roots:
        out.append(dict(split=split,root=r.seed,n=r.n,arm='risk',seed=0,cap=cap,eviction=eviction,**rollout(r,prior,'risk',cap=cap,eviction=eviction)))
        sampled=max(sampled,rss())
    conn.send((out,dict(cap=cap,eviction=eviction,rss_start=start,rss_sampled_peak=sampled,rss_end=rss(),ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))); conn.close()

def summarize(out):
    ans=[]
    for key in sorted({(r['split'],r['arm'],r['cap'],r['eviction']) for r in out}):
        a=[r for r in out if (r['split'],r['arm'],r['cap'],r['eviction'])==key]
        ans.append(dict(key=key,n=len(a),closed=sum(r['closed'] for r in a),work=mean(r['work'] for r in a),rejected_work=mean(r['rejected_work'] for r in a),rejects=sum(r['rejects'] for r in a),elapsed_ms=mean(r['elapsed_ns'] for r in a)/1e6,score_ms=mean(r['score_ns'] for r in a)/1e6,hits=sum(r['hits'] for r in a),peak=max(r['peak'] for r in a),object_peak=max(r['object_peak'] for r in a),pending_peak=max(r['pending_peak'] for r in a),scans=sum(r['scans'] for r in a)))
    return ans

def paired(out,arm,cap=0,eviction='lru',control='risk',control_cap=0,control_eviction='lru'):
    def collect(a,c,e):
        d=defaultdict(list)
        for row in out:
            if (row['split'],row['arm'],row['cap'],row['eviction'])==('ood',a,c,e): d[row['root']].append(row['work'])
        return d
    a,b=collect(arm,cap,eviction),collect(control,control_cap,control_eviction); assert a.keys()==b.keys() and len(a)==32
    ds=[mean(a[k])-mean(b[k]) for k in sorted(a)]; rng=random.Random(99282); boots=sorted(mean(rng.choices(ds,k=len(ds))) for _ in range(5000))
    return dict(arm=arm,cap=cap,eviction=eviction,control=control,control_cap=control_cap,control_eviction=control_eviction,mean=mean(ds),wins=sum(x<0 for x in ds),ties=sum(x==0 for x in ds),losses=sum(x>0 for x in ds),bootstrap95=[boots[125],boots[4874]])

def main():
    torch.set_num_threads(1); prior,rows=training_data(); ds={s:tensorize(prior,rows,s) for s in (False,True)}
    models={(a,s):fit(ds[a!='scale_free'],a=='gru',s) for a in ('scale_free','scale_aware','gru') for s in SEEDS}
    print(f'trained {len(rows)} candidate/history examples',file=sys.stderr,flush=True)
    out=[]; roots=[]; bank_ns=0
    for split,base,sizes,count in [('iid',120000,(8,12,16,20),24),('ood',130000,(24,32,40,48),32)]:
        for i in range(count):
            r=Root(base+i,sizes[i%4]); start=time.perf_counter_ns()
            for l in range(r.n):
                for rr in range(l+1,r.n+1):
                    r.poly((l,rr))
                    if rr-l>1: r.bank((l,rr))
            bank_ns+=time.perf_counter_ns()-start; roots.append((split,r))
            arms=[(a,s,m) for (a,s),m in models.items()]+[('reset',s,models['gru',s]) for s in SEEDS]+[(a,0,None) for a in ('heuristic','structural','risk')]
            random.Random(r.seed+555).shuffle(arms)
            for a,s,m in arms: out.append(dict(split=split,root=r.seed,n=r.n,arm=a,seed=s,cap=0,eviction='lru',**rollout(r,prior,a,m)))
    print('routing evaluation complete',file=sys.stderr,flush=True)
    ctx=mp.get_context('fork'); memory=[]
    for cap,eviction in [(0,'lru')]+[(c,e) for c in (32,64,128,256,512) for e in ('lru','pending')]:
        recv,send=ctx.Pipe(duplex=False); proc=ctx.Process(target=cache_worker,args=(send,roots,prior,cap,eviction)); proc.start(); send.close(); a,mem=recv.recv(); proc.join(); assert proc.exitcode==0; memory.append(mem)
        if cap: out.extend(a)
        else:
            assert all(x['work']==next(y['work'] for y in out if y['root']==x['root'] and y['arm']=='risk' and y['cap']==0) for x in a)
            zero_cache=a
        print(f'cache {cap} {eviction} complete',file=sys.stderr,flush=True)
    comparisons=[paired(out,a) for a in ('heuristic','structural','scale_free','scale_aware','gru','reset')]
    comparisons.extend(paired(out,'risk',c,'pending','risk',c,'lru') for c in (32,64,128,256,512))
    print(json.dumps(dict(examples=len(rows),prior_buckets=len(prior),params={a:sum(p.numel() for p in models[a,17].parameters()) for a in ('scale_free','scale_aware','gru')},bank_ms=bank_ns/1e6,summary=summarize(out),paired=comparisons,rss=memory,zero_cache=zero_cache,runs=out),sort_keys=True))

if __name__=='__main__':
    main()
