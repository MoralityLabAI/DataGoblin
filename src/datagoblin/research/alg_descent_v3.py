"""Masked Bellman residual and reuse-aware receipt cache."""
import json, random, time
from collections import Counter, OrderedDict
from statistics import mean
import torch
from torch import nn
try:
    from .alg_descent_v2 import Root, verify
except ImportError:
    from alg_descent_v2 import Root, verify
W,H,MAXA,HIST=10,8,8,3
SEEDS=(13,29,47)
def eligible(a):
    return bool(a[3][6] and a[3][7])
def heuristic(a):
    return a[2]+sum((r-l)**2 for l,r in a[0])/2
class Residual(nn.Module):
    def __init__(self,recurrent):
        super().__init__()
        self.cell=nn.GRUCell(W,H) if recurrent else None
        width=24 if recurrent else 70
        self.head=nn.Sequential(nn.Linear(W+(H if recurrent else 0),width),nn.Tanh(),nn.Linear(width,2))
    def forward(self,x,hx,hm):
        if self.cell is not None:
            h=x.new_zeros((x.shape[0],H))
            for i in range(HIST):
                h=torch.where(hm[:,i:i+1],self.cell(hx[:,i],h),h)
            x=torch.cat((x,h[:,None,:].expand(-1,x.shape[1],-1)),-1)
        return self.head(x)
def examples():
    rows=[]
    for i in range(60):
        r=Root(51000+i,8+i%13); r.oracle((0,r.n)); rng=random.Random(81000+i)
        for s,(best,_,vals) in sorted(r.dp.items()):
            bank=r.bank(s); n=s[1]-s[0]; fs=[a[3] for a in bank]
            mask=[eligible(a) for a in bank]; valid=[v is not None for v in vals]
            assert all(m or not v for m,v in zip(mask,valid))
            ys=[(v if v is not None else a[2]+best)/n**2 for a,v in zip(bank,vals)]
            bs=[heuristic(a)/n**2 for a in bank]
            bad=[j for j,v in enumerate(valid) if mask[j] and not v]; rng.shuffle(bad)
            for k in range(min(HIST,len(bad))+1):
                rows.append((fs,[m and j not in bad[:k] for j,m in enumerate(mask)],valid,ys,bs,[fs[j] for j in bad[:k]]))
    x=torch.zeros(len(rows),MAXA,W); mask=torch.zeros(len(rows),MAXA,dtype=torch.bool)
    v,y,b=[torch.zeros(len(rows),MAXA) for _ in range(3)]
    hx=torch.zeros(len(rows),HIST,W); hm=torch.zeros(len(rows),HIST,dtype=torch.bool)
    for i,(fs,m,vs,ys,bs,hist) in enumerate(rows):
        n=len(fs); x[i,:n]=torch.tensor(fs); mask[i,:n]=torch.tensor(m)
        v[i,:n]=torch.tensor(vs); y[i,:n]=torch.tensor(ys); b[i,:n]=torch.tensor(bs)
        if hist: hx[i,:len(hist)]=torch.tensor(hist); hm[i,:len(hist)]=True
    return x,mask,v,y,b,hx,hm
def train(data,recurrent,seed):
    torch.manual_seed(seed); model=Residual(recurrent); opt=torch.optim.AdamW(model.parameters(),lr=.003)
    x,mask,v,y,b,hx,hm=data
    for _ in range(12):
        for idx in torch.randperm(len(x)).split(128):
            pred=model(x[idx],hx[idx],hm[idx]); m=mask[idx]
            loss=nn.functional.smooth_l1_loss((pred[...,0]+b[idx])[m],y[idx][m],beta=.1)
            loss+=.15*nn.functional.binary_cross_entropy_with_logits(pred[...,1][m],v[idx][m])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()
@torch.no_grad()
def rollout(r,arm,model=None,cap=0,eviction='lru'):
    memo=OrderedDict(); pending=Counter(); seen=Counter()
    work=calls=rejects=hits=peak=used=scans=infer_ns=multi_history=max_history=0
    budget,callcap=3*r.n**2,8*r.n
    def utility(key,entry):
        size,saved=entry
        if eviction=='work': return saved/size
        return (pending[key]+(seen[key]-1)/(1+seen[key]))*saved/size
    def insert(p,paid):
        nonlocal used,peak,scans
        size=len(p)+16
        if size>cap or p in memo: return
        if eviction=='lru':
            while used+size>cap:
                _,old=memo.popitem(last=False); used-=old[0]
        else:
            scans+=len(memo)+1; victims=[]; available=cap-used
            for key in sorted(memo,key=lambda k:utility(k,memo[k])):
                if available>=size: break
                victims.append(key); available+=memo[key][0]
            if utility(p,(size,paid))<sum(utility(k,memo[k]) for k in victims): return
            for k in victims: used-=memo.pop(k)[0]
        memo[p]=(size,paid); used+=size; peak=max(peak,used)
    def visit(s):
        nonlocal work,calls,rejects,hits,infer_ns,multi_history,max_history
        p=r.poly(s)
        if s[1]-s[0]==1: return len(p)==2 and p[-1]==1
        seen[p]+=1
        if p in memo and work<budget:
            memo.move_to_end(p); hits+=1; work+=1; return True
        begin=work; bank=r.bank(s); remain={i for i,a in enumerate(bank) if eligible(a)}; assert remain
        history=[]
        if model is not None:
            start=time.perf_counter_ns(); x=torch.tensor([[a[3] for a in bank]])
            base=torch.tensor([heuristic(a)/(s[1]-s[0])**2 for a in bank]); infer_ns+=time.perf_counter_ns()-start
        while remain:
            if model is None: i=min(remain,key=lambda j:(heuristic(bank[j]),j))
            else:
                start=time.perf_counter_ns(); hx=torch.zeros(1,HIST,W); hm=torch.zeros(1,HIST,dtype=torch.bool)
                hist=history[-HIST:] if arm!='reset' else []
                if arm=='reverse': hist=hist[::-1]
                if hist: hx[0,:len(hist)]=torch.tensor(hist); hm[0,:len(hist)]=True
                scores=(model(x,hx,hm)[0,:,0]+base).tolist(); i=min(remain,key=lambda j:(scores[j],j))
                infer_ns+=time.perf_counter_ns()-start
            multi_history+=len(history)>=2; max_history=max(max_history,len(history))
            remain.remove(i); spans,cs,cost,features=bank[i]
            if work+cost>budget or calls>=callcap: return False
            ok,paid=verify(p,cs,r.p); assert paid==cost; calls+=1; work+=paid
            if not ok: rejects+=1; history.append(features); continue
            assert all(c==r.poly(t) for c,t in zip(cs,spans))
            pending.update(cs)
            for j,(t,c) in enumerate(zip(spans,cs)):
                pending[c]-=1
                if not visit(t): pending.subtract(cs[j+1:]); return False
            insert(p,work-begin); return True
        return False
    start=time.perf_counter_ns(); closed=visit((0,r.n)); elapsed=time.perf_counter_ns()-start
    assert work<=budget and calls<=callcap and peak<=cap and not any(pending.values())
    return dict(closed=closed,work=work,calls=calls,rejects=rejects,hits=hits,peak=peak,scans=scans,infer_ns=infer_ns,elapsed_ns=elapsed,multi_history=multi_history,max_history=max_history)
def summarize(out):
    result=[]
    for key in sorted({(r['split'],r['arm'],r['cap'],r['eviction']) for r in out}):
        a=[r for r in out if (r['split'],r['arm'],r['cap'],r['eviction'])==key]
        result.append(dict(key=key,n=len(a),closed=sum(r['closed'] for r in a),work=mean(r['work'] for r in a),rejects=sum(r['rejects'] for r in a),hits=sum(r['hits'] for r in a),peak=max(r['peak'] for r in a),scans=sum(r['scans'] for r in a),infer_ms=mean(r['infer_ns'] for r in a)/1e6,elapsed_ms=mean(r['elapsed_ns'] for r in a)/1e6,multi_history=sum(r['multi_history'] for r in a)))
    return result
def paired(out,split,arm,cap=0,eviction='lru',control='heuristic',control_cap=0,control_eviction='lru'):
    def group(a,c,e):
        d={}
        for row in out:
            if (row['split'],row['arm'],row['cap'],row['eviction'])==(split,a,c,e): d.setdefault(row['root'],[]).append(row)
        return d
    a,b=group(arm,cap,eviction),group(control,control_cap,control_eviction); assert a.keys()==b.keys()
    ds=[mean(r['work'] for r in a[k])-mean(r['work'] for r in b[k]) for k in sorted(a)]
    rng=random.Random(9988); boot=sorted(mean(rng.choices(ds,k=len(ds))) for _ in range(5000))
    return dict(split=split,arm=arm,control=control,cap=cap,eviction=eviction,control_cap=control_cap,control_eviction=control_eviction,mean=mean(ds),wins=sum(d<0 for d in ds),ties=sum(d==0 for d in ds),losses=sum(d>0 for d in ds),bootstrap95=[boot[125],boot[4874]])
def main():
    torch.set_num_threads(1); data=examples(); models={(k,s):train(data,k=='gru',s) for k in ('mlp','gru') for s in SEEDS}; out=[]
    for split,base,sizes in [('iid',61000,(8,12,16,20)),('ood',71000,(24,32,40,48))]:
        for i in range(24):
            r=Root(base+i,sizes[i%4])
            for l in range(r.n):
                for rr in range(l+2,r.n+1): r.bank((l,rr))
            for arm,seed,m in [(k,s,m) for (k,s),m in models.items()]+[(a,s,models['gru',s]) for a in ('reset','reverse') for s in SEEDS]:
                out.append(dict(split=split,root=r.seed,n=r.n,arm=arm,seed=seed,cap=0,eviction='lru',**rollout(r,arm,m)))
            for cap,e in ((0,'lru'),(128,'lru'),(128,'work'),(128,'reuse'),(512,'lru'),(512,'work'),(512,'reuse')):
                out.append(dict(split=split,root=r.seed,n=r.n,arm='heuristic',seed=0,cap=cap,eviction=e,**rollout(r,'heuristic',cap=cap,eviction=e)))
    ps=[paired(out,s,a) for s in ('iid','ood') for a in ('mlp','gru','reset','reverse')]
    ps.extend(paired(out,s,'heuristic',c,'reuse','heuristic',c,'lru') for s in ('iid','ood') for c in (128,512))
    print(json.dumps(dict(examples=len(data[0]),params={k:sum(p.numel() for p in models[k,13].parameters()) for k in ('mlp','gru')},summary=summarize(out),paired=ps,runs=out),sort_keys=True))
if __name__=='__main__': main()
