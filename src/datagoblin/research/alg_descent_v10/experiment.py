"""ALG-DESCENT-10: joint labels, exact registry control and closure cache."""
import json, random, sys, time, math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from statistics import mean
import torch
import numpy as np
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
try:
    from alg_descent_v7.risk import FamilyRoot, data, Model, metrics, prob, bucket, structural, features, ev, extra, verify
except ImportError:
    from v7.risk import FamilyRoot, data, Model, metrics, prob, bucket, structural, features, ev, extra, verify
from certificate_contract import CertificateReceipt, verify_bound
SEEDS=(19,41,67); VERSION='bound-v10'; W=20


def public_features(r,s,a):
    _,cs,_,_=a; parent=r.poly(s); p=r.p
    flat=[v for c in cs for v in c[:-1]]
    return features(r,a,True)+(sum(v==0 for v in parent)/len(parent),mean(flat)/(p-1),mean(v==p-1 for v in flat),mean(c[0]==0 for c in cs),mean(sum(v==0 for v in c)/len(c) for c in cs),max(sum(v==0 for v in c)/len(c) for c in cs),mean(ev(c,1,p)==0 for c in cs),mean(ev(c,0,p)==0 for c in cs))


class BoundRoot(FamilyRoot):
    def __init__(self,seed,n,family):
        super().__init__(seed,n,family); self.joint={}; self.twins={}; self.cover_joint=Counter()
    def expected(self,a): return tuple(self.poly(t) for t in a[0])
    def bank(self,s):
        if s in self.joint: return self.original[s]
        old=list(super().bank(s)); parent=self.poly(s)
        rows=[(a,x,verify_bound(parent,a[1],self.expected(a),self.p),0) for a,x in zip(old,self.x[s])]
        for j,(a,x,rec,_) in enumerate(rows):
            if not rec.accepted: continue
            cs=a[1]; pair=next(((i,k) for i in range(len(cs)) for k in range(i+1,len(cs)) if len(cs[i])==len(cs[k]) and cs[i]!=cs[k]),None)
            if pair is None: continue
            i,k=pair; swapped=list(cs); swapped[i],swapped[k]=swapped[k],swapped[i]; swapped=tuple(swapped)
            decoy=(a[0],swapped,a[2],a[3]); rr=verify_bound(parent,swapped,self.expected(decoy),self.p)
            assert rr.product_valid and not rr.child_bound and rr.work==rec.work
            assert all(extra(parent,swapped,z,self.p) for z in range(4))
            assert public_features(self,s,decoy)==x and bucket(self,decoy)==bucket(self,a) and structural(decoy)==structural(a)
            rows[j]=(a,x,rec,1); rows.append((decoy,x,rr,2)); break
        random.Random(self.seed*700001+s[0]*701+s[1]*13).shuffle(rows)
        self.original[s]=[a for a,_,_,_ in rows]; self.x[s]=[x for _,x,_,_ in rows]; self.joint[s]=[v for _,_,v,_ in rows]
        self.labels[s]=[int(v.accepted) for v in self.joint[s]]
        idx={tag:i for i,(_,_,_,tag) in enumerate(rows) if tag}
        if idx: self.twins[s]=(idx[1],idx[2])
        for v in self.joint[s]:
            self.cover_joint['eligible']+=1; self.cover_joint['product_invalid']+=not v.product_valid
            self.cover_joint['unbound_product_valid']+=v.product_valid and not v.child_bound
            self.cover_joint['accepted']+=v.accepted; self.cover_joint['bound_product_invalid']+=v.child_bound and not v.product_valid
        assert any(self.labels[s]); return self.original[s]
    def oracle(self,s):
        if s[1]-s[0]==1:return 0,-1,()
        if s not in self.dp:
            bank=self.bank(s); vals=[a[2]+sum(self.oracle(t)[0] for t in a[0]) if rec.accepted else None for a,rec in zip(bank,self.joint[s])]
            v,i=min((v,i) for i,v in enumerate(vals) if v is not None);self.dp[s]=(v,i,vals)
        return self.dp[s]


def roots(split):
    base,count,ds,fs={'train':(510000,16,tuple(range(8,21)),('adjacent','separated')),'iid':(520000,8,(12,16,20),('adjacent','separated')),'ood':(530000,12,(24,32,40,48),('second','two_child'))}[split]
    return [BoundRoot(base+1000*j+i,ds[i%len(ds)],f) for j,f in enumerate(fs) for i in range(count)]


def joint_data(rr,prior=None):
    rows=[];counts=defaultdict(lambda:[0,0]);slices=[]
    for r in rr:
        start=len(rows)
        for l in range(r.n):
            for t in range(l+2,r.n+1):
                s=(l,t);bank=r.bank(s);xs=r.x[s];ys=r.labels[s]
                neg=[j for j,v in enumerate(ys) if not v];random.Random(r.seed*99991+l*503+t).shuffle(neg)
                for j,(a,x,y) in enumerate(zip(bank,xs,ys)):
                    b=bucket(r,a);counts[b][0]+=y;counts[b][1]+=1
                    hist=[xs[i] for i in neg if i!=j][:2]
                    for k in range(len(hist)+1):rows.append((x,hist[:k],y,b))
        slices.append((r.seed,r.family,start,len(rows),dict(r.cover)))
    if prior is None:prior=dict(counts)
    x=torch.tensor([v[0] for v in rows]);zero=(0.,)*W
    h=torch.tensor([tuple(v[1])+(zero,)*(2-len(v[1])) for v in rows])
    mask=torch.tensor([[float(i<len(v[1])) for i in range(2)] for v in rows])
    y=torch.tensor([v[2] for v in rows],dtype=torch.float32);p=torch.tensor([prob(prior,v[3]) for v in rows]).logit()
    return (x,h,mask,y,p),prior,slices


class Dense(nn.Module):
    def __init__(self,passes):
        super().__init__();self.passes=passes;self.enc=nn.Linear(W,12);self.mat=nn.Parameter(torch.randn(12,12)/12**.5);self.out=nn.Linear(12,1)
    def forward(self,x,h,m):
        hist=(h*m[:,:,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
        inp=self.enc(x)+nn.functional.linear(hist,self.enc.weight);state=inp.tanh()
        for _ in range(self.passes):state=(inp+state@self.mat.T).tanh()
        return self.out(state).squeeze(-1)


def fit(d,arm,seed):
    torch.manual_seed(seed);model=Model('linear') if arm=='linear' else Dense(1 if arm=='one' else 3)
    opt=torch.optim.AdamW(model.parameters(),lr=.004);x,h,m,y,p=d
    for _ in range(8):
        for ix in torch.randperm(len(x)).split(512):
            loss=nn.functional.binary_cross_entropy_with_logits(p[ix]+model(x[ix],h[ix],m[ix]),y[ix]);opt.zero_grad();loss.backward();opt.step()
    return model.eval()


@torch.no_grad()
def predict(d,model):
    x,h,m,_,p=d
    return torch.cat([(p[i] if model is None else p[i]+model(x[i],h[i],m[i])).sigmoid().clamp(.05,.995) for i in torch.arange(len(x)).split(8192)])


class ClosureCache:
    def __init__(self,cap,scope):self.cap=cap;self.scope=scope;self.items=OrderedDict();self.used=self.peak=self.admitted=0
    def key(self,identity):return self.scope,identity
    def hit(self,identity):
        k=self.key(identity)
        if k not in self.items:return False
        self.items.move_to_end(k);return True
    def admit(self,identity,receipt,children_closed,size):
        if not (receipt.accepted and children_closed):return False
        if not self.cap or size>self.cap:return False
        k=self.key(identity)
        if k in self.items:return True
        while self.used+size>self.cap:
            _,old=self.items.popitem(last=False);self.used-=old
        self.items[k]=size;self.used+=size;self.peak=max(self.peak,self.used);self.admitted+=1;return True


@torch.no_grad()
def route(r,prior,arm,model=None,cap=0,identity='poly'):
    work=calls=product_invalid=unbound=hits=bind_calls=bind_coeff=bind_ns=prefilter_rejects=score_ns=0
    cache=ClosureCache(cap,(r.seed,r.p,VERSION));trace=[]
    def binding(a):
        nonlocal bind_calls,bind_coeff,bind_ns
        start=time.perf_counter_ns();expected=r.expected(a);cs=a[1];bind_calls+=1
        ok=len(cs)==len(expected)
        for c,e in zip(cs,expected):
            ok &= len(c)==len(e)
            for x,y in zip(c,e):bind_coeff+=1;ok &= x==y
        bind_ns+=time.perf_counter_ns()-start;return bool(ok)
    def visit(s):
        nonlocal work,calls,product_invalid,unbound,hits,prefilter_rejects,score_ns
        parent=r.poly(s)
        if len(parent)==2:return True
        key=parent if identity=='poly' else s
        if cap and work<3*r.n*r.n and cache.hit(key):work+=1;hits+=1;trace.append((s,'hit'));return True
        bank=r.bank(s);xs=r.x[s];remain=list(range(len(bank)));history=[]
        if arm=='binding':
            remain=[j for j in remain if binding(bank[j])];prefilter_rejects+=len(bank)-len(remain)
        st=[structural(a) for a in bank]
        if arm in ('bucket','linear','one','three'):
            base=torch.tensor([prob(prior,bucket(r,a)) for a in bank]).logit();xt=torch.tensor(xs)
        while remain:
            start=time.perf_counter_ns()
            if arm=='oracle': i=r.oracle(s)[1]
            elif arm in ('binding','structural'): i=min(remain,key=lambda j:(st[j],j))
            else:
                h=torch.zeros(len(bank),2,W);m=torch.zeros(len(bank),2);hist=history[-2:]
                if hist:h[:,:len(hist)]=torch.tensor(hist);m[:,:len(hist)]=1
                ps=(base if model is None else base+model(xt,h,m)).sigmoid().clamp(.05,.995).tolist()
                i=min(remain,key=lambda j:(st[j]+bank[j][2]*(1/ps[j]-1),j))
            score_ns+=time.perf_counter_ns()-start;remain.remove(i);a=bank[i]
            if work+a[2]>3*r.n*r.n or calls>=8*r.n:return False
            valid,paid=verify(parent,a[1],r.p);bound=binding(a);rec=CertificateReceipt(valid,bound,paid)
            assert rec==r.joint[s][i] and paid==a[2];work+=paid;calls+=1;trace.append((s,i,rec.accepted))
            if not rec.accepted:
                product_invalid+=not valid;unbound+=valid and not bound;history.append(xs[i]);continue
            done=all(visit(t) for t in a[0]);cache.admit(key,rec,done,(len(parent) if identity=='poly' else 4)+16);return done
        return False
    start=time.perf_counter_ns();closed=visit((0,r.n));elapsed=time.perf_counter_ns()-start
    assert work<=3*r.n*r.n and calls<=8*r.n and cache.peak<=cap
    return dict(closed=closed,work=work,calls=calls,product_invalid=product_invalid,unbound=unbound,hits=hits,bind_calls=bind_calls,bind_coeff=bind_coeff,bind_ns=bind_ns,prefilter_rejects=prefilter_rejects,score_ns=score_ns,elapsed_ns=elapsed,peak=cache.peak,admitted=cache.admitted,trace=trace)


def exact(parent,children,p):
    acc=np.array([1],dtype=np.int64);cost=0
    for q in children:
        cost+=len(acc)*len(q);acc=np.convolve(acc,np.asarray(q,dtype=np.int64))%p
    return tuple(acc.tolist())==parent,cost


def audit(r):
    checked=twins=0;dp={};cases=Counter()
    for d in range(1,r.n+1):
        for l in range(r.n-d+1):
            s=(l,l+d)
            if d==1:dp[s]=0;continue
            bank=r.bank(s);costs=[]
            for j,a in enumerate(bank):
                prod,cost=exact(r.poly(s),a[1],r.p);bound=all(q==r.poly(t) for t,q in zip(a[0],a[1]));assert (prod,bound,cost)==r.joint[s][j]
                cases['product_invalid']+=not prod;cases['unbound']+=prod and not bound;cases['bound_invalid']+=bound and not prod;checked+=1
                if prod and bound:costs.append(cost+sum(dp[t] for t in a[0]))
            dp[s]=min(costs);assert dp[s]==r.oracle(s)[0]
            if s in r.twins:
                a,b=r.twins[s];assert r.x[s][a]==r.x[s][b] and r.labels[s][a]==1 and r.labels[s][b]==0;twins+=1
    return dict(checked=checked,twins=twins,dp_states=len(dp)-r.n,**cases)


@torch.no_grad()
def twin_loss(rr,prior,model):
    x=[];p=[]
    for r in rr:
        for s,(a,b) in r.twins.items():
            assert r.x[s][a]==r.x[s][b];x.append(r.x[s][a]);p.append(prob(prior,bucket(r,r.bank(s)[a])))
    xt=torch.tensor(x);h=torch.zeros(len(x),2,W);m=torch.zeros(len(x),2);base=torch.tensor(p).logit()
    q=(base if model is None else base+model(xt,h,m)).sigmoid().clamp(.05,.995)
    loss=float((-0.5*(q.log()+(1-q).log())).mean());assert loss>=math.log(2)-1e-6
    return dict(pairs=len(x),loss=loss)


def main():
    torch.set_num_threads(1);tr_roots=roots('train');tr,prior,_=joint_data(tr_roots);ix=torch.randperm(len(tr[0]),generator=torch.Generator().manual_seed(778))[:20000];train=tuple(x[ix] for x in tr)
    print('train examples',len(tr[0]),'used',len(ix),flush=True)
    audit_train=[dict(root=r.seed,**audit(r)) for r in tr_roots];models={}
    for a in ('linear','one','three'):
        for seed in SEEDS:models[a,seed]=fit(train,a,seed);print('fit',a,seed,flush=True)
    prediction=[];runs=[];audits=[];coverage=[];twins=[]
    for split in ('iid','ood'):
        rr=roots(split);d,_,sl=joint_data(rr,prior);print(split,'examples',len(d[0]),flush=True)
        for r in rr:audits.append(dict(split=split,root=r.seed,**audit(r)));coverage.append(dict(split=split,root=r.seed,family=r.family,**r.cover_joint))
        print('audited',split,flush=True)
        arms=[('bucket',0,None)]+[(a,seed,m) for (a,seed),m in models.items()]
        for a,seed,m in arms:
            prediction.extend(dict(split=split,arm=a,seed=seed,**x) for x in metrics(predict(d,m),d[3],sl));twins.append(dict(split=split,arm=a,seed=seed,**twin_loss(rr,prior,m)))
        for r in rr:
            spec=[(a,seed,m,0,'poly') for a,seed,m in arms]+[(a,0,None,0,'poly') for a in ('binding','structural','oracle')]+[('binding',0,None,128,k) for k in ('poly','interval')]
            random.Random(r.seed+771).shuffle(spec)
            for a,seed,m,cap,key in spec:
                res=route(r,prior,a,m,cap,key);assert a!='oracle' or res['work']==r.oracle((0,r.n))[0]
                runs.append(dict(split=split,root=r.seed,family=r.family,n=r.n,arm=a,seed=seed,cap=cap,identity=key,**res))
        print('routed',split,flush=True)
        (HERE/'partial.json').write_text(json.dumps(dict(prediction=prediction,runs=runs,audits=audits,coverage=coverage,twins=twins),sort_keys=True))
    out=dict(train_examples=len(tr[0]),train_used=len(ix),params={a:sum(p.numel() for p in models[a,SEEDS[0]].parameters()) for a in ('linear','one','three')},train_audit=audit_train,coverage=coverage,audits=audits,prediction=prediction,twins=twins,runs=runs)
    (HERE/'receipt.json').write_text(json.dumps(out,sort_keys=True));torch.save(dict(prior=prior,models={f'{a}_{s}':m.state_dict() for (a,s),m in models.items()}),HERE/'models.pt')

if __name__=='__main__':main()
