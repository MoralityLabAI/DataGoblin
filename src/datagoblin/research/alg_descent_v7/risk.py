"""ALG-DESCENT-07: synthetic certificate risk, shared-factor intervention."""
import json, random, time, sys
from pathlib import Path
from collections import defaultdict
from statistics import mean
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from alg_descent_v2 import Root
from alg_descent_v4 import structural,bucket,features,verify
SEEDS=(19,41,67); FAMILIES=('adjacent','separated','second','two_child')
W,H,R,HIST=20,12,12,2

def ev(poly,z,p):
    out=0
    for c in reversed(poly):out=(out*z+c)%p
    return out

def extra(parent,cs,z,p):
    val=1
    for c in cs:val=val*ev(c,z,p)%p
    return val==ev(parent,z,p)

class FamilyRoot(Root):
    def __init__(self,seed,n,family):
        super().__init__(seed,n); self.family=family;self.original={};self.labels={};self.x={};self.cover=defaultdict(int)
    def bank(self,s):
        if s in self.original:return self.original[s]
        old=super().bank(s); parent=self.poly(s); p=self.p; l,r=s; d=r-l
        rng=random.Random(self.seed*200003+l*1013+r*41)
        out=[]; ys=[]; xs=[]
        for spans,_,cost,_ in old:
            cs=[list(self.poly(t)) for t in spans]; ds=[len(c)-1 for c in cs];k=len(cs)
            balanced=tuple(ds)==tuple(d*(i+1)//k-d*i//k for i in range(k))
            if not (k==2 and balanced) and rng.random()<self.rate:
                inds=[i for i,c in enumerate(cs) if len(c)>=4]
                if self.family=='adjacent' and inds:
                    c=cs[rng.choice(inds)]; a=rng.randrange(1,len(c)-2); b=a+1; q=rng.randrange(1,p)
                    c[a]=(c[a]+q)%p;c[b]=(c[b]-q)%p
                elif self.family=='separated':
                    inds=[i for i,c in enumerate(cs) if len(c)>=5]
                    if inds:
                        c=cs[rng.choice(inds)];a,b=rng.choice([(i,j) for i in range(1,len(c)-1) for j in range(i+2,len(c)-1)]);q=rng.randrange(1,p)
                        c[a]=(c[a]+q)%p;c[b]=(c[b]-q)%p
                elif self.family=='second':
                    inds=[i for i,c in enumerate(cs) if len(c)>=5]
                    if inds:
                        c=cs[rng.choice(inds)];a=rng.randrange(1,len(c)-3);q=rng.randrange(1,p)
                        for j,v in enumerate((q,-2*q,q)):c[a+j]=(c[a+j]+v)%p
                elif self.family=='two_child' and len(inds)>=2:
                    for i in rng.sample(inds,2):
                        c=cs[i];a,b=sorted(rng.sample(range(1,len(c)-1),2));q=rng.randrange(1,p)
                        c[a]=(c[a]+q)%p;c[b]=(c[b]-q)%p
            cs=tuple(map(tuple,cs)); ok,paid=verify(parent,cs,p); assert paid==cost
            assert extra(parent,cs,0,p) and extra(parent,cs,1,p)
            mask=extra(parent,cs,2,p) and extra(parent,cs,3,p); assert mask or not ok
            self.cover['all']+=1;self.cover['invalid']+=not ok;self.cover['masked']+=not mask;self.cover['eligible_invalid']+=mask and not ok
            if not mask:continue
            f=(d/48,k/8,max(ds)/d,min(ds)/d,sum(t*t for t in ds)/d**2,cost/d**2,1.,1.,p/11,sum(v==0 for c in cs for v in c)/sum(map(len,cs)))
            a=(spans,cs,cost,f); flat=[v for c in cs for v in c[:-1]]
            ff=features(self,a,True)+(sum(v==0 for v in parent)/len(parent),mean(flat)/(p-1),mean(v==p-1 for v in flat),mean(c[0]==0 for c in cs),mean(sum(v==0 for v in c)/len(c) for c in cs),max(sum(v==0 for v in c)/len(c) for c in cs),mean(ev(c,1,p)==0 for c in cs),mean(ev(c,0,p)==0 for c in cs))
            assert len(ff)==W
            out.append(a);ys.append(int(ok));xs.append(ff)
        assert any(ys)
        self.original[s]=out;self.labels[s]=ys;self.x[s]=xs
        return out

def roots(split):
    defs={'train':(210000,48,tuple(range(8,21)),FAMILIES[:2]),'cal':(220000,12,(12,16,20),FAMILIES[:2]),'iid':(230000,12,(12,16,20),FAMILIES[:2]),'ood':(240000,16,(24,32,40,48),FAMILIES[2:])}
    base,count,degrees,fams=defs[split]
    for j,f in enumerate(fams):
        for i in range(count):yield FamilyRoot(base+j*1000+i,degrees[i%len(degrees)],f)

def prob(prior,b):
    yes,n=prior.get(b,(0,0));return min(.995,max(.05,(yes+1)/(n+2)))

def data(rr,prior=None):
    rows=[];counts=defaultdict(lambda:[0,0]);slices=[]
    for r in rr:
        start=len(rows)
        for l in range(r.n):
            for t in range(l+2,r.n+1):
                s=(l,t);bank=r.bank(s);xs=r.x[s];ys=r.labels[s]
                neg=[j for j,v in enumerate(ys) if not v];random.Random(r.seed*99991+l*503+t).shuffle(neg)
                for j,(a,x,y) in enumerate(zip(bank,xs,ys)):
                    b=bucket(r,a);counts[b][0]+=y;counts[b][1]+=1
                    history=[xs[i] for i in neg if i!=j][:HIST]
                    for k in range(len(history)+1):rows.append((x,history[:k],y,b))
        slices.append((r.seed,r.family,start,len(rows),dict(r.cover)))
    if prior is None:prior=dict(counts)
    x=torch.tensor([v[0] for v in rows]);h=torch.zeros(len(rows),HIST,W);mask=torch.zeros(len(rows),HIST)
    for i,(_,hist,_,_) in enumerate(rows):
        if hist:h[i,:len(hist)]=torch.tensor(hist);mask[i,:len(hist)]=1
    y=torch.tensor([v[2] for v in rows],dtype=torch.float32);p=torch.tensor([prob(prior,v[3]) for v in rows]).logit()
    return (x,h,mask,y,p),prior,slices

class Model(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='linear': self.net=nn.Linear(W*2+1,1)
        elif kind=='mlp':self.net=nn.Sequential(nn.Linear(W*2+1,12),nn.Tanh(),nn.Linear(12,1))
        else:
            self.enc=nn.Linear(W,H);self.u=nn.Parameter(torch.randn(H,R)*.15);self.v=nn.Parameter(torch.randn(R,H)*.15);self.out=nn.Linear(H,1)
    def forward(self,x,h,m,edit=None):
        if self.kind!='factor':
            hist=(h*m[:,:,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1.)
            return self.net(torch.cat((x,hist,m.sum(1,keepdim=True)/HIST),1)).squeeze(-1)
        mat=self.u@self.v
        if edit is not None:mat=mat+edit
        state=x.new_zeros(len(x),H)
        for i in range(HIST):state=torch.where(m[:,i:i+1].bool(),torch.tanh(self.enc(h[:,i])+state@mat.T),state)
        return self.out(torch.tanh(self.enc(x)+state@mat.T)).squeeze(-1)
    def atom(self,i):return self.u[:,i,None]@self.v[i,None,:]

def fit(d,kind,seed):
    torch.manual_seed(seed);model=Model(kind);opt=torch.optim.AdamW(model.parameters(),lr=.004)
    x,h,m,y,p=d
    for _ in range(12):
        for ix in torch.randperm(len(x)).split(512):
            logits=p[ix]+model(x[ix],h[ix],m[ix]);loss=nn.functional.binary_cross_entropy_with_logits(logits,y[ix]);opt.zero_grad();loss.backward();opt.step()
    return model.eval()

@torch.no_grad()
def preds(d,model=None,edit=None):
    x,h,m,y,p=d;out=[]
    for ix in torch.arange(len(x)).split(8192):out.append((p[ix] if model is None else p[ix]+model(x[ix],h[ix],m[ix],edit)).sigmoid().clamp(.05,.995))
    return torch.cat(out)

def metrics(p,y,slices):
    loss=-(y*p.log()+(1-y)*(1-p).log()); b=(p-y).square()
    return [dict(root=root,family=f,n=t-s,neg=int((1-y[s:t]).sum()),logloss=float(loss[s:t].mean()),brier=float(b[s:t].mean())) for root,f,s,t,_ in slices]

@torch.no_grad()
def route(r,prior,model=None,edit=None):
    work=calls=rejects=score_ns=0
    def visit(s):
        nonlocal work,calls,rejects,score_ns
        parent=r.poly(s)
        if len(parent)==2:return True
        bank=r.bank(s);xs=r.x[s];history=[];remain=set(range(len(bank)))
        start=time.perf_counter_ns();base=torch.tensor([prob(prior,bucket(r,a)) for a in bank]).logit();xt=torch.tensor(xs);st=[structural(a) for a in bank];score_ns+=time.perf_counter_ns()-start
        while remain:
            start=time.perf_counter_ns()
            if model is None:ps=base.sigmoid().tolist()
            else:
                h=torch.zeros(len(bank),HIST,W);m=torch.zeros(len(bank),HIST);hist=history[-HIST:]
                if hist:h[:,:len(hist)]=torch.tensor(hist);m[:,:len(hist)]=1
                ps=(base+model(xt,h,m,edit)).sigmoid().clamp(.05,.995).tolist()
            i=min(remain,key=lambda j:(st[j]+bank[j][2]*(1/ps[j]-1),j));score_ns+=time.perf_counter_ns()-start
            remain.remove(i);spans,cs,cost,_=bank[i]
            if work+cost>3*r.n*r.n or calls>=8*r.n:return False
            ok,paid=verify(parent,cs,r.p);assert paid==cost;work+=paid;calls+=1
            if not ok:rejects+=1;history.append(xs[i]);continue
            assert all(c==r.poly(t) for c,t in zip(cs,spans))
            return all(visit(t) for t in spans)
        return False
    start=time.perf_counter_ns();closed=visit((0,r.n));elapsed=time.perf_counter_ns()-start
    assert work<=3*r.n*r.n and calls<=8*r.n
    return dict(closed=closed,work=work,calls=calls,rejects=rejects,score_ns=score_ns,elapsed_ns=elapsed)

def main():
    torch.set_num_threads(1);torch.manual_seed(77)
    tr,prior,tr_s=data(list(roots('train')));print('train',len(tr[0]),flush=True)
    ca,_,ca_s=data(list(roots('cal')),prior);models={};edits={};choices=[]
    for kind in ('linear','mlp','factor'):
        for seed in SEEDS:
            model=fit(tr,kind,seed);models[kind,seed]=model
            if kind=='factor':
                baseline=mean(x['logloss'] for x in metrics(preds(ca,model),ca[3],ca_s));effects=[mean(x['logloss'] for x in metrics(preds(ca,model,-model.atom(j).detach()),ca[3],ca_s))-baseline for j in range(R)]
                j=max(range(R),key=lambda i:effects[i]);atom=model.atom(j).detach();norm=float(atom.norm());rng=torch.Generator().manual_seed(seed+1000)
                edits[seed]={'ablate':-atom,'enhance':atom}
                for sh in range(8):
                    u=torch.randn(H,generator=rng);v=torch.randn(H,generator=rng);mat=u[:,None]@v[None,:];edits[seed][f'sham{sh}']=-mat*(norm/mat.norm())
                choices.append(dict(seed=seed,atom=j,norm=norm,effects=effects,sham_norms=[float(edits[seed][f'sham{k}'].norm()) for k in range(8)]))
            print('trained',kind,seed,flush=True)
    prediction=[];routes=[];coverage=[]
    for split in ('iid','ood'):
        rr=list(roots(split));d,_,sl=data(rr,prior);print(split,len(d[0]),flush=True)
        for root,f,a,b,cov in sl:coverage.append(dict(split=split,root=root,family=f,examples=b-a,**cov))
        arms=[('bucket',0,None,None)]+[(kind,seed,m,None) for (kind,seed),m in models.items()]+[(arm,seed,models['factor',seed],edit) for seed,es in edits.items() for arm,edit in es.items()]
        for arm,seed,m,edit in arms:prediction.extend(dict(split=split,arm=arm,seed=seed,**q) for q in metrics(preds(d,m,edit),d[3],sl))
        for r in rr:
            order=list(arms);random.Random(r.seed+123).shuffle(order)
            for arm,seed,m,edit in order:routes.append(dict(split=split,arm=arm,seed=seed,root=r.seed,family=r.family,n=r.n,**route(r,prior,m,edit)))
        print('routed',split,flush=True)
    output=dict(train_examples=len(tr[0]),cal_examples=len(ca[0]),train_coverage=[dict(root=root,family=f,examples=b-a,**cov) for root,f,a,b,cov in tr_s],params={k:sum(p.numel() for p in models[k,SEEDS[0]].parameters()) for k in ('linear','mlp','factor')},choices=choices,prediction=prediction,routes=routes,coverage=coverage)
    (HERE/'receipt.json').write_text(json.dumps(output,sort_keys=True));torch.save({f'{k}_{s}':m.state_dict() for (k,s),m in models.items()},HERE/'models.pt')
if __name__=='__main__':main()
