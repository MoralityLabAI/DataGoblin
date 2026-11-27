"""Frozen recurrent risk router with exact rotated factors and sparse gates."""
import json, random, sys, time, hashlib
from pathlib import Path
from statistics import mean
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
try:
    from alg_descent_v7.risk import FamilyRoot,data,Model,metrics,prob,bucket,structural,verify
except ImportError:
    from v7.risk import FamilyRoot,data,Model,metrics,prob,bucket,structural,verify
SEEDS=(19,41,67); H=12; W=20; T=3

def roots(split):
    base,count,ds,fams={'train':(310000,32,tuple(range(8,21)),('adjacent','separated')),'cal':(320000,8,(12,16,20),('adjacent','separated')),'iid':(330000,8,(12,16,20),('adjacent','separated')),'ood':(340000,12,(24,32,40,48),('second','two_child'))}[split]
    return [FamilyRoot(base+1000*j+i,ds[i%len(ds)],f) for j,f in enumerate(fams) for i in range(count)]

class Dense(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=nn.Linear(W,H); self.mat=nn.Parameter(torch.randn(H,H)/H**.5); self.out=nn.Linear(H,1)
    def trajectory(self,x,h,m,edit=None,reset=False,dict_=None):
        hist=(h*m[:,:,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
        inp=self.enc(x)+torch.nn.functional.linear(hist,self.enc.weight)
        h0=inp.tanh(); state=h0; ins=[]; states=[]; logits=[]; gates=[]
        for t in range(T):
            if reset: state=h0
            ins.append(state)
            if dict_ is None:
                w=self.mat if edit is None else self.mat+edit[t]; out=state@w.T
            else:
                u,v,g=dict_; gate=g(state).sigmoid();gates.append(gate);out=((state@v.T)*gate)@u.T
            state=(inp+out).tanh(); states.append(state); logits.append(self.out(state).squeeze(-1))
        return torch.stack(logits,1),torch.stack(states,1),torch.stack(ins,1),gates
    def forward(self,x,h,m,edit=None,reset=False):return self.trajectory(x,h,m,edit,reset)[0][:,-1]

def subset(d,cap,seed):
    ix=torch.randperm(len(d[0]),generator=torch.Generator().manual_seed(seed))[:cap]
    return tuple(a[ix] for a in d)

def fit(d,kind,seed):
    torch.manual_seed(seed); model=Dense() if kind=='dense' else Model(kind);opt=torch.optim.AdamW(model.parameters(),lr=.004)
    x,h,m,y,p=d
    for _ in range(10):
        for ix in torch.randperm(len(x)).split(512):
            loss=nn.functional.binary_cross_entropy_with_logits(p[ix]+model(x[ix],h[ix],m[ix]),y[ix]);opt.zero_grad();loss.backward();opt.step()
    return model.eval()

class Decomp(nn.Module):
    def __init__(self,mat):
        super().__init__();u,s,v=torch.linalg.svd(mat.detach());self.register_buffer('u0',u*s.sqrt());self.register_buffer('v0',s.sqrt()[:,None]*v)
        self.skew=nn.Parameter(torch.zeros(H,H));self.g=nn.Linear(H,H);nn.init.zeros_(self.g.weight);nn.init.constant_(self.g.bias,2.)
    def factors(self):
        q=torch.matrix_exp(self.skew-self.skew.T); return self.u0@q,q.T@self.v0
    def atom(self,j):
        u,v=self.factors();return u[:,j,None]@v[j,None,:]

def decompose(model,d,seed):
    for p in model.parameters():p.requires_grad_(False)
    torch.manual_seed(seed); dec=Decomp(model.mat);opt=torch.optim.AdamW(dec.parameters(),lr=.005)
    x,h,m,_,_=subset(d,12000,889)
    with torch.no_grad():cl,cs,_,_=model.trajectory(x,h,m)
    for _ in range(8):
        for ix in torch.randperm(len(x)).split(512):
            u,v=dec.factors(); l,s,_,gs=model.trajectory(x[ix],h[ix],m[ix],dict_=(u,v,dec.g))
            loss=(l-cl[ix]).square().mean()+.1*(s-cs[ix]).square().mean()+.02*torch.stack(gs,1).amax(1).mean()
            opt.zero_grad();loss.backward();opt.step()
    return dec.eval()

@torch.no_grad()
def predict(d,model=None,edit=None,reset=False):
    x,h,m,y,p=d;out=[]
    for ix in torch.arange(len(x)).split(8192):
        q=p[ix] if model is None else p[ix]+(model(x[ix],h[ix],m[ix],edit,reset) if isinstance(model,Dense) else model(x[ix],h[ix],m[ix]))
        out.append(q.sigmoid().clamp(.05,.995))
    return torch.cat(out)

@torch.no_grad()
def clean_states(d,model):
    x,h,m,_,_=d
    return torch.cat([model.trajectory(x[ix],h[ix],m[ix])[2] for ix in torch.arange(len(x)).split(8192)])

def rms(states,edit):return torch.einsum('bti,toi->bto',states,edit).square().mean((0,2)).sqrt()

@torch.no_grad()
def choose(model,dec,d,sl,seed):
    base=mean(q['logloss'] for q in metrics(predict(d,model),d[3],sl));u,v=dec.factors()
    effects=[]
    for j in range(H):
        edit=-dec.atom(j).repeat(T,1,1);effects.append(mean(q['logloss'] for q in metrics(predict(d,model,edit),d[3],sl))-base)
    j=max(range(H),key=lambda j:effects[j]);atom=dec.atom(j);edits={'ablate':-atom.repeat(T,1,1),'enhance':atom.repeat(T,1,1)}
    states=clean_states(d,model);target=rms(states,edits['ablate']);gen=torch.Generator().manual_seed(seed+801)
    for i in range(8):
        a=torch.randn(H,generator=gen);b=torch.randn(H,generator=gen);w=-(a[:,None]@b[None,:]).repeat(T,1,1)
        edits[f'sham{i}']=w*(target/rms(states,w).clamp_min(1e-10))[:,None,None]
    return edits,dict(seed=seed,atom=j,effects=effects,weight_error=float((u@v-model.mat).abs().max()),cal_rms=target.tolist(),cal_sham_relative_max=max(float(((rms(states,e)-target)/target.clamp_min(1e-10)).abs().max()) for k,e in edits.items() if k.startswith('sham')))

@torch.no_grad()
def fidelity(d,model,dec,edit):
    x,h,m,_,_=d;u,v=dec.factors();sq=0.;maxerr=0.;ng=0;gate=0.;maxgate=0.;states=[];exacterr=0.
    for ix in torch.arange(len(x)).split(8192):
        l,_,s,_=model.trajectory(x[ix],h[ix],m[ix]);g,_,_,gs=model.trajectory(x[ix],h[ix],m[ix],dict_=(u,v,dec.g));err=g-l;sq+=float(err.square().sum());maxerr=max(maxerr,float(err.abs().max()));ng+=err.numel();gg=torch.stack(gs,1);gate+=float(gg.mean((1,2)).sum());maxgate+=float(gg.amax(1).mean(1).sum());states.append(s)
        ee=model.trajectory(x[ix],h[ix],m[ix],edit=(u@v-model.mat).repeat(T,1,1))[0];exacterr=max(exacterr,float((ee-l).abs().max()))
    state=torch.cat(states);zero=m.sum(1)==0;write=state[zero,0]@edit[0].T
    return dict(logit_rms=(sq/ng)**.5,maxerr=maxerr,exact_logit_max=exacterr,mean_gate=gate/len(x),max_gate=maxgate/len(x),zero_history=int(zero.sum()),exposed=int((write.norm(dim=1)>1e-8).sum()),states=state)

@torch.no_grad()
def route(r,prior,model=None,edit=None,reset=False):
    work=calls=rejects=0;first=[];trace=[]
    def visit(s):
        nonlocal work,calls,rejects
        parent=r.poly(s)
        if len(parent)==2:return True
        bank=r.bank(s);xs=r.x[s];history=[];remain=set(range(len(bank)));base=torch.tensor([prob(prior,bucket(r,a)) for a in bank]).logit();xt=torch.tensor(xs);st=[structural(a) for a in bank]
        while remain:
            h=torch.zeros(len(bank),2,W);m=torch.zeros(len(bank),2);hist=history[-2:]
            if hist:h[:,:len(hist)]=torch.tensor(hist);m[:,:len(hist)]=1
            q=base if model is None else base+(model(xt,h,m,edit,reset) if isinstance(model,Dense) else model(xt,h,m));ps=q.sigmoid().clamp(.05,.995).tolist()
            i=min(remain,key=lambda j:(st[j]+bank[j][2]*(1/ps[j]-1),j))
            if not history:first.append((s,i))
            remain.remove(i);spans,cs,cost,_=bank[i]
            if work+cost>3*r.n*r.n or calls>=8*r.n:return False
            ok,paid=verify(parent,cs,r.p);assert paid==cost;work+=paid;calls+=1;trace.append((s,i,ok))
            if not ok:rejects+=1;history.append(xs[i]);continue
            assert all(c==r.poly(t) for c,t in zip(cs,spans));return all(visit(t) for t in spans)
        return False
    start=time.perf_counter_ns();closed=visit((0,r.n));elapsed=time.perf_counter_ns()-start
    assert work<=3*r.n*r.n and calls<=8*r.n
    return dict(closed=closed,work=work,calls=calls,rejects=rejects,elapsed_ns=elapsed,first=first[0][1],trace=hashlib.sha256(repr(trace).encode()).hexdigest())

def main():
    torch.set_num_threads(1);tr,prior,tr_s=data(roots('train'));dtrain=subset(tr,30000,778);ca,_,ca_s=data(roots('cal'),prior);print('data',len(tr[0]),len(dtrain[0]),len(ca[0]),flush=True)
    models={};decs={};edits={};choices=[]
    for kind in ('linear','mlp','dense'):
        for seed in SEEDS:
            model=fit(dtrain,kind,seed);models[kind,seed]=model
            if kind=='dense':
                dec=decompose(model,dtrain,seed);decs[seed]=dec;edits[seed],choice=choose(model,dec,ca,ca_s,seed);choices.append(choice)
            print('trained',kind,seed,flush=True)
    prediction=[];routes=[];coverage=[];fidel=[]
    for split in ('iid','ood'):
        rr=roots(split);d,_,sl=data(rr,prior);print(split,len(d[0]),flush=True)
        for root,f,a,b,cov in sl:coverage.append(dict(split=split,root=root,family=f,examples=b-a,**cov))
        arms=[('bucket',0,None,None,False)]+[(kind,seed,m,None,False) for (kind,seed),m in models.items()]+[(arm,seed,models['dense',seed],e,False) for seed,es in edits.items() for arm,e in es.items()]+[('reset',seed,models['dense',seed],None,True) for seed in SEEDS]
        for arm,seed,m,e,reset in arms:prediction.extend(dict(split=split,arm=arm,seed=seed,**q) for q in metrics(predict(d,m,e,reset),d[3],sl))
        for seed in SEEDS:
            f=fidelity(d,models['dense',seed],decs[seed],edits[seed]['ablate']);state=f.pop('states');target=rms(state,edits[seed]['ablate']);fidel.append(dict(split=split,seed=seed,**f,target_rms=target.tolist(),sham_relative_max=max(float(((rms(state,e)-target)/target.clamp_min(1e-10)).abs().max()) for k,e in edits[seed].items() if k.startswith('sham'))))
        for r in rr:
            order=list(arms);random.Random(r.seed+123).shuffle(order)
            for arm,seed,m,e,reset in order:routes.append(dict(split=split,arm=arm,seed=seed,root=r.seed,family=r.family,n=r.n,**route(r,prior,m,e,reset)))
        print('routed',split,flush=True)
    output=dict(train_examples=len(tr[0]),train_used=len(dtrain[0]),cal_examples=len(ca[0]),params={k:sum(p.numel() for p in models[k,SEEDS[0]].parameters()) for k in ('linear','mlp','dense')},choices=choices,prediction=prediction,routes=routes,coverage=coverage,fidelity=fidel)
    (HERE/'receipt.json').write_text(json.dumps(output,sort_keys=True));torch.save(dict(models={f'{k}_{s}':m.state_dict() for (k,s),m in models.items()},decs={s:m.state_dict() for s,m in decs.items()},edits=edits,prior=prior),HERE/'models.pt')
if __name__=='__main__':main()
