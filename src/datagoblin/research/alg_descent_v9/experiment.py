"""ALG-DESCENT-09: frozen basis comparison and physical matched nulls."""
import json, random, sys
from pathlib import Path
from statistics import mean
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
try:
    from alg_descent_v8.experiment import Dense,Decomp,Model,FamilyRoot,data,metrics,predict,route,roots as train_roots,subset,SEEDS,H,T
except ImportError:
    from v8.experiment import Dense,Decomp,Model,FamilyRoot,data,metrics,predict,route,roots as train_roots,subset,SEEDS,H,T

def fresh(split):
    base,count,ds,fams={'cal':(410000,8,(12,16,20),('adjacent','separated')),'ood':(420000,12,(24,32,40,48),('second','two_child'))}[split]
    return [FamilyRoot(base+1000*j+i,ds[i%len(ds)],f) for j,f in enumerate(fams) for i in range(count)]

class OnePass(Dense):
    def forward(self,x,h,m,edit=None,reset=False):
        hist=(h*m[:,:,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
        inp=self.enc(x)+torch.nn.functional.linear(hist,self.enc.weight)
        return self.out((inp+inp.tanh()@self.mat.T).tanh()).squeeze(-1)

def fit_one(d,seed):
    torch.manual_seed(seed); model=OnePass(); opt=torch.optim.AdamW(model.parameters(),lr=.004)
    x,h,m,y,p=d
    for _ in range(10):
        for ix in torch.randperm(len(x)).split(512):
            loss=nn.functional.binary_cross_entropy_with_logits(p[ix]+model(x[ix],h[ix],m[ix]),y[ix]);opt.zero_grad();loss.backward();opt.step()
    return model.eval()

@torch.no_grad()
def covariances(d,model):
    x,h,m,_,_=d; c=torch.zeros(T,H,H,dtype=torch.float64); g=torch.zeros(H*H,H*H,dtype=torch.float64)
    n=0; w=model.mat.double(); read=model.out.weight.double()
    for ix in torch.arange(len(x)).split(4096):
        _,states,ins,_=model.trajectory(x[ix],h[ix],m[ix]); ins=ins.double(); states=states.double()
        c+=torch.einsum('bti,btj->tij',ins,ins)
        delta=read*(1-states[:,-1].square()); grad=torch.zeros(len(ix),H,H,dtype=torch.float64)
        for t in reversed(range(T)):
            grad+=delta[:,:,None]*ins[:,t,None,:]
            if t: delta=(delta@w)*(1-states[:,t-1].square())
        v=grad.flatten(1);g+=v.T@v;n+=len(ix)
    return c/n,g/n

def magnitudes(edits,c,g):
    e=edits.double();flat=e.flatten(1)
    norm=flat.norm(dim=1)
    write=(torch.einsum('aoi,tij,aoj->at',e,c,e)/H).clamp_min(0).sqrt()
    jac=((flat@g)*flat).sum(1).clamp_min(0).sqrt()
    return torch.cat((norm[:,None],write,jac[:,None]),1)

@torch.no_grad()
def choose(model,dec,d,sl,seed):
    baseline=mean(q['logloss'] for q in metrics(predict(d,model),d[3],sl)); u0,v0=dec.u0,dec.v0
    u,v=dec.factors(); bases={'learned':(u,v),'svd':(u0,v0)}
    gen=torch.Generator().manual_seed(9000+seed)
    for k in range(4):
        a=torch.randn(H,H,generator=gen);q,r=torch.linalg.qr(a);q=q*torch.sign(r.diag());bases[f'haar{k}']=(u0@q,q.T@v0)
    edits={}; selections=[]
    for name,(u,v) in bases.items():
        err=float((u@v-model.mat).abs().max());assert err<1e-5
        atoms=[-(u[:,j,None]@v[j,None,:]) for j in range(H)]
        effects=[mean(z['logloss'] for z in metrics(predict(d,model,e.repeat(T,1,1)),d[3],sl))-baseline for e in atoms]
        j=max(range(H),key=lambda j:effects[j]);edits[name]=atoms[j];selections.append(dict(seed=seed,basis=name,atom=j,damage=effects[j],all_damage=effects,weight_error=err))
    c,g=covariances(d,model);target=magnitudes(edits['learned'][None],c,g)[0]
    a=torch.randn(2048,H,generator=gen);b=torch.randn(2048,H,generator=gen);e=a[:,:,None]*b[:,None,:]
    m=magnitudes(e,c,g);lr=(m.clamp_min(1e-15)/target.clamp_min(1e-15)).log();scale=(-(lr.amax(1)+lr.amin(1))/2).exp()
    e=e*scale.float()[:,None,None];ratios=magnitudes(e,c,g)/target;mis=ratios.log().abs().amax(1);chosen=torch.argsort(mis)[:16]
    for i,j in enumerate(chosen.tolist()):edits[f'sham{i}']=e[j]
    return edits,selections,dict(seed=seed,target=target.tolist(),indices=chosen.tolist(),cal_ratios=ratios[chosen].tolist(),cal_mismatch=mis[chosen].tolist())

@torch.no_grad()
def finite_difference(model,d,edit,seed):
    x,h,m,_,_=subset(d,256,seed);_,states,ins,_=model.trajectory(x,h,m);z=torch.zeros_like(ins[:,0]);w=model.mat
    for t in range(T): z=(z@w.T+ins[:,t]@edit.T)*(1-states[:,t].square())
    analytic=(z@model.out.weight.T).flatten(); eps=.001
    numeric=(model(x,h,m,(eps*edit).repeat(T,1,1))-model(x,h,m,(-eps*edit).repeat(T,1,1)))/(2*eps)
    return dict(n=len(x),rmse=float((analytic-numeric).square().mean().sqrt()),relative=float((analytic-numeric).norm()/analytic.norm().clamp_min(1e-10)),maxerr=float((analytic-numeric).abs().max()))

def main():
    torch.set_num_threads(1);v8=next(p for p in (HERE.parent/'alg_descent_v8',HERE.parent/'v8') if (p/'models.pt').exists())
    pack=torch.load(v8/'models.pt',weights_only=True);prior=pack['prior'];tr,check,_=data(train_roots('train'));assert check==prior
    tr=subset(tr,30000,778);ca,_,sl=data(fresh('cal'),prior);print('cal',len(ca[0]),flush=True)
    models={};edits={};selections=[];matching=[];diff=[]
    for seed in SEEDS:
        for kind in ('linear','mlp','dense'):
            m=Dense() if kind=='dense' else Model(kind);m.load_state_dict(pack['models'][f'{kind}_{seed}']);models[kind,seed]=m.eval()
        models['one',seed]=fit_one(tr,seed);m=models['dense',seed];dec=Decomp(m.mat);dec.load_state_dict(pack['decs'][seed]);dec.eval()
        es,sel,match=choose(m,dec,ca,sl,seed);edits[seed]=es;selections.extend(sel);matching.append(match)
        diff.append(dict(seed=seed,**finite_difference(m,ca,es['learned'],seed)));print('selected',seed,flush=True)
    (HERE/'selection.json').write_text(json.dumps(dict(selections=selections,matching=matching,finite_difference=diff),sort_keys=True))
    rr=fresh('ood');d,_,sl=data(rr,prior);print('ood',len(d[0]),flush=True)
    prediction=[];routes=[];coverage=[];arms=[('bucket',0,None,None)]+[(k,s,m,None) for (k,s),m in models.items()]+[(k,s,models['dense',s],e.repeat(T,1,1)) for s,es in edits.items() for k,e in es.items()]
    for root,f,a,b,cov in sl:coverage.append(dict(root=root,family=f,examples=b-a,**cov))
    for seed,match in zip(SEEDS,matching):
        c,g=covariances(d,models['dense',seed]);e=edits[seed];target=magnitudes(e['learned'][None],c,g)[0]
        ratios=(magnitudes(torch.stack([e[f'sham{i}'] for i in range(16)]),c,g)/target).tolist();match['ood_ratios']=ratios
        valid=lambda rs:all(.8<=x<=1.2 for x in rs)
        match['cal_count']=sum(map(valid,match['cal_ratios']));match['ood_count']=sum(map(valid,ratios));match['both']=[i for i,(a,b) in enumerate(zip(match['cal_ratios'],ratios)) if valid(a) and valid(b)]
    for k,s,m,e in arms:prediction.extend(dict(arm=k,seed=s,**z) for z in metrics(predict(d,m,e),d[3],sl))
    for r in rr:
        order=list(arms);random.Random(r.seed+123).shuffle(order)
        for k,s,m,e in order:routes.append(dict(arm=k,seed=s,root=r.seed,family=r.family,n=r.n,**route(r,prior,m,e)))
        print('routed',r.seed,flush=True)
    out=dict(cal_examples=len(ca[0]),ood_examples=len(d[0]),params={k:sum(p.numel() for p in models[k,SEEDS[0]].parameters()) for k in ('linear','mlp','dense','one')},selections=selections,matching=matching,finite_difference=diff,prediction=prediction,routes=routes,coverage=coverage)
    (HERE/'receipt.json').write_text(json.dumps(out,sort_keys=True));torch.save(dict(models={f'{k}_{s}':m.state_dict() for (k,s),m in models.items()},edits=edits,prior=prior),HERE/'models.pt')
if __name__=='__main__':main()
