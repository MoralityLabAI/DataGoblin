"""Independent frozen-checkpoint arithmetic, route, and receipt audit."""
import json, hashlib, random, statistics as st
from collections import defaultdict
import torch
from experiment import HERE, roots, data, prob, bucket, structural, T, H, W

def independent(x,h,m,w,kind,edit=None,reset=False):
    hist=(h*m[:,:,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
    if kind!='dense':
        z=torch.cat((x,hist,m.sum(1,keepdim=True)/2),1)
        if kind=='linear':return (z@w['net.weight'].T+w['net.bias']).flatten()
        z=(z@w['net.0.weight'].T+w['net.0.bias']).tanh()
        return (z@w['net.2.weight'].T+w['net.2.bias']).flatten()
    inp=x@w['enc.weight'].T+w['enc.bias']+hist@w['enc.weight'].T
    h0=inp.tanh();state=h0
    for t in range(T):
        mat=w['mat'] if edit is None else w['mat']+edit[t]
        state=(inp+(h0 if reset else state)@mat.T).tanh()
    return (state@w['out.weight'].T+w['out.bias']).flatten()

def exact(parent,children,p):
    acc=[1];paid=0
    for child in children:
        paid+=len(acc)*len(child)
        acc=[sum(acc[i]*child[k-i] for i in range(len(acc)) if 0<=k-i<len(child))%p for k in range(len(acc)+len(child)-1)]
    return tuple(acc)==parent,paid

def predict(x,h,m,b,arm,seed,pack):
    if arm=='bucket':return b.sigmoid().clamp(.05,.995)
    kind=arm if arm in ('linear','mlp','dense') else 'dense'
    edit=pack['edits'].get(seed,{}).get(arm)
    return (b+independent(x,h,m,pack['models'][f'{kind}_{seed}'],kind,edit,arm=='reset')).sigmoid().clamp(.05,.995)

def replay(r,arm,seed,pack):
    stack=[(0,r.n)];trace=[];work=calls=rejects=0;closed=True;first=None
    while stack:
        s=stack.pop();parent=r.poly(s)
        if len(parent)==2:continue
        bank=r.bank(s);xs=torch.tensor(r.x[s]);n=len(bank);history=[];remaining=list(range(n));b=torch.tensor([prob(pack['prior'],bucket(r,a)) for a in bank]).logit()
        while remaining:
            h=torch.zeros(n,2,W);m=torch.zeros(n,2)
            if history:h[:,:len(history)]=torch.stack(history);m[:,:len(history)]=1
            ps=predict(xs,h,m,b,arm,seed,pack).tolist()
            i=min(remaining,key=lambda j:(structural(bank[j])+bank[j][2]*(1/ps[j]-1),j));remaining.remove(i)
            if first is None:first=i
            spans,cs,cost,_=bank[i]
            if work+cost>3*r.n*r.n or calls>=8*r.n:closed=False;break
            ok,paid=exact(parent,cs,r.p);assert paid==cost;work+=paid;calls+=1;trace.append((s,i,ok))
            if ok:
                assert all(c==r.poly(t) for c,t in zip(cs,spans));stack.extend(reversed(spans));break
            rejects+=1;history=(history+[xs[i]])[-2:]
        else:closed=False
        if not closed:break
    return dict(closed=closed,work=work,calls=calls,rejects=rejects,first=first,trace=hashlib.sha256(repr(trace).encode()).hexdigest())

def paired(rows,a,b,key,family=None):
    g=[]
    for arm in (a,b):
        q=defaultdict(list)
        for x in rows:
            if x['split']=='ood' and x['arm']==arm and (family is None or x['family']==family):q[x['root']].append(x[key])
        g.append(q)
    assert g[0].keys()==g[1].keys()
    ds=[st.mean(g[0][k])-st.mean(g[1][k]) for k in sorted(g[0])];rng=random.Random(8);boot=sorted(st.mean(rng.choices(ds,k=len(ds))) for _ in range(5000))
    return dict(mean=st.mean(ds),wins=sum(x<0 for x in ds),ties=sum(x==0 for x in ds),losses=sum(x>0 for x in ds),bootstrap95=[boot[125],boot[4874]])

@torch.no_grad()
def main():
    torch.set_num_threads(1);r=json.loads((HERE/'receipt.json').read_text());pack=torch.load(HERE/'models.pt',weights_only=True);p=r['prediction'];q=r['routes'];assert len(p)==len(q)==1720
    assert all(x['closed'] and x['work']<=3*x['n']**2 and x['calls']<=8*x['n'] for x in q)
    _,prior,_=data(roots('train'));assert prior==pack['prior']
    weight_err=[];atom_err=[]
    for c in r['choices']:
        seed,j=c['seed'],c['atom'];d=pack['decs'][seed];a=d['skew'].double();rot=torch.matrix_exp(a-a.T);u=d['u0'].double()@rot;v=rot.T@d['v0'].double();w=pack['models'][f'dense_{seed}']['mat'].double();weight_err.append(float((u@v-w).abs().max()));atom_err.append(float((u[:,j,None]@v[j,None,:]+pack['edits'][seed]['ablate'][0].double()).abs().max()))
    assert max(weight_err+atom_err)<1e-5
    checked=metric_fields=route_fields=roots_count=0;max_error=0.;different=[]
    for split in ('iid','ood'):
        for root in roots(split):
            d,_,sl=data([root],prior);x,h,m,y,b=d
            for s,bank in root.original.items():
                for a,v in zip(bank,root.labels[s]):
                    ok,paid=exact(root.poly(s),a[1],root.p);assert ok==bool(v) and paid==a[2]
                    checked+=1
            known={(z['arm'],z['seed']):z for z in p if z['root']==root.seed}
            rr={(z['arm'],z['seed']):z for z in q if z['root']==root.seed}
            for (arm,seed),z in known.items():
                pred=predict(x,h,m,b,arm,seed,pack)
                for k,v in [('logloss',float((-(y*pred.log()+(1-y)*(1-pred).log())).mean())),('brier',float((pred-y).square().mean()))]:
                    err=abs(v-z[k]);max_error=max(max_error,err);assert err<1e-5;metric_fields+=1
                a=replay(root,arm,seed,pack);b0=rr[arm,seed];bad=[k for k,v in a.items() if v!=b0[k]];route_fields+=len(a)-len(bad)
                if bad:different.append(dict(root=root.seed,arm=arm,seed=seed,fields=bad))
            roots_count+=1
    contrasts={}
    for family in (None,'second','two_child'):
        for a,b0 in [('dense','linear'),('dense','mlp'),('dense','bucket'),('ablate','dense'),('enhance','dense'),('reset','dense')]:
            contrasts[str((family,a,b0,'logloss'))]=paired(p,a,b0,'logloss',family);contrasts[str((family,a,b0,'work'))]=paired(q,a,b0,'work',family)
    sham=[paired(p,f'sham{i}','dense','logloss')['mean'] for i in range(8)]
    out=dict(roots=roots_count,eligible_identities=checked,metric_fields=metric_fields,max_metric_error=max_error,route_fields=route_fields,route_discrepancies=different,weight_error_max=max(weight_err),selected_atom_error_max=max(atom_err),contrasts=contrasts,sham_damage=sham,median_sham_damage=st.median(sham))
    (HERE/'audit.json').write_text(json.dumps(out,sort_keys=True));print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
