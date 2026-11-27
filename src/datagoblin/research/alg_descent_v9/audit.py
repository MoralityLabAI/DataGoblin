"""Independent checkpoint inference, coefficient convolution and guarded iterative DFS."""
import json,hashlib,random,statistics as st
from collections import defaultdict
import torch
from experiment import HERE,fresh,data,SEEDS,H,T
try:
    from alg_descent_v8.experiment import prob,bucket,structural,W
except ImportError:
    from v8.experiment import prob,bucket,structural,W

def inference(x,h,m,b,arm,seed,pack):
    if arm=='bucket':return b.sigmoid().clamp(.05,.995)
    kind=arm if arm in ('linear','mlp','dense','one') else 'dense';w=pack['models'][f'{kind}_{seed}']
    hist=(h*m[:,:,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
    if kind in ('linear','mlp'):
        z=torch.cat((x,hist,m.sum(1,keepdim=True)/2),1)
        if kind=='linear':z=z@w['net.weight'].T+w['net.bias']
        else:z=(z@w['net.0.weight'].T+w['net.0.bias']).tanh()@w['net.2.weight'].T+w['net.2.bias']
    else:
        inp=x@w['enc.weight'].T+w['enc.bias']+hist@w['enc.weight'].T
        z=inp.tanh();e=pack['edits'].get(seed,{}).get(arm);mat=w['mat'] if e is None else w['mat']+e
        for _ in range(1 if kind=='one' else T):z=(inp+z@mat.T).tanh()
        z=z@w['out.weight'].T+w['out.bias']
    return (b+z.flatten()).sigmoid().clamp(.05,.995)

def exact(parent,children,p):
    acc=[1];paid=0
    for child in children:
        paid+=len(acc)*len(child)
        acc=[sum(acc[i]*child[k-i] for i in range(len(acc)) if 0<=k-i<len(child))%p for k in range(len(acc)+len(child)-1)]
    return tuple(acc)==parent,paid

def replay(r,arm,seed,pack):
    stack=[(0,r.n)];trace=[];work=calls=rejects=binding=0;closed=True;first=None
    while stack:
        s=stack.pop();parent=r.poly(s)
        if len(parent)==2:continue
        bank=r.bank(s);xs=torch.tensor(r.x[s]);n=len(bank);history=[];remaining=list(range(n));b=torch.tensor([prob(pack['prior'],bucket(r,a)) for a in bank]).logit()
        while remaining:
            h=torch.zeros(n,2,W);m=torch.zeros(n,2)
            if history:h[:,:len(history)]=torch.stack(history);m[:,:len(history)]=1
            ps=inference(xs,h,m,b,arm,seed,pack).tolist()
            i=min(remaining,key=lambda j:(structural(bank[j])+bank[j][2]*(1/ps[j]-1),j));remaining.remove(i)
            if first is None:first=i
            spans,cs,cost,_=bank[i]
            if work+cost>3*r.n*r.n or calls>=8*r.n:closed=False;break
            ok,paid=exact(parent,cs,r.p);assert paid==cost;work+=paid;calls+=1
            bound=ok and all(c==r.poly(t) for c,t in zip(cs,spans));binding+=int(ok and not bound);trace.append((s,i,bound))
            if bound:stack.extend(reversed(spans));break
            rejects+=1;history=(history+[xs[i]])[-2:]
        else:closed=False
        if not closed:break
    return dict(closed=closed,work=work,calls=calls,rejects=rejects,binding_rejects=binding,first=first,trace=hashlib.sha256(repr(trace).encode()).hexdigest())

def paired(rows,a,b,key,family=None):
    g=[]
    for arm in (a,b):
        q=defaultdict(list)
        for x in rows:
            if x['arm']==arm and (family is None or x['family']==family):q[x['root']].append(x[key])
        g.append(q)
    assert g[0].keys()==g[1].keys();ds=[st.mean(g[0][k])-st.mean(g[1][k]) for k in sorted(g[0])]
    rng=random.Random(8);boot=sorted(st.mean(rng.choices(ds,k=len(ds))) for _ in range(5000))
    return dict(mean=st.mean(ds),wins=sum(x<0 for x in ds),ties=sum(x==0 for x in ds),losses=sum(x>0 for x in ds),bootstrap95=[boot[125],boot[4874]])

@torch.no_grad()
def main():
    torch.set_num_threads(1);r=json.loads((HERE/'receipt.json').read_text());pack=torch.load(HERE/'models.pt',weights_only=True);p=r['prediction'];q=r['routes'];assert len(p)==len(q)==1896
    assert all(x['closed'] and x['work']<=3*x['n']**2 and x['calls']<=8*x['n'] for x in q)
    checked=metric_fields=route_fields=0;max_error=0.;different=[];bad_keys=[]
    for root in fresh('ood'):
        d,_,sl=data([root],pack['prior']);x,h,m,y,b=d
        for s,bank in root.original.items():
            for i,(a,v) in enumerate(zip(bank,root.labels[s])):
                ok,paid=exact(root.poly(s),a[1],root.p);assert ok==bool(v) and paid==a[2];checked+=1
                if ok and any(c!=root.poly(t) for c,t in zip(a[1],a[0])):bad_keys.append(dict(root=root.seed,span=s,candidate=i,children=a[0]))
        known={(z['arm'],z['seed']):z for z in p if z['root']==root.seed};rr={(z['arm'],z['seed']):z for z in q if z['root']==root.seed}
        for (arm,seed),z in known.items():
            pred=inference(x,h,m,b,arm,seed,pack)
            for k,v in [('logloss',float((-(y*pred.log()+(1-y)*(1-pred).log())).mean())),('brier',float((pred-y).square().mean()))]:
                err=abs(v-z[k]);max_error=max(max_error,err);assert err<1e-5;metric_fields+=1
            a=replay(root,arm,seed,pack);b0=rr[arm,seed];bad=[k for k,v in a.items() if v!=b0[k]];route_fields+=len(a)-len(bad)
            if bad:different.append(dict(root=root.seed,arm=arm,seed=seed,fields=bad))
        print('audited',root.seed,flush=True)
    contrasts={}
    for family in (None,'second','two_child'):
        for a,b0 in [('dense','one'),('dense','linear'),('dense','mlp'),('dense','bucket'),('learned','dense'),('learned','svd'),('svd','dense')]:
            for key,rows in [('logloss',p),('work',q)]:contrasts[str((family,a,b0,key))]=paired(rows,a,b0,key,family)
    sham=[paired(p,f'sham{i}','dense','logloss')['mean'] for i in range(16)]
    out=dict(eligible_identities=checked,binding_invalid_count=len(bad_keys),binding_invalid_examples=bad_keys[:20],metric_fields=metric_fields,max_metric_error=max_error,route_fields=route_fields,route_discrepancies=different,contrasts=contrasts,sham_damage=sham,median_sham_damage=st.median(sham))
    (HERE/'audit.json').write_text(json.dumps(out,sort_keys=True));print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
