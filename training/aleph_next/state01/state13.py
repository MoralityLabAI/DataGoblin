"""STATE13: cost-sensitive routing of certified synthetic rewrite search."""
import argparse,csv,hashlib,heapq,itertools,json,random,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
import state12 as s
SEED=2026112519
SEEDS=(313,317,331)
CAP=64
NCAP=256
ROUTES={'search_first':('search','norm'),'normalize_first':('norm','search'),'decompose_first':('decompose','search','norm'),'shortnorm_first':('shortnorm','search','norm')}
TOKENS=('+','*','++','rev','mf','mg','0','1','nil')
def nf(d,x):return s.polynomial(x) if d=='nat' else s.list_word(x)
def generate():
    prior={(d,x,y) for d,x,y,*_ in s.panel()};prior|={(d,y,x) for d,x,y in prior}
    rng=random.Random(s.s.SEED);seen=set()
    for d,g in [('nat',{'struct','id'}),('nat',{'dist','id'}),('list',{'append','rev'}),('list',{'append','map'})]:s.s.generate(rng,d,g,120,False,seen)
    for d,g in [('nat',{'struct','id','dist'}),('list',{'append','rev','map'})]:s.s.generate(rng,d,g,32,True,seen)
    prior|=seen|{(d,y,x) for d,x,y in seen};rng=random.Random(SEED);rows=[]
    def walk(d,depth,steps):
        while True:
            x=s.s.expr(rng,d,depth,{'struct','id','dist','append','rev','map'});y=x;visited={x}
            for _ in range(steps):
                opts=[a for a in s.moves(d,y,True) if a[0] not in visited]
                if not opts:break
                y,_,_,_=rng.choice(opts);visited.add(y)
            if y!=x:return x,y
    def wrap(d,x,y,depth):
        for _ in range(depth):
            op=rng.choice(('+','*') if d=='nat' else ('++','rev','mf','mg'));a=rng.choice(('x','y','z','1') if d=='nat' else ('xs','ys','zs'))
            if op in ('+','*','++'):
                if rng.randrange(2):x,y=(op,x,a),(op,y,a)
                else:x,y=(op,a,x),(op,a,y)
            else:x,y=(op,x),(op,y)
        return x,y
    def sample(d,f):
        if f=='shallow':return walk(d,3,3)
        if f=='local':return wrap(d,*walk(d,2,2),2)
        if f=='long':return walk(d,4,7)
        if f=='deep_context':return wrap(d,*walk(d,2,3),7)
        if f=='two_holes':
            x,y=walk(d,2,3);a,b=walk(d,2,3);op='+' if d=='nat' else '++';return (op,x,a),(op,y,b)
        raise ValueError(f)
    for split,fs,n in [('train',('shallow','local'),24),('held',('long','deep_context','two_holes'),16)]:
        for d in ('nat','list'):
            for fam in fs:
                count=0
                while count<n:
                    x,y=sample(d,fam)
                    if rng.randrange(2):x,y=y,x
                    if (d,x,y) in prior or x==y or nf(d,x)!=nf(d,y):continue
                    bad=('+',y,'1') if d=='nat' else ('++',y,'xs')
                    if (d,x,bad) in prior:continue
                    prior.update(((d,x,y),(d,y,x),(d,x,bad),(d,bad,x)))
                    rows.append(dict(split=split,domain=d,family=fam,x=x,y=y,truth=True))
                    if count%3==0:
                        assert nf(d,x)!=nf(d,bad);rows.append(dict(split=split,domain=d,family=fam,x=x,y=bad,truth=False))
                    count+=1
    return rows

def search(d,x,y,budget):
    if x==y:return [],0,0,0
    seq=itertools.count();ends=(x,y);parents=[{x:()},{y:()}];queues=[[(0,next(seq),x)],[(0,next(seq),y)]];n=e=t=0
    while n<budget and any(queues):
        side=n%2
        if not queues[side]:side=1-side
        _,_,u=heapq.heappop(queues[side]);n+=1;t+=s.s.size(u)*len(s.DOMAIN[d])
        for edge in s.moves(d,u,True):
            z=edge[0]
            if z in parents[side]:continue
            e+=1;p=parents[side][u]+(edge,);parents[side][z]=p
            if z in parents[1-side]:return list(parents[0][z])+s.reverse_path(y,parents[1][z]),n,e,t
            heapq.heappush(queues[side],(len(p)+s.s.distance(z,ends[1-side])/4,next(seq),z))
    return None,n,e,t

def decompose(d,x,y):
    n=e=t=v=0;path=[];cur=x
    def visit(p,target):
        nonlocal cur,n,e,t,v
        v+=1;u=s.s.get(cur,p)
        if u==target:return True
        if not isinstance(u,str) and not isinstance(target,str) and u[0]==target[0] and len(u)==len(target) and all(nf(d,a)==nf(d,b) for a,b in zip(u[1:],target[1:])):
            return all(visit(p+(j,),a) for j,a in enumerate(target[1:],1))
        if not p:return False
        z,nn,ee,tt=search(d,u,target,CAP-n);n+=nn;e+=ee;t+=tt
        if z is None:return False
        for sub,i,q,r in z:cur=s.s.replace(cur,p,sub);path.append((cur,i,p+q,r))
        return True
    good=visit((),y);return path if good and cur==y else None,n,e,t,v

def attempt(d,x,y,a):
    start=time.process_time();s.moves.cache_clear();s.s.distance.cache_clear();n=e=t=v=ns=0
    if a=='search':path,n,e,t=search(d,x,y,CAP)
    elif a=='decompose':path,n,e,t,v=decompose(d,x,y)
    else:path,ns,v,t,_=s.normal(d,x,y,16 if a=='shortnorm' else NCAP)
    if path is not None:
        assert s.verify(d,x,y,path);base=nf(d,x);assert all(nf(d,z)==base for z,*_ in path)
    return dict(ok=path is not None,steps=len(path) if path is not None else 0,expansions=n,edges=e,match_tests=t,visits=v,normal_steps=ns,cpu_seconds=time.process_time()-start,work=t+v+n+e+ns),path

def route(r,name,weight=32):
    if not r['gate']:return dict(ok=False,work=0,steps=0,cost=0,cpu_seconds=0,actions=[])
    work=steps=0;cpu=0;acts=[];ok=False
    for a in ROUTES[name]:
        v=r['attempts'][a];acts.append(a);work+=v['work'];cpu+=v['cpu_seconds']
        if v['ok']:steps=v['steps'];ok=True;break
    return dict(ok=ok,work=work,steps=steps,cost=work+weight*steps,cpu_seconds=cpu,actions=acts)

def features(d,x,y):
    def depth(t):return 0 if isinstance(t,str) else 1+max(depth(a) for a in t[1:])
    def same(a,b):
        if a==b:return s.s.size(a)
        if isinstance(a,str) or isinstance(b,str) or a[0]!=b[0]:return 0
        return 1+sum(same(u,v) for u,v in zip(a[1:],b[1:]))
    a,b=s.s.bag(x),s.s.bag(y);n,m=s.s.size(x),s.s.size(y)
    return [float(d=='nat'),n/64,m/64,depth(x)/16,depth(y)/16,s.s.distance(x,y)/64,same(x,y)/max(n,m),float(not isinstance(x,str) and not isinstance(y,str) and x[0]==y[0])]+[v[k]/32 for v in (a,b) for k in TOKENS]

def main(out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out.mkdir(parents=True,exist_ok=True);start=time.process_time();rows=generate();paths={}
    for j,r in enumerate(rows):
        d,x,y=r['domain'],r['x'],r['y'];t=time.process_time();r['gate']=nf(d,x)==nf(d,y);r['gate_cpu_seconds']=time.process_time()-t;assert r['gate']==r['truth'];r['features']=features(d,x,y);r['attempts']={}
        if r['gate']:
            for a in ('search','decompose','norm','shortnorm'):
                v,p=attempt(d,x,y,a);r['attempts'][a]=v;paths[j,a]=p
        if (j+1)%32==0:print('attempts',j+1,'/',len(rows),flush=True)
    names=list(ROUTES);weight=32;penalty=10**9
    def objective(r,k):
        v=route(r,k,weight);return v['cost']+penalty*int(r['truth'] and not v['ok'])
    tr=[r for r in rows if r['split']=='train' and r['truth']];he=[r for r in rows if r['split']=='held'];x=torch.tensor([r['features'] for r in tr]);cost=np.array([[objective(r,k) for k in names] for r in tr],float);y=torch.tensor(np.log1p(cost)-np.log1p(cost.min(1,keepdims=True)),dtype=torch.float32)
    pol={k:[k]*len(he) for k in names};best=min(names,key=lambda k:sum(objective(r,k) for r in tr));pol['train_best_static']=[best]*len(he);pol['oracle_route']=[min(names,key=lambda k:objective(r,k)) for r in he];rng=random.Random(SEED+1);pol['random_route']=[rng.choice(names) for _ in he]
    fits=[]
    for seed in SEEDS:
        torch.manual_seed(seed);m=nn.Sequential(nn.Linear(x.shape[1],32),nn.GELU(),nn.Linear(32,len(names)));opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01);t=time.process_time()
        for _ in range(400):
            opt.zero_grad();loss=nn.functional.smooth_l1_loss(m(x),y);loss.backward();opt.step()
        fit=time.process_time()-t;t=time.process_time()
        with torch.no_grad():pred=m(torch.tensor([r['features'] for r in he])).argmin(1).tolist()
        inference=time.process_time()-t;pol[f'learned_{seed}']=[names[i] for i in pred];fits.append(dict(seed=seed,params=sum(p.numel() for p in m.parameters()),train_cpu_seconds=fit,inference_cpu_seconds=inference,loss=float(loss.detach())))
    summary=[]
    for policy,choices in pol.items():
        for fam in ('all','long','deep_context','two_holes'):
            selected=[(r,k) for r,k in zip(he,choices) if fam=='all' or r['family']==fam];vs=[route(r,k) for r,k in selected]
            z=dict(policy=policy,family=fam,n=len(vs),positive=sum(r['truth'] for r,_ in selected),solved=sum(v['ok'] for v in vs),false_accept=sum(v['ok'] and not r['truth'] for (r,_),v in zip(selected,vs)),work=sum(v['work'] for v in vs),proof_steps=sum(v['steps'] for v in vs),cost=sum(v['cost'] for v in vs),attempt_cpu_seconds=sum(v['cpu_seconds'] for v in vs),gate_cpu_seconds=sum(r['gate_cpu_seconds'] for r,_ in selected),choices=dict(Counter(k for r,k in selected if r['truth'])));summary.append(z)
    sensitivity={str(w):{k:sum(route(r,choice,w)['cost'] for r,choice in zip(he,choices)) for k,choices in pol.items()} for w in (0,32,128,512)}
    for z in summary:
        if z['family']=='all':print(z,flush=True)
    # Replay certificates for all held successful routes, compactly storing only one path per primitive.
    cert=[]
    for j,r in enumerate(rows):
        if r['split']!='held' or not r['gate']:continue
        for a in ('search','decompose','norm','shortnorm'):
            p=paths[j,a]
            if p is not None:cert.append(dict(row=j,action=a,path=p))
    import gzip
    data=gzip.compress(json.dumps(cert,separators=(',',':')).encode(),mtime=0);(out/'STATE13_CERTIFICATES.json.gz').write_bytes(data)
    receipt=dict(seed=SEED,seeds=SEEDS,weight=weight,route_failure_penalty=penalty,train_best_static=best,train_positive=len(tr),held_rows=len(he),training_total=len(rows)-len(he),routes=ROUTES,summary=summary,choices=pol,fits=fits,sensitivity=sensitivity,records=rows,experiment_cpu_seconds=time.process_time()-start,certificate_sha256=hashlib.sha256(data).hexdigest(),lean_kernel_verified=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(out/'STATE13_RECEIPTS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (out/'STATE13_SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
