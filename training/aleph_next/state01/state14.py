"""STATE14: charged search and normalization probes, synthetic proof certificates."""
import argparse,csv,gzip,hashlib,heapq,itertools,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import state13 as q
s=q.s
SEED=2026112520
SEEDS=(337,347,349)
CAP,PROBE,NPROBE,NCAP,W=64,4,8,512,32
FAMILIES=('long','deep_context','two_holes','global')

def panel():
    old=q.SEED; rows=[]
    prior={(r['domain'],r['x'],r['y']) for r in q.generate()};prior|={(d,y,x) for d,x,y in prior}
    for i,seed in enumerate(range(SEED,SEED+3)):
        q.SEED=seed
        for r in q.generate():
            key=r['domain'],r['x'],r['y']
            if r['split']!='held' or key in prior:continue
            prior.update((key,(key[0],key[2],key[1])));rows.append(dict(r,split='held' if i==2 else 'train'))
    q.SEED=old;rng=random.Random(SEED+100)
    def fold(op,items):
        v=items[-1]
        for u in reversed(items[:-1]):v=(op,u,v)
        return v
    for split,n in [('train',32),('held',16)]:
        for d in ('nat','list'):
            for j in range(n):
                for _ in range(20000):
                    if d=='nat':
                        fs=[('+',rng.choice('xyz'),rng.choice('xyz')) for _ in range(3 if split=='held' else 2)]
                        x=fold('*',fs);ts=[fold('*',sorted(t)) for t in itertools.product(*(a[1:] for a in fs))];rng.shuffle(ts);y=fold('+',ts)
                    else:
                        it=[rng.choice(('xs','ys','zs')) for _ in range(5 if split=='held' else 4)];x=('rev',('mf',('mg',fold('++',it))));y=fold('++',[('mf',('mg',('rev',v))) for v in reversed(it)])
                    if rng.randrange(2):x,y=y,x
                    bad=('+',y,'1') if d=='nat' else ('++',y,'xs')
                    if (d,x,y) not in prior and (d,x,bad) not in prior:break
                else:raise RuntimeError('panel exhausted')
                prior.update(((d,x,y),(d,y,x),(d,x,bad),(d,bad,x)))
                assert q.nf(d,x)==q.nf(d,y) and q.nf(d,x)!=q.nf(d,bad)
                rows.append(dict(split=split,domain=d,family='global',x=x,y=y,truth=True))
                if j%4==0:rows.append(dict(split=split,domain=d,family='global',x=x,y=bad,truth=False))
    return rows

def check(d,x,y,p):
    if p is not None:
        assert s.verify(d,x,y,p);base=q.nf(d,x);assert all(q.nf(d,z)==base for z,*_ in p)

def record(p,work,start,**kw):return dict(ok=p is not None,work=work,steps=len(p) if p is not None else 0,cpu=time.process_time()-start,**kw)

def search(d,x,y):
    s.moves.cache_clear();s.s.distance.cache_clear();start=time.process_time();seq=itertools.count();ends=(x,y);parents=[{x:()},{y:()}];queues=[[(0,next(seq),x)],[(0,next(seq),y)]]
    initial=s.s.distance(x,y);gap=initial;n=e=t=0;p=None;probe=None
    def snap():return record(p,n+e+t,start,expansions=n,edges=e,gap=gap,initial_gap=initial,frontier=sum(map(len,queues)))
    while n<CAP and any(queues) and p is None:
        side=n%2
        if not queues[side]:side=1-side
        _,_,u=heapq.heappop(queues[side]);n+=1;t+=s.s.size(u)*len(s.DOMAIN[d])
        for edge in s.moves(d,u,True):
            z=edge[0]
            if z in parents[side]:continue
            e+=1;pp=parents[side][u]+(edge,);parents[side][z]=pp;dist=s.s.distance(z,ends[1-side]);gap=min(gap,dist)
            if z in parents[1-side]:p=list(parents[0][z])+s.reverse_path(y,parents[1][z]);break
            heapq.heappush(queues[side],(len(pp)+dist/4,next(seq),z))
        if n==PROBE or (n<PROBE and p is not None):probe=snap()
    if probe is None:probe=snap()
    full=snap();check(d,x,y,p);return probe,full,p

def attempts(r):
    d,x,y=r['domain'],r['x'],r['y'];start=time.process_time();s.moves.cache_clear();s.s.distance.cache_clear();p,n,e,t,v=q.decompose(d,x,y);check(d,x,y,p);dc=record(p,n+e+t+v,start)
    sp,sf,ps=search(d,x,y)
    start=time.process_time();a,b=s.Normalizer(x,d,NPROBE),s.Normalizer(y,d,NPROBE);done=[]
    for z in (a,b):
        try:z.run();done.append(True)
        except s.Limit:done.append(False)
    pn=a.path+s.reverse_path(y,b.path) if all(done) and a.x==b.x else None;check(d,x,y,pn)
    npb=record(pn,len(a.path)+len(b.path)+a.visits+b.visits+a.tests+b.tests,start,done=sum(done),growth=(s.s.size(a.x)+s.s.size(b.x))/max(1,s.s.size(x)+s.s.size(y)),gap=s.s.distance(a.x,b.x))
    start=time.process_time();pf,ns,v,t,status=s.normal(d,x,y,NCAP);check(d,x,y,pf);nf=record(pf,ns+v+t,start,status=status)
    return dict(decompose=dc,search_probe=sp,search=sf,norm_probe=npb,norm=nf),dict(decompose=p,search=ps,norm_probe=pn,norm=pf)

def route(r,mode,b=0):
    if not r['truth']:return dict(ok=False,work=0,steps=0,cost=0,cpu=0,actions=[])
    a=r['attempts'];work=steps=paid=0;cpu=pcpu=0.;actions=[];ok=False
    order=([] if mode=='global' else ['decompose'])+(['search_probe'] if mode in ('single','dual') else [])+(['norm_probe'] if mode=='dual' else [])+(['norm','search'] if b else ['search','norm'])
    for k in order:
        v=a[k];actions.append(k);work+=v['work']-(paid if k=='search' else 0);cpu+=v['cpu']-(pcpu if k=='search' else 0)
        if k=='search_probe':paid=v['work'];pcpu=v['cpu']
        if v['ok']:ok=True;steps=v['steps'];break
    return dict(ok=ok,work=work,steps=steps,cost=work+W*steps,cpu=cpu,actions=actions)

def progress(r):
    p=r['attempts']['search_probe'];return 1-p['gap']/max(1,p['initial_gap'])

def features(r):
    a=r['attempts'];p=a['search_probe'];n=a['norm_probe']
    return q.features(r['domain'],r['x'],r['y'])+[progress(r),p['edges']/128,p['work']/2048,n['done']/2,n['growth'],n['gap']/64,n['work']/2048]

def main(out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.process_time();rows=panel();paths={};out.mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(rows):
        assert (q.nf(r['domain'],r['x'])==q.nf(r['domain'],r['y']))==r['truth']
        if r['truth']:r['attempts'],paths[i]=attempts(r);r['features']=features(r)
        if (i+1)%64==0:print('attempts',i+1,'/',len(rows),flush=True)
    train=[r for r in rows if r['split']=='train' and r['truth']];held=[r for r in rows if r['split']=='held'];res=[r for r in train if not any(r['attempts'][k]['ok'] for k in ('decompose','search_probe','norm_probe'))]
    def obj(r,mode,b):
        v=route(r,mode,b);return v['cost']+10**9*int(r['truth'] and not v['ok'])
    def dual(r,th,growth):return int(r['truth'] and progress(r)<=th and r['attempts']['norm_probe']['growth']<=growth)
    def single(r,th):return int(r['truth'] and progress(r)<=th)
    th,growth=min(itertools.product((0,.1,.25,.5,.75,1.),(1.,1.25,1.5,2.,100.)),key=lambda z:sum(obj(r,'dual',dual(r,*z)) for r in train))
    st=min((-.1,0,.1,.25,.5,.75,1.),key=lambda z:sum(obj(r,'single',single(r,z)) for r in train))
    policies={k:(m,[b]*len(held)) for k,m,b in [('decompose_first','static',0),('decompose_normal_first','static',1),('global_search_first','global',0),('dual_continue','dual',0),('dual_normal','dual',1)]}
    policies['adaptive_symbolic']=('dual',[dual(r,th,growth) for r in held]);policies['single_symbolic']=('single',[single(r,st) for r in held]);fits=[]
    if res:
        x=torch.tensor([r['features'] for r in res]);cost=np.array([[obj(r,'dual',b) for b in (0,1)] for r in res],float);y=torch.tensor(np.log1p(cost)-np.log1p(cost.min(1,keepdims=True)),dtype=torch.float32)
        for seed in SEEDS:
            torch.manual_seed(seed);m=nn.Sequential(nn.Linear(x.shape[1],32),nn.GELU(),nn.Linear(32,2));opt=torch.optim.AdamW(m.parameters(),lr=.003);t=time.process_time()
            for _ in range(400):
                opt.zero_grad();loss=nn.functional.smooth_l1_loss(m(x),y);loss.backward();opt.step()
            tc=time.process_time()-t;t=time.process_time()
            with torch.no_grad():pred=m(torch.tensor([r['features'] if r['truth'] else [0.]*x.shape[1] for r in held])).argmin(1).tolist()
            fits.append(dict(seed=seed,parameters=sum(p.numel() for p in m.parameters()),train_cpu=tc,inference_cpu=time.process_time()-t));policies[f'learned_{seed}']=('dual',pred)
    outcomes={name:[route(r,mode,b) for r,b in zip(held,bits)] for name,(mode,bits) in policies.items()};options=[('static',0),('static',1),('global',0),('single',0),('single',1),('dual',0),('dual',1)];outcomes['oracle']=[route(r,*min(options,key=lambda p:obj(r,*p))) for r in held];summary=[]
    for name,vs in outcomes.items():
        for fam in ('all',*FAMILIES):
            z=[(r,v) for r,v in zip(held,vs) if fam=='all' or r['family']==fam];summary.append(dict(policy=name,family=fam,n=len(z),positives=sum(r['truth'] for r,v in z),solved=sum(v['ok'] for r,v in z),false_accept=sum(v['ok'] and not r['truth'] for r,v in z),work=sum(v['work'] for r,v in z),steps=sum(v['steps'] for r,v in z),cost=sum(v['cost'] for r,v in z),solver_cpu=sum(v['cpu'] for r,v in z)))
    cert=[dict(row=i,action=k,path=p) for i,r in enumerate(rows) if r['split']=='held' and r['truth'] for k,p in paths[i].items() if p is not None];data=gzip.compress(json.dumps(cert,separators=(',',':')).encode(),mtime=0);(out/'STATE14_CERTIFICATES.json.gz').write_bytes(data)
    receipt=dict(seed=SEED,seeds=SEEDS,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),train_positive=len(train),held_count=len(held),residual_train=len(res),residual_held=sum(r['truth'] and not any(r['attempts'][k]['ok'] for k in ('decompose','search_probe','norm_probe')) for r in held),threshold=[th,growth],single_threshold=st,weight=W,fit=fits,summary=summary,outcomes=outcomes,rows=rows,certificate_sha256=hashlib.sha256(data).hexdigest(),lean_kernel_verified=False,cpu_seconds=time.process_time()-start)
    (out/'STATE14_RECEIPTS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (out/'STATE14_SUMMARY.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    for z in summary:
        if z['family']=='all':print(z,flush=True)
    print('residual',len(res),receipt['residual_held'],'thresholds',th,growth,st,'cpu',receipt['cpu_seconds'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
