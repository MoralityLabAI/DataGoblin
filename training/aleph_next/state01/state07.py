"""STATE-07: costed adaptive witness allocation; CPU-only public diagnostic."""
import itertools, json, math, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

Q=8; N=64; SEEDS=(277,281,283); BUDGETS=(4,8,16)
OLD_SPLIT=2026112508; FRESH_SPLIT=2026112510; TRAIN_SEED=2026112511
PANEL_SEED=2026112512; OPEN_SEED=2026112513
N_PER_CLASS=64; OPEN_N=128; TRAIN_TASKS=640; STEPS=500; BATCH=128
SAMPLE=32; SWITCH=4; SMALL=32
X,Y=np.indices((Q,Q)); X=X.ravel(); Y=Y.ravel()

def library():
    base=np.array([(a*X+b*Y+c)%Q for a,b,c in itertools.product(range(Q),repeat=3)],dtype=np.uint8); lib={}
    def add(rows,k,t,ood=False):
        for f in rows: lib.setdefault(f.tobytes(),(k,t,ood))
    add(base,0,'affine')
    for t,g,o in [('x>=4',X>=4,False),('y>=4',Y>=4,False),('x>y',X>Y,True)]:
        for d in range(1,Q): add((base+d*g).astype(np.uint8)%Q,1,t,o)
    for t,g,o in [('xy',X*Y,False),('x^2',X*X,False),('y^2',Y*Y,True)]:
        for d in range(1,Q): add((base+d*g).astype(np.uint8)%Q,2,t,o)
    lx=np.array([(a*X+c)%Q for a,c in itertools.product(range(Q),repeat=2)],dtype=np.uint8)
    ry=np.array([(b*Y+c)%Q for b,c in itertools.product(range(Q),repeat=2)],dtype=np.uint8)
    for t,fn,o in [('xor',np.bitwise_xor,False),('and',np.bitwise_and,False),('or',np.bitwise_or,True)]:
        add(fn(lx[:,None,:],ry[None,:,:]).reshape(-1,N),3,t,o)
    f=np.stack([np.frombuffer(k,np.uint8) for k in lib]); v=list(lib.values())
    return f,np.array([a[0] for a in v]),np.array([a[2] for a in v])

def split(cl,ood):
    # Reconstruct STATE-06 held IDs, then exclude from both new training and test.
    r=np.random.default_rng(OLD_SPLIT); old=[]
    for k in range(4):
        old.extend(r.permutation(np.flatnonzero((cl==k)&~ood))[:96])
        if k: old.extend(r.permutation(np.flatnonzero((cl==k)&ood))[:96])
    old=np.array(old); available=np.ones(len(cl),bool); available[old]=False
    r=np.random.default_rng(FRESH_SPLIT); train=[]; iid=[]; outer=[]
    for k in range(4):
        a=r.permutation(np.flatnonzero((cl==k)&~ood&available)); iid.extend(a[:N_PER_CLASS]); train.append(a[N_PER_CLASS:])
        if k: outer.extend(r.permutation(np.flatnonzero((cl==k)&ood&available))[:N_PER_CLASS])
    flat=np.concatenate(train); assert not (set(flat)&set(iid+outer+old.tolist()))
    assert not (set(iid+outer)&set(old)); assert len(set(iid+outer))==448
    return train,np.array(iid),np.array(outer)

def open_panel(f):
    known={row.tobytes() for row in f}; seen=set(); r=np.random.default_rng(OPEN_SEED); out=[]
    # Sparse deviations from known functions stress false certainty; plus compositions.
    for i in range(OPEN_N):
        while True:
            a=f[r.integers(len(f))].copy()
            if i<OPEN_N//2:
                x=r.integers(N); a[x]=(int(a[x])+int(r.integers(1,Q)))%Q
            else:
                b=f[r.integers(len(f))]; a=((a.astype(int)+b+((X+Y)%3==0)*int(r.integers(1,Q)))%Q).astype(np.uint8)
            key=a.tobytes()
            if key not in known and key not in seen: seen.add(key);out.append(a);break
    return np.stack(out)

def H(p):
    return -np.sum(p*np.log2(np.maximum(p,1e-300)),axis=-1)

class Space:
    def __init__(self,f,cl):
        self.f=f;self.cl=cl;self.prior=1/np.bincount(cl)[cl];self.all=np.arange(len(f))
        self.first={}
        for kind in ('function','class'):
            self.first[kind]=int(np.argmax(self.entropy(self.all,np.arange(N),kind)))
    def entropy(self,m,cols,kind='function',sampled=False):
        values=self.f[np.ix_(m,cols)].astype(np.int64); k=len(cols)
        w=np.ones(len(m))/len(m) if sampled else self.prior[m]/self.prior[m].sum()
        if kind=='function':
            idx=values+Q*np.arange(k)[None,:]
            p=np.bincount(idx.ravel(),weights=np.repeat(w,k),minlength=k*Q).reshape(k,Q)
            return H(p)
        idx=values+Q*(self.cl[m,None]+4*np.arange(k)[None,:])
        p=np.bincount(idx.ravel(),weights=np.repeat(w,k),minlength=k*4*Q).reshape(k,4,Q)
        return H(p.sum(1))+H(p.sum(2))-H(p.reshape(k,-1))
    def update(self,m,x,y): return m[self.f[m,x]==y]

def encode(obs): return F.one_hot(torch.as_tensor(obs,dtype=torch.long),Q+1).float().reshape(-1,N*(Q+1))

class Policy(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(N*(Q+1),96),nn.LayerNorm(96),nn.GELU(),nn.Linear(96,96),nn.GELU(),nn.Linear(96,N))
    def forward(self,x): return self.net(x)

def training_examples(space,groups):
    r=np.random.default_rng(TRAIN_SEED); obs=[]; targets=[]; work=0
    start=time.process_time()
    for i in range(TRAIN_TASKS):
        truth=space.f[r.choice(groups[i%4])]; m=space.all; z=np.zeros(N,dtype=np.int64)
        # Alternate random and greedy function queries; teacher targets only on train functions.
        for step in range(8):
            cols=np.flatnonzero(z==0)
            if len(m)<=1: break
            score=space.entropy(m,cols);work+=len(m)*len(cols)
            if step>0:
                target=np.zeros(N,dtype=np.float32); p=np.exp((score-score.max())/0.25);target[cols]=p/p.sum()
                obs.append(z.copy());targets.append(target)
            x=cols[score.argmax()] if (i%2==0 or step==0) else r.choice(cols)
            z[x]=int(truth[x])+1;m=space.update(m,x,truth[x]);assert len(m)>0
    return np.stack(obs),np.stack(targets),work,time.process_time()-start

def fit(seed,obs,targets):
    torch.manual_seed(seed); r=np.random.default_rng(seed); m=Policy();opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01)
    x=encode(obs); y=torch.from_numpy(targets);mask=torch.as_tensor(obs!=0)
    start=time.process_time();losses=[]
    for step in range(STEPS):
        i=r.integers(len(obs),size=BATCH);z=m(x[i]).masked_fill(mask[i],-1e9);loss=-(F.log_softmax(z,dim=1)*y[i]).sum(1).mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step in (0,99,499): losses.append(float(loss.detach()))
    return m.eval(),dict(seed=seed,cpu_seconds=time.process_time()-start,losses=losses,params=sum(p.numel() for p in m.parameters()))

@torch.no_grad()
def query(space,m,z,policy,r,net):
    cols=np.flatnonzero(z==0); step=N-len(cols)
    if len(m)<=1: return int(r.choice(cols)),0,0
    if step==0: # common entropy first query, cached once; cost charged to all except random.
        if policy=='random': return int(r.choice(cols)),0,0
        return space.first['function'],0,0
    if policy=='random': return int(r.choice(cols)),0,0
    if policy=='learned' or (policy=='hybrid' and len(m)>SMALL):
        scores=net(encode(z[None,:]))[0].numpy();return int(cols[scores[cols].argmax()]),0,1
    if policy=='sample32' and len(m)>SAMPLE:
        w=space.prior[m]; sm=r.choice(m,SAMPLE,replace=True,p=w/w.sum()); scores=space.entropy(sm,cols,sampled=True);return int(cols[scores.argmax()]),len(sm)*len(cols),0
    kind='class' if policy=='class' or (policy=='switch' and step<SWITCH and len(set(space.cl[m]))>1) else 'function'
    if kind=='class' and len(set(space.cl[m]))==1: kind='function' # avoid wasting post-class-certainty queries
    scores=space.entropy(m,cols,kind);return int(cols[scores.argmax()]),len(m)*len(cols),0

def empty_metrics():
    return dict(n=0,class_correct=0,unique_function=0,unique_correct=0,wrong_singleton=0,rejected=0,unanimous_unqueried=0,false_unanimous_unqueried=0,false_certificate_cases=0,filter_cells=0,acquisition_cells=0,model_calls=0)

def evaluate(space,truths,labels,policy,seed,net=None):
    r=np.random.default_rng(seed);res={str(b):empty_metrics() for b in BUDGETS};start=time.process_time();rows=[]
    for truth,label in zip(truths,labels):
        m=space.all; z=np.zeros(N,dtype=np.int64); filter_work=acq=mc=0; row=[]
        for step in range(1,max(BUDGETS)+1):
            x,c,k=query(space,m,z,policy,r,net); acq+=c;mc+=k;filter_work+=len(m);z[x]=int(truth[x])+1;m=space.update(m,x,truth[x])
            if step not in BUDGETS: continue
            v=res[str(step)];v['n']+=1;v['filter_cells']+=filter_work;v['acquisition_cells']+=acq;v['model_calls']+=mc
            if not len(m): v['rejected']+=1;row.append((0,0,1));continue
            mass=np.bincount(space.cl[m],weights=space.prior[m],minlength=4);v['class_correct']+=int(mass.argmax()==label)
            unique=len(m)==1;match=unique and np.array_equal(space.f[m[0]],truth); v['unique_function']+=int(unique);v['unique_correct']+=int(match);v['wrong_singleton']+=int(unique and not match)
            agree=np.all(space.f[m]==space.f[m[:1]],axis=0)&(z==0); false=agree&(space.f[m[0]]!=truth)
            v['unanimous_unqueried']+=int(agree.sum());v['false_unanimous_unqueried']+=int(false.sum());v['false_certificate_cases']+=int(false.any());row.append((int(match),int(unique and not match),0))
        rows.append(row)
    return dict(cpu_seconds=time.process_time()-start,counts=res,per_operator=rows)

def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    f,cl,ood=library(); assert len(f)==24224;groups,iid,outer=split(cl,ood);space=Space(f,cl)
    o=open_panel(f); obs,t,work,seconds=training_examples(space,groups)
    out=dict(grammar_size=len(f),train_pool=[len(g) for g in groups],first_query=space.first,training=dict(examples=len(obs),teacher_candidate_cells=work,cpu_seconds=seconds,steps=STEPS,batch=BATCH),panels={'iid':len(iid),'template_ood':len(outer),'open':len(o)},policies={},learned_training=[],config=dict(seeds=SEEDS,old_split=OLD_SPLIT,fresh_split=FRESH_SPLIT,train_seed=TRAIN_SEED,panel_seed=PANEL_SEED,open_seed=OPEN_SEED,sample=SAMPLE,switch=SWITCH,small=SMALL,torch=torch.__version__,numpy=np.__version__))
    print('train teacher',out['training'],flush=True)
    datasets={'iid':(f[iid],cl[iid]),'template_ood':(f[outer],cl[outer]),'open':(o,np.full(len(o),-1))}
    def run_policy(name,policy,seed,net=None):
        res={k:evaluate(space,a,b,policy,PANEL_SEED+i,net) for i,(k,(a,b)) in enumerate(datasets.items())};out['policies'][name]=res
        print(name,{k:(v['counts']['8']['unique_correct'],v['counts']['8']['wrong_singleton'],v['counts']['8']['acquisition_cells']) for k,v in res.items()},'cpu',sum(v['cpu_seconds'] for v in res.values()),flush=True)
    for p in ('random','function','class','switch','sample32'): run_policy(p,p,SEEDS[0])
    for seed in SEEDS:
        m,r=fit(seed,obs,t);out['learned_training'].append(r);print('fit',r,flush=True)
        run_policy(f'learned_{seed}','learned',seed,m);run_policy(f'hybrid_{seed}','hybrid',seed,m)
    return out

if __name__=='__main__':
    path=Path(__file__).with_name('STATE07_RECEIPTS.json');out=run();out['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();path.write_text(json.dumps(out,indent=2)+'\n')
