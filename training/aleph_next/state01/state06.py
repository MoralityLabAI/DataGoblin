"""CPU hypothesis-class identification; semantic-disjoint train and test."""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
Q=8; N=64; SEEDS=(263,269,271); BUDGETS=(4,8,16); STEPS=800; BATCH=128; WIDTH=96; PANEL_N=96
CLASSES=('affine','piecewise','quadratic','bitwise'); SPLIT_SEED=2026112508; PANEL_SEED=2026112509
X,Y=np.indices((Q,Q));X=X.ravel();Y=Y.ravel()
def library():
    base=np.array([(a*X+b*Y+c)%Q for a,b,c in itertools.product(range(Q),repeat=3)],dtype=np.uint8); lib={}
    def add(rows,k,t,ood=False):
        for f in rows:lib.setdefault(f.tobytes(),(k,t,ood))
    add(base,0,'affine')
    for t,g,o in [('x>=4',X>=4,False),('y>=4',Y>=4,False),('x>y',X>Y,True)]:
        for d in range(1,Q):add((base+d*g).astype(np.uint8)%Q,1,t,o)
    for t,g,o in [('xy',X*Y,False),('x^2',X*X,False),('y^2',Y*Y,True)]:
        for d in range(1,Q):add((base+d*g).astype(np.uint8)%Q,2,t,o)
    lx=np.array([(a*X+c)%Q for a,c in itertools.product(range(Q),repeat=2)],dtype=np.uint8);ry=np.array([(b*Y+c)%Q for b,c in itertools.product(range(Q),repeat=2)],dtype=np.uint8)
    for t,fn,o in [('xor',np.bitwise_xor,False),('and',np.bitwise_and,False),('or',np.bitwise_or,True)]:add(fn(lx[:,None,:],ry[None,:,:]).reshape(-1,N),3,t,o)
    vals=list(lib.values());f=np.stack([np.frombuffer(k,np.uint8) for k in lib]);cl=np.array([v[0] for v in vals]);t=np.array([v[1] for v in vals]);ood=np.array([v[2] for v in vals]);return f,cl,t,ood

def splits(cl,ood):
    r=np.random.default_rng(SPLIT_SEED);train=[];iid=[];outer=[]
    for k in range(4):
        a=r.permutation(np.flatnonzero((cl==k)&~ood));assert len(a)>PANEL_N;iid.append(a[:PANEL_N]);train.append(a[PANEL_N:])
        if k:
            a=r.permutation(np.flatnonzero((cl==k)&ood));assert len(a)>=PANEL_N;outer.append(a[:PANEL_N])
    assert not set(np.concatenate(train))&set(np.concatenate(iid+outer));return train,iid,outer
def observations(r,n,b):return np.argsort(r.random((n,N)),axis=1)[:,:b]
def encode(f,idx):
    a=np.zeros((len(f),N),dtype=np.int64);a[np.arange(len(f))[:,None],idx]=f[np.arange(len(f))[:,None],idx]+1;return F.one_hot(torch.from_numpy(a),Q+1).float().flatten(1)
def panels(f,cl,iid,outer):
    r=np.random.default_rng(PANEL_SEED);out={}
    for name,groups in [('iid',iid),('template_ood',outer)]:
        ids=np.concatenate(groups);ix=observations(r,len(ids),max(BUDGETS));p=r.permutation(len(ids));ids=ids[p];ix=ix[p]
        for b in BUDGETS:out[f'{name}/{b}']=(ids,ix[:,:b],encode(f[ids],ix[:,:b]),cl[ids])
    return out
def stats(p,y):
    p=np.asarray(p);ok=p.argmax(1)==y;return dict(n=len(y),correct=int(ok.sum()),by_class={CLASSES[k]:dict(n=int((y==k).sum()),correct=int((ok&(y==k)).sum())) for k in range(4)},nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()))
def symbolic(f,cl,panel):
    prior=1/np.bincount(cl,minlength=4)[cl];out={}
    for key,(ids,ix,_,y) in panel.items():
        probs=[];sizes=[];unique=unique_class=certified=0;affine_compatible=[]
        for i,cols in zip(ids,ix):
            m=np.arange(len(f))
            for c in cols:m=m[f[m,c]==f[i,c]]
            assert i in m
            mass=np.bincount(cl[m],weights=prior[m],minlength=4);probs.append(mass/mass.sum());sizes.append(len(m));unique+=len(m)==1;unique_class+=np.count_nonzero(mass)==1;certified+=int(np.all(f[m]==f[m[:1]],axis=0).sum());affine_compatible.append(np.any(cl[m]==0))
        r=stats(probs,y);r.update(unique_function=unique,unique_class=unique_class,mean_candidates=float(np.mean(sizes)),certified_coordinates=certified,coordinate_total=len(ids)*N,affine_compatible=int(sum(affine_compatible)),affine_false_compatible=int((np.array(affine_compatible)&(y!=0)).sum()));out[key]=r
    return out

class Selector(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(N*(Q+1),WIDTH),nn.LayerNorm(WIDTH),nn.GELU(),nn.Linear(WIDTH,WIDTH),nn.GELU(),nn.Linear(WIDTH,4))
    def forward(self,x):return self.net(x)
def train(seed,f,cl,groups,panel):
    torch.manual_seed(seed);r=np.random.default_rng(seed+10000);m=Selector();opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01);start=time.process_time()
    for step in range(STEPS):
        b=BUDGETS[step%3];ids=np.concatenate([r.choice(g,BATCH//4) for g in groups]);ix=observations(r,len(ids),b);x=encode(f[ids],ix);y=torch.from_numpy(cl[ids]);loss=F.cross_entropy(m(x),y);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    cpu=time.process_time()-start;m.eval();res={}
    with torch.no_grad():
        for k,(_,_,x,y) in panel.items():res[k]=stats(m(x).softmax(1).numpy(),y)
    return dict(seed=seed,params=sum(p.numel() for p in m.parameters()),cpu_seconds=cpu,train_episodes=STEPS*BATCH,train_witnesses=sum(BUDGETS[i%3]*BATCH for i in range(STEPS)),held=res)

def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);f,cl,t,ood=library();g,i,o=splits(cl,ood);p=panels(f,cl,i,o)
    print('Library',len(f),[(CLASSES[k],int((cl==k).sum()),int(((cl==k)&ood).sum()),len(g[k])) for k in range(4)],flush=True)
    start=time.process_time();sym=symbolic(f,cl,p);cpu=time.process_time()-start;print('symbolic',cpu,flush=True);runs=[]
    for seed in SEEDS:
        a=train(seed,f,cl,g,p);runs.append(a);print(seed,a['cpu_seconds'],{k:v['correct'] for k,v in a['held'].items()},flush=True)
    return dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__,numpy=np.__version__,seeds=SEEDS,steps=STEPS,batch=BATCH,budgets=BUDGETS,split_seed=SPLIT_SEED,panel_seed=PANEL_SEED,panel_n_per_class=PANEL_N,grammar_size=len(f),grammar_counts={CLASSES[k]:dict(total=int((cl==k).sum()),ood=int(((cl==k)&ood).sum()),train=len(g[k]),iid=len(i[k]),held_template=len(o[k-1]) if k else 0) for k in range(4)},templates={s:int((t==s).sum()) for s in sorted(set(t))},symbolic_cpu_seconds=cpu,symbolic=sym,runs=runs)
if __name__=='__main__':Path(__file__).with_name('STATE06_RECEIPTS.json').write_text(json.dumps(run(),indent=2,default=lambda x:x.item())+'\n')
