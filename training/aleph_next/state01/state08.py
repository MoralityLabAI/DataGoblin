"""STATE08: bounded counterexample-guided repair on the STATE07 grammar."""
import itertools, json, math, time, hashlib
from pathlib import Path
import numpy as np
from state07 import library, split, open_panel, X, Y, Q, N, OLD_SPLIT
SEED=2026112514
BUDGETS=(8,16,32,64)
POLICIES=(('frozen','sample32'),('patch','sample32'),('expand','sample32'),('combined','sample32'),('combined','exact'),('combined','random'),('table','random'))
N_PER=48

def panel():
    base,cl,ood=library(); known={v.tobytes() for v in base}; extra=[]; tags=[]
    affine=np.array([(a*X+b*Y+c)%Q for a,b,c in itertools.product(range(Q),repeat=3)],dtype=np.uint8)
    for name,gate in [('threshold',X+Y>=8),('diagonal',X==Y)]:
        for d in range(1,Q):
            for v in (affine+d*gate).astype(np.uint8)%Q:
                if v.tobytes() not in known:known.add(v.tobytes());extra.append(v);tags.append(name)
    extra=np.stack(extra);tags=np.array(tags);full=np.concatenate([base,extra]);r=np.random.default_rng(SEED)
    _,i,o=split(cl,ood);old=set(np.concatenate([i,o]).tolist());ro=np.random.default_rng(OLD_SPLIT)
    for k in range(4):
        old.update(ro.permutation(np.flatnonzero((cl==k)&~ood))[:96].tolist())
        if k:old.update(ro.permutation(np.flatnonzero((cl==k)&ood))[:96].tolist())
    used={v.tobytes() for v in open_panel(base)};out=[]
    def add(name,v):
        assert v.tobytes() not in used;used.add(v.tobytes());out.append((name,v.copy()))
    for j in r.choice(np.array(sorted(set(range(len(base)))-old)),N_PER,replace=False):add('base',base[j])
    for name in ('threshold','diagonal'):
        for j in r.choice(np.flatnonzero(tags==name),N_PER,replace=False):add(name,extra[j])
    for k in (1,2):
        for _ in range(N_PER):
            while True:
                v=base[r.integers(len(base))].copy();c=r.choice(N,k,replace=False);v[c]=(v[c].astype(int)+r.integers(1,Q,k))%Q
                if v.tobytes() not in known and v.tobytes() not in used:break
            add(f'sparse{k}',v)
    for _ in range(N_PER):
        while True:
            v=((affine[r.integers(len(affine))].astype(int)+r.integers(1,Q)*((X+Y)%3==0))%Q).astype(np.uint8)
            if v.tobytes() not in known and v.tobytes() not in used and np.count_nonzero(full!=v,axis=1).min()>2:break
        add('alien',v)
    return base,cl,extra,out

def entropy(base,ids,cols,w):
    v=base[np.ix_(ids,cols)].astype(np.int64);k=len(cols)
    p=np.bincount((v+Q*np.arange(k)[None,:]).ravel(),weights=np.repeat(w,k),minlength=k*Q).reshape(k,Q)
    return -(p*np.log2(np.maximum(p,1e-300))).sum(1)

def episode(base,cl,extra,truth,mode,acq,seed):
    r=np.random.default_rng(seed);order=r.permutation(N);y=np.full(N,-1,dtype=np.int16)
    bm=np.zeros(len(base),dtype=np.int16);em=np.zeros(len(extra),dtype=np.int16) if mode in ('expand','combined') else None
    prior=1/np.bincount(cl)[cl];reads=queries=checks=calls=ce=acq_cells=filter_cells=proposals=0
    candidate=bits=kind=None;out={};bb=math.ceil(math.log2(len(base)));be=math.ceil(math.log2(len(extra)))
    def observe(x,query):
        nonlocal reads,queries,checks,filter_cells
        assert y[x]<0;y[x]=truth[x];reads+=1;queries+=query;checks+=not query
        if mode!='table':
            bm[:]+=base[:,x]!=y[x];filter_cells+=len(base)
            if em is not None:em[:]+=extra[:,x]!=y[x];filter_cells+=len(extra)
    def propose():
        nonlocal candidate,bits,kind,proposals
        opts=[];b=int(bm.argmin());k=int(bm[b])
        if k==0:opts.append((2+bb,0,b,'base'))
        if em is not None:
            e=int(em.argmin())
            if em[e]==0:opts.append((2+4+be,1,e,'expanded'))
        if mode in ('patch','combined') and 0<k<=2:opts.append((2+bb+2+9*k,2,b,'patch'))
        if not opts:candidate=bits=kind=None;return False
        bits,_,idx,kind=min(opts);candidate=(extra if kind=='expanded' else base)[idx].copy()
        if kind=='patch':candidate[y>=0]=y[y>=0]
        assert np.array_equal(candidate[y>=0],y[y>=0]);proposals+=1;return True
    def snap(b):
        exact=candidate is not None and bool(np.array_equal(candidate,truth));cert=candidate is not None and reads==N
        assert not cert or exact
        out[str(b)]=dict(reads=reads,queries=queries,checks=checks,verify_calls=calls,counterexamples=ce,acquisition_cells=acq_cells,filter_cells=filter_cells,proposals=proposals,proposed=int(candidate is not None),exact=int(exact),certified=int(cert),false_proposal=int(candidate is not None and not exact),kind=kind,bits=bits)
    if mode=='table':
        for x in order:
            observe(x,True)
            if reads==N:candidate=y.astype(np.uint8);bits=2+N*3;kind='table'
            if reads in BUDGETS:snap(reads)
        return out
    for _ in range(8):
        cols=np.flatnonzero(y<0);ids=np.flatnonzero(bm==0)
        if acq=='random' or len(ids)<=1:x=int(r.choice(cols))
        else:
            w=prior[ids];w=w/w.sum()
            if acq=='sample32' and len(ids)>32:ids=r.choice(ids,32,replace=True,p=w);w=np.full(32,1/32)
            acq_cells+=len(ids)*len(cols);x=int(cols[entropy(base,ids,cols,w).argmax()])
        observe(x,True)
    stopped=not propose();snap(8);active=False
    for budget in BUDGETS[1:]:
        while not stopped and reads<budget:
            if not active:calls+=1;active=True
            x=int(next(x for x in order if y[x]<0));pred=int(candidate[x]);observe(x,False)
            if pred!=truth[x]:ce+=1;active=False;stopped=not propose()
        snap(budget)
    return out

def run():
    base,cl,extra,p=panel();out=dict(seed=SEED,base_size=len(base),expanded_size=len(extra),n_per_family=N_PER,budgets=BUDGETS,policies={},panel_hash=hashlib.sha256(b''.join(v.tobytes() for _,v in p)).hexdigest())
    for mode,acq in POLICIES:
        t=time.process_time();rows=[dict(family=f,id=i,results=episode(base,cl,extra,v,mode,acq,SEED+1000+i)) for i,(f,v) in enumerate(p)];seconds=time.process_time()-t
        out['policies'][mode+'_'+acq]=dict(cpu_seconds=seconds,rows=rows)
        print(mode,acq,round(seconds,3),{b:(sum(r['results'][str(b)]['exact'] for r in rows),sum(r['results'][str(b)]['certified'] for r in rows)) for b in BUDGETS},flush=True)
    out['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return out

if __name__=='__main__':
    Path(__file__).with_name('STATE08_RECEIPTS.json').write_text(json.dumps(run(),indent=2)+'\n')
