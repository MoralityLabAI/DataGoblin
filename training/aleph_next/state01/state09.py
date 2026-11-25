"""STATE09: finite-domain certified operator transfer. Requires state07/08."""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from state08 import panel,entropy,SEED as OLD
SEED=2026112515

def op(d,p):
    t,a,b,c,*r=d;x,y=divmod(p,8)
    if 7<=t<=9:
        l=(a*x+b)%8;v=(c*y+r[0])%8
        return (l^v,l&v,l|v)[t-7]
    z=a*x+b*y+c
    if t:z+=r[0]*(x>=4,y>=4,x>y,x*y,x*x,y*y,0,0,0,x+y>=8,x==y)[t-1]
    return z%8

def catalog():
    seen={};c3=list(itertools.product(range(8),repeat=3))
    def add(d):seen.setdefault(bytes(op(d,p) for p in range(64)),d)
    for c in c3:add((0,*c))
    for t in range(1,7):
        for k in range(1,8):
            for c in c3:add((t,*c,k))
    for t in range(7,10):
        for c in itertools.product(range(8),repeat=4):add((t,*c))
    n=len(seen)
    for t in (10,11):
        for k in range(1,8):
            for c in c3:add((t,*c,k))
    return seen,n

def encode(d=None,patch=(),raw=None):
    fields=[(2,2)]+[(v,3) for v in raw] if raw is not None else [(int(bool(patch)),2),(d[0],4)]+[(v,3) for v in d[1:]]
    if patch:fields += [(len(patch)-1,1)]+[(v,w) for p,y in patch for v,w in ((p,6),(y,3))]
    return ''.join(format(int(v),f'0{w}b') for v,w in fields)

def decode(s):
    i=0
    def read(w):
        nonlocal i
        v=int(s[i:i+w],2);i+=w;return v
    k=read(2)
    if k==2:d=None;patch={};raw=tuple(read(3) for _ in range(64))
    else:
        assert k in (0,1);t=read(4);assert t<=11
        d=(t,*(read(3) for _ in range(3 if t==0 else 4)));patch={read(6):read(3) for _ in range(read(1)+1)} if k else {};raw=None
    assert i==len(s)
    return d,patch,raw

def execute(d,p):
    desc,patch,raw=d
    if raw is not None:return raw[p]
    if p in patch:return patch[p]
    return op(desc,p)

def compile_all(base,cl,extra,panel_rows):
    cat,n=catalog();alltab=np.concatenate([base,extra]);assert n==len(base) and list(cat)==[v.tobytes() for v in alltab]
    desc=list(cat.values());length=np.array([15 if d[0]==0 else 18 for d in desc]);codes=[];early=[];rows=[];prior=1/np.bincount(cl)[cl]
    for i,(fam,f) in enumerate(panel_rows):
        dist=np.count_nonzero(alltab!=f,axis=1);opts=[]
        for j in np.flatnonzero(dist==0):opts.append((int(length[j]),int(j),()))
        for j in np.flatnonzero((dist[:n]>0)&(dist[:n]<=2)):
            patch=tuple((int(p),int(f[p])) for p in np.flatnonzero(base[j]!=f));opts.append((int(length[j])+1+9*len(patch),int(j),patch))
        if opts:
            bits,j,patch=min(opts);code=encode(desc[j],patch);assert bits==len(code)
        else:code=encode(raw=f)
        actual=bytes(execute(decode(code),p) for p in range(64));assert actual==f.tobytes();codes.append(code)
        r=np.random.default_rng(OLD+1000+i);r.permutation(64);obs=np.full(64,-1);bm=np.zeros(n,dtype=int)
        for _ in range(8):
            cols=np.flatnonzero(obs<0);ids=np.flatnonzero(bm==0)
            if len(ids)<=1:p=int(r.choice(cols))
            else:
                w=prior[ids];w=w/w.sum()
                if len(ids)>32:ids=r.choice(ids,32,replace=True,p=w);w=np.full(32,1/32)
                p=int(cols[entropy(base,ids,cols,w).argmax()])
            obs[p]=f[p];bm+=base[:,p]!=obs[p]
        cols=np.flatnonzero(obs>=0);em=np.count_nonzero(extra[:,cols]!=obs[cols],axis=1);j=int(bm.argmin());k=int(bm[j]);choices=[]
        if k==0:choices.append((17,0,j))
        if em.min()==0:choices.append((19,1,int(em.argmin())))
        if 0<k<=2:choices.append((19+9*k,2,j))
        v=None
        if choices:
            _,tag,j=min(choices);v=(extra if tag==1 else base)[j].copy()
            if tag==2:v[cols]=obs[cols]
            assert np.array_equal(v[cols],f[cols]);
        early.append(v);rows.append(dict(id=i,family=fam,bits=len(code),code=code,raw=decode(code)[2] is not None,truth_sha256=hashlib.sha256(actual).hexdigest(),certificate_sha256=hashlib.sha256(code.encode()+b':'+actual).hexdigest(),early_exact=v is not None and bool(np.array_equal(v,f)),early_observations={int(p):int(obs[p]) for p in cols}))
    return codes,early,rows

def tasks(n):
    r=np.random.default_rng(SEED);out=[]
    for l in (8,32,64):
        for _ in range(144):out.append(('register',l,r.integers(n,size=l),r.integers(2,size=l),int(r.integers(8))))
        for i in range(96):
            a=r.integers(n,size=8);b=a.copy();j=int(r.integers(8))
            if i%2:b[j]=(b[j]+int(r.integers(1,n)))%n
            out.append(('graph',l,a,b,int(r.integers(8))))
    return out

def evaluate(mode,tasks,truths,codes,early,rows,ref=None):
    dec=[decode(c) for c in codes];cache={};reads=steps=bad=abstain=0;out=[];cells={}
    if mode=='guarded':cache={(i,p):v for i,row in enumerate(rows) for p,v in row['early_observations'].items()}
    def f(i,x,y):
        nonlocal reads,steps,bad,abstain
        steps+=1;p=x*8+y;key=(i,p)
        if mode=='oracle':reads+=1;return int(truths[i][p])
        if mode=='raw':return int(truths[i][p])
        if mode=='compact':return execute(dec[i],p)
        if mode=='early':
            if early[i] is None:abstain+=1;return -1
            return int(early[i][p])
        if key not in cache:
            reads+=1;v=int(truths[i][p]);cache[key]=v
            if mode=='guarded' and early[i] is not None:bad+=int(early[i][p]!=v)
        return cache[key]
    t=time.process_time()
    for idx,(fam,l,a,b,c) in enumerate(tasks):
        traces=[]
        if fam=='register':
            for s in range(8):
                x,y=s,c;z=[]
                for i,k in zip(a,b):
                    v=f(int(i),x,y)
                    if v<0:break
                    if k==0:x=v
                    else:y=v
                    z.append(x*8+y)
                traces.append(tuple(z))
        else:
            for rules in (a,b):
                for s in range(8):
                    x=s;z=[]
                    for _ in range(l):
                        x=f(int(rules[x]),x,c)
                        if x<0:break
                        z.append(x)
                    traces.append(tuple(z))
        result=tuple(traces);out.append(result)
        if ref is None:continue
        rr=ref[idx];key=f'{fam}/{l}';v=cells.setdefault(key,dict(tasks=0,full=0,terminal=0,answers=0,answer_total=0,trace_hits=0,trace_total=0,changed=0,changed_correct=0,unchanged=0,unchanged_correct=0));v['tasks']+=1;v['full']+=int(result==rr)
        ok=[len(x)==l and x[-1]==y[-1] for x,y in zip(result,rr)];v['terminal']+=int(all(ok));v['answers']+=sum(ok);v['answer_total']+=len(rr);v['trace_hits']+=sum(sum(x==y for x,y in zip(a,b)) for a,b in zip(result,rr));v['trace_total']+=len(rr)*l
        if fam=='graph':
            for i in range(8):
                k='changed' if rr[i][-1]!=rr[i+8][-1] else 'unchanged';v[k]+=1;v[k+'_correct']+=int(ok[i] and ok[i+8])
    return out,dict(cpu_seconds=time.process_time()-t,reads=reads,steps=steps,cached=len(cache),guard_disagreements=bad,abstentions=abstain,cells=cells)

def main():
    t=time.process_time();base,cl,extra,p=panel();truth=[v for _,v in p];codes,early,rows=compile_all(base,cl,extra,p);compile_time=time.process_time()-t
    ts=tasks(len(p));ref,r=evaluate('oracle',ts,truth,codes,early,rows);result={'oracle':r}
    for mode in ('raw','compact','lazy','early','guarded'):
        ans,r=evaluate(mode,ts,truth,codes,early,rows,ref);result[mode]=r
        if mode!='early':assert ans==ref
    rng=np.random.default_rng(SEED+1)
    for i in rng.choice(len(p),64,replace=False):
        new=truth[i].copy();q=int(rng.integers(64));new[q]=(int(new[q])+1)%8;assert bytes(execute(decode(codes[i]),j) for j in range(64))!=new.tobytes();assert bytes(execute(decode(encode(raw=new)),j) for j in range(64))==new.tobytes()
    out=dict(seed=SEED,panel_hash=hashlib.sha256(b''.join(v.tobytes() for v in truth)).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),operators=len(p),tasks=len(ts),full_verification_reads=len(p)*64,early_witness_reads=len(p)*8,code_bits=sum(map(len,codes)),raw_bits=len(p)*194,compile_cpu_seconds=compile_time,revision_invalidations=64,revision_reverification_reads=4096,records=rows,results=result)
    Path(__file__).with_name('STATE09_RECEIPTS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('records','results')},indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
