"""STATE11: synthetic Lean-core rewrite search, with optional kernel replay."""
import argparse,csv,hashlib,heapq,itertools,json,random,re,shutil,subprocess,time
from collections import Counter
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
SEED=2026112517; SEEDS=(293,307,311); BUDGETS=(4,16,64)
a,b,c='?a','?b','?c'
R=[('Nat.add_assoc','struct',('+',('+',a,b),c),('+',a,('+',b,c))),('Nat.add_comm','struct',('+',a,b),('+',b,a)),('Nat.mul_assoc','struct',('*',('*',a,b),c),('*',a,('*',b,c))),('Nat.mul_comm','struct',('*',a,b),('*',b,a)),('Nat.add_zero','id',('+',a,'0'),a),('Nat.zero_add','id',('+','0',a),a),('Nat.mul_one','id',('*',a,'1'),a),('Nat.one_mul','id',('*','1',a),a),('Nat.mul_zero','id',('*',a,'0'),'0'),('Nat.zero_mul','id',('*','0',a),'0'),('Nat.mul_add','dist',('*',a,('+',b,c)),('+',('*',a,b),('*',a,c))),('Nat.add_mul','dist',('*',('+',a,b),c),('+',('*',a,c),('*',b,c))),('List.append_assoc','append',('++',('++',a,b),c),('++',a,('++',b,c))),('List.nil_append','append',('++','nil',a),a),('List.append_nil','append',('++',a,'nil'),a),('List.reverse_append','rev',('rev',('++',a,b)),('++',('rev',b),('rev',a))),('List.reverse_reverse','rev',('rev',('rev',a)),a),('List.map_append','map',('mf',('++',a,b)),('++',('mf',a),('mf',b))),('List.map_append','map',('mg',('++',a,b)),('++',('mg',a),('mg',b)))]
NAT=tuple(range(12)); LIST=tuple(range(12,len(R))); TOKENS=('+','*','++','rev','mf','mg','x','y','z','0','1','xs','ys','zs','nil')
def positions(x,p=()):
    yield p,x
    if not isinstance(x,str):
        for i,v in enumerate(x[1:],1):yield from positions(v,p+(i,))
@lru_cache(maxsize=100000)
def size(x):return sum(1 for _ in positions(x))
def replace(x,p,y):
    if not p:return y
    return tuple(replace(v,p[1:],y) if i==p[0] else v for i,v in enumerate(x))
def get(x,p):
    for i in p:x=x[i]
    return x
def match(p,x,b):
    if isinstance(p,str):
        if p.startswith('?'):
            if p in b:return b[p]==x
            b[p]=x;return True
        return p==x
    return isinstance(x,tuple) and len(p)==len(x) and p[0]==x[0] and all(match(v,w,b) for v,w in zip(p[1:],x[1:]))
def subst(p,b):return b.get(p,p) if isinstance(p,str) else (p[0],*(subst(x,b) for x in p[1:]))
@lru_cache(maxsize=100000)
def moves(x,d):
    out=[];seen=set()
    for p,v in positions(x):
        for i in (NAT if d=='nat' else LIST):
            bb={};lhs,rhs=R[i][2:]
            if match(lhs,v,bb):
                z=replace(x,p,subst(rhs,bb))
                if z!=x and z not in seen and size(z)<=55:seen.add(z);out.append((z,i,p))
    return tuple(out)
def expr(r,d,depth,g):
    atoms=('x','y','z','0','1') if d=='nat' else ('xs','ys','zs','nil')
    if depth<=0 or r.random()<.20:return r.choice(atoms)
    ops=['+','*'] if d=='nat' else ['++','++']+(['rev'] if 'rev' in g else [])+(['mf','mg'] if 'map' in g else [])
    op=r.choice(ops);return (op,*(expr(r,d,depth-1,g) for _ in range(2 if op in ('+','*','++') else 1)))
def generate(r,d,g,n,mixed,seen):
    rows=[];tries=0;req=({'struct','dist'} if d=='nat' else {'rev','map'}) if mixed else set()
    while len(rows)<n:
        tries+=1;assert tries<200000
        x=expr(r,d,r.choice((3,4)),g);cur=x;path=[];visited={x}
        for _ in range(r.randint(3,5)):
            opts=[v for v in moves(cur,d) if R[v[1]][1] in g and v[0] not in visited]
            if not opts:break
            z,i,p=r.choice(opts);path.append((z,i,p));visited.add(z);cur=z
        if len(path)<3 or not req<=set(R[i][1] for _,i,_ in path) or (d,x,cur) in seen:continue
        seen.add((d,x,cur));rows.append((d,x,cur,path))
    return rows
@lru_cache(maxsize=100000)
def bag(x):return Counter(v if isinstance(v,str) else v[0] for _,v in positions(x))
@lru_cache(maxsize=200000)
def distance(x,y):
    if x==y:return 0
    if isinstance(x,str) or isinstance(y,str) or x[0]!=y[0]:return max(size(x),size(y))
    return sum(distance(v,w) for v,w in zip(x[1:],y[1:]))
def features(x,y,z,i,p):
    bx,by,bz=bag(x),bag(y),bag(z);dx,dz=distance(x,y),distance(z,y)
    return [size(x)/32,size(y)/32,size(z)/32,dx/32,dz/32,(dx-dz)/32,len(p)/5,float(z==y)]+[float(j==i) for j in range(len(R))]+[(bx[k]-by[k])/16 for k in TOKENS]+[(bz[k]-by[k])/16 for k in TOKENS]
def fit(seed,x,y):
    torch.manual_seed(seed);m=nn.Sequential(nn.Linear(x.shape[1],48),nn.GELU(),nn.Linear(48,1));opt=torch.optim.AdamW(m.parameters(),lr=.003);g=torch.Generator().manual_seed(seed+1);start=time.process_time()
    for _ in range(240):
        ids=torch.randint(len(x),(128,),generator=g);z=m(x[ids]).squeeze(-1);loss=F.binary_cross_entropy_with_logits(z,y[ids],pos_weight=torch.tensor(4.));opt.zero_grad();loss.backward();opt.step()
    return m.eval(),dict(seed=seed,params=sum(p.numel() for p in m.parameters()),seconds=time.process_time()-start)
def check(d,x,y,path):
    for z,i,p in path:
        bb={};lhs,rhs=R[i][2:]
        if i not in (NAT if d=='nat' else LIST) or not match(lhs,get(x,p),bb) or replace(x,p,subst(rhs,bb))!=z:return False
        x=z
    return x==y
@torch.no_grad()
def search(d,x,y,arm,m=None):
    seq=itertools.count();todo=[(0,next(seq),x,())];visited={x};n=edges=0
    while todo and n<BUDGETS[-1]:
        _,_,cur,path=heapq.heappop(todo);n+=1;opts=[v for v in moves(cur,d) if v[0] not in visited];edges+=len(opts)
        scores=m(torch.tensor([features(cur,y,*v) for v in opts])).flatten().tolist() if m is not None and opts else [0.]*len(opts)
        for (z,i,p),s in zip(opts,scores):
            visited.add(z);pp=path+((z,i,p),)
            if z==y:return pp,n,edges
            score=len(pp) if arm=='bfs' else len(pp)+distance(z,y)/4-.8*s
            heapq.heappush(todo,(score,next(seq),z,pp))
    return None,n,edges
def lean(x):
    if isinstance(x,str):return '([] : List Nat)' if x=='nil' else x
    op,*xs=x
    if op in ('+','*','++'):return '('+lean(xs[0])+' '+op+' '+lean(xs[1])+')'
    return '('+{'rev':'List.reverse','mf':'List.map f','mg':'List.map g'}[op]+' '+lean(xs[0])+')'
def declaration(name,row,path):
    d,x,y,_=row;ty='Nat' if d=='nat' else 'List Nat';args='(x y z : Nat)' if d=='nat' else '(xs ys zs : List Nat) (f g : Nat → Nat)'
    lines=[f'theorem {name} {args} : {lean(x)} = {lean(y)} := by','  calc']
    for z,i,p in path:
        u,v=get(x,p),get(z,p);term=f'(show {lean(u)} = {lean(v)} from by simp only [{R[i][0]}])'
        if p:term=f'congrArg (fun hole : {ty} => {lean(replace(x,p,"hole"))}) {term}'
        lines.append(f'    {lean(x)} = {lean(z)} := {term}');x=z
    return '\n'.join(lines)+'\n'
def run(out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);r=random.Random(SEED);seen=set();train=[];held=[];start=time.process_time()
    for d,g in [('nat',{'struct','id'}),('nat',{'dist','id'}),('list',{'append','rev'}),('list',{'append','map'})]:train+=generate(r,d,g,120,False,seen)
    for d,g in [('nat',{'struct','id','dist'}),('list',{'append','rev','map'})]:held+=generate(r,d,g,32,True,seen)
    xx=[];yy=[]
    for d,x,y,path in train:
        for z,i,p in path:
            for q,j,v in moves(x,d):xx.append(features(x,y,q,j,v));yy.append(float(q==z))
            x=z
    X,Y=torch.tensor(xx),torch.tensor(yy);models=[];fits=[]
    for seed in SEEDS:m,f=fit(seed,X,Y);models.append(m);fits.append(f)
    records=[];decl=[];names=[];timing={}
    for k,row in enumerate(held):
        assert check(*row);name=f'gold_{k}';decl.append(declaration(name,row,row[3]));names.append(name)
    for arm,m in [('bfs',None),('symbolic',None),*((f'learned_{s}',m) for s,m in zip(SEEDS,models))]:
        ts=time.process_time()
        for k,row in enumerate(held):
            d,x,y,_=row;path,n,e=search(d,x,y,arm,m);records.append(dict(arm=arm,id=k,domain=d,found=path is not None,expansions=n,edges=e))
            if path is not None:
                assert check(d,x,y,path);name=f'found_{arm}_{k}';decl.append(declaration(name,row,path));names.append(name)
        timing[arm]=time.process_time()-ts;print(arm,sum(z['found'] for z in records if z['arm']==arm),round(timing[arm],3),flush=True)
    summary=[]
    for arm in timing:
        for d in ('nat','list','all'):
            q=[z for z in records if z['arm']==arm and (d=='all' or z['domain']==d)]
            for b in BUDGETS:summary.append(dict(arm=arm,domain=d,budget=b,goals=len(q),solved=sum(z['found'] and z['expansions']<=b for z in q),expansions_at_64=sum(z['expansions'] for z in q),edges_at_64=sum(z['edges'] for z in q)))
    out.mkdir(exist_ok=True,parents=True)
    with (out/'STATE11_SUMMARY.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    source='set_option autoImplicit false\n\n'+'\n'.join(decl)+'\n'+'\n'.join('#print axioms '+n for n in names)+'\n';(out/'STATE11_POSITIVE.lean').write_text(source)
    neg=[]
    for k,row in enumerate(held):
        d,x,y,_=row;args='(x y z : Nat)' if d=='nat' else '(xs ys zs : List Nat) (f g : Nat → Nat)';vals='x y z' if d=='nat' else 'xs ys zs f g';bad=f'({lean(x)} + 1)' if d=='nat' else f'({lean(x)} ++ [0])'
        neg.append(dict(id=k,source=declaration(f'gold_{k}',row,row[3]),invalid=f'theorem invalid_{k} {args} : {lean(x)} = {bad} := gold_{k} {vals}\n'))
    (out/'STATE11_NEGATIVE.json').write_text(json.dumps(neg,indent=2))
    receipt=dict(seed=SEED,train_tasks=len(train),train_edges=len(X),positive_train_edges=int(Y.sum()),held_tasks=len(held),held_sha256=hashlib.sha256(repr(held).encode()).hexdigest(),fits=fits,timings=timing,summary=summary,rows=records,lean_declarations=len(names),lean_source_sha256=hashlib.sha256(source.encode()).hexdigest(),lean_kernel_verified=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cpu_seconds=time.process_time()-start)
    (out/'STATE11_RECEIPTS.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
def replay(out):
    exe=shutil.which('lean');assert exe,'Lean unavailable';start=time.perf_counter()
    def check_file(path):return subprocess.run([exe,'-j','1',str(path)],capture_output=True,text=True,timeout=240)
    p=check_file(out/'STATE11_POSITIVE.lean');log=p.stdout+p.stderr;(out/'STATE11_LEAN.log').write_text(log)
    receipt=json.loads((out/'STATE11_RECEIPTS.json').read_text());expected=receipt['lean_declarations'];matches=re.findall(r"^'([^']+)' (?:does not depend on any axioms|depends on axioms: \[(.*)\])$",log,re.M)
    assert p.returncode==0 and len(matches)==expected and len({n for n,_ in matches})==expected,(p.returncode,len(matches),expected)
    assert all(set(x.strip() for x in a.split(',') if x.strip())<={'propext','Quot.sound','Classical.choice'} for _,a in matches)
    negatives=0
    for r in json.loads((out/'STATE11_NEGATIVE.json').read_text()):
        path=out/'invalid_transport.lean';path.write_text('set_option autoImplicit false\n'+r['source']+r['invalid']);p=check_file(path)
        assert p.returncode!=0 and 'type mismatch' in (p.stdout+p.stderr).lower(),(r['id'],p.stdout,p.stderr);negatives+=1
    (out/'STATE11_LEAN_REPLAY.json').write_text(json.dumps(dict(version=subprocess.check_output([exe,'--version'],text=True).strip(),kernel_verified=expected,negative_rejections=negatives,source_sha256=receipt['lean_source_sha256'],log_sha256=hashlib.sha256(log.encode()).hexdigest(),wall_seconds=time.perf_counter()-start),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--lean',action='store_true');p.add_argument('--replay-only',action='store_true');a=p.parse_args()
    if not a.replay_only:run(a.out)
    if a.lean or a.replay_only:replay(a.out)
