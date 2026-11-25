"""STATE12: proof-producing normalization and two-frontier rewrite search."""
import argparse,csv,hashlib,heapq,itertools,json,random,time
from collections import Counter
from functools import lru_cache
from pathlib import Path
import state11 as s
from state11_normal_forms import polynomial,list_word
SEED=2026112518
CAP=4096
RULES=tuple(s.R)+(
 ('List.reverse_nil','rev',('rev','nil'),'nil'),
 ('List.map_nil','map',('mf','nil'),'nil'),
 ('List.map_nil','map',('mg','nil'),'nil'),
 ('List.map_reverse','mixed',('rev',('mf',s.a)),('mf',('rev',s.a))),
 ('List.map_reverse','mixed',('rev',('mg',s.a)),('mg',('rev',s.a))),)
DOMAIN={'nat':tuple(range(12)),'list':tuple(range(12,len(RULES)))}
def apply(x,i,p,rev=False):
 a,b=RULES[i][2:]
 if rev:a,b=b,a
 bindings={}
 if not s.match(a,s.get(x,p),bindings):raise ValueError('match failed')
 z=s.subst(b,bindings)
 if any(isinstance(v,str) and v.startswith('?') for _,v in s.positions(z)):raise ValueError('unbound')
 return s.replace(x,p,z)
def verify(d,x,y,path):
 for z,i,p,r in path:
  if i not in DOMAIN[d] or (apply(z,i,p)!=x if r else apply(x,i,p)!=z):return False
  x=z
 return x==y
def reverse_path(x,path):
 before=[x]+[v[0] for v in path]
 return [(before[j],i,p,not r) for j,(_,i,p,r) in reversed(list(enumerate(path)))]
class Limit(Exception):pass
class Normalizer:
 def __init__(self,x,d,cap):self.x=x;self.d=d;self.cap=cap;self.path=[];self.visits=0;self.tests=0
 def step(self,i,p,r=False):
  if len(self.path)>=self.cap:raise Limit('step cap')
  z=apply(self.x,i,p,r)
  if s.size(z)>CAP:raise Limit('term cap')
  assert z!=self.x;self.path.append((z,i,p,r));self.x=z
 def sweep(self,ids):
  while True:
   hit=None
   for p,v in s.positions(self.x):
    self.visits+=1
    for i in ids:
     self.tests+=1
     if s.match(RULES[i][2],v,{}):hit=(i,p);break
    if hit is not None:break
   if hit is None:return
   self.step(*hit)
 def sort(self,op,assoc,comm):
  def visit(p):
   self.visits+=1;x=s.get(self.x,p)
   if isinstance(x,str):return
   if x[0]!=op:
    for j in range(1,len(x)):visit(p+(j,))
    return
   q=p
   while not isinstance(v:=s.get(self.x,q),str) and v[0]==op:visit(q+(1,));q+=(2,)
   visit(q)
   while True:
    q=p;changed=False
    while not isinstance(v:=s.get(self.x,q),str) and v[0]==op:
     self.visits+=1;l,r=v[1:];nested=not isinstance(r,str) and r[0]==op;nxt=r[1] if nested else r
     if repr(l)>repr(nxt):
      if nested:self.step(assoc,q,True);self.step(comm,q+(1,));self.step(assoc,q)
      else:self.step(comm,q)
      changed=True;break
     q+=(2,)
    if not changed:break
  visit(())
 def run(self):
  if self.d=='nat':self.sweep((8,9,4,5,6,7,10,11,0,2));self.sort('*',2,3);self.sort('+',0,1)
  else:self.sweep((13,14,19,20,21,16,15,22,23,17,18,12))
  return self.x
def normal(d,x,y,cap=CAP):
 a,b=Normalizer(x,d,cap),Normalizer(y,d,cap)
 try:
  ca,cb=a.run(),b.run();path=a.path+reverse_path(y,b.path) if ca==cb else None;status='equal' if path is not None else 'different'
 except Limit as e:path,status=None,str(e)
 return path,len(a.path)+len(b.path),a.visits+b.visits,a.tests+b.tests,status
@lru_cache(maxsize=25000)
def moves(d,x,expanded):
 out=[];seen=set()
 for p,v in s.positions(x):
  for i in DOMAIN[d]:
   if not expanded and i>=len(s.R):continue
   b={}
   if s.match(RULES[i][2],v,b):
    z=s.replace(x,p,s.subst(RULES[i][3],b))
    if z!=x and z not in seen and s.size(z)<=55:seen.add(z);out.append((z,i,p,False))
 return tuple(out)
def search(d,x,y,mode,budget=64):
 if x==y:return [],0,0
 dual=mode=='two_frontier';seq=itertools.count();ends=(x,y);parents=[{x:()},{y:()}];queues=[[(0,next(seq),x)],[(0,next(seq),y)]];work=edges=0
 while work<budget and any(queues[:2 if dual else 1]):
  side=work%2 if dual else 0
  if not queues[side]:side=1-side
  _,_,u=heapq.heappop(queues[side]);work+=1
  for edge in moves(d,u,mode!='original'):
   z=edge[0]
   if z in parents[side]:continue
   edges+=1;path=parents[side][u]+(edge,);parents[side][z]=path
   if z in parents[1-side]:return list(parents[0][z])+reverse_path(y,parents[1][z]),work,edges
   heapq.heappush(queues[side],(len(path)+s.distance(z,ends[1-side])/4,next(seq),z))
 return None,work,edges
def panel():
 seen=set();r=random.Random(s.SEED)
 for d,g in [('nat',{'struct','id'}),('nat',{'dist','id'}),('list',{'append','rev'}),('list',{'append','map'})]:s.generate(r,d,g,120,False,seen)
 for d,g in [('nat',{'struct','id','dist'}),('list',{'append','rev','map'})]:s.generate(r,d,g,32,True,seen)
 old=set(seen);old|={(d,y,x) for d,x,y in old};r=random.Random(SEED);rows=[]
 for d,g in [('nat',{'struct','id','dist'}),('list',{'append','rev','map'})]:
  count=0
  for _,x,y,_ in s.generate(r,d,g,64,True,seen):
   bad=('+',y,'1') if d=='nat' else ('++',y,'xs')
   if any(v in old for v in ((d,x,y),(d,y,x),(d,x,bad))):continue
   rows.extend(((d,x,y,True,'forward'),(d,y,x,True,'reverse'),(d,x,bad,False,'negative')));old.update(((d,x,y),(d,y,x),(d,x,bad)));count+=1
   if count==48:break
  assert count==48
 assert len({r[:3] for r in rows})==len(rows)
 return rows
def declaration(name,d,x,y,path):
 ty='Nat' if d=='nat' else 'List Nat';args='(x y z : Nat)' if d=='nat' else '(xs ys zs : List Nat) (f g : Nat → Nat)';lines=[f'theorem {name} {args} : {s.lean(x)} = {s.lean(y)} := by']
 if not path:return '\n'.join(lines+['  rfl'])+'\n'
 lines.append('  calc')
 for z,i,p,r in path:
  u,v=s.get(x,p),s.get(z,p);a,b=(v,u) if r else (u,v);t=f'(show {s.lean(a)} = {s.lean(b)} from by simp only [{RULES[i][0]}])'
  if r:t=f'({t}).symm'
  if p:t=f'congrArg (fun hole : {ty} => {s.lean(s.replace(x,p,"hole"))}) ({t})'
  lines.append(f'    {s.lean(x)} = {s.lean(z)} := {t}');x=z
 return '\n'.join(lines)+'\n'
def run(out):
 out.mkdir(exist_ok=True,parents=True);t=time.process_time();rows=panel();records=[];certs=[];decl=[];timings={}
 for d,x,y,truth,_ in rows:
  nf=polynomial if d=='nat' else list_word;assert (nf(x)==nf(y))==truth
 for mode in ('original','enhanced','two_frontier','normal','hybrid'):
  moves.cache_clear();s.distance.cache_clear();start=time.process_time()
  for j,(d,x,y,truth,kind) in enumerate(rows):
   n=edges=steps=visits=tests=0;status='search'
   if mode in ('normal','hybrid'):
    path,steps,visits,tests,status=normal(d,x,y,16 if mode=='hybrid' else CAP)
    if mode=='hybrid' and path is None and status!='different':path,n,edges=search(d,x,y,'two_frontier');status='fallback:'+status
   else:path,n,edges=search(d,x,y,mode)
   ok=path is not None
   if ok:
    assert truth and verify(d,x,y,path),(mode,j)
    nf=polynomial if d=='nat' else list_word;assert all(nf(z)==nf(x) for z,*_ in path)
    if mode=='normal' or (mode=='hybrid' and status.startswith('fallback')):
     name=f'{mode}_{j}';decl.append(declaration(name,d,x,y,path));certs.append(dict(name=name,domain=d,x=x,y=y,path=path))
   records.append(dict(mode=mode,id=j,domain=d,kind=kind,truth=truth,solved=ok,expansions=n,edges=edges,normal_steps=steps,node_visits=visits,match_tests=tests,proof_steps=len(path) if ok else None,status=status))
  timings[mode]=time.process_time()-start;q=[r for r in records if r['mode']==mode];print(mode,'solved',sum(r['solved'] for r in q),'exp',sum(r['expansions'] for r in q),'cpu',round(timings[mode],3),flush=True)
 summary=[]
 for mode in timings:
  for dom in ('nat','list','all'):
   for kind in ('forward','reverse','negative','all'):
    q=[r for r in records if r['mode']==mode and (dom=='all' or r['domain']==dom) and (kind=='all' or r['kind']==kind)]
    summary.append(dict(mode=mode,domain=dom,kind=kind,n=len(q),positive=sum(r['truth'] for r in q),solved=sum(r['solved'] for r in q),false_accept=sum(r['solved'] and not r['truth'] for r in q),**{k:sum(r[k] or 0 for r in q) for k in ('expansions','edges','normal_steps','node_visits','match_tests','proof_steps')}))
 with (out/'STATE12_SUMMARY.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (out/'STATE12_POSITIVE.lean').write_text('set_option autoImplicit false\n\n'+'\n'.join(decl)+'\n'+'\n'.join('#print axioms '+v['name'] for v in certs)+'\n')
 result=dict(seed=SEED,panel_sha256=hashlib.sha256(repr(rows).encode()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),counts=dict(Counter(r[0]+'_'+r[4] for r in rows)),records=records,summary=summary,timings=timings,cpu_seconds=time.process_time()-t,lean_kernel_verified=False,lean_declarations=len(certs),certificates=certs)
 (out/'STATE12_RECEIPTS.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
