"""Conditional regret factors inside a bounded last-use proof tree."""
from collections import OrderedDict
from heapq import heappush, heappop
from math import prod
from itertools import product
from time import process_time_ns

from sg28_integer import domains, split, conditioned, infeasible, check_witness, fixed_schedule
from sg28_lp import Model
from sg29_factor import column, partition, estimate, choose, combine, encode, relaxed
from datagoblin.training_gym.spectral_quotient import score, validate, indices
from datagoblin.training_gym.spectral_routing import TaskError

WORK=100_000
CACHE=64


def best(z,p,lo,hi,i=None,on=False,ref=False):
    """Conditional mask maximum, including impossible read conditions."""
    vals=[0] if lo==0 and not on else []
    prefix=0
    for l in range(hi):
        if l+1>=lo and (l!=i or on) and (not on or l>=i):
            a=sum((z[h] if on else 0) if h==i else max(0,z[h]) for h in range(l)) if ref else prefix
            vals.append(a+z[l]-p[l+1])
        prefix+=(z[l] if on else 0) if l==i else max(0,z[l])
    return max(vals) if vals else None


def screen(q,box,e,d,L,ref=False):
    # Conditioned verifies the multiplier representation.
    conditioned(q,box,e,d)
    cols=[column(q,j,e,d) for j in range(q['d'])]
    M=[best(z,p,*b,ref=ref) for (z,p),b in zip(cols,box,strict=True)]
    B=sum(x*k for x,k in zip(e,q['k'],strict=True))+sum(M); H=B-d*(L+1)
    must=[];may=[]
    if H>=0:
        for (z,p),b,m in zip(cols,box,M,strict=True):
            a=c=0
            for i in range(q['T']):
                x=best(z,p,*b,i,False,ref);y=best(z,p,*b,i,True,ref)
                if x is None or m-x>H:a|=1<<i
                if y is not None and m-y<=H:c|=1<<i
            must.append(a);may.append(c)
    res=[k-sum(bool(a&(1<<i)) for a in must) for i,k in enumerate(q['k'])]
    bad=H>=0 and (any(a&~b for a,b in zip(must,may,strict=True)) or any(k<0 or k>sum(bool((b^a)&(1<<i)) for a,b in zip(must,may,strict=True)) for i,k in enumerate(res)))
    return cols,M,B,H,must,may,res,bad


def signature(js,rows,sep,res,must,may,box,cols,M):
    """Exact task-local key; untruncated profiles do not depend on cutoff H."""
    active=tuple(i for i in sep if any((may[j]^must[j])&(1<<i) for j in js))
    axes=tuple(sorted(set(rows)|set(active)))
    return (tuple(rows),active,axes,tuple(res[i] for i in axes),tuple((j,must[j],may[j],box[j],tuple(cols[j][0]),tuple(cols[j][1]),M[j]) for j in js))


def table(key,ref=False):
    rows,sep,axes,lim,labels=key; pos={i:n for n,i in enumerate(axes)}
    dp={tuple(0 for _ in axes):(0,())}
    for j,a,b,(lo,hi),z,p,M in reversed(labels) if ref else labels:
        free=a^b; opts=[]
        if ref:
            masks=(sum(1<<i for i,on in zip(list(indices(free)),flags,strict=True) if on) for flags in product((0,1),repeat=free.bit_count()))
        else:
            def subsets():
                s=free
                while True:
                    yield s
                    if not s:break
                    s=(s-1)&free
            masks=subsets()
        for s in masks:
            m=a|s
            if not lo<=m.bit_length()<=hi:continue
            r=M-(sum(z[i] for i in range(len(z)) if m&(1<<i))-p[m.bit_length()]);assert r>=0
            opts.append((tuple(int(bool(s&(1<<i))) for i in axes),r,m))
        nxt={}
        for c,(v,path) in dp.items():
            for add,r,m in opts:
                nc=tuple(x+y for x,y in zip(c,add,strict=True));w=v+r
                if any(x>y for x,y in zip(nc,lim,strict=True)):continue
                if nc not in nxt or w<nxt[nc][0]:nxt[nc]=(w,(*path,(j,m)))
        dp=nxt
    out={}
    for c,(v,path) in dp.items():
        if any(c[pos[i]]!=lim[pos[i]] for i in rows):continue
        k=tuple(c[pos[i]] if i in pos else 0 for i in sep)
        if k not in out or v<out[k][0]:out[k]=(v,path)
    return out


def expand(tab,active,sep):
    pos={i:n for n,i in enumerate(active)}
    return {tuple(k[pos[i]] if i in pos else 0 for i in sep):v for k,v in tab.items()}


class Factors:
    def __init__(self,memo,cap):
        self.memo=memo;self.cap=cap;self.cache=OrderedDict();self.bank=[];self.ids={}
        self.work=self.computed=self.hits=self.calls=self.accepted=0

    def attempt(self,q,box,e,d,L):
        self.calls+=1
        cols,M,B,H,a,b,res,bad=screen(q,box,e,d,L)
        if H<0:return None,None
        proof=dict(eta=e,den=d,cutoff=L,upper=L,sep=[],parts=[])
        if bad:self.accepted+=1;return proof,None
        remaining=self.cap-self.work
        choice=choose(a,b,res,2,remaining)
        if choice is None:return None,None
        _,cost,_,sep,parts,costs,conv=choice
        self.work+=conv;self.computed+=conv;tabs=[];proof['sep']=list(sep);complete=True
        for (js,rows),w in zip(parts,costs,strict=True):
            if self.work+w>self.cap:
                proof['parts'].append([js,rows,None]);tabs.append(relaxed(sep,res));complete=False;continue
            self.work+=w
            key=signature(js,rows,sep,res,a,b,box,cols,M)
            if self.memo and key in self.cache:
                tab=self.cache.pop(key);self.cache[key]=tab;self.hits+=1
            else:
                tab=table(key);self.computed+=w
                if self.memo:
                    self.cache[key]=tab
                    if len(self.cache)>CACHE:self.cache.popitem(last=False)
            # Proof table interning is independent of the computation cache.
            # Use an exact key, never a digest or approximate matching.
            if self.memo and key in self.ids:i=self.ids[key]
            else:
                i=len(self.bank);self.bank.append(encode(tab))
                if self.memo:self.ids[key]=i
            proof['parts'].append([js,rows,i]);tabs.append(expand(tab,key[1],sep))
        ans=combine(tabs,sep,res,H);schedule=None
        if ans is not None:
            proof['upper']=max(L,(B-ans[0])//d)
            if complete:
                schedule=[0]*q['T']
                for j,m in ans[1]:
                    for i in indices(m):schedule[i]|=1<<j
                assert score(q,schedule)['objective']==proof['upper']
        self.accepted+=1
        return proof,schedule


def solve(task,initial,root,*,memo=False,cap=63,work_cap=WORK,static=False,force=None):
    q=validate(task);tick=process_time_ns();model=Model(q);f=Factors(memo,work_cap)
    if type(cap) is not int or not 1<=cap<=127 or type(work_cap) is not int or not 1<=work_cap<=1_000_000:raise TaskError('invalid cap')
    schedule=list(initial);L=score(q,schedule)['objective'];heap=[];tree={};nodes=1;calls=0;seq=0
    def prepare(node,box,proposal,depth):
        nonlocal L,schedule,calls,seq
        w=infeasible(q,box)
        if w is not None:node.update(kind='infeasible',witness=w);return
        e,d=proposal;e=list(e);u=conditioned(q,box,e,d)
        factor=None
        # Fixed root vector makes exact local subproblems reusable across branches.
        if u>L and (not static or depth==0) and not (force is not None and depth==0):
            p,a=f.attempt(q,box,list(root[0]),root[1],L)
            if p is not None:
                if a is not None and score(q,a)['objective']>L:schedule=a;L=score(q,a)['objective']
                if p['upper']<=u:factor=p
                if p['upper']<=L:
                    node.update(kind='factor',**p);return
        h=None
        if u>L:
            p,h=model.solve(box);calls+=1
            if p is not None:
                v,c=p;b=conditioned(q,box,list(v),c)
                if b<=u:e,d,u=list(v),c,b
        node.update(kind='bound',eta=e,den=d,upper=u)
        if factor is not None and factor['upper']<u:
            u=factor['upper'];node.clear();node.update(kind='factor',**factor)
        if h is not None and all(abs(float(v)-round(float(v)))<=1e-6 for v in h.flat):
            last=[int(round(sum(h[:,j]))) for j in range(q['d'])]
            if all(a<=v<=b for v,(a,b) in zip(last,box,strict=True)):
                a=fixed_schedule(q,tuple((v,v) for v in last))
                if a is not None and score(q,a)['objective']>L:schedule=a;L=score(q,a)['objective']
        if u<=L:return
        a=fixed_schedule(q,box)
        if a is not None:
            v=score(q,a)['objective']
            if v>L:schedule=a;L=v
            e=[]
            for i,k in enumerate(q['k'],1):
                vals=sorted((q['u'][i-1][j] for j,(lo,_) in enumerate(box) if lo>i),reverse=True);n=k-sum(lo==i for lo,_ in box)
                e.append(vals[0] if n==0 and vals else vals[n-1] if n else 0)
            assert conditioned(q,box,e,1)==v;node.clear();node.update(kind='bound',eta=e,den=1,upper=v);return
        opts=[(abs(float(h[i-1,j])-.5),j,i-1) for j,(lo,hi) in enumerate(box) for i in range(lo+1,hi+1) if h is not None and 1e-6<h[i-1,j]<1-1e-6]
        if force is not None and depth==0:j,cut=force
        elif opts:_,j,cut=min(opts)
        else:j=max(range(q['d']),key=lambda j:box[j][1]-box[j][0]);lo,hi=box[j];cut=(lo+hi)//2
        seq+=1;heappush(heap,(-u,seq,node,box,j,cut,depth))
    prepare(tree,domains(q),root,0)
    while heap and nodes+2<=cap:
        neg,_,node,box,j,cut,depth=heappop(heap)
        if -neg<=L:break
        p=node['eta'],node['den'];children=[{},{}];node.clear();node.update(kind='split',j=j,cut=cut,children=children);nodes+=2
        for child,b in zip(children,split(q,box,j,cut),strict=True):prepare(child,b,p,depth+1)
    def aggregate(n):
        if n['kind']=='infeasible':return None
        if n['kind']!='split':return n['upper']
        vals=[v for c in n['children'] if (v:=aggregate(c)) is not None];return max(vals) if vals else None
    U=aggregate(tree);assert U is not None and U>=L
    # Discard tables from overwritten proof nodes and remap resident references.
    bank=[];mapping={}
    def compact(n):
        if n['kind']=='split':
            for child in n['children']:compact(child)
        elif n['kind']=='factor':
            for part in n['parts']:
                i=part[2]
                if i is None:continue
                if i not in mapping:mapping[i]=len(bank);bank.append(f.bank[i])
                part[2]=mapping[i]
    compact(tree)
    return dict(schedule=schedule,objective=L,upper=U,gap=U-L,tree=tree,bank=bank,nodes=nodes,lp_calls=calls,work=f.work,computed=f.computed,hits=f.hits,factor_calls=f.calls,factor_leaves=f.accepted,cpu_ns=process_time_ns()-tick)


def replay(task,r,cap=63,work_cap=WORK):
    q=validate(task);L=score(q,r['schedule'])['objective'];nodes=work=0;keys={};used=set();checks=0
    bank=r['bank']
    def walk(n,box):
        nonlocal nodes,work,checks
        nodes+=1
        if nodes>cap:raise TaskError('tree cap')
        if n['kind']=='split':
            if len(n['children'])!=2:raise TaskError('missing branch')
            vals=[v for c,b in zip(n['children'],split(q,box,n['j'],n['cut']),strict=True) if (v:=walk(c,b)) is not None]
            return max(vals) if vals else None
        if n['kind']=='infeasible':
            if not check_witness(q,box,n['witness']):raise TaskError('bad witness')
            return None
        e,d=n['eta'],n['den'];C=n.get('cutoff',L)
        cols,M,B,H,a,b,res,bad=screen(q,box,e,d,C,True)
        if n['kind']=='bound':U=B//d
        elif n['kind']=='factor':
            if type(C) is not int or C>L or H<0:raise TaskError('bad cutoff')
            sep=n['sep'];parts=n['parts'];U=C
            if bad:
                if sep or parts:raise TaskError('unexpected factors')
            else:
                if len(sep)>2 or sep!=sorted(set(sep)) or any(type(i) is not int or not 0<=i<q['T'] for i in sep):raise TaskError('bad separator')
                if [(j,i) for j,i,_ in parts]!=partition(a,b,q['T'],sep):raise TaskError('partition mismatch')
                work+=len(parts)*prod(res[i]+1 for i in sep)**2;tabs=[]
                for js,rows,i in parts:
                    if i is None:
                        tabs.append(relaxed(sep,res));continue
                    work+=estimate(js,rows,sep,a,b,res)
                    if work>work_cap or type(i) is not int or not 0<=i<len(bank):raise TaskError('factor cap or bad table id')
                    key=signature(js,rows,sep,res,a,b,box,cols,M)
                    if i in keys:
                        if keys[i][0]!=key:raise TaskError('table reference mismatch')
                        tab=keys[i][1]
                    else:
                        tab=table(key,True);checks+=1
                        if encode(tab)!=bank[i]:raise TaskError('table mismatch')
                        keys[i]=(key,tab)
                    used.add(i);tabs.append(expand(tab,key[1],sep))
                ans=combine(tabs,sep,res,H);U=C if ans is None else max(C,(B-ans[0])//d)
        else:raise TaskError('bad node')
        if U!=n['upper']:raise TaskError('bad upper')
        return U
    U=walk(r['tree'],domains(q))
    if U is None or U<L or (L,U,U-L,nodes)!=(r['objective'],r['upper'],r['gap'],r['nodes']) or not work<=r['work']<=work_cap or used!=set(range(len(bank))):raise TaskError('receipt mismatch')
    return dict(objective=L,upper=U,gap=U-L,nodes=nodes,work=work,unique_tables=checks)
