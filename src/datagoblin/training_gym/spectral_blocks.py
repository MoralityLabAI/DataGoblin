"""SG-12: interruptible union-class search with monotone upper certificates."""
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush

from datagoblin.training_gym.spectral_quotient import coordinate, indices, score, validate
from datagoblin.training_gym.spectral_routing import TaskError


@dataclass(frozen=True)
class Block:
    t: int
    suffix: int
    include: int
    exclude: int
    value: int
    tail: tuple[int, ...]
    upper: int


def multiplier_vectors(q, extra):
    cut = [sorted(row, reverse=True)[k-1] if k else max(row)+1
           for row, k in zip(q['u'], q['k'], strict=True)]
    return [[0]*q['T'], *[[v+shift for v in cut] for shift in (-6,-3,0,3,6)], *extra]


def root_vector(q):
    e = [0]*q['T']
    for i in range(q['T']-1,-1,-1):
        e[i] = coordinate(q,q['T'],0,e,i)
    return e


def normalize(q,t,s,r,e):
    """Return a canonical feasible block, or None. Only outside bits are fixed."""
    k = q['k'][t-1]
    outside = ((1 << q['d'])-1) ^ s
    if r & e or (r | e) & ~outside or r.bit_count()>k or q['d']-e.bit_count()<k:
        return None
    if r.bit_count()==k:
        e = outside ^ r
    elif q['d']-e.bit_count()==k:
        r = outside ^ e
    return r,e


def legal_action(q,t,s,r):
    """At a resolved union class, optimize the remaining slots inside S."""
    row=q['u'][t-1]
    inside=sorted(indices(s),key=lambda j:(-row[j],j))[:q['k'][t-1]-r.bit_count()]
    mask=r|sum(1<<j for j in inside)
    if mask.bit_count()!=q['k'][t-1]:
        raise AssertionError('incomplete class')
    return mask


class PrefixRelaxation:
    """Keep the current cardinality exact; relax only earlier visits."""
    def __init__(self,q,extra):
        self.q=q
        self.tables=[[] for _ in range(q['T'])]
        for eta in multiplier_vectors(q,extra):
            a,b,c=[0]*q['d'],[0]*q['d'],0
            for i in range(q['T']):
                self.tables[i].append((c,tuple(a),tuple(b)))
                c+=eta[i]*q['k'][i]
                for j,bit in enumerate(q['bits']):
                    z=q['u'][i][j]-eta[i]
                    b[j]=max(b[j],a[j]+z-q['lambda']*bit*(q['x'][i+1]-q['x'][0]))
                    a[j]+=max(0,z)
        self.context=lru_cache(maxsize=256)(self._context)

    def _context(self,t,s):
        q=self.q
        row=q['u'][t-1]; price=q['lambda']*(q['x'][t]-q['x'][0]); out=[]
        for c,a,b in self.tables[t-1]:
            base=c; delta=[]
            for j,v in enumerate(row):
                active=bool(s&(1<<j))
                base+=a[j] if active else b[j]
                delta.append(v if active else a[j]-b[j]+v-price*q['bits'][j])
            order=tuple(sorted(range(q['d']),key=lambda j:(-delta[j],j)))
            out.append((base,tuple(delta),order))
        return tuple(out)

    def bound(self,t,s,r,e):
        count=self.q['k'][t-1]-r.bit_count()
        result=None
        for base,d,order in self.context(t,s):
            value=base+sum(d[j] for j in indices(r)); n=count
            if n:
                for j in order:
                    if not (r|e)&(1<<j):
                        value+=d[j];n-=1
                        if not n:break
            if n:raise AssertionError('infeasible block')
            result=value if result is None else min(result,value)
        return result

    def choose(self,b):
        outside=((1<<self.q['d'])-1) & ~(b.suffix|b.include|b.exclude)
        order=self.context(b.t,b.suffix)[-1][2]
        return next(j for j in order if outside&(1<<j))


def reference_bound(q,t,s,r,e,extra):
    """Independent arithmetic: enumerate last-use choices for each label.

    No recurrence, cached tables, or sorted-prefix context used by the engine.
    """
    candidates=[]
    k=q['k'][t-1]
    for eta in multiplier_vectors(q,extra):
        c=sum(eta[i]*q['k'][i] for i in range(t-1)); absent=[]; gain=[]
        for j,bit in enumerate(q['bits']):
            z=[q['u'][i][j]-eta[i] for i in range(t-1)]
            active=sum(max(0,v) for v in z)
            unused=max([0,*[z[l]+sum(max(0,v) for v in z[:l])
                             -q['lambda']*bit*(q['x'][l+1]-q['x'][0]) for l in range(t-1)]])
            free=bool(s&(1<<j))
            without=active if free else unused
            with_=active+q['u'][t-1][j]-(0 if free else q['lambda']*bit*(q['x'][t]-q['x'][0]))
            absent.append(without);gain.append(with_-without)
        forced=sum(gain[j] for j in range(q['d']) if r&(1<<j))
        optional=sorted((gain[j] for j in range(q['d']) if not (r|e)&(1<<j)),reverse=True)
        candidates.append(c+sum(absent)+forced+sum(optional[:k-r.bit_count()]))
    return min(candidates)


def check_vectors(q,extra):
    if not isinstance(extra,list) or len(extra)>4 or any(not isinstance(e,list) or len(e)!=q['T'] or
            any(type(v) is not int or abs(v)>100_000_000 for v in e) for e in extra):
        raise TaskError('invalid extra multiplier vectors')
    return [list(e) for e in extra]


class BlockSearch:
    """Each event covers a complete binary partition or one resolved class.

    Every surviving obligation inherits the parent's upper bound. A cap may stop
    at any event boundary without losing any unvisited part of the action space.
    """
    def __init__(self,task,initial,*,event_cap=2048,extra=None):
        self.q=validate(task)
        if type(event_cap) is not int or not 1<=event_cap<=100000:
            raise TaskError('invalid event cap')
        self.cap=event_cap;self.extra=check_vectors(self.q,[] if extra is None else extra)
        self.relax=PrefixRelaxation(self.q,self.extra)
        self.schedule=list(initial);self.incumbent=score(self.q,self.schedule)['objective']
        self.live={};self.heap=[];self.seen={};self.events=[];self.next_id=0
        self.splits=self.leaves=self.terminals=self.bound_calls=0;self.peak=0
        self._add(self.q['T'],0,0,0,0,(),None,whole=True)

    def _add(self,t,s,r,e,g,tail,cap,*,whole=False):
        norm=normalize(self.q,t,s,r,e)
        if norm is None:return
        r,e=norm
        key=(t,s)
        if whole and key in self.seen and g<=self.seen[key]:return
        self.bound_calls+=1
        ub=g+self.relax.bound(t,s,r,e)
        if cap is not None:ub=min(ub,cap)
        if ub<=self.incumbent:return
        if whole:self.seen[key]=g
        i=self.next_id;self.next_id+=1
        b=Block(t,s,r,e,g,tail,ub);self.live[i]=b
        heappush(self.heap,(-ub,t,i));self.peak=max(self.peak,len(self.live))

    def _valid(self,b):
        return b.upper>self.incumbent and b.value==self.seen.get((b.t,b.suffix))

    def _clean(self):
        while self.heap:
            i=self.heap[0][2]
            if i in self.live and self._valid(self.live[i]):break
            heappop(self.heap);self.live.pop(i,None)

    def upper(self):
        self._clean()
        return max(self.incumbent,-self.heap[0][0]) if self.heap else self.incumbent

    def best(self):
        self._clean()
        return self.heap[0][2] if self.heap else None

    def advance(self,i,component=None):
        if type(i) is not int or i not in self.live or not self._valid(self.live[i]) or len(self.events)>=self.cap:
            raise TaskError('invalid live block or event cap')
        b=self.live[i];q=self.q;t,s,r,e=b.t,b.suffix,b.include,b.exclude
        free=((1<<q['d'])-1)&~(s|r|e)
        if t>1 and free:
            if type(component) is not int or not 0<=component<q['d'] or not free&(1<<component):
                raise TaskError('invalid split component')
        elif component is not None:
            raise TaskError('resolved block cannot split')
        self.live.pop(i)
        self.events.append({'id':i,'component':component})
        if t==1:
            # Exact terminal solve, respecting all fixed outside bits.
            row=q['u'][0];price=q['lambda']*(q['x'][1]-q['x'][0])
            adj=[v-(0 if s&(1<<j) else price*q['bits'][j]) for j,v in enumerate(row)]
            js=sorted((j for j in range(q['d']) if not (r|e)&(1<<j)),key=lambda j:(-adj[j],j))
            m=r|sum(1<<j for j in js[:q['k'][0]-r.bit_count()]);self.terminals+=1
        elif free:
            bit=1<<component;self.splits+=1
            self._add(t,s,r|bit,e,b.value,b.tail,b.upper)
            self._add(t,s,r,e|bit,b.value,b.tail,b.upper)
            return
        else:
            m=legal_action(q,t,s,r);self.leaves+=1
        price=q['lambda']*(q['x'][t]-q['x'][0])
        g=b.value+sum(q['u'][t-1][j] for j in indices(m))-price*sum(q['bits'][j] for j in indices(m&~s))
        tail=(m,*b.tail)
        if t==1:
            if score(q,list(tail))['objective']!=g or g>b.upper:raise AssertionError('terminal accounting')
            if g>self.incumbent:self.incumbent,self.schedule=g,list(tail)
        else:self._add(t-1,s|m,0,0,g,tail,b.upper,whole=True)

    def run(self):
        while len(self.events)<self.cap and (i:=self.best()) is not None:
            before=self.upper();b=self.live[i]
            free=((1<<self.q['d'])-1)&~(b.suffix|b.include|b.exclude)
            self.advance(i,self.relax.choose(b) if b.t>1 and free else None)
            if self.upper()>before:raise AssertionError('upper increased')
        return self.receipt()

    def receipt(self):
        upper=self.upper()
        return dict(**score(self.q,self.schedule),schedule=list(self.schedule),upper=upper,gap=upper-self.incumbent,
                    event_count=len(self.events),splits=self.splits,leaves=self.leaves,terminals=self.terminals,
                    bound_calls=self.bound_calls,events=[dict(e) for e in self.events],
                    live_states=sum(self._valid(b) for b in self.live.values()),peak_stored_blocks=self.peak)
