"""SG-20: exact differential dual and primal state, immutable snapshots."""
from dataclasses import dataclass

from datagoblin.training_gym.spectral_quotient import indices, validate
from datagoblin.training_gym.spectral_routing import TaskError


def integer(v, lo, hi):
    if type(v) is not int or not lo <= v <= hi:
        raise TaskError('integer outside registered bounds')
    return v


@dataclass(frozen=True)
class Task:
    d: int
    t: int
    columns: tuple
    k: tuple
    x: tuple
    bits: tuple
    lam: int

    @classmethod
    def from_task(cls, task):
        q = validate(task)
        return cls(q['d'], q['T'], tuple(zip(*q['u'], strict=True)), tuple(q['k']),
                   tuple(q['x']), tuple(q['bits']), q['lambda'])

    def thaw(self):
        return dict(d=self.d, T=self.t, u=[list(r) for r in zip(*self.columns, strict=True)],
                    k=list(self.k), x=list(self.x), bits=list(self.bits), **{'lambda':self.lam})

    def patch(self, edits):
        """Absolute replacements; the patch is the entire change, not a hint."""
        if not isinstance(edits, dict) or set(edits)-{'u','bits','k','x','lambda'}:
            raise TaskError('invalid patch keys')
        columns, k, x, bits = list(self.columns), list(self.k), list(self.x), list(self.bits)
        changed = set()
        for name, arity in (('u',3),('bits',2),('k',2),('x',2)):
            rows = edits.get(name, [])
            cap = self.d*self.t if name=='u' else len({'bits':bits,'k':k,'x':x}[name])
            if not isinstance(rows,list) or len(rows)>cap:
                raise TaskError('invalid patch length')
            seen = set()
            for r in rows:
                if not isinstance(r,list) or len(r)!=arity:
                    raise TaskError('invalid patch entry')
                if name=='u':
                    i,j,v = r
                    integer(i,0,self.t-1); integer(j,0,self.d-1); integer(v,-10000,10000)
                    key = (i,j)
                    if key in seen: raise TaskError('duplicate edit')
                    if columns[j][i]!=v:
                        col=list(columns[j]); col[i]=v; columns[j]=tuple(col); changed.add(j)
                else:
                    i,v=r; vec={'bits':bits,'k':k,'x':x}[name]
                    integer(i,0,len(vec)-1)
                    integer(v,*{'bits':(1,100),'k':(0,self.d),'x':(-1000,1000)}[name])
                    key=i
                    if key in seen: raise TaskError('duplicate edit')
                    vec[i]=v
                seen.add(key)
        lam=integer(edits.get('lambda',self.lam),0,100)
        if x!=sorted(x): raise TaskError('nonmonotone coordinates')
        old_pos=tuple(v-self.x[0] for v in self.x[1:]); new_pos=tuple(v-x[0] for v in x[1:])
        for j in range(self.d):
            a,b=self.lam*self.bits[j],lam*bits[j]
            if a!=b or (a and old_pos!=new_pos):
                if tuple(a*v for v in old_pos)!=tuple(b*v for v in new_pos): changed.add(j)
        return Task(self.d,self.t,tuple(columns),tuple(k),tuple(x),tuple(bits),lam),changed


def changed_columns(a,b):
    """Full-snapshot comparison: never trust a caller's change-set hint."""
    if (a.d,a.t)!=(b.d,b.t): raise TaskError('incompatible shape')
    ax=tuple(v-a.x[0] for v in a.x[1:]); bx=tuple(v-b.x[0] for v in b.x[1:])
    return {j for j in range(a.d) if a.columns[j]!=b.columns[j] or
            tuple(a.lam*a.bits[j]*v for v in ax)!=tuple(b.lam*b.bits[j]*v for v in bx)}


def component(q,j,eta,den):
    a=b=0; delta=[]
    for i,u in enumerate(q.columns[j]):
        price=den*q.lam*q.bits[j]*(q.x[i+1]-q.x[0]); z=den*u-eta[i]
        delta.append(den*u+a-b-price)
        b=max(b,a+z-price);a+=max(0,z)
    return b,tuple(delta)


def visits(q,schedule):
    if len(schedule)!=q.t: raise TaskError('invalid schedule length')
    out=[0]*q.d
    for i,m in enumerate(schedule):
        integer(m,0,(1<<q.d)-1)
        if m.bit_count()!=q.k[i]: raise TaskError('invalid cardinality')
        for j in indices(m): out[j]|=1<<i
    return tuple(out)


def contribution(q,j,m):
    return sum(q.columns[j][i] for i in indices(m))-q.lam*q.bits[j]*(q.x[m.bit_length()]-q.x[0])


def repair_budgets(old,q):
    """Keep all unchanged rows; optimize changed rows conditional on the others."""
    schedule=list(old.schedule); masks=list(old.visits)
    for i,(a,b) in enumerate(zip(old.q.k,q.k,strict=True)):
        if a==b: continue
        bit=1<<i; before=schedule[i]
        v=[q.columns[j][i]-q.lam*q.bits[j]*max(0,q.x[i+1]-q.x[(m&~bit).bit_length()])
           for j,m in enumerate(masks)]
        after=sum(1<<j for j in sorted(range(q.d),key=lambda j:(-v[j],not bool(before&(1<<j)),j))[:b])
        for j in indices(before^after): masks[j]^=bit
        schedule[i]=after
    return tuple(schedule),tuple(masks)


@dataclass(frozen=True)
class State:
    q: Task
    eta: tuple
    den: int
    schedule: tuple
    visits: tuple
    tables: tuple
    primal: tuple
    bound: int
    objective: int

    @classmethod
    def build(cls,q,eta,den,schedule):
        integer(den,1,1_000_000)
        if len(eta)!=q.t: raise TaskError('invalid multiplier length')
        eta=tuple(integer(v,-10**15,10**15) for v in eta)
        masks=visits(q,schedule)
        tables=tuple(component(q,j,eta,den) for j in range(q.d))
        primal=tuple(contribution(q,j,m) for j,m in enumerate(masks))
        return cls(q,eta,den,tuple(schedule),masks,tables,primal,
                   sum(a*k for a,k in zip(eta,q.k,strict=True))+sum(v[0] for v in tables),sum(primal))

    def certificate(self):
        gap=self.bound//self.den-self.objective
        if gap<0: raise AssertionError('invalid bound')
        return dict(objective=self.objective,bound_num=self.bound,den=self.den,integer_upper=self.bound//self.den,
                    gap=gap,certified=gap==0,exact_face=self.objective*self.den==self.bound)

    def _update(self,q,changed,*,full=False):
        # Internal API: changed is computed by Task.patch or changed_columns.
        schedule,masks=repair_budgets(self,q)
        if full:
            return State.build(q,self.eta,self.den,schedule),q.d,q.d
        tables=list(self.tables); primal=list(self.primal)
        b=self.bound+sum(e*(new-old) for e,new,old in zip(self.eta,q.k,self.q.k,strict=True)); obj=self.objective
        for j in changed:
            v=component(q,j,self.eta,self.den); b+=v[0]-tables[j][0]; tables[j]=v
        affected=changed|{j for j,(a,c) in enumerate(zip(self.visits,masks,strict=True)) if a!=c}
        for j in affected:
            v=contribution(q,j,masks[j]);obj+=v-primal[j];primal[j]=v
        return State(q,self.eta,self.den,schedule,masks,tuple(tables),tuple(primal),b,obj),len(changed),len(affected)

    def apply(self,edits,*,full=False):
        q,changed=self.q.patch(edits)
        return self._update(q,changed,full=full)

    def replace(self,task):
        q=Task.from_task(task)
        return self._update(q,changed_columns(self.q,q))

    def with_schedule(self,schedule):
        masks=visits(self.q,schedule);primal=list(self.primal);obj=self.objective
        for j,(a,b) in enumerate(zip(self.visits,masks,strict=True)):
            if a!=b:
                v=contribution(self.q,j,b);obj+=v-primal[j];primal[j]=v
        return State(self.q,self.eta,self.den,tuple(schedule),masks,self.tables,tuple(primal),self.bound,obj)

    def rounded(self):
        """Reuse cached prefix differences for exact SG-14 backward rounding."""
        q=self.q; s=0; out=[0]*q.t
        for i in range(q.t-1,-1,-1):
            values=[self.den*q.columns[j][i] if s&(1<<j) else self.tables[j][1][i] for j in range(q.d)]
            out[i]=sum(1<<j for j in sorted(range(q.d),key=lambda j:(-values[j],j))[:q.k[i]])
            s|=out[i]
        return out
