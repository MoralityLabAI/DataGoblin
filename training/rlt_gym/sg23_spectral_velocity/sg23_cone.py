"""Active dual-face velocity constraints, with exact rational checking."""
from math import lcm

from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from sg20_delta import State, integer
from datagoblin.training_gym.spectral_face import face, rational_vector
from datagoblin.training_gym.spectral_quotient import indices
from datagoblin.training_gym.spectral_routing import TaskError


def direction(old, new):
    if (old.d, old.t, old.k) != (new.d, new.t, new.k):
        raise TaskError('fixed dimensions and budgets required')
    du = [tuple(y-x for x,y in zip(a,b,strict=True)) for a,b in zip(old.columns,new.columns,strict=True)]
    dp = [tuple(new.lam*new.bits[j]*(y-new.x[0])-old.lam*old.bits[j]*(x-old.x[0])
                for x,y in zip(old.x,new.x,strict=True)) for j in range(old.d)]
    return du,dp


def slope_value(du,dp,mask):
    return sum(du[i] for i in indices(mask))-dp[mask.bit_length()]


class Cone:
    """Compact epigraph of all inequalities against anchor-optimal masks."""
    def __init__(self, anchor, target):
        if not anchor.certificate()['exact_face']:
            raise TaskError('velocity cone needs exact primal/dual equality')
        self.anchor = anchor
        self.du,self.dp=direction(anchor.q,target)
        self.t=anchor.q.t
        _,self.boxes=face(anchor.q.thaw(),list(anchor.eta),anchor.den)
        self.rows={};self.aux={};self.lower=[None]*self.t;self.upper=[None]*self.t
        self.impossible=False
        for j,(mask,boxes) in enumerate(zip(anchor.visits,self.boxes,strict=True)):
            own=next((a,b) for a,b in boxes if mask&a==a and mask&~b==0)
            optional=own[1]^own[0]
            # Zero reduced utilities before the selected last visit have a
            # prescribed directional sign. Eliminate these epigraph entries.
            for i in indices(optional):
                v=self.du[j][i]
                if mask&(1<<i): self.add({i:1},v)
                else:self.add({i:-1},-v)
            chosen=slope_value(self.du[j],self.dp[j],mask)
            for a,b in boxes:
                opt=b^a
                req=a|(opt&optional&mask)
                row={i:1 for i in indices(mask&~req)}
                row.update({i:-1 for i in indices(req&~mask)})
                for i in indices(opt&~optional):
                    key=(j,i)
                    if key not in self.aux:
                        self.aux[key]=self.t+len(self.aux)
                        self.add({i:-1,self.aux[key]:-1},-self.du[j][i])
                    row[self.aux[key]]=1
                # req retains the last visit of the alternative box.
                self.add(row,chosen-slope_value(self.du[j],self.dp[j],req))
        if any(a is not None and b is not None and a>b for a,b in zip(self.lower,self.upper,strict=True)):
            self.impossible=True

    def add(self,row,rhs):
        key=tuple(sorted(row.items()))
        if not key:
            self.impossible |= rhs<0
            return
        if len(key)==1 and key[0][0]<self.t:
            i,c=key[0]
            if c==1:self.upper[i]=rhs if self.upper[i] is None else min(self.upper[i],rhs)
            else:self.lower[i]=-rhs if self.lower[i] is None else max(self.lower[i],-rhs)
            return
        self.rows[key]=min(rhs,self.rows.get(key,rhs))

    def gap(self,v,den):
        """Independent box formula: sum of nonnegative directional regrets."""
        integer(den,1,1_000_000)
        if len(v)!=self.t:raise TaskError('invalid velocity length')
        total=0
        for j,(mask,boxes) in enumerate(zip(self.anchor.visits,self.boxes,strict=True)):
            z=[den*x-y for x,y in zip(self.du[j],v,strict=True)]
            chosen=sum(z[i] for i in indices(mask))-den*self.dp[j][mask.bit_length()]
            best=max(sum(z[i] for i in indices(a))+sum(max(0,z[i]) for i in indices(b^a))
                     -den*self.dp[j][a.bit_length()] for a,b in boxes)
            total+=best-chosen
        return total

    def sweep(self, v):
        """Exact coordinate minimization of the directional gap on the active face."""
        v=list(v)
        for i in range(self.t-1,-1,-1):
            bit=1<<i; thresholds=[]; forced=0
            for j,boxes in enumerate(self.boxes):
                z=[a-b for a,b in zip(self.du[j],v,strict=True)]
                absent=present=None
                for a,b in boxes:
                    val=sum(z[l] for l in indices(a&~bit))+sum(max(0,z[l]) for l in indices((b^a)&~bit))-self.dp[j][a.bit_length()]
                    if not a&bit:absent=val if absent is None else max(absent,val)
                    if b&bit:
                        w=val+self.du[j][i]
                        present=w if present is None else max(present,w)
                if absent is None:forced+=1
                elif present is not None:thresholds.append(present-absent)
            k=self.anchor.q.k[i]-forced
            assert 0<=k<=len(thresholds)
            thresholds.sort(reverse=True)
            if k<len(thresholds):v[i]=max(v[i],thresholds[k])
            if k:v[i]=min(v[i],thresholds[k-1])
        return v

    def propose(self,guess,mode):
        if mode not in ('clamp','sweep','lp'):raise TaskError('invalid cone mode')
        info=dict(rows=len(self.rows),aux=len(self.aux),lp=0,status='infeasible_bounds')
        if self.impossible:return None,info
        g=[max(a,v) if a is not None else v for a,v in zip(self.lower,guess,strict=True)]
        g=[min(b,v) if b is not None else v for b,v in zip(self.upper,g,strict=True)]
        if self.gap(g,1)==0:
            return (g,1),dict(**{**info,'status':'clamp'})
        if mode=='clamp':return None,dict(**{**info,'status':'clamp_failed'})
        if mode=='sweep':
            g=self.sweep(g)
            return ((g,1),dict(**{**info,'status':'sweep_exact'})) if self.gap(g,1)==0 else (None,dict(**{**info,'status':'sweep_failed'}))
        rr,cc,vv,rhs=[],[],[],[]
        n=self.t+len(self.aux)
        def add(entries,b):
            r=len(rhs);rhs.append(b)
            for i,v in entries:rr.append(r);cc.append(i);vv.append(v)
        for row,b in self.rows.items():add(row,b)
        # L1 distance to the clamped seed limits arbitrary velocity choices.
        for i,v in enumerate(g):
            add(((i,1),(n+i,-1)),v);add(((i,-1),(n+i,-1)),-v)
        mat=coo_matrix((vv,(rr,cc)),shape=(len(rhs),n+self.t)).tocsr()
        z=linprog([0]*n+[1]*self.t,A_ub=mat,b_ub=rhs,
                  bounds=list(zip(self.lower,self.upper,strict=True))+[(0,None)]*(len(self.aux)+self.t),
                  method='highs',options={'time_limit':0.1})
        info.update(lp=1,status='solver_'+str(z.status))
        if not z.success:return None,info
        e,den=rational_vector(z.x[:self.t])
        if self.gap(e,den):return None,dict(**{**info,'status':'rational_rejected'})
        return (e,den),dict(**{**info,'status':'lp_exact'})


def endpoint(anchor,target,velocity,den):
    """A tangent direction is only a proposal for a finite edit."""
    common=lcm(anchor.den,den)
    integer(common,1,1_000_000)
    e=[a*(common//anchor.den)+b*(common//den) for a,b in zip(anchor.eta,velocity,strict=True)]
    return State.build(target,e,common,anchor.schedule)
