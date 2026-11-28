"""SG-21: exact integer certificate intervals along affine task rays."""
from dataclasses import dataclass

from sg20_delta import Task
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_quotient import indices
from datagoblin.training_gym.spectral_routing import TaskError


@dataclass(frozen=True)
class Ray:
    base: Task
    horizon: int
    edits: tuple
    terms: tuple

    @classmethod
    def build(cls, first, last, horizon):
        if type(horizon) is not int or not 1 <= horizon <= 10000:
            raise TaskError('invalid ray horizon')
        a, b = Task.from_task(first), Task.from_task(last)
        if (a.d, a.t, a.k) != (b.d, b.t, b.k):
            raise TaskError('ray requires fixed shape and budgets')
        # Effective prices are affine when at most one factor changes.
        if sum((a.bits != b.bits, a.x != b.x, a.lam != b.lam)) > 1:
            raise TaskError('ray requires affine prices')
        edits = []
        def slope(x, y):
            v, rem = divmod(y-x, horizon)
            if rem:
                raise TaskError('noninteger ray step')
            return v
        def add(key, path, x, y):
            v = slope(x, y)
            if v:
                edits.append((key, path, x, v))
            return v
        terms = []
        for j in range(a.d):
            du = tuple(add('u', (i, j), a.columns[j][i], b.columns[j][i]) for i in range(a.t))
            p = tuple(a.lam*a.bits[j]*(x-a.x[0]) for x in a.x)
            dp = tuple(slope(v, b.lam*b.bits[j]*(x-b.x[0])) for v,x in zip(p,b.x,strict=True))
            if any(du) or any(dp):
                terms.append((j, a.columns[j], du, p, dp))
        for key in ('bits', 'x'):
            for i,(x,y) in enumerate(zip(getattr(a,key),getattr(b,key),strict=True)):
                add(key,(i,),x,y)
        add('lambda',(),a.lam,b.lam)
        return cls(a,horizon,tuple(edits),tuple(terms))

    def patch(self, step):
        if type(step) is not int or not 0 <= step <= self.horizon:
            raise TaskError('invalid ray position')
        patch = {}
        for key,path,x,v in self.edits:
            value = x+step*v
            if key == 'lambda': patch[key]=value
            else: patch.setdefault(key,[]).append([*path,value])
        return patch

    def at(self, step):
        return self.base.patch(self.patch(step))[0]


def region(ray, state, start):
    """Maximal integer prefix certified by this fixed schedule and multiplier."""
    if state.q != ray.at(start) or not state.certificate()['certified']:
        raise TaskError('uncertified or mismatched ray anchor')
    den, eta = state.den, state.eta
    slope = sum(sum(du[i] for i in indices(state.visits[j]))-dp[state.visits[j].bit_length()]
                for j,_,du,_,dp in ray.terms)
    terms = []
    for j,u,du,p,dp in ray.terms:
        terms.append((state.tables[j][0], tuple(den*v-e for v,e in zip(u,eta,strict=True)),
                      tuple(den*v for v in du), tuple(den*v for v in p[1:]), tuple(den*v for v in dp[1:])))
    probes = 0
    def evaluate(step):
        nonlocal probes
        probes += 1
        bound = state.bound
        for old,z,dz,p,dp in terms:
            a = b = 0
            for u,v,c,w in zip(z,dz,p,dp,strict=True):
                x = u+step*v
                b = max(b,a+x-c-step*w)
                a += max(0,x)
            bound += b-old
        obj = state.objective+(step-start)*slope
        if bound < den*obj:
            raise AssertionError('invalid bound')
        return bound,obj
    lo, hi = start, ray.horizon
    best = (state.bound,state.objective)
    if hi > lo:
        b,o = evaluate(hi)
        if b < den*(o+1):
            lo,best = hi,(b,o)
        else:
            # The strict sublevel of a convex gap along this ray is an interval.
            while hi-lo > 1:
                mid = (lo+hi)//2
                b,o = evaluate(mid)
                if b < den*(o+1): lo,best = mid,(b,o)
                else: hi = mid
    return dict(start=start,end=lo,eta=list(eta),den=den,schedule=list(state.schedule),
                bounds=[state.bound,best[0]],objectives=[state.objective,best[1]],probes=probes,
                changed=len(terms))


def interval_value(proof, step):
    """Use only the chord certificate, without evaluating a dual component."""
    a,b = proof['start'],proof['end']
    if type(step) is not int or not a <= step <= b:
        raise TaskError('position outside certified interval')
    if a == b:
        return proof['objectives'][0], proof['bounds'][0]//proof['den']
    w,v = b-step,step-a
    objective, rem = divmod(w*proof['objectives'][0]+v*proof['objectives'][1],b-a)
    if rem:
        raise TaskError('noninteger primal interpolation')
    upper = (w*proof['bounds'][0]+v*proof['bounds'][1])//((b-a)*proof['den'])
    if objective != upper:
        raise TaskError('interval not closed')
    return objective,upper


def verify_region(ray, proof):
    """Independent endpoint arithmetic; Ray.build validates the affine path."""
    a,b=proof['start'],proof['end']
    if type(a) is not int or type(b) is not int or not 0 <= a <= b <= ray.horizon:
        raise TaskError('invalid interval')
    for i,s in enumerate((a,b)):
        c=verify_certificate(ray.at(s).thaw(),proof['schedule'],proof['eta'],proof['den'])
        if not c['certified'] or c['objective']!=proof['objectives'][i] or c['bound_num']!=proof['bounds'][i]:
            raise TaskError('endpoint certificate mismatch')
    return True
