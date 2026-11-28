"""Exact affine multiplier paths; proposals never certify themselves."""
from dataclasses import dataclass
from math import gcd, lcm

from sg20_delta import State, integer
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_quotient import coordinate
from datagoblin.training_gym.spectral_routing import TaskError


@dataclass(frozen=True)
class Line:
    origin: int
    eta: tuple
    slope: tuple
    den: int

    @classmethod
    def secant(cls, a, b):
        s, e, d = a
        t, f, c = b
        if type(s) is not int or type(t) is not int or not s < t or len(e) != len(f):
            raise TaskError('invalid secant points')
        integer(d, 1, 1_000_000)
        integer(c, 1, 1_000_000)
        for v in (*e, *f):
            integer(v, -10**15, 10**15)
        common = lcm(d, c)
        scale = t-s
        den = common*scale
        slope = tuple(y*(common//c)-x*(common//d) for x,y in zip(e,f,strict=True))
        eta = tuple(y*(common//c)*scale for y in f)
        g = gcd(den, *eta, *slope)
        den //= g
        if den > 1_000_000:
            raise TaskError('secant denominator too large')
        return cls(t, tuple(x//g for x in eta), tuple(x//g for x in slope), den)

    def at(self, step):
        if type(step) is not int:
            raise TaskError('invalid line position')
        integer(self.den, 1, 1_000_000)
        return [integer(x+(step-self.origin)*v,-10**15,10**15)
                for x,v in zip(self.eta,self.slope,strict=True)], self.den


def correct(q, eta, den):
    """One reverse exact coordinate sweep on the scaled integer objective."""
    z = q.thaw()
    z['u'] = [[v*den for v in row] for row in z['u']]
    z['lambda'] *= den
    e = list(eta)
    for i in range(q.t-1,-1,-1):
        e[i] = coordinate(z,q.t,0,e,i)
    return [integer(v,-10**15,10**15) for v in e],den


def region(ray, state, start, line):
    """Find maximal forward integer interval for fixed schedule and affine eta."""
    if state.q != ray.at(start) or not state.certificate()['certified']:
        raise TaskError('invalid anchor')
    e, den = line.at(start)
    if len(e)!=state.q.t or any(a*den!=b*state.den for a,b in zip(state.eta,e,strict=True)):
        raise TaskError('line misses anchor')
    # Check validity of the complete multiplier path before searching it.
    line.at(ray.horizon)
    # Same exact rational bound with the line's common denominator.
    base, rem = divmod(state.bound*den,state.den)
    if rem:
        raise TaskError('nonintegral scaled anchor bound')
    lo,hi=start,ray.horizon
    best=(base,state.objective)
    probes=0
    def evaluate(step):
        nonlocal probes
        probes+=1
        e,_=line.at(step)
        s=State.build(ray.at(step),e,den,state.schedule)
        return s.bound,s.objective
    if hi>lo:
        b,o=evaluate(hi)
        if b<den*(o+1):
            lo,best=hi,(b,o)
        else:
            while hi-lo>1:
                mid=(lo+hi)//2
                b,o=evaluate(mid)
                if b<den*(o+1):lo,best=mid,(b,o)
                else:hi=mid
    return dict(start=start,end=lo,origin=line.origin,eta=list(line.eta),slope=list(line.slope),den=den,
                schedule=list(state.schedule),bounds=[base,best[0]],objectives=[state.objective,best[1]],probes=probes)


def line_from_proof(proof):
    return Line(proof['origin'],tuple(proof['eta']),tuple(proof['slope']),proof['den'])


def value(proof, step):
    a,b=proof['start'],proof['end']
    if type(step) is not int or not a<=step<=b:
        raise TaskError('step outside certificate')
    if a==b:
        objective=proof['objectives'][0]
        upper=proof['bounds'][0]//proof['den']
    else:
        w,v=b-step,step-a
        objective,rem=divmod(w*proof['objectives'][0]+v*proof['objectives'][1],b-a)
        if rem:raise TaskError('noninteger objective')
        upper=(w*proof['bounds'][0]+v*proof['bounds'][1])//((b-a)*proof['den'])
    if objective!=upper:raise TaskError('unclosed chord')
    e,den=line_from_proof(proof).at(step)
    return dict(objective=objective,upper=upper,gap=0,schedule=list(proof['schedule']),eta=e,den=den)


def verify_region(ray, proof):
    a,b=proof['start'],proof['end']
    if type(a) is not int or type(b) is not int or not 0<=a<=b<=ray.horizon:
        raise TaskError('invalid interval')
    line=line_from_proof(proof)
    for i,s in enumerate((a,b)):
        e,den=line.at(s)
        c=verify_certificate(ray.at(s).thaw(),proof['schedule'],e,den)
        if not c['certified'] or c['objective']!=proof['objectives'][i] or c['bound_num']!=proof['bounds'][i]:
            raise TaskError('invalid endpoint')
    return True
