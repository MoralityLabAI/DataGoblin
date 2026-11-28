"""SG-25: exact convex certificates over a two-dimensional affine task plane."""
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm

from sg20_delta import Task, integer
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_routing import TaskError


@dataclass(frozen=True)
class Plane:
    base: Task
    horizon: int
    edits: tuple
    price: tuple

    @classmethod
    def build(cls, task, horizon, directions, price):
        q = Task.from_task(task)
        integer(horizon, 1, 100)
        if len(directions) != 2 or len(price) != 2:
            raise TaskError('two directions required')
        ds = {}
        for axis, changes in enumerate(directions):
            seen = set()
            for i, j, v in changes:
                integer(i, 0, q.t-1); integer(j, 0, q.d-1); integer(v, -10000, 10000)
                if (i, j) in seen:
                    raise TaskError('duplicate direction entry')
                seen.add((i, j)); ds.setdefault((i, j), [0, 0])[axis] = v
        price = tuple(integer(v, -100, 100) for v in price)
        out = cls(q, horizon, tuple((i, j, a, b) for (i, j), (a, b) in sorted(ds.items())), price)
        # All scalar task fields are affine over a box. Valid corners imply
        # every intermediate integer task meets the same registered bounds.
        for x in (0, horizon):
            for y in (0, horizon):
                out.at((x, y))
        return out

    def point(self, point):
        if len(point) != 2:
            raise TaskError('invalid plane point')
        return tuple(integer(v, 0, self.horizon) for v in point)

    def at(self, point):
        x, y = self.point(point)
        patch = {'u': [[i, j, self.base.columns[j][i]+x*a+y*b] for i,j,a,b in self.edits],
                 'lambda': self.base.lam+x*self.price[0]+y*self.price[1]}
        return self.base.patch(patch)[0]


@dataclass(frozen=True)
class Vertex:
    ident: int
    point: tuple
    schedule: tuple
    eta: tuple
    den: int
    bound: int
    objective: int

    @classmethod
    def checked(cls, plane, ident, point, schedule, eta, den):
        point = plane.point(point)
        c = verify_certificate(plane.at(point).thaw(), list(schedule), list(eta), den)
        if not c['certified']:
            raise TaskError('unclosed vertex certificate')
        return cls(ident, point, tuple(schedule), tuple(eta), den, c['bound_num'], c['objective'])


def det(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def weights(points, p):
    """Signed barycentric numerators and positive common denominator."""
    if len(points) == 1:
        return ((1,), 1) if points[0] == p else None
    if len(points) == 2:
        a,b = points
        if a == b or det(a,b,p): return None
        k = 0 if a[0] != b[0] else 1
        d = b[k]-a[k]; n = p[k]-a[k]
        if d < 0: d,n = -d,-n
        return (d-n,n),d
    if len(points) == 3:
        a,b,c = points; d=det(a,b,c)
        if not d: return None
        w=(det(p,b,c),det(a,p,c),det(a,b,p))
        if d<0: w=tuple(-v for v in w); d=-d
        return w,d
    raise TaskError('unsupported simplex size')


def mix_eta(vertices, w, d):
    common = lcm(*(v.den for v in vertices))
    den = common*d
    eta = [sum(a*v.eta[i]*(common//v.den) for a,v in zip(w,vertices,strict=True))
           for i in range(len(vertices[0].eta))]
    g = gcd(den,*eta); den //= g; eta=[v//g for v in eta]
    integer(den,1,1_000_000)
    for v in eta: integer(v,-10**15,10**15)
    return eta,den


def certificate(vertices, point, w, d):
    if d<=0 or any(a<0 for a in w) or sum(w)!=d or any(v.schedule!=vertices[0].schedule for v in vertices):
        raise TaskError('invalid convex certificate')
    if any(sum(a*v.point[k] for a,v in zip(w,vertices,strict=True))!=d*point[k] for k in (0,1)):
        raise TaskError('invalid barycentric point')
    common=lcm(*(v.den for v in vertices))
    upper=sum(a*v.bound*(common//v.den) for a,v in zip(w,vertices,strict=True))//(common*d)
    objective,rem=divmod(sum(a*v.objective for a,v in zip(w,vertices,strict=True)),d)
    if rem or objective!=upper: raise TaskError('unclosed interpolated certificate')
    e,den=mix_eta(vertices,w,d)
    return dict(schedule=list(vertices[0].schedule),eta=e,den=den,objective=objective,upper=upper,gap=0,
                proof=dict(vertices=[v.ident for v in vertices],weights=list(w),den=d))


def hull(vertices):
    """Monotone-chain extreme vertices, dropping redundant collinear points."""
    ordered=sorted(vertices,key=lambda v:v.point)
    def half(vs):
        h=[]
        for v in vs:
            while len(h)>1 and det(h[-2].point,h[-1].point,v.point)<=0: h.pop()
            h.append(v)
        return h
    if len(ordered)<=1: return ordered
    return half(ordered)[:-1]+half(reversed(ordered))[:-1]


class Memory:
    def __init__(self, cap=12):
        self.cap=integer(cap,0,64);self.vertices=[];self.groups={}

    def put(self, vertex):
        self.vertices=[v for v in self.vertices if v.point!=vertex.point]
        if self.cap: self.vertices.insert(0,vertex); del self.vertices[self.cap:]
        groups={}
        for v in self.vertices: groups.setdefault(v.schedule,[]).append(v)
        self.groups={k:hull(vs) for k,vs in groups.items()}

    def lookup(self, point):
        for h in self.groups.values():
            # Triangulate the convex hull by a fan; all vertices share a schedule.
            cells=[h] if len(h)<=2 else [(h[0],h[i],h[i+1]) for i in range(1,len(h)-1)]
            for vs in cells:
                v=weights([x.point for x in vs],point)
                if v is not None and min(v[0])>=0:
                    try: return certificate(vs,point,*v)
                    except TaskError: continue
        return None


def verify_mix(plane, point, result, records):
    """Independent arithmetic with Fraction, including full vertex rechecks."""
    point=plane.point(point); p=result['proof']; d=p['den']; ns=p['weights']; ids=p['vertices']
    if type(d) is not int or d<=0 or not 1<=len(ids)==len(ns)<=3 or len(set(ids))!=len(ids):
        raise TaskError('invalid proof format')
    if any(type(n) is not int or n<0 for n in ns) or sum(ns)!=d: raise TaskError('invalid convex weights')
    vs=[records[i] for i in ids]; ws=[Fraction(n,d) for n in ns]
    for v in vs:
        c=verify_certificate(plane.at(v.point).thaw(),list(v.schedule),list(v.eta),v.den)
        if not c['certified'] or c['bound_num']!=v.bound or c['objective']!=v.objective or list(v.schedule)!=result['schedule']:
            raise TaskError('invalid vertex')
    if tuple(sum(w*v.point[k] for w,v in zip(ws,vs,strict=True)) for k in (0,1))!=point:
        raise TaskError('invalid interpolated point')
    for j,e in enumerate(result['eta']):
        if Fraction(e,result['den'])!=sum(w*Fraction(v.eta[j],v.den) for w,v in zip(ws,vs,strict=True)):
            raise TaskError('invalid interpolated multiplier')
    obj=sum(w*v.objective for w,v in zip(ws,vs,strict=True)); ub=sum(w*Fraction(v.bound,v.den) for w,v in zip(ws,vs,strict=True))
    if obj!=result['objective'] or result['upper']!=ub.numerator//ub.denominator or obj!=result['upper'] or result['gap']!=0:
        raise TaskError('invalid interpolated objective')
    c=verify_certificate(plane.at(point).thaw(),result['schedule'],result['eta'],result['den'])
    if not c['certified'] or c['objective']!=obj: raise TaskError('point certificate mismatch')
    return True
