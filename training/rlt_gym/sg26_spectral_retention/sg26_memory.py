"""SG-26: bounded verified vertex memory and coverage-aware retention."""
from dataclasses import dataclass
from math import lcm
from struct import pack

from sg25_regions import Vertex, certificate, hull, weights
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_block_replay import verify
from datagoblin.training_gym.spectral_routing import TaskError


def payload(v, cost):
    """Fixed-width serialized vertex and cost proxy; not a Python RSS bound."""
    return pack('<IiiIqqI'+'I'*len(v.schedule)+'q'*len(v.eta),
                v.ident,*v.point,v.den,v.bound,v.objective,cost,*v.schedule,*v.eta)


def measure(h):
    if not h:
        return (0,0,0)
    if len(h)==1:
        return (0,0,1)
    if len(h)==2:
        a,b=(v.point for v in h)
        return (0,(a[0]-b[0])**2+(a[1]-b[1])**2,1)
    area=abs(sum(a.point[0]*b.point[1]-a.point[1]*b.point[0]
                 for a,b in zip(h,h[1:]+h[:1],strict=True)))
    return (area,0,1)


def groups(vertices):
    out={}
    for v in vertices:
        out.setdefault(v.schedule,[]).append(v)
    return out


def eviction_key(v, peers, weight):
    old=measure(hull(peers));new=measure(hull([u for u in peers if u.ident!=v.ident]))
    # Lexicographic dimension-aware loss: positive area first, then segment
    # coverage, then whether a singleton remains. An interior point loses none.
    return (weight*max(0,old[0]-new[0]),weight*max(0,old[1]-new[1]),
            weight*max(0,old[2]-new[2]))


@dataclass(frozen=True)
class Stored:
    vertex: Vertex
    cost: int
    wire: bytes


class AcceptedMemory:
    """One validated plane. Only the independent checker admits fresh vertices.

    The proposal view consists of immutable accepted vertices. Derived hits are
    not inserted. All proof references must be resident at time of acceptance.
    Trusted in-process code, not an adversarial Python sandbox.
    """
    def __init__(self,plane,cap,policy):
        if type(cap) is not int or not 0<=cap<=1_000_000 or policy not in ('lru','area','weighted'):
            raise TaskError('invalid cache configuration')
        self.plane,self.cap,self.policy=plane,cap,policy
        self.entries={};self.clock={};self.hulls={};self.used=0;self.peak=0

    def rebuild(self):
        self.hulls={k:hull(vs) for k,vs in groups([s.vertex for s in self.entries.values()]).items()}

    def admit(self,point,ident,result,cost):
        """Full independent verification, then enforce payload budget."""
        if type(ident) is not int or ident<0 or ident>=2**32 or ident in self.entries:
            raise TaskError('invalid vertex id')
        if type(cost) is not int or not 1<=cost<2**32:
            raise TaskError('invalid cost proxy')
        v=Vertex.checked(self.plane,ident,point,result['schedule'],result['eta'],result['den'])
        if v.objective!=result['objective'] or result['upper']!=v.objective or result['gap']!=0:
            raise TaskError('invalid fresh result')
        wire=payload(v,cost)
        dropped=[]
        if len(wire)>self.cap:return v,dropped
        for i,s in list(self.entries.items()):
            if s.vertex.point==v.point:
                self.used-=len(s.wire);del self.entries[i];del self.clock[i];dropped.append(i)
        self.entries[ident]=Stored(v,cost,wire);self.clock[ident]=ident;self.used+=len(wire)
        while self.used>self.cap:
            if self.policy=='lru':
                victim=min(self.entries,key=lambda i:(self.clock[i],i))
            else:
                by=groups([s.vertex for s in self.entries.values()])
                def key(i):
                    s=self.entries[i];w=s.cost if self.policy=='weighted' else 1
                    return (*eviction_key(s.vertex,by[s.vertex.schedule],w),self.clock[i],i)
                victim=min(self.entries,key=key)
            self.used-=len(self.entries[victim].wire);del self.entries[victim];del self.clock[victim];dropped.append(victim)
        self.peak=max(self.peak,self.used);self.rebuild()
        return v,dropped

    def lookup(self,point):
        for h in self.hulls.values():
            cells=[h] if len(h)<=2 else [(h[0],h[i],h[i+1]) for i in range(1,len(h)-1)]
            for vs in cells:
                w=weights([v.point for v in vs],point)
                if w is not None and min(w[0])>=0:
                    try:return certificate(vs,point,*w)
                    except TaskError:continue
        return None

    def accept(self,point,result,stamp,full=False):
        """Independent integer chord checker using only accepted resident IDs."""
        point=self.plane.point(point);p=result['proof'];ids,ns,d=p['vertices'],p['weights'],p['den']
        if (type(d) is not int or d<=0 or not 1<=len(ids)==len(ns)<=3
                or any(type(i) is not int or i not in self.entries for i in ids)
                or len(set(ids))!=len(ids) or any(type(n) is not int or n<0 for n in ns) or sum(ns)!=d):
            raise TaskError('invalid references or weights')
        vs=[self.entries[i].vertex for i in ids]
        if any(list(v.schedule)!=result['schedule'] for v in vs):raise TaskError('schedule mismatch')
        if any(sum(n*v.point[k] for n,v in zip(ns,vs,strict=True))!=d*point[k] for k in (0,1)):
            raise TaskError('point mismatch')
        den=result['den'];eta=result['eta'];t=self.plane.base.t
        if type(den) is not int or not 1<=den<=1_000_000 or len(eta)!=t or any(type(e) is not int or abs(e)>10**15 for e in eta):
            raise TaskError('invalid multiplier')
        common=lcm(*(v.den for v in vs)); factors=[n*(common//v.den) for n,v in zip(ns,vs,strict=True)]
        for i,e in enumerate(eta):
            if e*common*d!=den*sum(a*v.eta[i] for a,v in zip(factors,vs,strict=True)):
                raise TaskError('multiplier mismatch')
        obj,rem=divmod(sum(n*v.objective for n,v in zip(ns,vs,strict=True)),d)
        upper=sum(a*v.bound for a,v in zip(factors,vs,strict=True))//(common*d)
        if rem or obj!=upper or obj!=result['objective'] or upper!=result['upper'] or result['gap']!=0:
            raise TaskError('unclosed chord')
        if full:
            c=verify_certificate(self.plane.at(point).thaw(),result['schedule'],eta,den)
            if not c['certified'] or c['objective']!=obj:raise TaskError('full certificate mismatch')
        for i in ids:self.clock[i]=stamp
        return True


def check_fresh(q,result,stage):
    c=verify_certificate(q,result['schedule'],result['eta'],result['den'])
    upper=c['integer_upper']
    if stage['block']:
        b=stage['block'];r=b['receipt']
        replay=verify(q,b['initial'],b['extra'],r['events'],event_cap=2048)
        if replay!={k:v for k,v in r.items() if k!='peak_stored_blocks'} or r['schedule']!=result['schedule']:
            raise TaskError('block mismatch')
        upper=min(upper,r['upper'])
    if c['objective']!=result['objective'] or upper!=result['upper'] or upper-c['objective']!=result['gap']:
        raise TaskError('fresh result mismatch')
    return c
