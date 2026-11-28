"""SG-27: online density-aware retention with verified bounded vertex memory."""
import argparse
import json
import random
import statistics
from collections import Counter, deque
from pathlib import Path
from unittest.mock import patch

import sg26
import sg20
from sg20 import audit, finish
from sg20_delta import State
from sg25_regions import Plane, Vertex, det, hull
from sg26_memory import AcceptedMemory, eviction_key, groups, payload
from datagoblin.training_gym.spectral_blocks import BlockSearch, root_vector
from datagoblin.training_gym.spectral_routing import TaskError
import sg11

ARMS = ('step', 'lru', 'area', 'weighted', 'density')
CAP, HISTORY, REPEATS, BLOCK_CAP = 964, 8, 2, 256


def contains(h, p):
    if not h:
        return False
    if len(h) == 1:
        return h[0].point == p
    if len(h) == 2:
        a, b = (v.point for v in h)
        return det(a, b, p) == 0 and all(min(x, y) <= z <= max(x, y) for x, y, z in zip(a,b,p,strict=True))
    return all(det(a.point,b.point,p) >= 0 for a,b in zip(h,h[1:]+h[:1],strict=True))


def samples(history, horizon):
    """Fixed, causal five-point stencil on the last eight observed positions."""
    out = Counter()
    for x,y in history:
        for dx,dy,w in ((0,0,2),(1,0,1),(-1,0,1),(0,1,1),(0,-1,1)):
            a,b = x+dx,y+dy
            if 0 <= a <= horizon and 0 <= b <= horizon:
                out[a,b] += w
    return out


class DensityMemory(AcceptedMemory):
    """Reserve 64 serialized bytes for eight int32 point pairs; no future order."""
    def __init__(self, plane, cap):
        if cap < 8*HISTORY:
            raise TaskError('density budget too small')
        super().__init__(plane, cap-8*HISTORY, 'area')
        self.history = deque(maxlen=HISTORY)

    def lookup(self, point):
        point = self.plane.point(point)
        self.history.append(point)
        return super().lookup(point)

    def victim(self):
        by = groups([s.vertex for s in self.entries.values()])
        density = samples(self.history,self.plane.horizon)
        old_hulls = {k:hull(vs) for k,vs in by.items()}
        before = {k:{p for p in density if contains(h,p)} for k,h in old_hulls.items()}
        # Credit coverage only when no other schedule group covers the sample.
        counts = Counter(p for pts in before.values() for p in pts)
        def key(i):
            v = self.entries[i].vertex
            peers = by[v.schedule]
            after = hull([u for u in peers if u.ident != i])
            lost = sum(density[p] for p in before[v.schedule] if counts[p]==1 and not contains(after,p))
            return (lost,*eviction_key(v,peers,1),self.clock[i],i)
        return min(self.entries,key=key)

    def admit(self,point,ident,result,cost):
        # Reuse the original independent admission and deduplication. Give it
        # room for one transient vertex, then apply our own bounded eviction.
        cap=self.cap
        self.cap=cap+36+12*self.plane.base.t
        old_peak=self.peak
        try:
            v,dropped=super().admit(point,ident,result,cost)
        finally:
            self.cap=cap
            self.peak=old_peak
        while self.used>cap:
            i=self.victim();self.used-=len(self.entries[i].wire)
            del self.entries[i];del self.clock[i];dropped.append(i)
        self.peak=max(self.peak,self.used);self.rebuild()
        return v,dropped


def trajectory(seed,reg,mixed,drift):
    q,dirs,price,_=sg26.trajectory(seed,reg,mixed,False)
    r=random.Random(seed+271);h=sg26.H
    available=[(x,y) for x in range(h+1) for y in range(h+1) if (x,y)!=(0,0)]
    a=(r.choice((1,2,3)),r.choice((1,2,3)))
    b=(h-a[0],h-a[1])
    points=[]
    for i in range(sg26.N):
        center=b if drift and i>=sg26.N//2 else a
        w=[1/(1+abs(x-center[0])+abs(y-center[1]))**3 for x,y in available]
        p=r.choices(available,weights=w,k=1)[0]
        available.remove(p);points.append(p)
    return q,dirs,price,points


def capped_search(task,initial,**kw):
    kw['event_cap']=min(BLOCK_CAP,kw.get('event_cap',BLOCK_CAP))
    return BlockSearch(task,initial,**kw)


def run(plane,points,tasks,anchor,arm):
    def factory(p,cap,policy):
        return DensityMemory(p,cap) if arm=='density' else AcceptedMemory(p,cap,policy)
    with patch.object(sg26,'CAP',CAP),patch.object(sg26,'AcceptedMemory',factory),patch.object(sg20,'BlockSearch',capped_search):
        rows=sg26.run(plane,points,tasks,anchor,arm)
    for row in rows:
        row['charged_bytes']=row['bytes']+(8*HISTORY if arm=='density' else 0)
        assert row['charged_bytes']<=CAP
    return rows


def summary(roots):
    out={}
    for arm in ARMS:
        by=[[z for rep in r['arms'][arm] for z in rep] for r in roots];flat=[z for g in by for z in g]
        times=[statistics.mean(z['cpu_ns'] for z in g)/1e6 for g in by]
        base=[statistics.mean(z['cpu_ns'] for rep in r['arms']['lru'] for z in rep)/1e6 for r in roots]
        out[arm]=dict(runs=len(flat),closed=sum(z['gap']==0 for z in flat),mean_gap=statistics.mean(z['gap'] for z in flat),
                      cpu_ms=statistics.mean(times),hits=sum(z['hit'] for z in flat),lp_calls=sum(z['lp_calls'] for z in flat),
                      peak_bytes=max(z['charged_bytes'] for z in flat),wins_lru=sum(a<b for a,b in zip(times,base,strict=True)),
                      evictions=sum(len(z['evicted']) for z in flat))
    return out


def campaign(resume=False,limit=12):
    roots=json.loads(Path('sg27_progress.json').read_text()) if resume else []
    done={r['seed'] for r in roots}
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(4):
            seed=2710000+j*1000+i
            if seed in done or len(roots)>=limit:continue
            mixed=bool(i%2);drift=bool(i//2)
            q,dirs,price,points=trajectory(seed,reg,mixed,drift)
            plane=Plane.build(q,sg26.H,dirs,price);tasks=[plane.at(p) for p in points]
            with patch.object(sg20,'BlockSearch',capped_search):
                _,a=finish(State.build(plane.base,root_vector(q),1,sg11.initial(q)))
            audit(q,a)
            anchor=dict(**a,objective=a['certificate']['objective']);arms={arm:[] for arm in ARMS}
            for rep in range(REPEATS):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for arm in order:arms[arm].append(run(plane,points,tasks,anchor,arm))
                for rr in zip(*(arms[arm][-1] for arm in ARMS),strict=True):
                    assert max(z['objective'] for z in rr)<=min(z['upper'] for z in rr)
            roots.append(dict(seed=seed,regime=reg,mixed=mixed,drift=drift,base=q,directions=dirs,price=price,points=points,anchor=anchor,arms=arms))
            Path('sg27_progress.tmp').write_text(json.dumps(roots));Path('sg27_progress.tmp').replace('sg27_progress.json')
            print(seed,summary(roots[-1:]),flush=True)
    return dict(version='SG-27/v0.1',horizon=sg26.H,cap=CAP,block_cap=BLOCK_CAP,roots=roots,summary=summary(roots),
                by_drift={str(d):summary([r for r in roots if r['drift']==d]) for d in (False,True) if any(r['drift']==d for r in roots)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int,default=12);a=p.parse_args()
    z=campaign(a.resume,a.limit);Path('sg27_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='roots'},indent=2))
