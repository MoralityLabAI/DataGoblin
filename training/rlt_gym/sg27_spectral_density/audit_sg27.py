"""Full receipt replay and independent density-retention decision audit."""
import json
from collections import Counter, deque
from itertools import combinations
from pathlib import Path

from audit_sg26 import replay
from sg25_regions import Vertex, Plane, weights
from sg26_memory import eviction_key, payload


def covered(vertices, point):
    # Exhaustive simplex membership, independent of hull containment.
    for n in range(1,min(3,len(vertices))+1):
        for vs in combinations(vertices,n):
            w=weights([v.point for v in vs],point)
            if w is not None and min(w[0])>=0:return True
    return False


def replay_density(z):
    checks=evictions=blocks=0
    for root in z['roots']:
        p=Plane.build(root['base'],z['horizon'],root['directions'],root['price']);a=root['anchor']
        if a['block']:
            assert a['block']['receipt']['event_count']<=z['block_cap']
            blocks+=1
        for arm,reps in root['arms'].items():
            for rows in reps:
                for row in rows:
                    expected=row['bytes']+(64 if arm=='density' else 0)
                    assert row['charged_bytes']==expected<=z['cap'];checks+=1
                    if row['stage'] and row['stage']['block']:
                        assert row['stage']['block']['receipt']['event_count']<=z['block_cap'];blocks+=1
        for rows in root['arms']['density']:
            resident={};cost={};clock={};history=deque(maxlen=8)
            if a['certificate']['certified']:
                resident[0]=Vertex.checked(p,0,(0,0),a['schedule'],a['eta'],a['den']);cost[0]=1+4*a['lp_calls'];clock[0]=0
            for row in rows:
                pt=tuple(row['point']);ident=row['ident'];history.append(pt)
                if row['hit']:
                    for i in row['proof']['vertices']:clock[i]=ident
                elif row['stage']['certificate']['certified'] and not row['stage']['block']:
                    resident[ident]=Vertex.checked(p,ident,pt,row['schedule'],row['eta'],row['den']);cost[ident]=1+4*row['lp_calls'];clock[ident]=ident
                lost=[]
                while sum(len(payload(v,cost[i])) for i,v in resident.items())>z['cap']-64:
                    groups={}
                    for v in resident.values():groups.setdefault(v.schedule,[]).append(v)
                    density=Counter()
                    for x,y in history:
                        for dx,dy,w in [(0,0,2),(0,1,1),(0,-1,1),(1,0,1),(-1,0,1)]:
                            xx,yy=x+dx,y+dy
                            if 0<=xx<=p.horizon and 0<=yy<=p.horizon:density[xx,yy]+=w
                    before={pt for pt in density if any(covered(vs,pt) for vs in groups.values())}
                    def rank(i):
                        v=resident[i];peers=groups[v.schedule]
                        loss=sum(density[pt] for pt in before if not any(covered([u for u in vs if u.ident!=i],pt) for vs in groups.values()))
                        return (loss,*eviction_key(v,peers,1),clock[i],i)
                    i=min(resident,key=rank);del resident[i];del cost[i];del clock[i];lost.append(i);evictions+=1
                assert lost==row['evicted'] and list(resident)==row['resident']
    return dict(charged_bytes_checks=checks,density_evictions=evictions,capped_block_receipts=blocks)


if __name__=='__main__':
    z=json.loads(Path('sg27_results.json').read_text())
    out=dict(receipts=replay(z),density=replay_density(z))
    Path('audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
