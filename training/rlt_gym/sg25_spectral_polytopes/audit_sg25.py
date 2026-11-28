"""Independent replay and amortized vertex-verification comparison."""
import json
import random
import statistics
import time
from fractions import Fraction
from pathlib import Path

from sg20 import audit
from sg25_regions import Plane, Vertex
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_quotient import validate
from datagoblin.training_gym.spectral_routing import TaskError


def reconstruct(root, point):
    q=json.loads(json.dumps(root['base']))
    for v,changes in zip(point,root['directions'],strict=True):
        for i,j,s in changes:q['u'][i][j]+=v*s
    q['lambda']+=sum(v*s for v,s in zip(point,root['price'],strict=True))
    return validate(q)


def chord(plane, point, result, verified):
    """Verify a convex proof from previously verified immutable vertices.

    The verified map is constructed only by successful full checks. This
    helper never accepts arbitrary serialized vertex claims as trusted input.
    """
    point=plane.point(point);p=result['proof'];ids,ns,d=p['vertices'],p['weights'],p['den']
    if type(d) is not int or d<=0 or not 1<=len(ids)==len(ns)<=3 or len(set(ids))!=len(ids):
        raise TaskError('invalid proof')
    if any(type(i) is not int or i not in verified for i in ids) or any(type(n) is not int or n<0 for n in ns) or sum(ns)!=d:
        raise TaskError('invalid vertex references or weights')
    vs=[verified[i] for i in ids];w=[Fraction(n,d) for n in ns]
    if any(list(v.schedule)!=result['schedule'] for v in vs):raise TaskError('schedule mismatch')
    if any(sum(a*v.point[k] for a,v in zip(w,vs,strict=True))!=point[k] for k in (0,1)):
        raise TaskError('point mismatch')
    if type(result['den']) is not int or not 1<=result['den']<=1_000_000 or len(result['eta'])!=plane.base.t or any(type(e) is not int or abs(e)>10**15 for e in result['eta']):
        raise TaskError('invalid multiplier representation')
    for i,e in enumerate(result['eta']):
        if Fraction(e,result['den'])!=sum(a*Fraction(v.eta[i],v.den) for a,v in zip(w,vs,strict=True)):
            raise TaskError('multiplier mismatch')
    obj=sum(a*v.objective for a,v in zip(w,vs,strict=True));ub=sum(a*Fraction(v.bound,v.den) for a,v in zip(w,vs,strict=True))
    if obj.denominator!=1 or obj!=result['objective'] or result['upper']!=ub.numerator//ub.denominator or result['gap']!=0 or obj!=result['upper']:
        raise TaskError('invalid chord gap')
    return True


def replay(z):
    counts=dict(roots=0,anchors=0,queries=0,final=0,stages=0,candidates=0,hits=0,strict_triangles=0,pairs=0,blocks=0)
    seen=set();timings=[]
    for root in z['roots']:
        plane=Plane.build(root['base'],z['horizon'],root['directions'],root['price']);a=root['anchor'];audit(root['base'],a)
        init=Vertex.checked(plane,0,(0,0),a['schedule'],a['eta'],a['den']);counts['anchors']+=1;counts['roots']+=1
        tasks=[reconstruct(root,p) for p in root['points']]
        for p,q in zip(root['points'],tasks,strict=True):
            assert plane.at(p).thaw()==q
            sig=json.dumps(q,sort_keys=True);assert sig not in seen;seen.add(sig);counts['queries']+=1
        root_times=[]
        for arm,reps in root['arms'].items():
            for rep,rows in enumerate(reps):
                resident={0:init} if arm!='step' else {}
                for p,q,r in zip(root['points'],tasks,rows,strict=True):
                    ident=r['ident'];assert r['point']==p and r['gap']==0
                    c=verify_certificate(q,r['schedule'],r['eta'],r['den']);assert c['objective']==r['objective']==r['upper']
                    if r['hit']:
                        assert c['certified'] and chord(plane,p,r,resident);counts['hits']+=1
                        counts['strict_triangles']+=len(r['proof']['vertices'])==3 and min(r['proof']['weights'])>0
                        if arm=='region':
                            # Compare independent checking of a single query, using
                            # already accepted vertices, not rebuilding the task.
                            t={'full':[],'chord':[]}
                            for i in range(5):
                                order=['full','chord'];random.Random(root['seed']+ident*37+i+rep*99).shuffle(order)
                                for name in order:
                                    start=time.process_time_ns()
                                    if name=='full':assert verify_certificate(q,r['schedule'],r['eta'],r['den'])['certified']
                                    else:assert chord(plane,p,r,resident)
                                    t[name].append(time.process_time_ns()-start)
                            root_times.append({k:statistics.median(v) for k,v in t.items()})
                    else:
                        stage=r['stage'];audit(q,stage);counts['stages']+=len(stage['log']);counts['blocks']+=stage['block'] is not None
                        assert stage['certificate']['objective']==r['objective'] and stage['upper']==r['upper']
                        if arm!='step' and c['certified']:
                            resident[ident]=Vertex(ident,tuple(p),tuple(r['schedule']),tuple(r['eta']),r['den'],c['bound_num'],c['objective'])
                            while len(resident)>z['cap']:del resident[next(iter(resident))]
                    assert len(resident)==r['resident']
                    for c in r['candidates']:
                        assert verify_certificate(q,c['schedule'],c['eta'],c['den'])==c['certificate'];counts['candidates']+=1
                    counts['final']+=1
                    if arm=='region':
                        ref=root['arms']['recompute'][rep][ident-1];assert all(ref[k]==r[k] for k in r if k not in ('cpu_ns','lookup_ns','audit_ns'));counts['pairs']+=1
        timings.append(dict(seed=root['seed'],rows=root_times))
    flat=[v for r in timings for v in r['rows']]
    ms={k:statistics.mean(v[k] for v in flat)/1000 for k in ('full','chord')}
    return dict(counts=counts,verification_us=ms,verification_hits=len(flat),timings=timings)


if __name__=='__main__':
    z=replay(json.loads(Path('sg25_results.json').read_text()));Path('audit.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='timings'},indent=2))
