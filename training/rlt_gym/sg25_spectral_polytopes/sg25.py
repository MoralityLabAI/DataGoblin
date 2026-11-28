"""Paired, online two-dimensional certificate-memory campaign."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import sg07
import sg11
from sg20 import audit, finish, repair
from sg20_delta import State
from sg22_move import correct
from sg25_regions import Memory, Plane, Vertex, mix_eta, verify_mix, weights
from datagoblin.training_gym.spectral_blocks import root_vector
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_routing import TaskError

ARMS = ('step','recompute','region')
CAP,H = 12,6


def trajectory(seed, regime, mixed, corners):
    rng=random.Random(seed)
    q=sg07.task(seed,regime,32,12);q['k']=[7+(seed+i)%2 for i in range(q['T'])]
    # Two independent sparse utility directions. Mixed roots also vary lambda.
    js=rng.sample(range(q['d']),4)
    directions=[[[rng.randrange(q['T']),j,rng.choice((-1,1))] for j in js[2*a:2*a+2]] for a in range(2)]
    price=[1,0] if mixed else [0,0]
    points=[(a,b) for a in range(H+1) for b in range(H+1) if (a,b)!=(0,0)]
    if corners:
        first=[(H,0),(0,H),(H,H)]; interior=[(a,b) for a in range(1,H) for b in range(1,H)]
        rng.shuffle(first);rng.shuffle(interior);points=first+interior[:21]
    else:
        rng.shuffle(points);points=points[:24]
    return q,directions,price,points


def prediction(history, point):
    for size in (3,2):
        if len(history)<size:continue
        vs=history[-size:];w=weights([v.point for v in vs],point)
        if w is not None:
            try:return mix_eta(vs,*w)
            except TaskError:pass
    return list(history[-1].eta),history[-1].den


def merge(a,b):
    best=b.schedule if b.objective>a.objective else a.schedule
    return (b if b.bound*a.den<a.bound*b.den else a).with_schedule(best)


def run(plane, points, tasks, initial, arm):
    history=[initial];mem=Memory(CAP);records={initial.ident:initial};rows=[]
    if arm!='step':mem.put(initial)
    for ident,(point,q) in enumerate(zip(points,tasks,strict=True),1):
        start=time.process_time_ns();lookup_ns=0;calls=0;stage=None;candidate=[];hit=False
        result=None
        if arm!='step':
            tick=time.process_time_ns();result=mem.lookup(point);lookup_ns=time.process_time_ns()-tick
        if result is not None:
            hit=True
            if arm=='recompute':
                s=State.build(q,result['eta'],result['den'],result['schedule']);c=s.certificate()
                assert c['certified'] and c['objective']==result['objective']
        else:
            last=history[-1];s=State.build(q,last.eta,last.den,last.schedule)
            if not s.certificate()['certified']:
                try:
                    e,den=prediction(history,point)
                    p=repair(State.build(q,e,den,s.schedule));candidate.append(dict(schedule=list(p.schedule),eta=list(p.eta),den=p.den,certificate=p.certificate()))
                    if not p.certificate()['certified']:
                        e,den=correct(q,e,den);c=repair(State.build(q,e,den,p.schedule));candidate.append(dict(schedule=list(c.schedule),eta=list(c.eta),den=c.den,certificate=c.certificate()));p=merge(p,c)
                    s=merge(s,p)
                except TaskError:pass
            s,stage=finish(s);calls=stage['lp_calls'];result=dict(schedule=stage['schedule'],eta=stage['eta'],den=stage['den'],objective=stage['certificate']['objective'],upper=stage['upper'],gap=stage['gap'])
            if arm!='step' and s.certificate()['certified']:
                v=Vertex.checked(plane,ident,point,s.schedule,s.eta,s.den);mem.put(v);records[ident]=v
        elapsed=time.process_time_ns()-start
        tick=time.process_time_ns()
        if hit:assert verify_mix(plane,point,result,records)
        else:audit(q.thaw(),stage)
        c=verify_certificate(q.thaw(),result['schedule'],result['eta'],result['den'])
        for z in candidate:assert verify_certificate(q.thaw(),z['schedule'],z['eta'],z['den'])==z['certificate']
        assert c['objective']==result['objective'] and result['gap']>=0
        # History consists of recent certified outputs; region memory admits
        # only independently checked fresh vertices, not recursively derived hits.
        history=([*history,Vertex(ident,point,tuple(result['schedule']),tuple(result['eta']),result['den'],c['bound_num'],c['objective'])][-3:] if c['certified'] else [history[-1]])
        rows.append(dict(**result,point=list(point),ident=ident,cpu_ns=elapsed,lookup_ns=lookup_ns,audit_ns=time.process_time_ns()-tick,
                         hit=hit,lp_calls=calls,stage=stage,candidates=candidate,resident=len(mem.vertices)))
    return rows


def summary(roots):
    out={}
    for arm in ARMS:
        groups=[[z for rep in r['arms'][arm] for z in rep] for r in roots];flat=[z for g in groups for z in g]
        times=[statistics.mean(z['cpu_ns'] for z in g)/1e6 for g in groups]
        base=[statistics.mean(z['cpu_ns'] for rep in r['arms']['step'] for z in rep)/1e6 for r in roots]
        out[arm]=dict(runs=len(flat),cpu_ms=statistics.mean(times),certified=sum(z['gap']==0 for z in flat),gap=statistics.mean(z['gap'] for z in flat),
                      lp_calls=sum(z['lp_calls'] for z in flat),hits=sum(z['hit'] for z in flat),
                      interior=sum(z['hit'] and len(z['proof']['vertices'])==3 and min(z['proof']['weights'])>0 for z in flat),
                      lookup_us=statistics.mean(z['lookup_ns'] for z in flat)/1e3,audit_ms=statistics.mean(z['audit_ns'] for z in flat)/1e6,
                      peak=max(z['resident'] for z in flat),wins=sum(a<b for a,b in zip(times,base,strict=True)))
    return out


def campaign(resume=False,limit=24):
    roots=json.loads(Path('sg25_progress.json').read_text()) if resume else [];done={r['seed'] for r in roots}
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(8):
            seed=2510000+j*1000+i
            if seed in done or len(roots)>=limit:continue
            mixed=bool(i%2);corners=bool(i%4<2)
            q,dirs,price,points=trajectory(seed,reg,mixed,corners);plane=Plane.build(q,H,dirs,price);tasks=[plane.at(p) for p in points]
            s,a=finish(State.build(plane.base,root_vector(q),1,sg11.initial(q)));audit(q,a)
            initial=Vertex.checked(plane,0,(0,0),s.schedule,s.eta,s.den);arms={arm:[] for arm in ARMS}
            for rep in range(2):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for arm in order:arms[arm].append(run(plane,points,tasks,initial,arm))
                for a0,b,c in zip(*(arms[arm][-1] for arm in ARMS),strict=True):
                    assert max(z['objective'] for z in (a0,b,c))<=min(z['upper'] for z in (a0,b,c))
                    assert all(b[k]==c[k] for k in b if k not in ('cpu_ns','lookup_ns','audit_ns'))
            roots.append(dict(seed=seed,regime=reg,mixed=mixed,corners=corners,base=q,directions=dirs,price=price,points=points,anchor=a,arms=arms))
            Path('sg25_progress.tmp').write_text(json.dumps(roots));Path('sg25_progress.tmp').replace('sg25_progress.json')
            print(seed,[(a,round(v['cpu_ms'],3),v['lp_calls'],v['hits']) for a,v in summary(roots[-1:]).items()],flush=True)
    groups={f'mixed={m},corners={c}':summary([r for r in roots if r['mixed']==m and r['corners']==c]) for m in (False,True) for c in (False,True) if any(r['mixed']==m and r['corners']==c for r in roots)}
    return dict(version='SG-25/v0.1',horizon=H,cap=CAP,roots=roots,summary=summary(roots),by_mode=groups)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int,default=24);a=p.parse_args()
    z=campaign(a.resume,a.limit);Path('sg25_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='roots'},indent=2))
