"""SG-26 paired online retention with independent acceptance in the timed path."""
import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import sg07
import sg11
from sg20 import audit, finish, repair
from sg20_delta import State
from sg22_move import correct
from sg25 import merge, prediction
from sg25_regions import Plane, Vertex
from audit_sg25 import chord, reconstruct
from sg26_memory import AcceptedMemory, check_fresh
from datagoblin.training_gym.spectral_blocks import root_vector
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_routing import TaskError

ARMS=('step','lru','area','weighted','area_full')
CAP,H,N,REPEATS=900,8,32,2


@dataclass(frozen=True)
class History:
    point: tuple
    eta: tuple
    den: int


def trajectory(seed,reg,mixed,corners):
    rng=random.Random(seed);q=sg07.task(seed,reg,32,12);q['k']=[7+(seed+i)%2 for i in range(q['T'])]
    js=rng.sample(range(q['d']),4)
    dirs=[[[rng.randrange(q['T']),j,rng.choice((-1,1))] for j in js[2*a:2*a+2]] for a in range(2)]
    price=[1,0] if mixed else [0,0]
    points=[(a,b) for a in range(H+1) for b in range(H+1) if (a,b)!=(0,0)]
    if corners:
        first=[(H,0),(0,H),(H,H)];rng.shuffle(first)
        rest=[(a,b) for a in range(1,H) for b in range(1,H)];rng.shuffle(rest)
        points=first+rest[:N-3]
    else:
        rng.shuffle(points);points=points[:N]
    return q,dirs,price,points


def run(plane,points,tasks,anchor,arm):
    policy='area' if arm=='area_full' else arm
    mem=AcceptedMemory(plane,CAP if arm!='step' else 0,policy if arm!='step' else 'lru')
    if arm!='step' and anchor['certificate']['certified']:mem.admit((0,0),0,anchor,1+4*anchor['lp_calls'])
    history=[History((0,0),tuple(anchor['eta']),anchor['den'])];last_schedule=tuple(anchor['schedule']);rows=[]
    for ident,(point,q) in enumerate(zip(points,tasks,strict=True),1):
        start=time.process_time_ns();lookup=verify_ns=0;calls=0;stage=None;candidate=[];dropped=[];hit=False
        result=None
        if arm!='step':
            tick=time.process_time_ns();result=mem.lookup(point);lookup=time.process_time_ns()-tick
        if result is not None:
            hit=True;tick=time.process_time_ns();mem.accept(point,result,ident,full=arm=='area_full');verify_ns=time.process_time_ns()-tick
            closed=True
        else:
            last=history[-1];s=State.build(q,last.eta,last.den,last_schedule)
            if not s.certificate()['certified']:
                try:
                    e,den=prediction(history,point)
                    p=repair(State.build(q,e,den,s.schedule));candidate.append(dict(schedule=list(p.schedule),eta=list(p.eta),den=p.den,certificate=p.certificate()))
                    if not p.certificate()['certified']:
                        e,den=correct(q,e,den);c=repair(State.build(q,e,den,p.schedule));candidate.append(dict(schedule=list(c.schedule),eta=list(c.eta),den=c.den,certificate=c.certificate()));p=merge(p,c)
                    s=merge(s,p)
                except TaskError:pass
            s,stage=finish(s);calls=stage['lp_calls']
            result=dict(schedule=stage['schedule'],eta=stage['eta'],den=stage['den'],objective=stage['certificate']['objective'],upper=stage['upper'],gap=stage['gap'])
            tick=time.process_time_ns()
            if arm!='step' and s.certificate()['certified'] and not stage['block']:
                _,dropped=mem.admit(point,ident,result,1+4*calls);closed=True
            else:closed=check_fresh(q.thaw(),result,stage)['certified']
            verify_ns=time.process_time_ns()-tick
        if closed:
            history=[*history,History(tuple(point),tuple(result['eta']),result['den'])][-3:]
            last_schedule=tuple(result['schedule'])
        row=dict(**result,point=list(point),ident=ident,hit=hit,lp_calls=calls,stage=stage,candidates=candidate,
                 resident=list(mem.entries),bytes=mem.used,evicted=dropped,lookup_ns=lookup,verify_ns=verify_ns)
        row['cpu_ns']=time.process_time_ns()-start;rows.append(row)
        # Post-timing audit checks every output, including every derived hit.
        c=verify_certificate(q.thaw(),result['schedule'],result['eta'],result['den'])
        if hit:assert c['certified'] and chord(plane,point,result,{i:s.vertex for i,s in mem.entries.items()})
        else:audit(q.thaw(),stage)
        assert c['objective']==result['objective'] and result['gap']>=0
        for z in candidate:assert verify_certificate(q.thaw(),z['schedule'],z['eta'],z['den'])==z['certificate']
    return rows


def summary(roots):
    out={}
    for arm in ARMS:
        groups=[[z for rep in r['arms'][arm] for z in rep] for r in roots];flat=[z for g in groups for z in g]
        times=[statistics.mean(z['cpu_ns'] for z in g)/1e6 for g in groups]
        base=[statistics.mean(z['cpu_ns'] for rep in r['arms']['step'] for z in rep)/1e6 for r in roots]
        lru=[statistics.mean(z['cpu_ns'] for rep in r['arms']['lru'] for z in rep)/1e6 for r in roots]
        hits=[z for z in flat if z['hit']]
        out[arm]=dict(runs=len(flat),cpu_ms=statistics.mean(times),certified=sum(z['gap']==0 for z in flat),
                      gap=statistics.mean(z['gap'] for z in flat),lp_calls=sum(z['lp_calls'] for z in flat),hits=len(hits),
                      interior=sum(len(z['proof']['vertices'])==3 and min(z['proof']['weights'])>0 for z in hits),
                      verify_us=statistics.mean(z['verify_ns'] for z in flat)/1e3,
                      hit_verify_us=statistics.mean(z['verify_ns'] for z in hits)/1e3 if hits else None,
                      lookup_us=statistics.mean(z['lookup_ns'] for z in flat)/1e3,
                      evictions=sum(len(z['evicted']) for z in flat),peak_bytes=max(z['bytes'] for z in flat),
                      mean_bytes=statistics.mean(z['bytes'] for z in flat),wins_step=sum(a<b for a,b in zip(times,base,strict=True)),
                      wins_lru=sum(a<b for a,b in zip(times,lru,strict=True)))
    return out


def campaign(resume=False,limit=24):
    roots=json.loads(Path('sg26_progress.json').read_text()) if resume else [];done={r['seed'] for r in roots}
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(8):
            seed=2610000+j*1000+i
            if seed in done or len(roots)>=limit:continue
            mixed=bool(i%2);corners=bool(i%4<2)
            q,dirs,price,points=trajectory(seed,reg,mixed,corners);plane=Plane.build(q,H,dirs,price);tasks=[plane.at(p) for p in points]
            s,a=finish(State.build(plane.base,root_vector(q),1,sg11.initial(q)));audit(q,a)
            anchor=dict(**a,objective=a['certificate']['objective']);arms={arm:[] for arm in ARMS}
            for rep in range(REPEATS):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for arm in order:arms[arm].append(run(plane,points,tasks,anchor,arm))
                for rr in zip(*(arms[arm][-1] for arm in ARMS),strict=True):
                    assert max(z['objective'] for z in rr)<=min(z['upper'] for z in rr)
                for b,c in zip(arms['area'][-1],arms['area_full'][-1],strict=True):
                    assert all(b[k]==c[k] for k in b if k not in ('cpu_ns','lookup_ns','verify_ns'))
            roots.append(dict(seed=seed,regime=reg,mixed=mixed,corners=corners,base=q,directions=dirs,price=price,points=points,anchor=anchor,arms=arms))
            Path('sg26_progress.tmp').write_text(json.dumps(roots));Path('sg26_progress.tmp').replace('sg26_progress.json')
            print(seed,[(arm,round(v['cpu_ms'],3),v['lp_calls'],v['hits']) for arm,v in summary(roots[-1:]).items()],flush=True)
    mode={f'mixed={m},corners={c}':summary([r for r in roots if r['mixed']==m and r['corners']==c]) for m in (False,True) for c in (False,True) if any(r['mixed']==m and r['corners']==c for r in roots)}
    return dict(version='SG-26/v0.1',horizon=H,cap=CAP,roots=roots,summary=summary(roots),by_mode=mode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int,default=24);a=p.parse_args()
    z=campaign(a.resume,a.limit);Path('sg26_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='roots'},indent=2))
