"""SG-24: paired continuation with binary versus exact event boundaries."""
import json
import random
import statistics
import time
from pathlib import Path

import sg11
from sg20 import audit, finish, repair
from sg20_delta import State
from sg21_ray import Ray
from sg22 import H, trajectory
from sg22_move import Line, correct, region as binary_region, value, verify_region
from sg23 import merge, record
from sg24_events import region as event_region
from datagoblin.training_gym.spectral_blocks import root_vector
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_routing import TaskError

ARMS = ('correct', 'binary', 'events')


def run(ray,tasks,initial,arm):
    state=initial;anchor=0;history=[(0,initial.eta,initial.den)];previous=initial.schedule
    proofs=[];rows=[];proof=None
    for step,q in enumerate(tasks,1):
        tick=time.process_time_ns();planning=0;gate=0;calls=0;candidates=[];reused=False
        if proof is not None and step<=proofs[proof]['end']:
            r=value(proofs[proof],step);r['proof']=proof;reused=True
        else:
            proof=None
            if anchor!=step-1:
                old=rows[-1]
                state=State.build(ray.at(step),old['eta'],old['den'],old['schedule'])
            else:state,_,_=state.apply(ray.patch(step))
            if not state.certificate()['certified'] and len(history)==2:
                try:
                    e,den=Line.secant(*history).at(step)
                    p=repair(State.build(state.q,e,den,state.schedule));candidates.append(record(p,'extrap'))
                    if not p.certificate()['certified']:
                        e,den=correct(state.q,e,den)
                        c=repair(State.build(state.q,e,den,p.schedule));candidates.append(record(c,'correct'))
                        p=merge(p,c)
                    state=merge(state,p)
                except TaskError:pass
            state,z=finish(state);anchor=step;calls=z['lp_calls']
            r=dict(schedule=z['schedule'],eta=z['eta'],den=z['den'],objective=z['certificate']['objective'],upper=z['upper'],gap=z['gap'],full=z)
            # Common gate: equal consecutive schedules and a certified next-step secant.
            if arm!='correct' and history and tuple(previous)==state.schedule and ray.horizon-step>=2 and state.certificate()['certified']:
                t=time.process_time_ns()
                try:
                    line=Line.secant(history[-1],(step,state.eta,state.den));e,den=line.at(step+1)
                    gate=1
                    if State.build(ray.at(step+1),e,den,state.schedule).certificate()['certified']:
                        p=(event_region if arm=='events' else binary_region)(ray,state,step,line)
                        proofs.append(p)
                        if p['end']>step:proof=len(proofs)-1
                except TaskError:pass
                planning=time.process_time_ns()-t
        elapsed=time.process_time_ns()-tick
        c=verify_certificate(q,r['schedule'],r['eta'],r['den'])
        assert c['objective']==r['objective'] and r['gap']>=0
        if reused:assert c['certified'] and r['upper']==c['integer_upper']
        else:audit(q,r['full'])
        for z in candidates:assert verify_certificate(q,z['schedule'],z['eta'],z['den'])==z['certificate']
        history=([*history,(step,tuple(r['eta']),r['den'])][-2:] if c['certified'] else [])
        previous=tuple(r['schedule'])
        rows.append(dict(**r,step=step,cpu_ns=elapsed,planning_ns=planning,gate=gate,reused=reused,lp_calls=calls,candidates=candidates))
    for p in proofs:assert verify_region(ray,p)
    return dict(rows=rows,proofs=proofs)


def summarize(roots):
    out={}
    for arm in ARMS:
        groups=[[z for rep in r['arms'][arm] for z in rep['rows']] for r in roots];flat=[z for g in groups for z in g]
        times=[statistics.mean(z['cpu_ns'] for z in g)/1e6 for g in groups]
        base=[statistics.mean(z['cpu_ns'] for rep in r['arms']['correct'] for z in rep['rows'])/1e6 for r in roots]
        proofs=[p for r in roots for rep in r['arms'][arm] for p in rep['proofs']]
        out[arm]=dict(runs=len(flat),certified=sum(z['gap']==0 for z in flat),mean_gap=statistics.mean(z['gap'] for z in flat),
                      cpu_ms=statistics.mean(times),planning_us=statistics.mean(z['planning_ns'] for z in flat)/1e3,
                      lp_calls=sum(z['lp_calls'] for z in flat),gates=sum(z['gate'] for z in flat),reused=sum(z['reused'] for z in flat),
                      proofs=len(proofs),probes=sum(p['probes'] for p in proofs),lines=sum(p.get('lines',0) for p in proofs),
                      crossings=sum(p.get('crossing') is not None for p in proofs),max_span=max((p['end']-p['start'] for p in proofs),default=0),
                      wins=sum(a<b for a,b in zip(times,base,strict=True)))
    return out


def campaign():
    roots=[]
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(8):
            seed=2410000+j*1000+i;q,end=trajectory(seed,reg,bool(i%2));ray=Ray.build(q,end,H)
            tasks=[ray.at(s).thaw() for s in range(1,H+1)]
            initial,receipt=finish(State.build(ray.base,root_vector(q),1,sg11.initial(q)));audit(q,receipt)
            assert initial.certificate()['certified']
            arms={a:[] for a in ARMS}
            for rep in range(2):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for arm in order:arms[arm].append(run(ray,tasks,initial,arm))
                a,b,c=(arms[arm][-1] for arm in ARMS)
                for x,y,z in zip(a['rows'],b['rows'],c['rows'],strict=True):
                    assert max(r['objective'] for r in (x,y,z))<=min(r['upper'] for r in (x,y,z))
                    assert all(y[k]==z[k] for k in ('objective','upper','schedule','eta','den','gap','lp_calls','reused'))
                assert len(b['proofs'])==len(c['proofs'])
                for x,y in zip(b['proofs'],c['proofs'],strict=True):
                    assert all(x[k]==y[k] for k in x if k!='probes')
            roots.append(dict(seed=seed,regime=reg,mixed=bool(i%2),base=q,end=end,initial=receipt,arms=arms))
            Path('sg24_progress.tmp').write_text(json.dumps(roots));Path('sg24_progress.tmp').replace('sg24_progress.json')
            print(seed,[(a,round(v['cpu_ms'],3),v['lp_calls'],v['reused']) for a,v in summarize(roots[-1:]).items()],flush=True)
    return dict(version='SG-24/v0.1',horizon=H,roots=roots,summary=summarize(roots),
                by_mode={str(m):summarize([r for r in roots if r['mixed']==m]) for m in (False,True)})


if __name__=='__main__':
    z=campaign();Path('sg24_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='roots'},indent=2))
