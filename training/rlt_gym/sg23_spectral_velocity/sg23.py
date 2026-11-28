"""Fresh-root SG-23 velocity cone comparison with corrected extrapolation."""
import argparse
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
from sg22_move import Line, correct
from sg23_cone import Cone, endpoint
from datagoblin.training_gym.spectral_blocks import root_vector
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_routing import TaskError

ARMS=('correct','clamp','sweep','cone_lp','lp_every')


def record(s,kind):
    return dict(kind=kind,schedule=list(s.schedule),eta=list(s.eta),den=s.den,certificate=s.certificate())


def merge(a,b):
    best=b.schedule if b.objective>a.objective else a.schedule
    return (b if b.bound*a.den<a.bound*b.den else a).with_schedule(best)


def run(ray,tasks,initial,arm):
    state=initial;history=[(0,state.eta,state.den)] if state.certificate()['certified'] else []
    rows=[]
    for step,q in enumerate(tasks,1):
        tick=time.process_time_ns();old=state
        state,_,_=state.apply(ray.patch(step));candidate=[];proposal=None;info=None;accepted=False
        if arm in ('clamp','sweep','cone_lp') and not state.certificate()['certified'] and old.certificate()['exact_face']:
            try:
                guess=[0]*old.q.t
                if len(history)==2:
                    try:
                        line=Line.secant(*history)
                        guess=[round(v/line.den) for v in line.slope]
                    except TaskError:pass
                c=Cone(old,state.q);p,info=c.propose(guess,'lp' if arm=='cone_lp' else arm)
                if p is not None:
                    v,den=p;assert c.gap(v,den)==0
                    prediction=endpoint(old,state.q,v,den)
                    raw=prediction.certificate();prediction=repair(prediction)
                    candidate.append(record(prediction,'cone'))
                    proposal=dict(velocity=v,den=den,raw=raw)
                    if prediction.certificate()['certified']:
                        state=merge(state,prediction);accepted=True
            except TaskError:
                info=dict(status='representation_cap',lp=0)
        # On failure, the same corrected extrapolation path, from the same
        # state as the control. Failed cone candidates are not retained.
        if arm!='lp_every' and not state.certificate()['certified'] and len(history)==2:
            try:
                e,den=Line.secant(*history).at(step)
                prediction=repair(State.build(state.q,e,den,state.schedule));candidate.append(record(prediction,'extrap'))
                if not prediction.certificate()['certified']:
                    e,den=correct(state.q,e,den)
                    c=repair(State.build(state.q,e,den,prediction.schedule));candidate.append(record(c,'correct'))
                    prediction=merge(prediction,c)
                state=merge(state,prediction)
            except TaskError:pass
        state,r=finish(state,force_lp=arm=='lp_every');elapsed=time.process_time_ns()-tick
        audit(q,r)
        for z in candidate:assert verify_certificate(q,z['schedule'],z['eta'],z['den'])==z['certificate']
        c=r['certificate'];history=([*history,(step,state.eta,state.den)][-2:] if c['certified'] else [])
        rows.append(dict(full=r,step=step,cpu_ns=elapsed,objective=c['objective'],upper=r['upper'],gap=r['gap'],
                         lp_calls=r['lp_calls'],cone=info,proposal=proposal,accepted=accepted,candidates=candidate))
    return rows


def summary(roots):
    out={}
    for arm in ARMS:
        groups=[[z for rep in r['arms'][arm] for z in rep] for r in roots];flat=[z for g in groups for z in g]
        times=[statistics.mean(z['cpu_ns'] for z in g)/1e6 for g in groups]
        base=[statistics.mean(z['cpu_ns'] for rep in r['arms']['correct'] for z in rep)/1e6 for r in roots]
        out[arm]=dict(runs=len(flat),cpu_ms=statistics.mean(times),certified=sum(z['gap']==0 for z in flat),
                      gap=statistics.mean(z['gap'] for z in flat),lp_calls=sum(z['lp_calls'] for z in flat),
                      cone_attempts=sum(z['cone'] is not None for z in flat),cone_lp=sum((z['cone'] or {}).get('lp',0) for z in flat),
                      directions=sum(z['proposal'] is not None for z in flat),accepted=sum(z['accepted'] for z in flat),
                      raw_closures=sum(z['proposal'] is not None and z['proposal']['raw']['certified'] for z in flat),
                      wins=sum(a<b for a,b in zip(times,base,strict=True)))
    return out


def campaign(resume=False,limit=24):
    roots=json.loads(Path('sg23_progress.json').read_text()) if resume else [];done={r['seed'] for r in roots}
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(8):
            seed=2310000+j*1000+i
            if seed in done:continue
            if len(roots)>=limit:break
            q,end=trajectory(seed,reg,bool(i%2));ray=Ray.build(q,end,H);tasks=[ray.at(s).thaw() for s in range(1,H+1)]
            initial,receipt=finish(State.build(ray.base,root_vector(q),1,sg11.initial(q)));audit(q,receipt)
            arms={a:[] for a in ARMS}
            for rep in range(2):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for arm in order:arms[arm].append(run(ray,tasks,initial,arm))
                for pos in range(H):
                    rr=[arms[a][-1][pos] for a in ARMS]
                    assert max(z['objective'] for z in rr)<=min(z['upper'] for z in rr)
            roots.append(dict(seed=seed,regime=reg,mixed=bool(i%2),base=q,end=end,initial=receipt,arms=arms))
            Path('sg23_progress.tmp').write_text(json.dumps(roots));Path('sg23_progress.tmp').replace('sg23_progress.json')
            print(seed,[(a,round(v['cpu_ms'],3),v['lp_calls'],v['accepted']) for a,v in summary(roots[-1:]).items()],flush=True)
    return dict(version='SG-23/v0.1',horizon=H,roots=roots,summary=summary(roots),
                by_mode={str(m):summary([r for r in roots if r['mixed']==m]) for m in (False,True) if any(r['mixed']==m for r in roots)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int,default=24);a=p.parse_args()
    z=campaign(a.resume,a.limit);Path('sg23_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='roots'},indent=2))
