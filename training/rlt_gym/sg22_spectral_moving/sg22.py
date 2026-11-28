"""SG-22 global-price affine multiplier transfer campaign."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import sg07
import sg11
from sg20 import audit, finish, repair
from sg20_delta import State, Task
from sg21_ray import Ray
from sg21 import run as fixed_run
from sg22_move import Line, correct, line_from_proof, region, value, verify_region
from datagoblin.training_gym.spectral_blocks import root_vector
from datagoblin.training_gym.spectral_face import verify_certificate
from datagoblin.training_gym.spectral_routing import TaskError

ARMS=('delta','lp_every','fixed','extrap','correct','moving')
H=16


def trajectory(seed,reg,mixed):
    r=random.Random(seed)
    q=sg07.task(seed,reg,32,12);q['k']=[7+(seed+i)%2 for i in range(q['T'])]
    end=Task.from_task(q).thaw();end['lambda']+=H
    if mixed:
        for j in r.sample(range(q['d']),4):
            end['u'][r.randrange(q['T'])][j]+=r.choice((-1,1))*H
    return q,end


def run(ray,tasks,initial,arm):
    if arm=='fixed':return fixed_run(ray,tasks,initial,'ray')
    state=initial; anchor=0; history=[(0,initial.eta,initial.den)] if initial.certificate()['certified'] else []
    proofs=[];rows=[];proof=None
    for step,q in enumerate(tasks,1):
        tick=time.process_time_ns();planned=calls=probes=corrections=0;reused=False;transfer=False;proposal=None;candidate=[]
        if proof is not None and step<=proofs[proof]['end']:
            r=value(proofs[proof],step);r['proof']=proof;reused=True
        else:
            proof=None
            if anchor!=step-1:
                old=rows[-1]
                state=State.build(ray.at(step),old['eta'],old['den'],old['schedule'])
            else:
                state,_,_=state.apply(ray.patch(step))
            # Retained-state and candidate arithmetic use the same exact verifier.
            if arm in ('extrap','correct','moving') and not state.certificate()['certified'] and len(history)==2:
                try:
                    line=Line.secant(*history);e,den=line.at(step)
                    prediction=State.build(state.q,e,den,state.schedule);probes+=1
                    prediction=repair(prediction)
                    candidate.append(dict(kind='extrap',schedule=list(prediction.schedule),eta=list(prediction.eta),den=prediction.den,certificate=prediction.certificate()))
                    if arm=='correct' and not prediction.certificate()['certified']:
                        e,den=correct(state.q,e,den);corrections+=1
                        c=repair(State.build(state.q,e,den,prediction.schedule))
                        candidate.append(dict(kind='correct',schedule=list(c.schedule),eta=list(c.eta),den=c.den,certificate=c.certificate()))
                        best=c.schedule if c.objective>prediction.objective else prediction.schedule
                        if c.bound*prediction.den<prediction.bound*c.den:prediction=c.with_schedule(best)
                        else:prediction=prediction.with_schedule(best)
                    best=prediction.schedule if prediction.objective>state.objective else state.schedule
                    if prediction.bound*state.den<state.bound*prediction.den:state=prediction.with_schedule(best)
                    else:state=state.with_schedule(best)
                    transfer=state.certificate()['certified']
                    if transfer and all(a*line.den==b*state.den for a,b in zip(state.eta,line.at(step)[0],strict=True)):
                        proposal=line
                except TaskError:
                    # Representation cap: fall back to the ordinary warm engine.
                    pass
            state,z=finish(state,force_lp=arm=='lp_every');anchor=step;calls=z['lp_calls']
            r=dict(schedule=z['schedule'],eta=z['eta'],den=z['den'],objective=z['certificate']['objective'],upper=z['upper'],gap=z['gap'],full=z)
            # Only search forward after a successful one-step extrapolation.
            if arm=='moving' and proposal is not None and ray.horizon-step>=2 and state.certificate()['certified']:
                try:
                    p=region(ray,state,step,proposal);planned=p['probes'];proofs.append(p)
                    if p['end']>step:proof=len(proofs)-1
                except TaskError:
                    pass
        elapsed=time.process_time_ns()-tick
        c=verify_certificate(q,r['schedule'],r['eta'],r['den'])
        assert c['objective']==r['objective'] and r['gap']>=0
        if reused:assert c['certified'] and r['upper']==c['integer_upper']
        else:audit(q,r['full'])
        for z in candidate:assert verify_certificate(q,z['schedule'],z['eta'],z['den'])==z['certificate']
        # Only independently checkable zero-gap dual endpoints enter history.
        history=([*history,(step,tuple(r['eta']),r['den'])][-2:] if c['certified'] else [])
        rows.append(dict(**r,step=step,cpu_ns=elapsed,reused=reused,probes=planned,lp_calls=calls,
                         predictions=probes,corrections=corrections,transfer=transfer,candidates=candidate))
    for p in proofs:assert verify_region(ray,p)
    return dict(rows=rows,proofs=proofs)


def summarize(roots):
    out={}
    for arm in ARMS:
        groups=[[z for rep in r['arms'][arm] for z in rep['rows']] for r in roots]
        flat=[z for group in groups for z in group]
        ms=[statistics.mean(z['cpu_ns'] for z in group)/1e6 for group in groups]
        base=[statistics.mean(z['cpu_ns'] for rep in r['arms']['delta'] for z in rep['rows'])/1e6 for r in roots]
        pp=[p for r in roots for rep in r['arms'][arm] for p in rep['proofs']]
        out[arm]=dict(runs=len(flat),certified=sum(z['gap']==0 for z in flat),gap=statistics.mean(z['gap'] for z in flat),
                      cpu_ms=statistics.mean(ms),lp_calls=sum(z['lp_calls'] for z in flat),reused=sum(z['reused'] for z in flat),
                      probes=sum(z['probes'] for z in flat),predictions=sum(z.get('predictions',0) for z in flat),
                      corrections=sum(z.get('corrections',0) for z in flat),transfers=sum(z.get('transfer',False) for z in flat),
                      regions=len(pp),max_span=max((p['end']-p['start'] for p in pp),default=0),wins=sum(x<y for x,y in zip(ms,base,strict=True)))
    return out


def campaign(resume=False):
    roots=json.loads(Path('sg22_progress.json').read_text()) if resume else []
    done={r['seed'] for r in roots}
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(8):
            seed=2210000+j*1000+i
            if seed in done:continue
            q,end=trajectory(seed,reg,bool(i%2));ray=Ray.build(q,end,H);tasks=[ray.at(s).thaw() for s in range(1,H+1)]
            initial,anchor_receipt=finish(State.build(ray.base,root_vector(q),1,sg11.initial(q)));audit(q,anchor_receipt)
            arms={arm:[] for arm in ARMS}
            for rep in range(2):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for arm in order:arms[arm].append(run(ray,tasks,initial,arm))
                for pos in range(H):
                    rr=[arms[arm][-1]['rows'][pos] for arm in ARMS]
                    assert max(z['objective'] for z in rr)<=min(z['upper'] for z in rr)
            roots.append(dict(seed=seed,regime=reg,mixed=bool(i%2),base=q,end=end,initial=anchor_receipt,arms=arms))
            Path('sg22_progress.json').write_text(json.dumps(roots))
            print(seed,[(a,round(summarize(roots[-1:])[a]['cpu_ms'],3)) for a in ARMS],flush=True)
    return dict(version='SG-22/v0.1',horizon=H,roots=roots,summary=summarize(roots),
                by_mode={str(m):summarize([r for r in roots if r['mixed']==m]) for m in (False,True)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    z=campaign(a.resume);Path('sg22_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='roots'},indent=2))
