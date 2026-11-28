"""SG-21B: sparse-ray gate, evaluated on a separate fresh cohort."""
import json
import random
import statistics
import time
from pathlib import Path

import sg11
from sg20 import audit, finish
from sg20_delta import State
from sg21 import H, MODES, run, trajectory
from sg21_ray import Ray, interval_value, region, verify_region
from datagoblin.training_gym.spectral_blocks import root_vector
from datagoblin.training_gym.spectral_face import verify_certificate

ARMS = ('delta', 'ray', 'gated')


def gated(ray, tasks, initial):
    state=initial; proof=None; proofs=[]; rows=[]
    for step,q in enumerate(tasks,1):
        tick=time.process_time_ns(); planned=calls=0; reused=False
        if proof is not None and step <= proofs[proof]['end']:
            p=proofs[proof];lo,hi=interval_value(p,step);reused=True
            receipt=dict(schedule=p['schedule'],eta=p['eta'],den=p['den'],objective=lo,upper=hi,gap=hi-lo,proof=proof)
        else:
            proof=None
            state,_,_=state.apply(ray.patch(step)); stable=state.certificate()['certified']
            state,z=finish(state);calls=z['lp_calls']
            receipt=dict(schedule=z['schedule'],eta=z['eta'],den=z['den'],objective=z['certificate']['objective'],
                         upper=z['upper'],gap=z['gap'],full=z)
            # A successful ordinary update supplies the next-step evidence.
            # No extra first-step probe, no lookahead on a dense direction.
            if stable and len(ray.terms)<=4 and ray.horizon-step>=2:
                p=region(ray,state,step);planned=p['probes'];proofs.append(p)
                if p['end']>step:proof=len(proofs)-1
        elapsed=time.process_time_ns()-tick
        tick=time.process_time_ns()
        if reused:
            c=verify_certificate(q,receipt['schedule'],receipt['eta'],receipt['den'])
            assert c['certified'] and c['objective']==lo==hi
        else:
            assert state.q.thaw()==q; audit(q,receipt['full'])
        rows.append(dict(**receipt,step=step,cpu_ns=elapsed,audit_ns=time.process_time_ns()-tick,
                         reused=reused,probes=planned,lp_calls=calls))
    for p in proofs:assert verify_region(ray,p)
    return dict(rows=rows,proofs=proofs)


def summary(roots):
    out={}
    for arm in ARMS:
        groups=[[z for rep in r['arms'][arm] for z in rep['rows']] for r in roots]
        flat=[z for a in groups for z in a]
        times=[statistics.mean(z['cpu_ns'] for z in a)/1e6 for a in groups]
        base=[statistics.mean(z['cpu_ns'] for rep in r['arms']['delta'] for z in rep['rows'])/1e6 for r in roots]
        proofs=[p for r in roots for rep in r['arms'][arm] for p in rep['proofs']]
        out[arm]=dict(runs=len(flat),cpu_ms=statistics.mean(times),certified=sum(z['gap']==0 for z in flat),
                      gap=statistics.mean(z['gap'] for z in flat),lp_calls=sum(z['lp_calls'] for z in flat),
                      reused=sum(z['reused'] for z in flat),probes=sum(z['probes'] for z in flat),proofs=len(proofs),
                      nontrivial_proofs=sum(p['end']>p['start'] for p in proofs),
                      wins=sum(a<b for a,b in zip(times,base,strict=True)))
    return out


def campaign():
    roots=[]
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(8):
            seed=2120000+j*1000+i;mode=MODES[i%4];q,end=trajectory(seed,reg,mode)
            ray=Ray.build(q,end,H);tasks=[ray.at(s).thaw() for s in range(1,H+1)]
            initial,z=finish(State.build(ray.base,root_vector(q),1,sg11.initial(q)));audit(q,z)
            arms={a:[] for a in ARMS}
            for rep in range(2):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for arm in order:
                    arms[arm].append(gated(ray,tasks,initial) if arm=='gated' else run(ray,tasks,initial,arm))
                for a,b,c in zip(*(arms[arm][-1]['rows'] for arm in ARMS),strict=True):
                    assert all(a[k]==b[k]==c[k] for k in ('schedule','eta','den','objective','upper','gap'))
            roots.append(dict(seed=seed,regime=reg,mode=mode,base=q,end=end,anchor=z,arms=arms))
            Path('sg21b_progress.json').write_text(json.dumps(roots))
            print(seed,mode,[(a,round(summary(roots[-1:])[a]['cpu_ms'],3)) for a in ARMS],flush=True)
    return dict(version='SG-21B/v0.1',horizon=H,roots=roots,summary=summary(roots),
                by_mode={m:summary([r for r in roots if r['mode']==m]) for m in MODES})


if __name__=='__main__':
    z=campaign();Path('sg21b_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='roots'},indent=2))
