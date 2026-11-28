"""SG-21: paired ray certificates versus ordinary differential updates."""
import json
import random
import statistics
import time
from pathlib import Path

import sg07
import sg11
from sg20 import audit, finish
from sg20_delta import State, Task
from sg21_ray import Ray, interval_value, region, verify_region
from datagoblin.training_gym.spectral_blocks import root_vector
from datagoblin.training_gym.spectral_face import verify_certificate

ARMS = ('delta', 'ray')
MODES = ('one_utility', 'four_utilities', 'bit_price', 'global_price')
H = 24


def trajectory(seed, regime, mode):
    r = random.Random(seed)
    q = sg07.task(seed, regime, 32, 12)
    q['k'] = [7+(seed+i)%2 for i in range(q['T'])]
    end = Task.from_task(q).thaw()
    if mode in MODES[:2]:
        n = 1 if mode=='one_utility' else 4
        for j in r.sample(range(q['d']),n):
            end['u'][r.randrange(q['T'])][j] += r.choice((-1,1))*H
    elif mode=='bit_price':
        end['bits'][r.randrange(q['d'])] += H
    else:
        end['lambda'] += H
    return q,end


def run(ray, tasks, initial, arm):
    state = initial; anchor = 0; proof = None; proofs = []; rows = []
    for step, q in enumerate(tasks,1):
        tick = time.process_time_ns(); planned = 0; calls = 0; reused = False
        if arm=='ray' and proof is None and state.certificate()['certified']:
            p = region(ray,state,anchor)
            planned = p['probes']
            proofs.append(p); proof = len(proofs)-1
        if proof is not None and step <= proofs[proof]['end']:
            p = proofs[proof]; lo,hi = interval_value(p,step); reused = True
            receipt = dict(schedule=p['schedule'],eta=p['eta'],den=p['den'],objective=lo,upper=hi,gap=hi-lo,proof=proof)
        else:
            proof = None
            state,_,_ = state.apply(ray.patch(step))
            state,z = finish(state);anchor = step; calls = z['lp_calls']
            receipt = dict(schedule=z['schedule'],eta=z['eta'],den=z['den'],objective=z['certificate']['objective'],
                           upper=z['upper'],gap=z['gap'],full=z)
        elapsed = time.process_time_ns()-tick
        # Audit every query independently, after stopping the timed interval.
        tick = time.process_time_ns()
        if reused:
            c = verify_certificate(q,receipt['schedule'],receipt['eta'],receipt['den'])
            assert c['certified'] and c['objective']==lo==hi
        else:
            assert state.q.thaw()==q
            audit(q,receipt['full'])
        rows.append(dict(**receipt,step=step,cpu_ns=elapsed,audit_ns=time.process_time_ns()-tick,
                         reused=reused,probes=planned,lp_calls=calls))
    for p in proofs:
        assert verify_region(ray,p)
    return dict(rows=rows,proofs=proofs)


def summarize(roots):
    out = {}
    for arm in ARMS:
        groups = [[z for rep in r['arms'][arm] for z in rep['rows']] for r in roots]
        flat = [z for group in groups for z in group]
        ms = [statistics.mean(z['cpu_ns'] for z in group)/1e6 for group in groups]
        baseline = [statistics.mean(z['cpu_ns'] for rep in r['arms']['delta'] for z in rep['rows'])/1e6 for r in roots]
        pp = [p for r in roots for rep in r['arms'][arm] for p in rep['proofs']]
        out[arm] = dict(runs=len(flat),cpu_ms=statistics.mean(ms),certified=sum(z['gap']==0 for z in flat),
                        gap=statistics.mean(z['gap'] for z in flat),lp_calls=sum(z['lp_calls'] for z in flat),
                        reused=sum(z['reused'] for z in flat),probes=sum(z['probes'] for z in flat),
                        proofs=len(pp),nontrivial_proofs=sum(p['end']>p['start'] for p in pp),
                        max_span=max((p['end']-p['start'] for p in pp),default=0),
                        wins=sum(a<b for a,b in zip(ms,baseline,strict=True)))
    return out


def campaign():
    roots = []
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(8):
            seed=2110000+j*1000+i; mode=MODES[i%4]; q,end=trajectory(seed,reg,mode)
            ray=Ray.build(q,end,H); tasks=[ray.at(s).thaw() for s in range(1,H+1)]
            initial,z=finish(State.build(ray.base,root_vector(q),1,sg11.initial(q))); audit(q,z)
            arms={a:[] for a in ARMS}
            for rep in range(2):
                order=list(ARMS); random.Random(seed+31*rep).shuffle(order)
                for arm in order: arms[arm].append(run(ray,tasks,initial,arm))
                for a,b in zip(arms['delta'][-1]['rows'],arms['ray'][-1]['rows'],strict=True):
                    assert all(a[k]==b[k] for k in ('schedule','eta','den','objective','upper','gap'))
            roots.append(dict(seed=seed,regime=reg,mode=mode,base=q,end=end,anchor=z,arms=arms))
            Path('sg21_progress.json').write_text(json.dumps(roots))
            print(seed,mode,[(a,round(summarize(roots[-1:])[a]['cpu_ms'],3)) for a in ARMS],flush=True)
    return dict(version='SG-21/v0.1',horizon=H,roots=roots,summary=summarize(roots),
                by_mode={m:summarize([r for r in roots if r['mode']==m]) for m in MODES})


if __name__=='__main__':
    z=campaign();Path('sg21_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='roots'},indent=2))
