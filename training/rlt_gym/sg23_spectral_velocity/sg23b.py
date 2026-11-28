"""Separate fresh cohort for the exact incremental sweep implementation."""
import argparse
import json
import random
import statistics
from pathlib import Path
from unittest.mock import patch

import sg11
import sg23
from sg20 import audit, finish
from sg20_delta import State
from sg21_ray import Ray
from sg22 import H, trajectory
from sg23_fast import FastCone
from datagoblin.training_gym.spectral_blocks import root_vector

ARMS=('correct','sweep','fast_sweep')


def run(ray,tasks,initial,arm):
    if arm!='fast_sweep':return sg23.run(ray,tasks,initial,arm)
    with patch.object(sg23,'Cone',FastCone):
        return sg23.run(ray,tasks,initial,'sweep')


def summary(roots):
    out={}
    for arm in ARMS:
        groups=[[z for rep in r['arms'][arm] for z in rep] for r in roots];flat=[z for g in groups for z in g]
        times=[statistics.mean(z['cpu_ns'] for z in g)/1e6 for g in groups]
        base=[statistics.mean(z['cpu_ns'] for rep in r['arms']['correct'] for z in rep)/1e6 for r in roots]
        slow=[statistics.mean(z['cpu_ns'] for rep in r['arms']['sweep'] for z in rep)/1e6 for r in roots]
        out[arm]=dict(runs=len(flat),cpu_ms=statistics.mean(times),certified=sum(z['gap']==0 for z in flat),
                      lp_calls=sum(z['lp_calls'] for z in flat),directions=sum(z['proposal'] is not None for z in flat),
                      accepted=sum(z['accepted'] for z in flat),raw_closures=sum(z['proposal'] is not None and z['proposal']['raw']['certified'] for z in flat),
                      wins=sum(a<b for a,b in zip(times,base,strict=True)),wins_vs_sweep=sum(a<b for a,b in zip(times,slow,strict=True)))
    return out


def campaign(resume=False,limit=24):
    roots=json.loads(Path('sg23b_progress.json').read_text()) if resume else [];done={r['seed'] for r in roots}
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(8):
            seed=2320000+j*1000+i
            if seed in done:continue
            if len(roots)>=limit:break
            q,end=trajectory(seed,reg,bool(i%2));ray=Ray.build(q,end,H);tasks=[ray.at(s).thaw() for s in range(1,H+1)]
            initial,receipt=finish(State.build(ray.base,root_vector(q),1,sg11.initial(q)));audit(q,receipt)
            arms={a:[] for a in ARMS}
            for rep in range(2):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for arm in order:arms[arm].append(run(ray,tasks,initial,arm))
                for pos in range(H):
                    a,b,c=(arms[a][-1][pos] for a in ARMS)
                    assert max(z['objective'] for z in (a,b,c))<=min(z['upper'] for z in (a,b,c))
                    assert {k:v for k,v in b.items() if k!='cpu_ns'}=={k:v for k,v in c.items() if k!='cpu_ns'}
            roots.append(dict(seed=seed,regime=reg,mixed=bool(i%2),base=q,end=end,initial=receipt,arms=arms))
            Path('sg23b_progress.tmp').write_text(json.dumps(roots));Path('sg23b_progress.tmp').replace('sg23b_progress.json')
            print(seed,[(a,round(v['cpu_ms'],3),v['lp_calls'],v['accepted']) for a,v in summary(roots[-1:]).items()],flush=True)
    return dict(version='SG-23B/v0.1',horizon=H,roots=roots,summary=summary(roots),
                by_mode={str(m):summary([r for r in roots if r['mixed']==m]) for m in (False,True) if any(r['mixed']==m for r in roots)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int,default=24);a=p.parse_args()
    z=campaign(a.resume,a.limit);Path('sg23b_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='roots'},indent=2))
