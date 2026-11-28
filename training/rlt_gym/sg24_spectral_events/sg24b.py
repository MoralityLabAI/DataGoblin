"""Fresh cohort comparing exact supporting-line and envelope boundary planners."""
import argparse
import json
import random
import statistics
from pathlib import Path
from unittest.mock import patch

import sg11
import sg24
from sg20 import audit, finish
from sg20_delta import State
from sg21_ray import Ray
from sg22 import trajectory
from sg24_support import region
from datagoblin.training_gym.spectral_blocks import root_vector

ARMS=('correct','binary','events','support')


def run(ray,tasks,initial,arm):
    if arm!='support':return sg24.run(ray,tasks,initial,arm)
    with patch.object(sg24,'event_region',region):
        return sg24.run(ray,tasks,initial,'events')


def summarize(roots):
    out={}
    for arm in ARMS:
        groups=[[z for rep in r['arms'][arm] for z in rep['rows']] for r in roots];flat=[z for g in groups for z in g]
        times=[statistics.mean(z['cpu_ns'] for z in g)/1e6 for g in groups]
        base=[statistics.mean(z['cpu_ns'] for rep in r['arms']['correct'] for z in rep['rows'])/1e6 for r in roots]
        binary=[statistics.mean(z['cpu_ns'] for rep in r['arms']['binary'] for z in rep['rows'])/1e6 for r in roots]
        proofs=[p for r in roots for rep in r['arms'][arm] for p in rep['proofs']]
        out[arm]=dict(runs=len(flat),certified=sum(z['gap']==0 for z in flat),cpu_ms=statistics.mean(times),
                      planning_us=statistics.mean(z['planning_ns'] for z in flat)/1e3,lp_calls=sum(z['lp_calls'] for z in flat),
                      gates=sum(z['gate'] for z in flat),reused=sum(z['reused'] for z in flat),proofs=len(proofs),
                      probes=sum(p['probes'] for p in proofs),support_evals=sum(p.get('support_evals',0) for p in proofs),
                      fallbacks=sum(p.get('fallback',False) for p in proofs),crossings=sum(p.get('crossing') is not None for p in proofs),
                      wins=sum(a<b for a,b in zip(times,base,strict=True)),wins_binary=sum(a<b for a,b in zip(times,binary,strict=True)))
    return out


def campaign(resume=False,limit=24):
    roots=json.loads(Path('sg24b_progress.json').read_text()) if resume else []
    done={r['seed'] for r in roots};H=32
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(8):
            seed=2420000+j*1000+i
            if seed in done:continue
            if len(roots)>=limit:break
            q,end=trajectory(seed,reg,bool(i%2))
            # Extend the identical affine direction from 16 to 32 positions.
            end['lambda']=q['lambda']+H
            end['u']=[[2*b-a for a,b in zip(u,v,strict=True)] for u,v in zip(q['u'],end['u'],strict=True)]
            ray=Ray.build(q,end,H);tasks=[ray.at(s).thaw() for s in range(1,H+1)]
            initial,receipt=finish(State.build(ray.base,root_vector(q),1,sg11.initial(q)));audit(q,receipt)
            assert initial.certificate()['certified'];arms={a:[] for a in ARMS}
            for rep in range(2):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for arm in order:arms[arm].append(run(ray,tasks,initial,arm))
                a,b,c,d=(arms[arm][-1] for arm in ARMS)
                for w,x,y,z in zip(a['rows'],b['rows'],c['rows'],d['rows'],strict=True):
                    assert max(r['objective'] for r in (w,x,y,z))<=min(r['upper'] for r in (w,x,y,z))
                    assert all(x[k]==y[k]==z[k] for k in ('objective','upper','schedule','eta','den','gap','lp_calls','reused'))
                for x,y,z in zip(b['proofs'],c['proofs'],d['proofs'],strict=True):
                    assert all(x[k]==y[k]==z[k] for k in x if k!='probes')
                    assert y['crossing']==z['crossing']
            roots.append(dict(seed=seed,regime=reg,mixed=bool(i%2),base=q,end=end,initial=receipt,arms=arms))
            Path('sg24b_progress.tmp').write_text(json.dumps(roots));Path('sg24b_progress.tmp').replace('sg24b_progress.json')
            print(seed,[(a,round(v['cpu_ms'],3),v['lp_calls'],v['reused']) for a,v in summarize(roots[-1:]).items()],flush=True)
    return dict(version='SG-24B/v0.1',horizon=H,roots=roots,summary=summarize(roots),
                by_mode={str(m):summarize([r for r in roots if r['mixed']==m]) for m in (False,True)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int,default=24);a=p.parse_args()
    z=campaign(a.resume,a.limit);Path('sg24b_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='roots'},indent=2))
