"""SG-15: exact face recovery on fresh finite routing tasks."""
import json
import random
import statistics
import time
from pathlib import Path

import sg07
import sg11
from sg15_lp import propose
from datagoblin.training_gym.spectral_blocks import BlockSearch, root_vector
from datagoblin.training_gym.spectral_block_replay import verify
from datagoblin.training_gym.spectral_face import recover_face, verify_certificate
from datagoblin.training_gym.spectral_primal import dual_round, recover, row_sweep
from datagoblin.training_gym.spectral_quotient import coordinate, score

ARMS=('blocks','root_face','sweep_face','lp_face')
NODE_CAP=4096


def initial(q):
    e=root_vector(q)
    a,_=recover(q,sg11.initial(q),e,'dual')
    return a,e


def round_rational(q,e,den):
    scaled={**q,'u':[[u*den for u in row] for row in q['u']], 'lambda':q['lambda']*den}
    a=dual_round(scaled,e)
    a,_=row_sweep(q,a)
    return a


def run(q,kind):
    tick=time.process_time_ns()
    a,e=initial(q);den=1
    if kind=='blocks':
        s=BlockSearch(q,a,event_cap=2048,extra=[e]);r=s.run();elapsed=time.process_time_ns()-tick
        assert verify(q,a,[e],r['events'],event_cap=2048)=={k:v for k,v in r.items() if k!='peak_stored_blocks'}
        return dict(objective=r['objective'],upper=r['upper'],gap=r['gap'],cpu_ns=elapsed,
                    events=r['event_count'],receipt=r,initial=a,eta=e)
    lp=None
    if kind=='sweep_face':
        for sweep in range(4):
            for i in (range(q['T']-1,-1,-1) if sweep%2==0 else range(q['T'])):
                e[i]=coordinate(q,q['T'],0,e,i)
    elif kind=='lp_face':
        v,lp=propose(q)
        if v is not None:e,den=v
    new=round_rational(q,e,den)
    if score(q,new)['objective']>score(q,a)['objective']:a=new
    before=verify_certificate(q,a,e,den)
    z=recover_face(q,e,den,node_cap=NODE_CAP)
    if z['schedule'] is not None:a=z['schedule']
    cert=verify_certificate(q,a,e,den)
    elapsed=time.process_time_ns()-tick
    assert cert['gap']<=before['gap'] and cert['bound_num']==z['bound_num']
    return dict(objective=cert['objective'],upper=cert['integer_upper'],gap=cert['gap'],cpu_ns=elapsed,
                nodes=z['nodes'],face_status=z['status'],reductions=z['reductions'],eta=e,den=den,
                certificate=cert,schedule=a,prior_gap=before['gap'],lp=lp)


def summarize(rows):
    return {m:dict(roots=len(rows),certified=sum(r['arms'][m]['gap']==0 for r in rows),
                   mean_gap=statistics.mean(r['arms'][m]['gap'] for r in rows),
                   mean_objective=statistics.mean(r['arms'][m]['objective'] for r in rows),
                   mean_cpu_ms=statistics.mean(r['arms'][m]['cpu_ns'] for r in rows)/1e6,
                   median_cpu_ms=statistics.median(r['arms'][m]['cpu_ns'] for r in rows)/1e6,
                   exact_faces=sum(r['arms'][m].get('face_status')=='exact' for r in rows),
                   infeasible_faces=sum(r['arms'][m].get('face_status')=='infeasible_face' for r in rows),
                   unknown_faces=sum(r['arms'][m].get('face_status')=='unknown' for r in rows),
                   total_nodes=sum(r['arms'][m].get('nodes',0) for r in rows),
                   max_nodes=max(r['arms'][m].get('nodes',0) for r in rows),
                   before_face_certified=sum(r['arms'][m].get('prior_gap')==0 for r in rows)) for m in ARMS}


def campaign():
    rows=[]
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(12):
            seed=1520000+j*1000+i
            q=sg07.task(seed,reg,32,12);q['k']=[7+(seed+v)%2 for v in range(q['T'])]
            order=list(ARMS);random.Random(seed).shuffle(order)
            arms={m:run(q,m) for m in order}
            lo=max(z['objective'] for z in arms.values());hi=min(z['upper'] for z in arms.values());assert lo<=hi
            rows.append(dict(seed=seed,regime=reg,task=q,arms=arms,lower=lo,upper=hi))
            Path('sg15_progress.json').write_text(json.dumps(rows))
            print(seed,[(m,arms[m]['gap'],round(arms[m]['cpu_ns']/1e6,2)) for m in ARMS],flush=True)
    exposed=json.loads(Path('sg15_exposed.json').read_text())
    for row in exposed:row['arms']={m:run(row['task'],m) for m in ARMS}
    return dict(version='SG-15/v0.1',roots=len(rows),node_cap=NODE_CAP,summary=summarize(rows),
                exact_cross_policy=sum(r['lower']==r['upper'] for r in rows),rows=rows,exposed=exposed)


if __name__=='__main__':
    z=campaign();Path('sg15_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k not in ('rows','exposed')},indent=2))
