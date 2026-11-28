"""SG-30 fresh, exposed, and forced-split controls with complete replay."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import sg07
import sg28
import sg30
from sg28_integer import replay as replay_tree
from sg29 import prepare
from test_sg29 import bridge_task, triangle_schedule

ARMS=('tree','static','dynamic','memo')


def case(q,a,p,seed,kind,force=None,opt=None,prep=0,repeats=1):
    arms=('dynamic','memo') if force is not None else ARMS
    out={arm:[] for arm in arms}
    for rep in range(repeats):
        order=list(arms);random.Random(seed+31*rep).shuffle(order)
        for arm in order:
            if arm=='tree':
                r=sg28.solve(q,a,root=p,cap=63,use_milp=False)
                tick=time.process_time_ns();c=replay_tree(q,r,63);ns=time.process_time_ns()-tick
                keys=('schedule','objective','upper','gap','tree','nodes','depth')
            else:
                r=sg30.solve(q,a,p,memo=arm=='memo',static=arm=='static',force=force)
                tick=time.process_time_ns();c=sg30.replay(q,r);ns=time.process_time_ns()-tick
                keys=('schedule','objective','upper','gap','tree','bank','nodes','work')
            size=len(json.dumps({k:r[k] for k in keys},separators=(',',':')).encode())
            if opt is not None:assert r['objective']<=opt<=r['upper']
            out[arm].append(dict(receipt=r,replay_ns=ns,proof_bytes=size,checked=c))
        rr=[out[arm][-1]['receipt'] for arm in arms]
        assert max(r['objective'] for r in rr)<=min(r['upper'] for r in rr)
        x,y=[out[m][-1]['receipt'] for m in ('dynamic','memo')]
        assert all(x[k]==y[k] for k in ('schedule','objective','upper','gap','nodes','lp_calls','work','factor_calls','factor_leaves'))
    return dict(task=q,initial=a,root=p,seed=seed,kind=kind,force=force,opt=opt,prep_ns=prep,arms=out)


def summary(rows):
    if not rows:return {}
    out={}
    for arm in rows[0]['arms']:
        flat=[v for r in rows for v in r['arms'][arm]];rr=[v['receipt'] for v in flat]
        times=[r['prep_ns']+statistics.median(v['receipt']['cpu_ns']+v['replay_ns'] for v in r['arms'][arm]) for r in rows]
        out[arm]=dict(roots=len(rows),runs=len(rr),closed=sum(r['gap']==0 for r in rr),mean_gap=statistics.mean(r['gap'] for r in rr),total_ms=statistics.mean(times)/1e6,
                      acquisition_ms=statistics.mean(r['cpu_ns'] for r in rr)/1e6,replay_ms=statistics.mean(v['replay_ns'] for v in flat)/1e6,
                      nodes=sum(r['nodes'] for r in rr),lp=sum(r['lp_calls'] for r in rr),factors=sum(r.get('factor_leaves',0) for r in rr),attempts=sum(r.get('factor_calls',0) for r in rr),
                      hits=sum(r.get('hits',0) for r in rr),logical_work=sum(r.get('work',0) for r in rr),computed_work=sum(r.get('computed',0) for r in rr),
                      tables=sum(len(r.get('bank',[])) for r in rr),proof_bytes=statistics.mean(v['proof_bytes'] for v in flat))
    return out


def campaign(kind,start=0,limit=100):
    path=Path(f'sg30_{kind}.json');rows=json.loads(path.read_text())['rows'] if start else []
    assert len(rows)==start
    inputs=[]
    if kind=='forced':
        for g in (2,3,4,5):
            for c in (1,2,3):
                q=bridge_task(g,c,0);a,p,ns=prepare(q)
                inputs.append((q,a,p,30000+g*10+c,(0,0),g*(30-2*c),ns,2))
    elif kind=='bridge':
        for g in (2,3,4):
            for c in (1,2,3):
                q=bridge_task(g,c,3);a,p,ns=prepare(q)
                inputs.append((q,a,p,30100+g*10+c,None,g*(30-2*c)+60,ns,2))
    elif kind=='fresh':
        for j,reg in enumerate(('persistent','rotating','mixed')):
            for i in range(6):
                seed=3010000+j*1000+i;q=sg07.task(seed,reg,32,12);q['k']=[7+(seed+v)%2 for v in range(q['T'])]
                a,p,ns=prepare(q);inputs.append((q,a,p,seed,None,None,ns,2))
    elif kind=='exposed':
        for i,r in enumerate(json.loads(Path('sg28_exposed_inputs.json').read_text())):
            q=r['task'];a,p=r['initial'],r['root'];ns=0
            if p is None:a,p,ns=prepare(q)
            inputs.append((q,a,p,i,None,None,ns,1))
    for q,a,p,seed,f,opt,ns,reps in inputs[start:limit]:
        rows.append(case(q,a,p,seed,kind,f,opt,ns,reps))
        out=dict(kind=kind,rows=rows,summary=summary(rows));tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out));tmp.replace(path)
        print(kind,seed,[(m,v[0]['receipt']['gap'],v[0]['receipt']['nodes'],v[0]['receipt'].get('hits',0)) for m,v in rows[-1]['arms'].items()],flush=True)
    print(json.dumps(summary(rows),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('fresh','bridge','forced','exposed'));p.add_argument('--start',type=int,default=0);p.add_argument('--limit',type=int,default=100);a=p.parse_args();campaign(a.kind,a.start,a.limit)
