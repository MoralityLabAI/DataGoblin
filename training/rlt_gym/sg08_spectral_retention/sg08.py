"""SG-08 trace replay and live CPU comparison. Run with SG-07 on PYTHONPATH."""
import bisect
import itertools
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sg07
from datagoblin.training_gym.spectral_dual import verify_dual_trace
from datagoblin.training_gym.spectral_retention import Retention, RetentionSession, features, fixed_keys
from datagoblin.training_gym.spectral_routing import oracle

CAPS = (6,12,24)
ONLINE = ('lru','lfu','weighted','fixed_first','learned','shuffled')


def trace_run(q,kind='lru',beta=(),cap=12,record=False):
    sessions = []
    def create(task,initial,**kw):
        kw.update(table_cap=cap,scalar_cap=16,kind=kind,beta=beta,record=record)
        s=RetentionSession(task,initial,**kw)
        sessions.append(s)
        return s
    with patch.object(sg07,'CachedDualSession',create):
        z=sg07.run(q,'lazy12')
    z['requests']=sessions[0].requests
    return z


def positions(trace):
    p=defaultdict(list)
    for i,k in enumerate(trace):
        p[k].append(i)
    return p


def belady(trace,cap):
    """Offline miss-count optimum under mandatory insertion."""
    pos=positions(trace); live=set(); misses=work=0
    for i,key in enumerate(trace):
        if key in live:
            continue
        misses+=1; work+=key[0]
        if not cap:
            continue
        if len(live)==cap:
            def nxt(k):
                a=pos[k]; j=bisect.bisect_right(a,i)
                return a[j] if j<len(a) else len(trace)+1
            live.remove(max(live,key=lambda k:(nxt(k),k)))
        live.add(key)
    return dict(misses=misses,depth_work=work,hits=len(trace)-misses)


def simulate(trace,cap,kind,fixed,beta=(),observer=None):
    s=Retention(cap,kind,fixed,beta,observer)
    for k in trace:
        s.access(k,lambda: None)
    return dict(misses=s.misses,depth_work=s.work,hits=s.hits,peak=s.peak)


def labels(trace,fixed,cap):
    p=positions(trace); rows=[]
    def collect(s,key):
        i=s.clock-1
        for k,e in s.entries.items():
            a=p[k]
            n=bisect.bisect_right(a,i+2*cap)-bisect.bisect_right(a,i)
            rows.append((features(k,e,key,s.clock,cap,fixed),k[0]*n))
    simulate(trace,cap,'lru',fixed,observer=collect)
    return rows


def fit(rows):
    a=np.asarray([x for x,y in rows],dtype=np.float64)
    b=np.asarray([y for x,y in rows],dtype=np.float64)
    c=np.linalg.solve(a.T@a+10*np.eye(a.shape[1]),a.T@b)
    return [int(round(v*1024)) for v in c]


def train():
    all_rows=[]; null=[]; counts=[]
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(12):
            seed=910000+1000*j+i
            q=sg07.task(seed,reg,8 if i<8 else 9,6 if i<8 else 7)
            trace=trace_run(q,record=True)['requests']; fixed=fixed_keys(q)
            root=list(itertools.chain.from_iterable(labels(trace,fixed,c) for c in CAPS))
            ys=[y for x,y in root]; random.Random(seed).shuffle(ys)
            all_rows.extend(root); null.extend((x,y) for (x,_),y in zip(root,ys,strict=True))
            counts.append(len(root))
    return dict(learned=fit(all_rows),shuffled=fit(null),roots=len(counts),labels=sum(counts),labels_per_root=counts)


def campaign():
    model=train(); rows=[]; replays=0
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(16):
            seed=920000+1000*j+i; d,t=(8,6) if i<8 else (9,7) if i<12 else (10,8)
            q=sg07.task(seed,reg,d,t); fixed=fixed_keys(q)
            baseline=trace_run(q,record=True); trace=baseline['requests']; ref=baseline['receipt']
            optimum=oracle(q,transition_cap=10_000_000)['objective']
            assert ref['objective']<=optimum<=ref['upper']
            offline={}
            for cap in CAPS:
                arm={k:simulate(trace,cap,k,fixed,model.get(k,())) for k in ONLINE}
                arm['belady']=belady(trace,cap)
                assert all(v['misses']>=arm['belady']['misses'] for v in arm.values())
                offline[cap]=arm
            live={k:[] for k in ('native_lru',*ONLINE)}
            for rep in range(3):
                order=list(live); random.Random(seed+13*rep).shuffle(order)
                for k in order:
                    z=sg07.run(q,'lazy12') if k=='native_lru' else trace_run(q,k,model.get(k,()),cap=12)
                    assert z['receipt']==ref
                    assert z['cache']['component_steps']==offline[12]['lru' if k=='native_lru' else k]['depth_work']*d
                    assert verify_dual_trace(q,sg07.initial(q),z['receipt']['events'],transition_cap=sg07.CAP,
                                             proposal_cap=sg07.PROPOSALS)==ref
                    replays+=1
                    live[k].append({a:b for a,b in z.items() if a not in ('receipt','requests')})
            rows.append(dict(seed=seed,regime=reg,task=q,optimum=optimum,receipt=ref,requests=trace,offline=offline,live=live))
            print(f'root {seed} complete',flush=True,file=__import__('sys').stderr)
    summary={}
    for cap in CAPS:
        summary[cap]={}
        for k in (*ONLINE,'belady'):
            summary[cap][k]=dict(misses=sum(r['offline'][cap][k]['misses'] for r in rows),
                component_steps=sum(r['offline'][cap][k]['depth_work']*r['task']['d'] for r in rows),
                hits=sum(r['offline'][cap][k]['hits'] for r in rows))
    timing={}; base=[statistics.median(z['cpu_ns'] for z in r['live']['native_lru']) for r in rows]
    for k in rows[0]['live']:
        a=[statistics.median(z['cpu_ns'] for z in r['live'][k]) for r in rows]
        timing[k]=dict(cpu_ms=statistics.mean(a)/1e6,wins=sum(x<y for x,y in zip(a,base,strict=True)))
    return dict(version='SG-08/v0.1',evidence='offline synthetic development; same receipts; table requests replayed',
                model=model,roots=len(rows),requests=sum(len(r['requests']) for r in rows),replays=replays,
                certified=sum(r['receipt']['gap']==0 for r in rows),summary=summary,timing=timing,rows=rows)


if __name__=='__main__':
    z=campaign(); Path('sg08_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='rows'},indent=2))
