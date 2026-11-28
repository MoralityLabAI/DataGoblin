"""SG-29: paired bounded factor certificates and last-use disjunction trees."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import sg07
import sg11
import sg28
import sg29_factor as factor
from sg17_lp import chain_lp
from sg20 import repair
from sg20_delta import State, Task
from sg28_integer import conditioned, domains, replay as tree_replay
from test_sg29 import bridge_task
from datagoblin.training_gym.spectral_quotient import score

ARMS = ('root','tree','factor0','factor1','factor2')
REPEATS = 2


def run(q, a, p, arm):
    e,d=p;tick=time.process_time_ns()
    if arm=='root':
        L=score(q,a)['objective'];U=conditioned(q,domains(q),e,d)
        r=dict(schedule=a,objective=L,upper=U,gap=U-L,nodes=1,depth=0,tree=dict(kind='bound',eta=e,den=d,upper=U))
    elif arm=='tree':
        r=sg28.solve(q,a,root=p,use_milp=False,cap=63)
    else:
        r=factor.solve(q,a,e,d,max_sep=int(arm[-1]))
    elapsed=r['cpu_ns'] if arm=='tree' else time.process_time_ns()-tick
    tick=time.process_time_ns()
    if arm in ('root','tree'):
        tree_replay(q,r,63);keys=('schedule','objective','upper','gap','tree','nodes','depth')
    else:
        factor.replay(q,r,max_sep=int(arm[-1]));keys=('schedule','objective','upper','gap','proof','work')
    replay_ns=time.process_time_ns()-tick
    size=len(json.dumps({k:r[k] for k in keys},separators=(',',':')).encode())
    return dict(receipt=r,acquire_ns=elapsed,replay_ns=replay_ns,proof_bytes=size)


def prepare(q):
    tick=time.process_time_ns();p,_=chain_lp(q)
    if p is None:raise RuntimeError('LP proposal failed')
    s=repair(State.build(Task.from_task(q),*p,sg11.initial(q)))
    return list(s.schedule),[list(s.eta),s.den],time.process_time_ns()-tick


def case(q, seed, kind, optimum=None, **kw):
    a,p,prep=prepare(q);arms={m:[] for m in ARMS}
    for rep in range(REPEATS):
        order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
        for arm in order:arms[arm].append(run(q,a,p,arm))
        rr=[v[-1]['receipt'] for v in arms.values()]
        assert max(z['objective'] for z in rr)<=min(z['upper'] for z in rr)
        if optimum is not None:assert all(z['objective']<=optimum<=z['upper'] for z in rr)
    return dict(task=q,seed=seed,kind=kind,optimum=optimum,initial=a,eta=p,prepare_ns=prep,arms=arms,**kw)


def summarize(rows):
    return {m:dict(roots=len(rows),runs=len(rows)*REPEATS,closed=sum(v['receipt']['gap']==0 for r in rows for v in r['arms'][m]),
                   mean_gap=statistics.mean(v['receipt']['gap'] for r in rows for v in r['arms'][m]),
                   acquire_ms=statistics.mean(statistics.median(v['acquire_ns'] for v in r['arms'][m]) for r in rows)/1e6,
                   replay_ms=statistics.mean(statistics.median(v['replay_ns'] for v in r['arms'][m]) for r in rows)/1e6,
                   total_ms=statistics.mean(r['prepare_ns']+statistics.median(v['acquire_ns']+v['replay_ns'] for v in r['arms'][m]) for r in rows)/1e6,
                   proof_bytes=statistics.mean(v['proof_bytes'] for r in rows for v in r['arms'][m]),
                   tree_nodes=sum(r['arms'][m][0]['receipt'].get('nodes',0) for r in rows),
                   factors=sum(len(r['arms'][m][0]['receipt'].get('proof',{}).get('parts',[])) for r in rows),
                   skipped_factors=sum(sum(p['table'] is None for p in r['arms'][m][0]['receipt'].get('proof',{}).get('parts',[])) for r in rows)) for m in ARMS}


def campaign(resume=False,limit=51):
    rows=json.loads(Path('sg29_progress.json').read_text()) if resume else []
    done={r['seed'] for r in rows}
    def add(r):
        rows.append(r);Path('sg29_progress.tmp').write_text(json.dumps(rows));Path('sg29_progress.tmp').replace('sg29_progress.json')
        print(r['seed'],r['kind'],[(m,v[0]['receipt']['gap'],v[0]['receipt'].get('nodes',v[0]['receipt'].get('work'))) for m,v in r['arms'].items()],flush=True)
    for bridges in (0,1,2):
        for g in (2,3,4,5):
            if 3*g+bridges>16:continue
            for c in (1,2,3):
                seed=29000+100*bridges+10*g+c
                if seed in done or len(rows)>=limit:continue
                q=bridge_task(g,c,bridges);opt=g*(30-2*c)+10*sum(range(1,bridges+1))
                add(case(q,seed,'planted',opt,groups=g,cost=c,bridges=bridges))
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(6):
            seed=2910000+j*1000+i
            if seed in done or len(rows)>=limit:continue
            q=sg07.task(seed,reg,32,12);q['k']=[7+(seed+v)%2 for v in range(q['T'])]
            add(case(q,seed,'random',regime=reg))
    groups={f'bridge{b}':summarize([r for r in rows if r.get('bridges')==b]) for b in (0,1,2) if any(r.get('bridges')==b for r in rows)}
    if any(r['kind']=='random' for r in rows):groups['random']=summarize([r for r in rows if r['kind']=='random'])
    out=dict(version='SG-29/v0.1',repeats=REPEATS,cap=factor.CAP,summary=groups,rows=rows)
    Path('sg29_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(groups,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int,default=51);args=p.parse_args();campaign(args.resume,args.limit)
