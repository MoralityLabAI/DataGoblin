"""Fresh SG-12 bounded-search development campaign."""
import argparse
import json
import random
import statistics
import time
from math import comb
from pathlib import Path

import sg07
import sg11
from datagoblin.training_gym.spectral_blocks import BlockSearch, root_vector
from datagoblin.training_gym.spectral_block_replay import verify
from datagoblin.training_gym.spectral_quotient import SparseSession, replay
from datagoblin.training_gym.spectral_routing import oracle

ARMS=('sg11','blocks256','blocks2048')


def run(q,initial,arm):
    tick=time.process_time_ns()
    if arm=='sg11':
        z=sg11.run(q,'sparse',raw_cap=None,class_cap=8192)
        elapsed=time.process_time_ns()-tick
        r=z['receipt']
        assert replay(q,initial,r['events'],class_cap=8192,raw_cap=None,proposal_cap=24)==r
        return dict(receipt=r,cpu_ns=elapsed,peak=z['peak_live'],stop=z['stop'])
    cap=int(arm[6:]);extra=[root_vector(q)]
    s=BlockSearch(q,initial,event_cap=cap,extra=extra)
    initial_upper=s.upper();r=s.run();elapsed=time.process_time_ns()-tick
    independently=verify(q,initial,extra,r['events'],event_cap=cap)
    assert independently=={k:v for k,v in r.items() if k!='peak_stored_blocks'}
    return dict(receipt=r,cpu_ns=elapsed,peak=r['peak_stored_blocks'],root_upper=initial_upper,
                extra=extra,stop='certified' if not r['gap'] else 'event_cap')


def campaign(group):
    rows=[]
    for j,regime in enumerate(('persistent','rotating','mixed')):
        for i in range(12 if group=='small' else 8):
            if group=='small':
                seed=993000+1000*j+i;d,t=(8,6) if i<6 else (9,7) if i<9 else (10,8)
            else:
                seed=997000+1000*j+i;d,t=(16,8) if i<3 else (24,10) if i<6 else (32,12)
            q=sg07.task(seed,regime,d,t)
            if group=='large':
                lo={16:4,24:5,32:7}[d]
                q['k']=[lo+(seed+v)%2 for v in range(t)]
            initial=sg11.initial(q)
            opt=oracle(q,transition_cap=10_000_000)['objective'] if group=='small' else None
            order=list(ARMS);random.Random(seed).shuffle(order);runs={}
            for arm in order:
                z=run(q,initial,arm);r=z['receipt']
                if opt is not None:assert r['objective']<=opt<=r['upper']
                runs[arm]=z
            a,b=runs['blocks256']['receipt'],runs['blocks2048']['receipt']
            assert a['events']==b['events'][:len(a['events'])] and b['gap']<=a['gap']
            lo=max(z['receipt']['objective'] for z in runs.values());hi=min(z['receipt']['upper'] for z in runs.values())
            assert lo<=hi
            rows.append(dict(seed=seed,regime=regime,task=q,initial=initial,optimum=opt,
                             root_classes=comb(d,q['k'][-1]),runs=runs))
            print(group,seed,d,[(a,runs[a]['receipt']['gap'],round(runs[a]['cpu_ns']/1e6,2)) for a in ARMS],flush=True)
    return rows


def summarize(rows):
    result={}
    for arm in ARMS:
        zz=[r['runs'][arm] for r in rows];rr=[z['receipt'] for z in zz]
        result[arm]=dict(roots=len(rr),certified=sum(r['gap']==0 for r in rr),
                         mean_gap=statistics.mean(r['gap'] for r in rr),
                         mean_objective=statistics.mean(r['objective'] for r in rr),
                         mean_cpu_ms=statistics.mean(z['cpu_ns'] for z in zz)/1e6,
                         median_cpu_ms=statistics.median(z['cpu_ns'] for z in zz)/1e6,
                         max_peak=max(z['peak'] for z in zz),
                         total_work=sum(r.get('event_count',r.get('classes',0)) for r in rr),
                         splits=sum(r.get('splits',0) for r in rr),leaves=sum(r.get('leaves',0) for r in rr),
                         terminals=sum(r.get('terminals',0) for r in rr),
                         no_expansion=sum(not r.get('event_count',len(r.get('trace',[]))) for r in rr),
                         total_proposals=sum(r.get('proposals',0) for r in rr),
                         total_bound_calls=sum(r.get('bound_calls',0) for r in rr))
        if rows[0]['optimum'] is not None:
            result[arm]['mean_regret']=statistics.mean(x['optimum']-r['objective'] for x,r in zip(rows,rr,strict=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=('small','large'));a=p.parse_args()
    rows=campaign(a.group)
    z=dict(version='SG-12/v0.1',group=a.group,roots=len(rows),independent_block_replays=2*len(rows),
           sg11_self_replays=len(rows),summary=summarize(rows),
           by_dimension={d:summarize([r for r in rows if r['task']['d']==d]) for d in sorted({r['task']['d'] for r in rows})},rows=rows)
    Path(f'sg12_{a.group}_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='rows'},indent=2))
