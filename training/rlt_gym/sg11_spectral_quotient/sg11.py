"""SG-11 development campaign: matched raw caps and sparse scaling."""
import json
import random
import statistics
import time
from itertools import combinations
from math import comb
from pathlib import Path

import sg07
from datagoblin.training_gym.spectral_batch import BatchedBankSession
from datagoblin.training_gym.spectral_dual import verify_dual_trace
from datagoblin.training_gym.spectral_quotient import SparseSession, coordinate, quotient, replay, score
from datagoblin.training_gym.spectral_routing import oracle


def initial(q):
    d,t=q['d'],q['T']
    def pick(i,fn):
        return sum(1<<j for j in sorted(range(d),key=lambda j:(-fn(j),j))[:q['k'][i]])
    pool=set(sorted(range(d),key=lambda j:(-sum(r[j] for r in q['u'])+q['lambda']*q['bits'][j]*(q['x'][-1]-q['x'][0]),j))[:max(q['k'])])
    a,s=[0]*t,0
    for i in range(t-1,-1,-1):
        a[i]=pick(i,lambda j:q['u'][i][j]-(0 if s&(1<<j) else q['lambda']*q['bits'][j]*(q['x'][i+1]-q['x'][0])))
        s|=a[i]
    return max([a,[pick(i,lambda j:q['u'][i][j]) for i in range(t)],
                [pick(i,lambda j:q['u'][i][j]+100000*(j in pool)) for i in range(t)]],key=lambda a:score(q,a)['objective'])


def run(q,arm,raw_cap=2048,class_cap=16384):
    start=time.process_time_ns(); init=initial(q)
    if arm=='sg10':
        s=BatchedBankSession(q,init,mode='batch',transition_cap=raw_cap,proposal_cap=24,table_cap=12,scalar_cap=16)
    else:
        s=SparseSession(q,init,raw_cap=raw_cap,class_cap=class_cap,proposal_cap=24)
    tried=set(); peak=len(s.live); stop='decision_cap'
    for _ in range(80):
        if not s.live:
            stop='certified';break
        t,m,_=min(s.frontier(),key=lambda a:(-a[2],a[0],a[1]))
        if (t,m) not in tried and s.proposals<24:
            e=[0]*t
            for i in range(t-1,-1,-1):
                e[i]=coordinate(q,t,m,e,i)
            s.propose(t,m,e);tried.add((t,m))
        elif (s.transitions+len(s.actions[t-1])<=raw_cap if arm=='sg10' else s.can_expand(t,m)):
            s.advance(t,m)
        else:
            stop='work_cap';break
        peak=max(peak,len(s.live))
    receipt=s.receipt()
    if not receipt['gap']:
        stop='certified'
    return dict(receipt=receipt,cpu_ns=time.process_time_ns()-start,peak_live=peak,stop=stop)


def micro():
    rows=[]
    for d,k in [(16,4),(20,5),(32,8)]:
        row=[(j*17)%13-6 for j in range(d)]
        for a in (0,d//2,d-4,d-1):
            if d==32 and a<d-4:
                continue
            s=(1<<a)-1
            def direct():
                return {u:(m,v) for u,m,v in quotient(row,k,s)}
            def raw():
                out={}
                for js in combinations(range(d),k):
                    m,v=sum(1<<j for j in js),sum(row[j] for j in js)
                    u=m|s
                    if u not in out or v>out[u][1]:
                        out[u]=(m,v)
                return out
            expected=direct(); rec=dict(d=d,k=k,active=a,raw=comb(d,k),classes=len(expected))
            for name,fn in [('direct',direct),*([('raw_enum',raw)] if d<32 else [])]:
                times=[]
                for _ in range(5):
                    tick=time.process_time_ns(); out=fn(); times.append(time.process_time_ns()-tick)
                    assert out==expected
                rec[name+'_ns']=statistics.median(times)
            rows.append(rec)
    return rows


def campaign():
    small=[];large=[]
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(12):
            seed=982000+j*1000+i; d,t=(8,6) if i<6 else (9,7) if i<9 else (10,8)
            q=sg07.task(seed,reg,d,t);opt=oracle(q,transition_cap=10_000_000)['objective']; init=initial(q)
            runs={a:[] for a in ('sg10','sparse')};ref=None
            for rep in range(3):
                order=list(runs);random.Random(seed+rep*31).shuffle(order)
                for arm in order:
                    z=run(q,arm);r=z['receipt']; parity={k:v for k,v in r.items() if k!='classes'}
                    if ref is None:ref=parity
                    assert parity==ref and r['objective']<=opt<=r['upper']
                    assert verify_dual_trace(q,init,r['events'],transition_cap=2048,proposal_cap=24)==ref
                    runs[arm].append(z)
            small.append(dict(seed=seed,regime=reg,task=q,optimum=opt,runs=runs))
        for i in range(8):
            seed=986000+j*1000+i;d,t=(16,8) if i<3 else (24,10) if i<6 else (32,12)
            q=sg07.task(seed,reg,d,t)
            q['k']=[2+int((seed+v)%3==0) for v in range(t)]
            runs={}
            for arm,raw in [('raw_cap',8192),('class_cap',None)]:
                z=run(q,'sparse',raw_cap=raw,class_cap=8192);r=z['receipt']
                assert replay(q,initial(q),r['events'],raw_cap=raw,class_cap=8192,proposal_cap=24)==r
                runs[arm]=z
            large.append(dict(seed=seed,regime=reg,task=q,runs=runs))
            print('large',seed,d,[(k,z['receipt']['gap'],z['receipt']['classes']) for k,z in runs.items()],flush=True)
    summary={}
    for a in ('sg10','sparse'):
        med=[statistics.median(z['cpu_ns'] for z in r['runs'][a]) for r in small]
        rr=[r['runs'][a][0]['receipt'] for r in small]
        summary[a]=dict(cpu_ms=statistics.mean(med)/1e6,certified=sum(z['gap']==0 for z in rr),raw=sum(z['transitions'] for z in rr),
                        classes=sum(z.get('classes',0) for z in rr),mean_gap=statistics.mean(z['gap'] for z in rr))
    scaling={}
    for d in (16,24,32):
        roots=[r for r in large if r['task']['d']==d];scaling[d]={}
        for a in ('raw_cap','class_cap'):
            z=[r['runs'][a] for r in roots];rr=[r['receipt'] for r in z]
            scaling[d][a]=dict(roots=len(rr),certified=sum(r['gap']==0 for r in rr),mean_gap=statistics.mean(r['gap'] for r in rr),
                               raw=sum(r['transitions'] for r in rr),classes=sum(r['classes'] for r in rr),cpu_ms=statistics.mean(r['cpu_ns'] for r in z)/1e6,
                               peak_live=max(r['peak_live'] for r in z),stopped=sum(r['stop']!='certified' for r in z))
    return dict(version='SG-11/v0.1',small_roots=len(small),small_old_replays=len(small)*6,large_roots=len(large),large_same_implementation_replays=len(large)*2,
                summary=summary,scaling=scaling,micro=micro(),small=small,large=large)


if __name__=='__main__':
    z=campaign();Path('sg11_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k not in ('small','large')},indent=2))
