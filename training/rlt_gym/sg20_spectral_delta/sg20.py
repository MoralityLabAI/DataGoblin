"""SG-20 paired full/differential certificate acquisition on evolving tasks."""
import copy
import json
import random
import statistics
import time
from pathlib import Path

import sg07
import sg11
from sg15_lp import propose
from sg17_lp import chain_lp
from sg20_delta import State, Task
from datagoblin.training_gym.spectral_blocks import BlockSearch, root_vector
from datagoblin.training_gym.spectral_block_replay import verify
from datagoblin.training_gym.spectral_face import recover_face, verify_certificate
from datagoblin.training_gym.spectral_primal import row_sweep
from datagoblin.training_gym.spectral_quotient import validate

ARMS=('full','delta','snapshot','lp_every')
PHASES=('one','column','four_columns','sixteen','bit','budget_up','mixed','coordinate','translation','lambda','budget_down','large')


def independent_patch(q,edits):
    q=copy.deepcopy(q)
    for i,j,v in edits.get('u',[]): q['u'][i][j]=v
    for name in ('k','x','bits'):
        for i,v in edits.get(name,[]): q[name][i]=v
    if 'lambda' in edits: q['lambda']=edits['lambda']
    return validate(q)


def trajectory(seed,reg):
    r=random.Random(seed);q=sg07.task(seed,reg,32,12);q['k']=[7+(seed+i)%2 for i in range(q['T'])]
    anchor=copy.deepcopy(q);seen={json.dumps(q,sort_keys=True)};rows=[]
    def utility(cols,n,scale=1):
        sites=r.sample([(i,j) for j in cols for i in range(q['T'])],n)
        return [[i,j,q['u'][i][j]+r.choice((-1,1))*r.randint(1,scale)] for i,j in sites]
    for phase in PHASES:
        if phase=='one': p={'u':utility(r.sample(range(32),1),1)}
        elif phase=='column': p={'u':utility(r.sample(range(32),1),4)}
        elif phase=='four_columns': p={'u':utility(r.sample(range(32),4),8)}
        elif phase=='sixteen': p={'u':utility(r.sample(range(32),16),16,2)}
        elif phase=='bit':
            j=r.randrange(32);p={'bits':[[j,q['bits'][j]+1]]}
        elif phase in ('budget_up','budget_down'):
            i=r.randrange(12);p={'k':[[i,q['k'][i]+(1 if phase=='budget_up' else -1)]]}
        elif phase=='mixed':
            j=r.randrange(32);i=r.randrange(12);p={'u':utility([j],2),'bits':[[j,q['bits'][j]+1]],'k':[[i,q['k'][i]+1]]}
        elif phase=='coordinate': p={'x':[[12,q['x'][-1]+1]]}
        elif phase=='translation': p={'x':[[i,v+1] for i,v in enumerate(q['x'])]}
        elif phase=='lambda': p={'lambda':q['lambda']+1}
        else: p={'u':utility(list(range(32)),48,3)}
        q=independent_patch(q,p);sig=json.dumps(q,sort_keys=True)
        assert sig not in seen;seen.add(sig);rows.append(dict(phase=phase,patch=p,task=q))
    return anchor,rows


def record(s,kind):
    return dict(kind=kind,schedule=list(s.schedule),eta=list(s.eta),den=s.den,certificate=s.certificate())


def repair(s):
    if s.certificate()['certified']: return s
    a=s.with_schedule(s.rounded())
    if a.objective>s.objective:s=a
    if not s.certificate()['certified']:
        a,_=row_sweep(s.q.thaw(),list(s.schedule));s=s.with_schedule(a)
    if not s.certificate()['certified']:
        z=recover_face(s.q.thaw(),list(s.eta),s.den,node_cap=512)
        if z['schedule'] is not None:s=s.with_schedule(z['schedule'])
    return s


def finish(s,force_lp=False):
    log=[record(s,'retained')];calls=0;block=None;transfer=False
    if not force_lp:
        s=repair(s);log.append(record(s,'warm_repair'));transfer=s.certificate()['certified']
    for kind,fn in (('chain',chain_lp),('direct',propose)):
        if (not force_lp or calls) and s.certificate()['certified']:break
        q=s.q.thaw();p,info=fn(q);calls+=1
        if p is None:continue
        e,den=p;z=repair(State.build(s.q,e,den,s.schedule));log.append(record(z,kind))
        best=z.schedule if z.objective>s.objective else s.schedule
        if z.bound*s.den<s.bound*z.den:s=z.with_schedule(best)
        else:s=s.with_schedule(best)
    if not s.certificate()['certified']:
        q=s.q.thaw();init=list(s.schedule);extra=[root_vector(q)]
        search=BlockSearch(q,init,event_cap=2048,extra=extra);r=search.run()
        s=s.with_schedule(r['schedule']);block=dict(initial=init,extra=extra,receipt=r)
    c=s.certificate();upper=min(c['integer_upper'],block['receipt']['upper']) if block else c['integer_upper']
    return s,dict(**record(s,'final'),upper=upper,gap=upper-s.objective,log=log,lp_calls=calls,
                  transfer=transfer,block=block)


def audit(q,r):
    for z in [r,*r['log']]:
        assert verify_certificate(q,z['schedule'],z['eta'],z['den'])==z['certificate']
    upper=r['certificate']['integer_upper']
    if r['block']:
        b=r['block'];z=b['receipt'];assert verify(q,b['initial'],b['extra'],z['events'],event_cap=2048)=={k:v for k,v in z.items() if k!='peak_stored_blocks'}
        upper=min(upper,z['upper']);assert z['schedule']==r['schedule']
    assert r['upper']==upper and r['gap']==upper-r['certificate']['objective']>=0


def run(old,row,arm):
    tick=time.process_time_ns()
    if arm=='snapshot': s,n,m=old.replace(row['task'])
    else:s,n,m=old.apply(row['patch'],full=arm in ('full','lp_every'))
    prep=time.process_time_ns()-tick;retained=s.certificate()['certified']
    s,r=finish(s,force_lp=arm=='lp_every');elapsed=time.process_time_ns()-tick
    check=time.process_time_ns();assert s.q.thaw()==row['task'];audit(row['task'],r)
    return s,dict(**r,phase=row['phase'],cpu_ns=elapsed,prepare_ns=prep,audit_ns=time.process_time_ns()-check,
                  dual_columns=n,primal_columns=m,retained_certified=retained)


def summarize(roots,phase=None):
    result={}
    for arm in ARMS:
        groups=[[z for rep in r['arms'][arm] for z in rep if phase is None or z['phase']==phase] for r in roots]
        flat=[z for zz in groups for z in zz]
        times=[statistics.mean(z['cpu_ns'] for z in zz) for zz in groups]
        base=[statistics.mean(z['cpu_ns'] for rep in r['arms']['full'] for z in rep if phase is None or z['phase']==phase) for r in roots]
        result[arm]=dict(runs=len(flat),certified=sum(z['gap']==0 for z in flat),mean_gap=statistics.mean(z['gap'] for z in flat),
                         cpu_ms=statistics.mean(times)/1e6,prepare_us=statistics.mean(z['prepare_ns'] for z in flat)/1e3,
                         audit_ms=statistics.mean(z['audit_ns'] for z in flat)/1e6,
                         dual_columns=sum(z['dual_columns'] for z in flat),primal_columns=sum(z['primal_columns'] for z in flat),
                         lp_calls=sum(z['lp_calls'] for z in flat),retained=sum(z['retained_certified'] for z in flat),
                         transfers=sum(z['transfer'] for z in flat),blocks=sum(z['block'] is not None for z in flat),
                         wins=sum(x<y for x,y in zip(times,base,strict=True)))
    return result


def campaign():
    roots=[]
    for j,reg in enumerate(('persistent','rotating','mixed')):
        for i in range(8):
            seed=2010000+j*1000+i;anchor,rows=trajectory(seed,reg)
            s,a=finish(State.build(Task.from_task(anchor),root_vector(anchor),1,sg11.initial(anchor)))
            audit(anchor,a);arms={arm:[] for arm in ARMS}
            for rep in range(2):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for arm in order:
                    state=s;res=[]
                    for row in rows:
                        state,z=run(state,row,arm);res.append(z)
                    arms[arm].append(res)
                for step in range(len(rows)):
                    zz=[arms[arm][-1][step] for arm in ARMS]
                    assert max(z['certificate']['objective'] for z in zz)<=min(z['upper'] for z in zz)
                    keys=('schedule','eta','den','certificate','upper','gap','log','lp_calls','transfer','block')
                    assert all(all(z[k]==zz[0][k] for k in keys) for z in zz[1:3])
            roots.append(dict(seed=seed,regime=reg,anchor=anchor,initial=a,queries=rows,arms=arms))
            Path('sg20_progress.json').write_text(json.dumps(roots))
            print(seed,flush=True)
    return dict(version='SG-20/v0.1',roots=roots,summary=summarize(roots),by_phase={p:summarize(roots,p) for p in PHASES})


if __name__=='__main__':
    z=campaign();Path('sg20_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='roots'},indent=2))
