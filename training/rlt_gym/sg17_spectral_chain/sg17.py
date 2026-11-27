"""SG-17: paired certificate acquisition with chain LP and cut generation."""
import json
import random
import statistics
import time
from pathlib import Path

import sg07
import sg11
from sg15_lp import propose
from sg17_lp import chain_lp, cut_lp
from datagoblin.training_gym.spectral_blocks import BlockSearch, root_vector
from datagoblin.training_gym.spectral_block_replay import verify
from datagoblin.training_gym.spectral_face import recover_face, verify_certificate
from datagoblin.training_gym.spectral_primal import dual_round, recover, row_sweep
from datagoblin.training_gym.spectral_quotient import score

ARMS = ('direct', 'chain', 'cuts8', 'cuts2_chain')


def run(q, arm):
    tick = time.process_time_ns()
    root = root_vector(q)
    a, _ = recover(q, sg11.initial(q), root, 'dual')
    eta, den = root, 1
    cert = verify_certificate(q, a, eta, den)
    log = []
    block = None

    def attempt(kind):
        nonlocal a, eta, den, cert
        if cert['certified']:
            return
        start = time.process_time_ns()
        if kind == 'direct':
            p, info = propose(q)
        elif kind == 'chain':
            p, info = chain_lp(q)
        else:
            p, info = cut_lp(q, a, root, round_cap=int(kind[4:]))
        ev = dict(kind=kind, info=info)
        if p is not None:
            e, z = p
            scaled = {**q, 'u': [[v*z for v in r] for r in q['u']], 'lambda': q['lambda']*z}
            candidate, _ = row_sweep(q, dual_round(scaled, e))
            if score(q, candidate)['objective'] > cert['objective']:
                a = candidate
            c = verify_certificate(q, a, e, z)
            if not c['certified']:
                f = recover_face(q, e, z, node_cap=4096)
                ev.update(face=f['status'], nodes=f['nodes'])
                if f['schedule'] is not None:
                    a = f['schedule']
                    c = verify_certificate(q, a, e, z)
            if c['bound_num']*den < cert['bound_num']*z:
                eta, den = e, z
            previous = cert
            cert = verify_certificate(q, a, eta, den)
            assert cert['objective'] >= previous['objective'] and cert['integer_upper'] <= previous['integer_upper']
            ev.update(schedule=list(a), eta=e, den=z, candidate=c, best=cert.copy())
        ev['cpu_ns'] = time.process_time_ns()-start
        log.append(ev)

    if arm == 'cuts2_chain':
        attempt('cuts2')
        attempt('chain')
    else:
        attempt(arm)
    primary = cert['certified']
    if not primary and arm != 'direct':
        attempt('direct')
    if not cert['certified']:
        init = list(a)
        s = BlockSearch(q, init, event_cap=2048, extra=[root])
        r = s.run()
        a = r['schedule']
        cert = verify_certificate(q, a, eta, den)
        block = dict(initial=init, extra=[root], receipt=r)
    upper = min(cert['integer_upper'], block['receipt']['upper']) if block else cert['integer_upper']
    elapsed = time.process_time_ns()-tick
    assert verify_certificate(q, a, eta, den) == cert
    for ev in log:
        if 'candidate' in ev:
            assert verify_certificate(q, ev['schedule'], ev['eta'], ev['den']) == ev['candidate']
    if block:
        r = block['receipt']
        assert verify(q, block['initial'], block['extra'], r['events'], event_cap=2048) == {
            k:v for k,v in r.items() if k != 'peak_stored_blocks'}
    return dict(objective=cert['objective'], upper=upper, gap=upper-cert['objective'],
                cpu_ns=elapsed, primary_certified=primary, schedule=a, eta=eta, den=den,
                certificate=cert, log=log, block=block)


def summarize(rows):
    out = {}
    for arm in ARMS:
        rr = [r['arms'][arm] for r in rows]
        flat = [z for a in rr for z in a]
        timings = [statistics.median(z['cpu_ns'] for z in a) for a in rr]
        base = [statistics.median(z['cpu_ns'] for z in r['arms']['direct']) for r in rows]
        out[arm] = dict(roots=len(rows), runs=len(flat), cpu_ms=statistics.mean(timings)/1e6,
                        gap=statistics.mean(z['gap'] for z in flat),
                        certified=sum(z['gap']==0 for z in flat), primary_certified=sum(z['primary_certified'] for z in flat),
                        fallback_direct=sum(any(e['kind']=='direct' for e in z['log']) for z in flat) if arm!='direct' else 0,
                        fallback_blocks=sum(z['block'] is not None for z in flat),
                        wins=sum(x<y for x,y in zip(timings,base,strict=True)),
                        solver_calls=sum(e['info'].get('rounds',1) for z in flat for e in z['log']))
    return out


def campaign():
    rows = []
    for j, reg in enumerate(('persistent','rotating','mixed')):
        for i in range(12):
            seed=1710000+j*1000+i
            t=12 if i<6 else 16
            q=sg07.task(seed,reg,32,t);q['k']=[7+(seed+v)%2 for v in range(t)]
            arms={m:[] for m in ARMS}
            for rep in range(2):
                order=list(ARMS);random.Random(seed+31*rep).shuffle(order)
                for m in order:
                    arms[m].append(run(q,m))
            flat=[z for a in arms.values() for z in a]
            assert max(z['objective'] for z in flat)<=min(z['upper'] for z in flat)
            rows.append(dict(seed=seed,regime=reg,task=q,arms=arms))
            Path('sg17_progress.json').write_text(json.dumps(rows))
            print(seed,[(m,round(arms[m][0]['cpu_ns']/1e6,2),arms[m][0]['gap']) for m in ARMS],flush=True)
    return dict(version='SG-17/v0.1',roots=len(rows),repeats=2,summary=summarize(rows),
                by_depth={t:summarize([r for r in rows if r['task']['T']==t]) for t in (12,16)},rows=rows)


if __name__=='__main__':
    z=campaign();Path('sg17_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='rows'},indent=2))
