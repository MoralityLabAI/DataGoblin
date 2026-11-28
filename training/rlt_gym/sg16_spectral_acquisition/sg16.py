"""SG-16 CPU development: learned strategy selection versus deterministic escalation."""
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import sg07
import sg11
from sg15_lp import propose
from datagoblin.training_gym.spectral_acquisition import Portfolio, replay
from datagoblin.training_gym.spectral_blocks import root_vector
from datagoblin.training_gym.spectral_primal import recover
from datagoblin.training_gym.spectral_quotient import coordinate, score

PLANS = ('lp', 'one', 'four', 'block')
ARMS = (*PLANS, 'gap_rule', 'learned', 'shuffled')
REGIMES = ('persistent', 'rotating', 'mixed')
CPU_NS = 30_000_000
BLOCK_CAP = 2048
NODE_CAP = 512
REPEATS = 2


def task(seed, regime):
    q = sg07.task(seed, regime, 32, 12)
    q['k'] = [7 + (seed+i)%2 for i in range(q['T'])]
    return q


def setup(q):
    e = root_vector(q)
    a, _ = recover(q, sg11.initial(q), e, 'dual')
    p = Portfolio(q, a)
    p.dual(e, recover=False)
    return p, a, e


def features(q, p, e):
    v = score(q, p.schedule)
    means = [sum(row[j] for row in q['u'])/q['T'] for j in range(q['d'])]
    used = 0
    for m in p.schedule:
        used |= m
    return [1., q['lambda']/3, statistics.mean(q['bits'])/3, (q['x'][-1]-q['x'][0])/24,
            v['utility']/500, v['cost']/500, v['objective']/500, min(4., p.gap/32),
            used.bit_count()/q['d'], statistics.mean(e)/16, statistics.pstdev(e)/16,
            statistics.pstdev(means)/8]


def run(q, arm, model, limit):
    start = time.process_time_ns()
    p, initial, eta = setup(q)
    original = eta.copy()
    plan = arm
    if arm == 'gap_rule':
        plan = 'one' if p.gap <= 8 else 'lp'
    elif arm in ('learned', 'shuffled'):
        x = features(q, p, eta)
        plan = min(PLANS, key=lambda m: sum(a*b for a,b in zip(model[arm][m], x, strict=True)))
    root_gap = p.gap
    stages, block_work = [], 0
    def expired():
        return bool(limit and time.process_time_ns()-start >= limit)
    def ready():
        return bool(p.gap and not expired())
    def timed(name, fn):
        tick = time.process_time_ns()
        old = p.gap
        detail = fn()
        assert p.gap <= old
        stages.append(dict(stage=name, cpu_ns=time.process_time_ns()-tick,
                           before=old, after=p.gap, detail=detail))
    def sweeps(n):
        for i in range(n):
            if not ready():
                break
            def action():
                for j in (range(q['T']-1,-1,-1) if i%2==0 else range(q['T'])):
                    eta[j] = coordinate(q,q['T'],0,eta,j)
                p.dual(eta, node_cap=NODE_CAP)
            timed('sweep', action)
    def blocks(cap):
        nonlocal block_work
        if ready() and cap > 0:
            def action():
                p.block([original, eta] if eta != original else [original], cap, deadline=expired)
                return p.records[-1]['receipt']['event_count']
            timed('block', action)
            block_work += stages[-1]['detail']
    def lp():
        if not ready():
            return
        def action():
            suggestion, info = propose(q)
            if suggestion is not None:
                p.dual(*suggestion, node_cap=NODE_CAP)
            return info
        timed('lp', action)
    if plan == 'one':
        sweeps(1)
    elif plan == 'four':
        sweeps(4)
    elif plan == 'block':
        blocks(256)
    lp()
    blocks(BLOCK_CAP-block_work)
    r = p.receipt()
    elapsed = time.process_time_ns()-start
    assert replay(q, initial, r['records']) == r
    return dict(receipt=r, initial=initial, plan=plan, root_gap=root_gap, stages=stages,
                cpu_ns=elapsed, overshoot_ns=max(0,elapsed-limit) if limit else 0,
                lp_calls=sum(s['stage']=='lp' for s in stages), block_events=block_work)


def train():
    xx, yy, rows = [], [], []
    for j, reg in enumerate(REGIMES):
        for i in range(8):
            seed = 1600000+j*1000+i
            q = task(seed,reg)
            p, _, eta = setup(q)
            order = list(PLANS)
            random.Random(seed).shuffle(order)
            arms = {m:run(q,m,{},0) for m in order}
            xx.append(features(q,p,eta))
            # Penalize unresolved gap. Time is measured once per training root.
            yy.append([arms[m]['cpu_ns']/1e6+100*arms[m]['receipt']['gap'] for m in PLANS])
            rows.append(dict(seed=seed,regime=reg,task=q,arms=arms))
    a,b = np.asarray(xx),np.asarray(yy)
    # Log target keeps outlier CPU observations from dominating the fit.
    b = np.log1p(b)
    order = list(range(len(xx)))
    random.Random(16099).shuffle(order)
    gram = a.T@a + 2*np.eye(a.shape[1])
    def fit(y):
        v = np.linalg.solve(gram,a.T@y)
        return {m:v[:,i].tolist() for i,m in enumerate(PLANS)}
    return dict(learned=fit(b),shuffled=fit(b[order]),training_rows=rows,
                train_mean_loss={m:float(np.expm1(b[:,i]).mean()) for i,m in enumerate(PLANS)})


def summarize(rows):
    out = {}
    for budget in ('work','cpu'):
        out[budget] = {}
        for m in ARMS:
            groups = [r['arms'][budget][m] for r in rows]
            flat = [z for group in groups for z in group]
            out[budget][m] = dict(cpu_ms=statistics.mean(statistics.median(z['cpu_ns'] for z in g) for g in groups)/1e6,
                gap=statistics.mean(z['receipt']['gap'] for z in flat),
                certified=sum(z['receipt']['gap']==0 for z in flat), runs=len(flat),
                lp_calls=sum(z['lp_calls'] for z in flat), block_events=sum(z['block_events'] for z in flat),
                max_overshoot_ms=max(z['overshoot_ns'] for z in flat)/1e6,
                choices={p:sum(z['plan']==p for z in flat) for p in PLANS})
    return out


def campaign():
    model=train(); rows=[]
    Path('sg16_progress.json').write_text(json.dumps(dict(model=model,rows=rows)))
    for j,reg in enumerate(REGIMES):
        for i in range(12):
            seed=1610000+j*1000+i; q=task(seed,reg)
            arms={b:{m:[] for m in ARMS} for b in ('work','cpu')}
            for budget in arms:
                for rep in range(REPEATS):
                    order=list(ARMS);random.Random(seed+31*rep+int(budget=='cpu')).shuffle(order)
                    for m in order:
                        arms[budget][m].append(run(q,m,model,CPU_NS if budget=='cpu' else 0))
            all_r=[z['receipt'] for b in arms.values() for a in b.values() for z in a]
            lo=max(r['objective'] for r in all_r); hi=min(r['upper'] for r in all_r)
            assert lo<=hi
            for m in ARMS:
                assert arms['work'][m][0]['receipt']==arms['work'][m][1]['receipt']
            rows.append(dict(seed=seed,regime=reg,task=q,arms=arms,lower=lo,upper=hi))
            Path('sg16_progress.json').write_text(json.dumps(dict(model=model,rows=rows)))
            print(seed,lo,hi,flush=True)
    return dict(version='SG-16/v0.1',roots=len(rows),model=model,repeats=REPEATS,cpu_cap_ns=CPU_NS,
                eval_replays=len(rows)*len(ARMS)*2*REPEATS,train_replays=len(model['training_rows'])*len(PLANS),
                exact_cross_policy=sum(r['lower']==r['upper'] for r in rows),summary=summarize(rows),rows=rows)


if __name__=='__main__':
    z=campaign();Path('sg16_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k not in ('rows','model')},indent=2))
