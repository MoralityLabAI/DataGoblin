"""SG-14: fresh-root primal recovery under equal soft CPU deadlines."""
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import sg07
import sg11
from datagoblin.training_gym.spectral_block_replay import verify
from datagoblin.training_gym.spectral_blocks import BlockSearch, root_vector
from datagoblin.training_gym.spectral_primal import recover
from datagoblin.training_gym.spectral_quotient import score

METHODS = ('base', 'row', 'dual', 'dual_row', 'pair')
ARMS = (*METHODS, 'learned', 'shuffled')
CAP, CPU_NS, REPEATS = 2048, 15_000_000, 2
REGIMES = ('persistent', 'rotating', 'mixed')


def task(seed, regime):
    q = sg07.task(seed, regime, 32, 12)
    q['k'] = [7 + (seed+i)%2 for i in range(q['T'])]
    return q


def features(q, initial, eta):
    v = score(q, initial)
    peaks = [max(row[j] for row in q['u']) for j in range(q['d'])]
    means = [sum(row[j] for row in q['u'])/q['T'] for j in range(q['d'])]
    union = 0
    for m in initial:
        union |= m
    switches = sum((a^b).bit_count() for a,b in zip(initial, initial[1:], strict=False))
    return [1., q['lambda']/3, statistics.mean(q['bits'])/3, (q['x'][-1]-q['x'][0])/24,
            v['utility']/500, v['cost']/500, v['objective']/500, union.bit_count()/q['d'],
            switches/(q['T']*q['d']), max(means)/16, statistics.pstdev(means)/8,
            statistics.mean(peaks)/16, statistics.mean(eta)/16, statistics.pstdev(eta)/16]


def run(q, arm, model, cpu_cap):
    start = time.process_time_ns()
    init = sg11.initial(q)
    eta = root_vector(q)
    method = arm
    if arm in ('learned', 'shuffled'):
        x = features(q, init, eta)
        method = max(METHODS, key=lambda m: sum(a*b for a,b in zip(model[arm][m], x, strict=True)))
    tick = time.process_time_ns()
    initial, gain = recover(q, init, eta, method)
    repair_ns = time.process_time_ns()-tick
    s = BlockSearch(q, initial, event_cap=CAP, extra=[eta])
    root_upper = s.upper()
    old = root_upper
    while len(s.events) < CAP and (i := s.best()) is not None:
        if cpu_cap and time.process_time_ns()-start >= cpu_cap:
            break
        b = s.live[i]
        free = ((1 << q['d'])-1) & ~(b.suffix|b.include|b.exclude)
        s.advance(i, s.relax.choose(b) if b.t > 1 and free else None)
        new = s.upper()
        assert new <= old
        old = new
    r = s.receipt()
    elapsed = time.process_time_ns()-start
    ref = verify(q, initial, [eta], r['events'], event_cap=CAP)
    assert ref == {k:v for k,v in r.items() if k!='peak_stored_blocks'}
    return dict(receipt=r, initial=initial, eta=eta, method=method, initial_gain=gain,
                initial_objective=score(q, initial)['objective'], baseline_objective=score(q, init)['objective'],
                root_upper=root_upper, cpu_ns=elapsed, repair_ns=repair_ns,
                overshoot_ns=max(0, elapsed-cpu_cap) if cpu_cap else 0)


def train():
    x, ys, rows = [], [], []
    for j, reg in enumerate(REGIMES):
        for i in range(8):
            seed = 1440000+j*1000+i
            q = task(seed, reg)
            order = list(METHODS)
            random.Random(seed).shuffle(order)
            arms = {m:run(q, m, {}, CPU_NS) for m in order}
            x.append(features(q, sg11.initial(q), root_vector(q)))
            ys.append([-arms[m]['receipt']['gap'] for m in METHODS])
            rows.append(dict(seed=seed, regime=reg, task=q, arms=arms))
    a, b = np.asarray(x), np.asarray(ys)
    order = list(range(len(x)))
    random.Random(14499).shuffle(order)
    gram = a.T@a + 2*np.eye(a.shape[1])
    def fit(y):
        v=np.linalg.solve(gram, a.T@y)
        return {m:v[:,i].tolist() for i,m in enumerate(METHODS)}
    return dict(learned=fit(b), shuffled=fit(b[order]), training_rows=rows,
                train_mean_gap={m:float(-b[:,i].mean()) for i,m in enumerate(METHODS)},
                fixed_train_choice=METHODS[int(b.mean(0).argmax())])


def summarize(rows):
    out = {}
    for budget in ('event', 'cpu'):
        out[budget] = {}
        for m in ARMS:
            a = [r['arms'][budget][m] for r in rows]
            flat = [z for zz in a for z in zz]
            out[budget][m] = dict(
                gap=statistics.mean(z['receipt']['gap'] for z in flat),
                objective=statistics.mean(z['receipt']['objective'] for z in flat),
                initial_gain=statistics.mean(z['initial_gain'] for z in flat),
                positive_initial=sum(z['initial_gain']>0 for z in flat),
                events=statistics.mean(z['receipt']['event_count'] for z in flat),
                cpu_ms=statistics.mean(statistics.median(z['cpu_ns'] for z in zz) for zz in a)/1e6,
                repair_ms=statistics.mean(statistics.median(z['repair_ns'] for z in zz) for zz in a)/1e6,
                certified=sum(z['receipt']['gap']==0 for z in flat),
                max_overshoot_ms=max(z['overshoot_ns'] for z in flat)/1e6,
                choices={v:sum(z['method']==v for z in flat) for v in METHODS})
    return out


def campaign():
    model = train()
    rows = []
    Path('sg14_progress.json').write_text(json.dumps(dict(model=model, rows=rows)))
    for j,reg in enumerate(REGIMES):
        for i in range(8):
            seed=1450000+j*1000+i
            q=task(seed,reg)
            arms={budget:{m:[] for m in ARMS} for budget in ('event','cpu')}
            for budget in arms:
                for rep in range(REPEATS):
                    order=list(ARMS)
                    random.Random(seed+31*rep+int(budget=='cpu')).shuffle(order)
                    for m in order:
                        arms[budget][m].append(run(q,m,model,CPU_NS if budget=='cpu' else 0))
            all_r=[z['receipt'] for b in arms.values() for v in b.values() for z in v]
            lo=max(r['objective'] for r in all_r);hi=min(r['upper'] for r in all_r)
            assert lo<=hi
            for m in ARMS:
                assert arms['event'][m][0]['receipt']==arms['event'][m][1]['receipt']
            rows.append(dict(seed=seed,regime=reg,task=q,arms=arms,best_lower=lo,best_upper=hi))
            Path('sg14_progress.json').write_text(json.dumps(dict(model=model, rows=rows)))
            print('root',seed,'complete',flush=True)
    q=json.loads(Path('sg13_exposed.json').read_text())
    exposed={m:run(q,m,model,0) for m in METHODS}
    return dict(version='SG-14/v0.1', evidence='offline synthetic development, same SG-12 proof engine',
                model=model, roots=len(rows), repeats=REPEATS, event_cap=CAP, cpu_cap_ns=CPU_NS,
                eval_replays=len(rows)*len(ARMS)*2*REPEATS,train_replays=len(model['training_rows'])*len(METHODS),
                exposed=exposed, exposed_task=q,
                exact_cross_policy_roots=sum(r['best_lower']==r['best_upper'] for r in rows),
                summary=summarize(rows),rows=rows)


if __name__=='__main__':
    z=campaign()
    Path('sg14_results.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k not in ('rows','model','exposed','exposed_task')},indent=2))
