"""SG-13 fresh-root branching campaign; no native model data."""
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
from datagoblin.training_gym.spectral_branching import BranchPolicy, free_bits, teacher_rows

ARMS = ('analytic', 'strong', 'pseudo', 'learned', 'shuffled')
CAP = 2048
CPU_NS = 15_000_000
REPEATS = 2


def task(seed, regime, d):
    q = sg07.task(seed, regime, d, {16: 8, 24: 10, 32: 12}[d])
    lo = {16: 4, 24: 5, 32: 7}[d]
    q['k'] = [lo + (seed + i) % 2 for i in range(q['T'])]
    return q


def train():
    x, y, null, weights, counts = [], [], [], [], []
    for j, reg in enumerate(('persistent', 'rotating', 'mixed')):
        for i in range(8):
            seed = 1300000 + j * 1000 + i
            q = task(seed, reg, (16, 24, 32)[i % 3])
            s = BlockSearch(q, sg11.initial(q), event_cap=CAP, extra=[root_vector(q)])
            rows, splits, states = [], 0, 0
            while len(s.events) < CAP and (ident := s.best()) is not None:
                b = s.live[ident]
                free = b.t > 1 and free_bits(q, b)
                if free:
                    if splits % 5 == 0 and states < 32:
                        rows.extend(teacher_rows(s, b))
                        states += 1
                    splits += 1
                s.advance(ident, s.relax.choose(b) if free else None)
            counts.append(dict(seed=seed, states=states, labels=len(rows)))
            if rows:
                yy = [v for _, v in rows]
                shuffled = yy.copy()
                random.Random(seed).shuffle(shuffled)
                x.extend(a for a, _ in rows)
                y.extend(yy)
                null.extend(shuffled)
                weights.extend([1 / len(rows)] * len(rows))
    a, w = np.asarray(x), np.asarray(weights)
    gram = a.T @ (a * w[:, None]) + .1 * np.eye(a.shape[1])
    return dict(counts=counts, labels=len(y), states=sum(v['states'] for v in counts),
                learned=np.linalg.solve(gram, a.T @ (w * y)).tolist(),
                shuffled=np.linalg.solve(gram, a.T @ (w * null)).tolist())


def run(q, kind, beta, cpu_cap):
    tick = time.process_time_ns()
    initial = sg11.initial(q)
    extra = [root_vector(q)]
    s = BlockSearch(q, initial, event_cap=CAP, extra=extra)
    policy = BranchPolicy(kind, beta)
    old = s.upper()
    reason = 'event_cap'
    policy_ns = 0
    while len(s.events) < CAP and (ident := s.best()) is not None:
        if cpu_cap and time.process_time_ns() - tick >= cpu_cap:
            reason = 'cpu_deadline'
            break
        b = s.live[ident]
        stamp = time.process_time_ns()
        j = policy.choose(s, b) if b.t > 1 and free_bits(q, b) else None
        policy_ns += time.process_time_ns() - stamp
        policy.advance(s, ident, j)
        new = s.upper()
        assert new <= old
        old = new
    r = s.receipt()
    elapsed = time.process_time_ns() - tick
    if r['gap'] == 0:
        reason = 'certified'
    checked = verify(q, initial, extra, r['events'], event_cap=CAP)
    assert checked == {k: v for k, v in r.items() if k != 'peak_stored_blocks'}
    return dict(receipt=r, cpu_ns=elapsed, policy_ns=policy_ns, probes=policy.probes,
                stop=reason, overshoot_ns=max(0, elapsed-cpu_cap) if cpu_cap else 0)


def campaign():
    model = train()
    rows = []
    for j, reg in enumerate(('persistent', 'rotating', 'mixed')):
        for i in range(8):
            seed = 1310000 + j * 1000 + i
            q = task(seed, reg, 32)
            arms = {budget: {arm: [] for arm in ARMS} for budget in ('event', 'cpu')}
            for budget in arms:
                for rep in range(REPEATS):
                    order = list(ARMS)
                    random.Random(seed + rep * 31 + (budget == 'cpu')).shuffle(order)
                    for arm in order:
                        z = run(q, arm, model.get(arm, ()), CPU_NS if budget == 'cpu' else 0)
                        arms[budget][arm].append(z)
            all_receipts = [z['receipt'] for b in arms.values() for a in b.values() for z in a]
            lo = max(r['objective'] for r in all_receipts)
            hi = min(r['upper'] for r in all_receipts)
            assert lo <= hi
            for arm in ARMS:
                a = arms['event'][arm]
                assert a[0]['receipt'] == a[1]['receipt']
            rows.append(dict(seed=seed, regime=reg, task=q, arms=arms, best_lower=lo, best_upper=hi))
            print('root', seed, 'complete', flush=True)
    summary = {}
    for budget in ('event', 'cpu'):
        summary[budget] = {}
        for arm in ARMS:
            rr = [r['arms'][budget][arm] for r in rows]
            means = {k: statistics.mean(statistics.mean(z['receipt'][k] for z in a) for a in rr)
                     for k in ('gap', 'objective', 'event_count', 'bound_calls')}
            means.update({k: statistics.mean(statistics.median(z[k] for z in a) for a in rr) / (1e6 if k.endswith('_ns') else 1)
                          for k in ('cpu_ns', 'policy_ns', 'probes')})
            means.update(certified_runs=sum(z['receipt']['gap'] == 0 for a in rr for z in a),
                         total_runs=len(rows)*REPEATS,
                         max_overshoot_ms=max(z['overshoot_ns'] for a in rr for z in a)/1e6,
                         max_peak=max(z['receipt']['peak_stored_blocks'] for a in rr for z in a))
            summary[budget][arm] = means
    return dict(version='SG-13/v0.1', model=model, roots=len(rows), repeats=REPEATS, cap=CAP,
                cpu_cap_ns=CPU_NS, replays=len(rows)*len(ARMS)*2*REPEATS,
                exact_cross_policy_roots=sum(r['best_lower']==r['best_upper'] for r in rows), summary=summary, rows=rows)


if __name__ == '__main__':
    z = campaign()
    Path('sg13_results.json').write_text(json.dumps(z, indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='rows'},indent=2))
