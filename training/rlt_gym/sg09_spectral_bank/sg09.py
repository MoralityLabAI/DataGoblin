"""SG-09 paired timing with exact certificate replay and batched integer evaluation."""
import json
import random
import statistics
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import sg07
from datagoblin.training_gym.spectral_bank import BankSession
from datagoblin.training_gym.spectral_dual import verify_dual_trace
from datagoblin.training_gym.spectral_routing import oracle

ARMS = ('lru12', 'lru128', 'prefix', 'split')


def run(q, arm):
    if arm.startswith('lru'):
        return sg07.run(q, 'lazy12' if arm == 'lru12' else 'lazy128')

    def make(task, initial, **kw):
        return BankSession(task, initial, split=arm == 'split', **kw)

    with patch.object(sg07, 'CachedDualSession', make):
        return sg07.run(q, 'lazy12')


def batch_micro(q):
    import numpy as np

    s = BankSession(q, sg07.initial(q), split=True)
    bank = s.bank
    data = np.frombuffer(bank.data, dtype=np.int64)
    t = q['T']
    masks = np.arange(1 << q['d'], dtype=np.int64)
    offsets = np.array(bank.rows[t], dtype=np.int64)[:, None]

    def batch():
        v = data[offsets + (masks & bank.low_mask)] + data[offsets + bank.low_n + (masks >> bank.cut)]
        return np.minimum(v.min(axis=0), s.weak[t])

    expected = bank.batch(t, masks.tolist(), s.weak[t])
    assert batch().tolist() == expected
    out = {}
    for label, fun in [('scalar', lambda: bank.batch(t, masks.tolist(), s.weak[t])), ('numpy', batch)]:
        times = []
        for _ in range(5):
            tick = time.process_time_ns()
            fun()
            times.append(time.process_time_ns() - tick)
        out[label] = statistics.median(times)
    return out


def campaign():
    rows = []
    replays = 0
    for j, regime in enumerate(('persistent', 'rotating', 'mixed')):
        for i in range(20):
            seed = 950000 + j * 1000 + i
            d, t = (8, 6) if i < 10 else (9, 7) if i < 15 else (10, 8)
            q = sg07.task(seed, regime, d, t)
            init = sg07.initial(q)
            optimum = oracle(q, transition_cap=10_000_000)['objective']
            runs = {a: [] for a in ARMS}
            ref = None
            for rep in range(3):
                order = list(ARMS)
                random.Random(seed + rep * 31).shuffle(order)
                for arm in order:
                    z = run(q, arm)
                    r = z['receipt']
                    if ref is None:
                        ref = r
                    assert r == ref
                    assert r['objective'] <= optimum <= r['upper']
                    assert verify_dual_trace(q, init, r['events'], transition_cap=sg07.CAP,
                                             proposal_cap=sg07.PROPOSALS) == r
                    replays += 1
                    runs[arm].append({k: v for k, v in z.items() if k != 'receipt'})
            memory = {}
            if i in (0, 10, 15):
                for arm in ARMS:
                    tracemalloc.start()
                    z = run(q, arm)
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    assert z['receipt'] == ref
                    memory[arm] = peak
            micro = batch_micro(q) if i in (0, 10, 15) else None
            rows.append(dict(seed=seed, regime=regime, task=q, optimum=optimum, receipt=ref,
                             runs=runs, memory=memory, batch=micro))
    summary = {}
    base = [statistics.median(z['cpu_ns'] for z in r['runs']['lru12']) for r in rows]
    for arm in ARMS:
        first = [r['runs'][arm][0] for r in rows]
        timing = [statistics.median(z['cpu_ns'] for z in r['runs'][arm]) for r in rows]
        summary[arm] = dict(cpu_ms=statistics.mean(timing)/1e6,
                            wins=sum(a < b for a, b in zip(timing, base, strict=True)),
                            peak_traced_bytes=statistics.mean(r['memory'][arm] for r in rows if r['memory']),
                            total_dynamic_steps=sum(z['cache']['component_steps'] for z in first),
                            total_bank_steps=sum(z['cache'].get('bank_steps',0) for z in first),
                            mean_bank_bytes=statistics.mean(z['cache'].get('bank_bytes',0) for z in first),
                            max_bank_bytes=max(z['cache'].get('bank_bytes',0) for z in first))
    batch = {k: statistics.mean(r['batch'][k] for r in rows if r['batch'])/1e6 for k in ('scalar','numpy')}
    return dict(version='SG-09/v0.1', roots=len(rows), replays=replays,
                certified=sum(r['receipt']['gap']==0 for r in rows),
                mean_gap=statistics.mean(r['receipt']['gap'] for r in rows),
                mean_regret=statistics.mean(r['optimum']-r['receipt']['objective'] for r in rows),
                summary=summary,batch_ms=batch,rows=rows)


if __name__ == '__main__':
    z = campaign()
    Path('sg09_results.json').write_text(json.dumps(z, indent=2)+'\n')
    print(json.dumps({k:v for k,v in z.items() if k!='rows'},indent=2))
