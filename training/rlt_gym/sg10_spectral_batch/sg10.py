"""SG-10 paired scalar, quotient and batch search timing."""
import json
import random
import statistics
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import sg07
from datagoblin.training_gym.spectral_bank import BankSession
from datagoblin.training_gym.spectral_batch import BatchedBankSession
from datagoblin.training_gym.spectral_dual import verify_dual_trace
from datagoblin.training_gym.spectral_routing import oracle

ARMS = ('scalar', 'dedup', 'batch', 'hybrid')


def run(q, arm):
    def make(task, initial, **kw):
        if arm == 'scalar':
            return BankSession(task, initial, split=True, **kw)
        return BatchedBankSession(task, initial, mode=arm, **kw)

    with patch.object(sg07, 'CachedDualSession', make):
        return sg07.run(q, 'lazy12')


def campaign():
    rows = []
    replays = 0
    for j, regime in enumerate(('persistent', 'rotating', 'mixed')):
        for i in range(20):
            seed = 970000 + j * 1000 + i
            d, t = (8, 6) if i < 10 else (9, 7) if i < 15 else (10, 8)
            q = sg07.task(seed, regime, d, t)
            init = sg07.initial(q)
            optimum = oracle(q, transition_cap=10_000_000)['objective']
            runs = {a: [] for a in ARMS}
            ref = None
            for rep in range(3):
                order = list(ARMS)
                random.Random(seed + 31 * rep).shuffle(order)
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
            rows.append(dict(seed=seed, regime=regime, task=q, optimum=optimum, receipt=ref,
                             runs=runs, memory=memory))
    summary = {}
    base = [statistics.median(z['cpu_ns'] for z in r['runs']['scalar']) for r in rows]
    for arm in ARMS:
        first = [r['runs'][arm][0] for r in rows]
        timing = [statistics.median(z['cpu_ns'] for z in r['runs'][arm]) for r in rows]
        totals = {k: sum(z['cache'].get(k, 0) for z in first) for k in (
            'bank_queries', 'component_steps', 'bank_steps', 'expansions', 'raw_actions',
            'unique_unions', 'dominated', 'bound_masks', 'vector_actions', 'vector_batches', 'vector_masks')}
        summary[arm] = dict(cpu_ms=statistics.mean(timing)/1e6,
                            wins=sum(a < b for a, b in zip(timing, base, strict=True)),
                            expanding_cpu_ms=statistics.mean(v for v, r in zip(timing, rows, strict=True)
                                                             if r['receipt']['transitions'])/1e6,
                            peak_traced_bytes=statistics.mean(r['memory'][arm] for r in rows if r['memory']),
                            counts=totals)
    return dict(version='SG-10/v0.1', roots=len(rows), replays=replays,
                certified=sum(r['receipt']['gap']==0 for r in rows),
                expanding_roots=sum(bool(r['receipt']['transitions']) for r in rows),
                mean_gap=statistics.mean(r['receipt']['gap'] for r in rows),
                mean_regret=statistics.mean(r['optimum']-r['receipt']['objective'] for r in rows),
                summary=summary, rows=rows)


if __name__ == '__main__':
    z = campaign()
    Path('sg10_results.json').write_text(json.dumps(z, indent=2)+'\n')
    print(json.dumps({k: v for k, v in z.items() if k != 'rows'}, indent=2))
