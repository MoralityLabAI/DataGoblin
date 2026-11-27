"""SG-07: compare exact arithmetic caching and lazy multiplier reuse."""
import hashlib
import json
import random
import statistics
import time
import tracemalloc
from collections import OrderedDict
from pathlib import Path

from datagoblin.training_gym.spectral_cache import CachedDualSession
from datagoblin.training_gym.spectral_dual import DualSearchSession, coordinate_minimizer, verify_dual_trace
from datagoblin.training_gym.spectral_routing import oracle, score

ARMS = ('original', 'cache128', 'cache12', 'lazy128', 'lazy12')
CAP, PROPOSALS, DECISIONS = 2048, 24, 80
REPEATS = 3


def task(seed, regime, d, t):
    r = random.Random(seed)
    common = r.sample(range(d), 2)
    x, u = [r.randrange(-2, 3)], []
    for _ in range(t):
        x.append(x[-1] + r.randrange(1, 3))
        rotating = r.sample(range(d), 2)
        hot = common if regime == 'persistent' else rotating if regime == 'rotating' else [common[0], rotating[0]]
        u.append([r.randrange(-2, 6) + (r.randrange(9, 14) if j in hot else 0) for j in range(d)])
    return dict(d=d, T=t, bits=[r.randrange(1, 4) for _ in range(d)], x=x,
                k=[r.randrange(2, min(d, 4) + 1) for _ in range(t)], u=u, **{'lambda': r.randrange(1, 4)})


def initial(q):
    d, t = q['d'], q['T']
    def pick(i, fn):
        return sum(1 << j for j in sorted(range(d), key=lambda j: (-fn(j), j))[:q['k'][i]])
    pool = set(sorted(range(d), key=lambda j: (-sum(row[j] for row in q['u']) + q['lambda'] * q['bits'][j]
                                              * (q['x'][-1] - q['x'][0]), j))[:max(q['k'])])
    a, suffix = [0] * t, 0
    for i in range(t - 1, -1, -1):
        a[i] = pick(i, lambda j: q['u'][i][j] - (0 if suffix & (1 << j) else q['lambda'] * q['bits'][j]
                                                * (q['x'][i + 1] - q['x'][0])))
        suffix |= a[i]
    candidates = [a, [pick(i, lambda j: q['u'][i][j]) for i in range(t)],
                  [pick(i, lambda j: q['u'][i][j] + 100000 * (j in pool)) for i in range(t)]]
    return max(candidates, key=lambda a: score(q, a)['objective'])


def run(q, arm):
    started = time.process_time_ns()
    if arm == 'original':
        s = DualSearchSession(q, initial(q), transition_cap=CAP, proposal_cap=PROPOSALS)
    else:
        small = arm.endswith('12')
        s = CachedDualSession(q, initial(q), transition_cap=CAP, proposal_cap=PROPOSALS,
                              table_cap=12 if small else 128, scalar_cap=16 if small else 128)
    lazy = arm.startswith('lazy')
    tried, pool = set(), OrderedDict()
    coordinates = reuses = candidates = 0
    peak_live = len(s.live)
    for _ in range(DECISIONS):
        if not s.live:
            break
        t, mask, upper = min(s.frontier(), key=lambda a: (-a[2], a[0], a[1]))
        can_dual = (t, mask) not in tried and s.proposals < s.proposal_cap
        can_expand = s.transitions + len(s.actions[t - 1]) <= s.cap
        if can_dual:
            e = None
            if lazy:
                node = s.live[t, mask]
                gap = upper - s.incumbent
                opts = []
                for depth, vec in pool:
                    if depth == t:
                        candidates += 1
                        v = s.candidate_bound(t, mask, list(vec)) + node.value
                        opts.append((v, vec))
                if opts:
                    v, vec = min(opts)
                    if upper - v >= max(1, (gap + 1) // 2):
                        e = list(vec)
                        reuses += 1
            if e is None:
                e = [0] * t
                for i in range(t - 1, -1, -1):
                    e[i] = coordinate_minimizer(s.task, t, mask, e, i)
                coordinates += t
            s.propose(t, mask, e)
            tried.add((t, mask))
            if lazy:
                key = (t, tuple(e))
                pool[key] = None
                pool.move_to_end(key)
                if len(pool) > 8:
                    pool.popitem(last=False)
        elif can_expand:
            s.advance(t, mask)
        else:
            break
        peak_live = max(peak_live, len(s.live))
    receipt = s.receipt()
    elapsed = time.process_time_ns() - started
    return dict(receipt=receipt, cpu_ns=elapsed, coordinates=coordinates, reuses=reuses,
                candidate_checks=candidates, peak_live=peak_live, cache=getattr(s, 'counts', {}))


def campaign():
    rows = []
    replays = 0
    for j, regime in enumerate(('persistent', 'rotating', 'mixed')):
        for i in range(20):
            seed = 901000 + j * 1000 + i
            d, t = (8, 6) if i < 12 else (9, 7) if i < 16 else (10, 8)
            q = task(seed, regime, d, t)
            init = initial(q)
            optimum = oracle(q, transition_cap=10_000_000)['objective']
            runs = {arm: [] for arm in ARMS}
            for rep in range(REPEATS):
                order = list(ARMS)
                random.Random(seed * 13 + rep).shuffle(order)
                for arm in order:
                    z = run(q, arm)
                    r = z['receipt']
                    assert r['objective'] <= optimum <= r['upper']
                    assert verify_dual_trace(q, init, r['events'], transition_cap=CAP, proposal_cap=PROPOSALS) == r
                    replays += 1
                    if runs[arm]:
                        assert r == runs[arm][0]['receipt']
                    runs[arm].append(z)
                assert runs['original'][-1]['receipt'] == runs['cache128'][-1]['receipt'] == runs['cache12'][-1]['receipt']
            memory = {}
            if i in (0, 12, 16):
                for arm in ARMS:
                    tracemalloc.start()
                    z = run(q, arm)
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    assert z['receipt'] == runs[arm][0]['receipt']
                    memory[arm] = peak
            rows.append(dict(seed=seed, regime=regime, task=q, optimum=optimum, initial=init, runs=runs, memory=memory))
            print(f'root {seed} complete', flush=True, file=__import__('sys').stderr)
    summary = {}
    for arm in ARMS:
        first = [r['runs'][arm][0] for r in rows]
        timings = [statistics.median(z['cpu_ns'] for z in r['runs'][arm]) for r in rows]
        baseline = [statistics.median(z['cpu_ns'] for z in r['runs']['original']) for r in rows]
        memory = [r['memory'][arm] for r in rows if r['memory']]
        summary[arm] = dict(cpu_ms=statistics.mean(timings)/1e6,
                            median_root_ms=statistics.median(timings)/1e6,
                            wins=sum(a < b for a, b in zip(timings, baseline, strict=True)),
                            certified=sum(z['receipt']['gap'] == 0 for z in first),
                            regret=statistics.mean(r['optimum']-z['receipt']['objective'] for r,z in zip(rows,first,strict=True)),
                            gap=statistics.mean(z['receipt']['gap'] for z in first),
                            transitions=statistics.mean(z['receipt']['transitions'] for z in first),
                            proposals=statistics.mean(z['receipt']['proposals'] for z in first),
                            coordinates=statistics.mean(z['coordinates'] for z in first),
                            reuses=sum(z['reuses'] for z in first), candidate_checks=sum(z['candidate_checks'] for z in first),
                            peak_traced_bytes=statistics.mean(memory),
                            cache_totals={k: sum(z['cache'][k] for z in first) for k in first[0]['cache']},
                            cache_max={k: max(z['cache'][k] for z in first) for k in first[0]['cache'] if k.startswith('peak')})
    return dict(version='SG-07/v0.1', evidence='offline synthetic development; three paired timings per root',
                roots=len(rows), repeats=REPEATS, replays=replays, summary=summary, rows=rows,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    result = campaign()
    Path('sg07_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))
