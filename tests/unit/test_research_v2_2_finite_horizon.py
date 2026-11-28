"""RV22-W: survival forecasts, causal prefixes, exact replay and bounded storage."""
import sys
from pathlib import Path
from random import Random
from unittest.mock import patch


def module():
    here = Path(__file__).resolve().parent
    scripts = here if (here / 'finite_horizon.py').exists() else here.parents[1] / 'scripts' / 'research_v2_2'
    with patch.object(sys, 'path', [str(scripts), *sys.path]):
        import finite_horizon as m
    return m


def test_conditional_lifetime_forecast():
    m = module()
    lengths = (24, 36, 48, 96)
    table = b''.join(m.COUNT.pack(sum(n >= t for n in lengths)) for t in range(m.MAX_AGE+1))
    for now in (1, 12, 24, 25, 48, 96, 97):
        alive = [n for n in lengths if n >= now]
        for interval in (1, 12, 40, 56):
            expected = sum((n-now)//interval for n in alive)/len(alive) if alive else 0.
            assert abs(m.forecast(table,now,interval)-expected) < 1e-12
            assert 0 <= m.forecast(table,now,interval,lower=True) <= expected+1e-12
            assert m.future_hits('known_end',table,now,interval,end=96) == max(0,(96-now)//interval)
    assert m.forecast(m.lifetime_table(),193,12) == 0


def test_online_prefix_independence_and_accounting():
    m = module()
    ds = m.generate(2026728801,'cycle12',72)
    table = m.lifetime_table()
    ref = m.evaluate(ds,'cold',4096,table)[1]
    for arm in m.ARMS:
        stats,vals = m.evaluate(ds,arm,4096,table)
        assert vals == ref and stats['hits']+stats['solves'] == len(ds)
        assert stats['checks'] == stats['admissions']
        assert stats['weighted_edges'] == 5*stats['solve_edges']+6*stats['verify_edges']
        assert stats['retained_bytes'] <= 4096
        if arm == 'known_end':
            continue
        traces = []
        for stream in (ds[:24],ds):
            checks = []
            original = m.check
            current = [0]
            original_key = m.binary_key

            def key(d, current=current, original_key=original_key):
                current[0] += 1
                return original_key(d)

            def check(s, checks=checks, current=current, original=original):
                if current[0] <= 24:
                    checks.append(current[0])
                return original(s)

            with patch.object(m,'binary_key',side_effect=key), patch.object(m,'check',side_effect=check):
                m.evaluate(stream,arm,4096,table)
            traces.append(checks)
        assert traces[0] == traces[1]


def test_collisions_and_no_storage():
    m = module()
    from cold_bypass import variant, world
    rng = Random(2301)
    roots = [world(rng,12),world(rng,16)]
    ds = [variant(roots[t%2],rng) for t in range(16)]
    table = m.lifetime_table()
    ref = m.evaluate(ds,'cold',4096,table)[1]
    arena = m.Arena(256)
    minimum = m.History().retained()+sys.getsizeof(table)+arena.retained()-len(arena.buf)+m.HEADER.size
    for arm in m.ARMS:
        with patch.object(m,'tag_of',return_value=1):
            stats,vals = m.evaluate(ds,arm,4096,table)
        assert vals == ref and stats['checks'] == stats['admissions']
        stats,vals = m.evaluate(ds,arm,minimum,table)
        assert vals == ref and stats['checks'] == stats['hits'] == 0
