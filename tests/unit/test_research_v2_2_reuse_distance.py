"""RV22-V: packed storage, recurrence evidence and checked admission regression."""
import sys
from functools import cache
from pathlib import Path
from random import Random
from unittest.mock import patch


def module():
    here = Path(__file__).resolve().parent
    scripts = here if (here / 'reuse_distance.py').exists() else here.parents[1] / 'scripts' / 'research_v2_2'
    with patch.object(sys, 'path', [str(scripts), *sys.path]):
        import reuse_distance as m
    return m


def oracle(d):
    hs, ts = d['hypotheses'], d['tests']
    c, r = d['correct_reward'], d['wrong_reward']

    @cache
    def go(indices, budget, used):
        w = sum(hs[i]['weight'] for i in indices)
        a = sum(hs[i]['weight'] for i in indices if hs[i]['class'] == 'A')
        best = max(0,a*c+(w-a)*r,(w-a)*c+a*r)
        for j,t in enumerate(ts):
            if j in used or t['cost'] > budget:
                continue
            z = tuple(i for i in indices if not t['bits'][i])
            o = tuple(i for i in indices if t['bits'][i])
            if z and o:
                u = tuple(sorted((*used,j)))
                best = max(best,go(z,budget-t['cost'],u)+go(o,budget-t['cost'],u)-t['cost']*w)
        return best

    return go(tuple(range(len(hs))),d['budget'],())


def test_packed_recurrence_and_arena():
    m = module()
    short,long = m.History(),m.History()
    assert short.retained() == long.retained()
    for t in range(112):
        x = short.observe(t%56,t,False)
        y = long.observe(t%56,t,True)
        assert x[0] == 0
        assert y[:3] == ((1,56,0) if t >= 56 else (0,0,0))
    rows = [(b'key',b'policy',1,56,2.5),(b'x',b'y',2,12,.125)]
    a = m.Arena(2048-short.retained())
    assert short.retained()+a.retained() <= 2048
    a.write(rows)
    assert a.rows() == rows
    a.write(rows[1:])
    assert a.rows() == rows[1:]
    a.write([])
    assert a.rows() == [] and m.HEADER.unpack_from(a.buf)[0] == 0


def test_oracle_and_all_arms():
    m = module()
    rng = Random(721)
    for i in range(12):
        d = m.world(rng,12 if i%2 == 0 else 16)
        for x in (d,m.variant(d,rng)):
            s = m.DP(x)
            v,p = s.run()
            assert v == oracle(x) == m.replay(x,p)
            assert m.check(s) == s.edges
    ds = m.generate(2026701100,'cycle40',120)
    ref = m.evaluate(ds,'cold',4096)[1]
    results = {}
    for arm in m.ARMS:
        stats,vals = m.evaluate(ds,arm,4096)
        assert vals == ref and stats['checks'] == stats['admissions']
        assert stats['hits']+stats['solves'] == len(ds)
        assert stats['retained_bytes'] <= 4096
        results[arm] = stats
    assert results['short_second']['long_repeat_hints'] == results['short_second']['checks'] == 0
    assert results['long_second']['long_repeat_hints'] == 80
    assert results['long_freeze']['hits'] > 0
    assert results['long_freeze']['checks'] < results['long_second']['checks']


def test_reset_collision_and_empty_arena():
    m = module()
    rng = Random(722)
    d = m.world(rng,12)
    ds = [m.variant(d,rng) for _ in range(12)] + [m.world(rng,12) for _ in range(8)]
    stats,ref = m.evaluate(ds,'distance_reset',4096)
    assert stats['resets'] == 1 and stats['reset_evictions'] == 1
    assert ref == m.evaluate(ds,'cold',4096)[1]
    ds = [m.world(rng,12),m.world(rng,16)]
    ds += [m.variant(ds[i%2],rng) for i in range(12)]
    ref = m.evaluate(ds,'cold',4096)[1]
    minimum = m.History().retained()+m.Arena(256).retained()-len(m.Arena(256).buf)+m.HEADER.size
    for arm in m.ARMS:
        with patch.object(m,'tag_of',return_value=1):
            stats,values = m.evaluate(ds,arm,4096)
        assert values == ref and stats['checks'] == stats['admissions']
        stats,values = m.evaluate(ds,arm,minimum)
        assert values == ref and stats['checks'] == stats['hits'] == 0
