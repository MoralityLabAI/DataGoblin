"""RV22-P: exact utility, collision and sampled bypass regression."""
import sys
from functools import cache
from pathlib import Path
from random import Random
from unittest.mock import patch

import pytest


def module():
    here = Path(__file__).resolve().parent
    scripts = here if (here / 'cold_bypass.py').exists() else here.parents[1] / 'scripts' / 'research_v2_2'
    with patch.object(sys, 'path', [str(scripts), *sys.path]):
        import cold_bypass as m
    return m


def oracle(d):
    hs, ts = d['hypotheses'], d['tests']
    c, r = d['correct_reward'], d['wrong_reward']

    @cache
    def go(indices, budget, used):
        w = sum(hs[i]['weight'] for i in indices)
        a = sum(hs[i]['weight'] for i in indices if hs[i]['class'] == 'A')
        best = max(0, a*c+(w-a)*r, (w-a)*c+a*r)
        for j,t in enumerate(ts):
            if j in used or t['cost'] > budget:
                continue
            z = tuple(i for i in indices if not t['bits'][i])
            o = tuple(i for i in indices if t['bits'][i])
            if z and o:
                u = tuple(sorted((*used,j)))
                best = max(best, go(z,budget-t['cost'],u)+go(o,budget-t['cost'],u)-t['cost']*w)
        return best

    return go(tuple(range(len(hs))),d['budget'],())


def test_exact_solver_and_certificate():
    m = module()
    rng = Random(6129)
    for i in range(12):
        d = m.world(rng, 12 if i%2 == 0 else 16)
        v = m.variant(d,rng)
        assert m.canonical(d)[0] == m.canonical(v)[0]
        for x in (d,v):
            s = m.DP(x)
            value,p = s.run()
            assert value == oracle(x) == m.replay(x,p)
            assert m.check(s) == s.edges
            root = s.full,x['budget']
            val,action = s.memo[root]
            s.memo[root] = val+1,action
            with pytest.raises(ValueError,match='invalid Bellman'):
                m.check(s)


def test_bypass_and_collision_safe_admission():
    m = module()
    rng = Random(6130)
    ds = [m.world(rng,12) for _ in range(48)]
    ref = m.evaluate(ds,'cold',[0.]*6)[1]
    with patch.object(m, 'canonical', wraps=m.canonical) as canon:
        stats,values = m.evaluate(ds,'checked_sample',[0.]*6)
        assert stats['keys'] == canon.call_count == 12
        assert stats['bypass'] == 36 and stats['checks'] == stats['hits'] == 0
        assert values == ref
    d = m.world(rng,12)
    ds = [m.variant(d,rng) for _ in range(48)]
    ref = m.evaluate(ds,'cold',[0.]*6)[1]
    for arm in m.ARMS:
        stats,values = m.evaluate(ds,arm,[1.]*6)
        assert values == ref and stats['hits']+stats['solves'] == len(ds)
        assert stats['peak_accounted'] <= m.CAP
        if arm.startswith('checked'):
            assert stats['checks'] == stats['admissions'] > 0
        else:
            assert stats['checks'] == 0
        if arm.endswith(('sample','ldt')):
            assert stats['activations'] > 0 and stats['hits'] > 0
    ds = [m.world(rng,12),m.world(rng,12)]
    ds += [m.variant(ds[0],rng),m.variant(ds[1],rng)]*4
    ref = m.evaluate(ds,'cold',[0.]*6)[1]
    with patch.object(m.hashlib,'blake2b') as digest:
        digest.return_value.digest.return_value = b'\x00'*8
        stats,values = m.evaluate(ds,'checked_ldt',[1.]*6)
    assert values == ref and stats['checks'] == stats['admissions'] == 2
    assert stats['hits'] > 0
    stats,values = m.evaluate(ds,'checked_ldt',[1.]*6,cap=m.RESERVE)
    assert values == ref and stats['checks'] == stats['hits'] == 0
