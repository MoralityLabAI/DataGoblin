"""RV22-R: invariants, repeat coverage, exact parity, collision and capacity checks."""
import sys
from functools import cache
from pathlib import Path
from random import Random
from unittest.mock import patch

import pytest


def module():
    here = Path(__file__).resolve().parent
    scripts = here if (here / 'fingerprint.py').exists() else here.parents[1] / 'scripts' / 'research_v2_2'
    with patch.object(sys, 'path', [str(scripts), *sys.path]):
        import fingerprint as m
    return m


def oracle(d):
    hs, ts = d['hypotheses'], d['tests']
    c, r = d['correct_reward'], d['wrong_reward']

    @cache
    def solve(indices, budget, used):
        w = sum(hs[i]['weight'] for i in indices)
        a = sum(hs[i]['weight'] for i in indices if hs[i]['class'] == 'A')
        v = max(0, a*c+(w-a)*r, (w-a)*c+a*r)
        for j,t in enumerate(ts):
            if j in used or t['cost'] > budget:
                continue
            zero = tuple(i for i in indices if not t['bits'][i])
            one = tuple(i for i in indices if t['bits'][i])
            if zero and one:
                u = tuple(sorted((*used,j)))
                v = max(v, solve(zero,budget-t['cost'],u)+solve(one,budget-t['cost'],u)-t['cost']*w)
        return v

    return solve(tuple(range(len(hs))),d['budget'],())


def test_fingerprint_invariance_and_exact_oracle():
    m = module()
    rng = Random(1129)
    for i in range(12):
        d = m.world(rng,12 if i%2 == 0 else 16)
        v = m.hidden_variant(d,rng)
        assert m.canonical(d)[0] == m.canonical(v)[0]
        for marginal in (False,True):
            assert m.fingerprint(d,marginal) == m.fingerprint(v,marginal)
        for x in (d,v):
            s = m.DP(x)
            value,p = s.run()
            assert value == oracle(x) == m.replay(x,p)
            assert m.check(s) == s.edges
    for kind in m.KINDS:
        ds = m.generate(2026301100+len(kind),kind,48)
        for c in m.audit_stream(ds).values():
            assert c['fn'] == 0 and sum(c.values()) == len(ds)


def test_coarse_collisions_exact_gate_and_capacity():
    m = module()
    ds = m.generate(2026302211,'collision',24)
    assert len({m.canonical(d)[0] for d in ds}) == len(ds)
    assert len({m.fingerprint(d,True) for d in ds}) == 1
    ref = m.evaluate(ds,'cold')[1]
    for arm in ('checked_mass','checked_marginal','checked_marginal_confirm'):
        stats,values = m.evaluate(ds,arm)
        assert values == ref and stats['hits'] == 0 and stats['keys'] == 23
        assert stats['checks'] == (0 if arm.endswith('confirm') else 23)
    rng = Random(1130)
    d,e = m.world(rng,12),m.world(rng,12)
    ds = [m.hidden_variant(d if j%2 == 0 else e,rng) for j in range(24)]
    ref = m.evaluate(ds,'cold')[1]
    with patch.object(m,'fingerprint_tag',return_value=1):
        for arm in m.ARMS[5:]:
            stats,values = m.evaluate(ds,arm)
            assert values == ref and stats['hits'] > 0
            assert stats['peak_accounted'] <= m.CAP
            if arm.startswith('checked'):
                assert stats['admissions'] == stats['checks'] == 2
            else:
                assert stats['admissions'] == 2 and stats['checks'] == 0
    for arm in m.ARMS:
        stats,values = m.evaluate(ds,arm)
        assert values == ref and stats['hits']+stats['solves'] == len(ds)
        assert stats['peak_accounted'] <= m.CAP
    stats,values = m.evaluate(ds,'checked_marginal',cap=m.RESERVE)
    assert values == ref and stats['checks'] == stats['hits'] == 0
    with pytest.raises(ValueError,match='metadata capacity'):
        m.evaluate(ds,'checked_marginal',cap=m.RESERVE-1)
