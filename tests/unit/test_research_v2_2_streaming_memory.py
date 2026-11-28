"""RV22-O: packed sketch, trust separation and exact utility regression."""
import sys
from functools import cache
from pathlib import Path
from random import Random
from unittest.mock import patch

import pytest


def oracle(d):
    hs, ts = d['hypotheses'], d['tests']
    correct, wrong = d['correct_reward'], d['wrong_reward']

    @cache
    def solve(indices, budget, used):
        mass = sum(hs[i]['weight'] for i in indices)
        a = sum(hs[i]['weight'] for i in indices if hs[i]['class'] == 'A')
        best = max(0, a*correct+(mass-a)*wrong, (mass-a)*correct+a*wrong)
        for j, test in enumerate(ts):
            if j in used or test['cost'] > budget:
                continue
            zero = tuple(i for i in indices if test['bits'][i] == 0)
            one = tuple(i for i in indices if test['bits'][i] == 1)
            if zero and one:
                nxt = tuple(sorted((*used, j)))
                q = (solve(zero, budget-test['cost'], nxt)+solve(one, budget-test['cost'], nxt)
                     -test['cost']*mass)
                best = max(best, q)
        return best

    return solve(tuple(range(len(hs))), d['budget'], ())


def module():
    # Also runs standalone beside the local source, outside the checkout.
    here = Path(__file__).resolve().parent
    scripts = here if (here / 'streaming_memory.py').exists() else here.parents[1] / 'scripts' / 'research_v2_2'
    with patch.object(sys, 'path', [str(scripts), *sys.path]):
        import streaming_memory as m
    return m


def test_packed_sketch_and_collision_safety():
    m = module()
    s = m.Sketch()
    assert len(s.buf) == 1032 and m.RESERVE == 1080
    assert s.observe(b'a', 0)[:3] == (0, 0, m.H+1)
    assert s.observe(b'a', 1)[:3] == (1, 1, 1)
    assert s.observe(b'a', 2+m.H)[:3] == (0, 0, m.H+1)
    rng = Random(601)
    ds = [m.world(rng, 12), m.world(rng, 12)]
    assert m.key(ds[0])[0] != m.key(ds[1])[0]
    ds += [m.variant(ds[0], rng), m.variant(ds[1], rng)]
    ref = m.evaluate(ds, 'cold', [0.]*6)[1]
    with patch.object(m.hashlib, 'blake2b') as digest:
        digest.return_value.digest.return_value = b'\x00'*8
        stats, values = m.evaluate(ds, 'checked_ldt', [100.] + [0.]*5)
    assert values == ref and stats['hits'] == 2 and stats['checks'] == 2
    assert stats['peak_accounted_bytes'] <= m.CAP
    with pytest.raises(ValueError, match='reserved metadata'):
        m.evaluate(ds, 'checked_ldt', [1.]*6, cap=m.RESERVE-1)


def test_exact_replay_and_all_arm_activation():
    m = module()
    rng = Random(602)
    for i in range(12):
        d = m.world(rng, 12 if i%2 == 0 else 16)
        v = m.variant(d, rng)
        assert m.key(d)[0] == m.key(v)[0]
        for x in (d, v):
            s = m.DP(x)
            value, tree = s.run()
            assert value == oracle(x) == m.replay(x, tree)
            assert m.check(s) == s.edges
    d = m.world(rng, 12)
    ds = [m.variant(d, rng) for _ in range(12)]
    ref = m.evaluate(ds, 'cold', [0.]*6)[1]
    for arm in m.ARMS:
        stats, values = m.evaluate(ds, arm, [100.] + [0.]*5)
        assert values == ref and stats['peak_accounted_bytes'] <= m.CAP
        assert stats['hits'] + stats['solves'] == len(ds)
        if arm.startswith('checked'):
            assert stats['checks'] == stats['admissions'] > 0
        else:
            assert stats['checks'] == 0
        if arm.endswith('second'):
            assert stats['hits'] == 10
        if arm.endswith(('ldt', 'shuffled', 'lru', 'matched')):
            assert stats['hits'] == 11
    stats, values = m.evaluate(ds, 'checked_ldt', [100.] + [0.]*5, cap=m.RESERVE)
    assert values == ref and stats['hits'] == stats['checks'] == stats['admissions'] == 0
