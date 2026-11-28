"""RV22-S: exact identity and low-order collision regression."""
import sys
from pathlib import Path
from random import Random
from unittest.mock import patch


def module():
    here = Path(__file__).resolve().parent
    scripts = here if (here / 'refinement.py').exists() else here.parents[1] / 'scripts' / 'research_v2_2'
    with patch.object(sys, 'path', [str(scripts), *sys.path]):
        import refinement as m
    return m


def test_parity_collision_witness():
    m = module()
    for n in range(1, 6):
        independent, parity = m.witness(n)
        assert m.moments(independent, n-1) == m.moments(parity, n-1)
        assert m.moments(independent, n) != m.moments(parity, n)
        assert m.packed_key(independent)[0] != m.packed_key(parity)[0]
        for d, expected in ((independent, 0), (parity, 12-n)):
            solver = m.DP(d)
            value, tree = solver.run()
            assert value == expected*sum(h['weight'] for h in d['hypotheses'])
            assert m.replay(d, tree) == value
            assert m.check(solver) == solver.edges


def test_packed_equivalence_and_exact_safety():
    m = module()
    from fingerprint import hidden_variant
    rng = Random(2729)
    for n in range(1, 5):
        for d in m.witness(n):
            v = hidden_variant(d, rng)
            assert m.canonical(d)[0] == m.canonical(v)[0]
            assert m.packed_key(d)[0] == m.packed_key(v)[0]
            assert m.moments(d, 2) == m.moments(v, 2)
    for kind in m.KINDS:
        ds = m.stream(2026411100+len(kind), kind, 24)
        old_new, new_old = {}, {}
        ref = m.evaluate(ds, 'cold')[1]
        for d in ds:
            a, s = m.canonical(d)
            b, t = m.packed_key(d)
            assert s == t and old_new.setdefault(a,b) == b and new_old.setdefault(b,a) == a
        for arm in m.ARMS:
            stats, vals = m.evaluate(ds, arm)
            assert vals == ref and stats['peak_accounted'] <= m.CAP
            assert stats['hits']+stats['solves'] == len(ds)
    ds = m.stream(2026415500,'pair_alias',24)
    assert len({m.moments(d,2) for d in ds}) == 1
    assert len({m.packed_key(d)[0] for d in ds}) == len(ds)
    ref = m.evaluate(ds,'cold')[1]
    for arm in ('ambiguous_guard','progressive_pair'):
        stats, vals = m.evaluate(ds,arm)
        assert vals == ref and stats['checks'] == 1 and stats['hits'] == 0
    # Force coarse and strong digest collisions: full key remains authoritative.
    with patch.object(m,'fingerprint_tag',return_value=1), patch.object(m,'tag',return_value=1):
        for arm in m.ARMS:
            stats, vals = m.evaluate(ds,arm)
            assert vals == ref and stats['hits'] == 0
    d = m.witness(3)[1]
    ds = [hidden_variant(d,rng) for _ in range(12)]
    ref = m.evaluate(ds,'cold')[1]
    for arm in m.ARMS:
        stats, vals = m.evaluate(ds,arm,cap=m.RESERVE)
        assert vals == ref and stats['hits'] == stats['checks'] == stats['admissions'] == 0
