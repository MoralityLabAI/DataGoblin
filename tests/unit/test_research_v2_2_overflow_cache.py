"""RV22-U: exact utility, admission guards and accounting."""
import sys
from pathlib import Path
from random import Random
from unittest.mock import patch


def module():
    here = Path(__file__).resolve().parent
    scripts = here if (here / 'overflow_cache.py').exists() else here.parents[1] / 'scripts' / 'research_v2_2'
    with patch.object(sys, 'path', [str(scripts), *sys.path]):
        import overflow_cache as m
    return m


def test_overflow_and_phase_recovery():
    m = module()
    ds = m.stream(2026591001, 'cycle12', 72)
    ref = m.evaluate(ds, 'cold', 2048)[1]
    results = {}
    for arm in m.ARMS:
        stats, values = m.evaluate(ds, arm, 2048)
        assert values == ref
        assert stats['hits']+stats['solves'] == len(ds)
        assert stats['checks'] == stats['admissions']
        assert stats['peak_accounted'] <= 2048
        results[arm] = stats
    assert results['second']['hits'] == 0
    assert results['frequency']['hits'] > 0
    assert results['frequency']['checks'] < results['second']['checks']
    assert results['freeze']['evictions'] == 0
    ds = m.stream(2026591002, 'phase', 72)
    fixed, ref = m.evaluate(ds, 'freeze', 2048)
    adaptive, vals = m.evaluate(ds, 'frequency', 2048)
    assert vals == ref and adaptive['second_hits'] > fixed['second_hits'] == 0


def test_collision_and_zero_payload():
    m = module()
    rng = Random(482)
    roots = [m.world(rng, 16) for _ in range(3)]
    ds = [m.variant(roots[i%3], rng) for i in range(15)]
    ref = m.evaluate(ds, 'cold', 2048)[1]
    for arm in m.ARMS:
        with patch.object(m, 'digest', return_value=b'\x00'*8):
            stats, vals = m.evaluate(ds, arm, 2048)
        assert vals == ref and stats['checks'] == stats['admissions']
        assert stats['peak_accounted'] <= 2048
        stats, vals = m.evaluate(ds, arm, m.RESERVE)
        assert vals == ref and stats['checks'] == stats['hits'] == 0
