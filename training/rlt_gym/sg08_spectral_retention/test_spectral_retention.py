from functools import lru_cache
from itertools import product

import pytest

import sg07
import sg08
from datagoblin.training_gym.spectral_retention import Retention, fixed_keys
from datagoblin.training_gym.spectral_routing import TaskError


def test_belady_exhaustive():
    keys = ((1, (0,)), (1, (1,)), (2, (0, 0)))
    for trace in product(keys, repeat=6):
        for cap in (0, 1, 2, 3):
            @lru_cache(None)
            def best(i, live):
                if i == len(trace):
                    return 0
                key = trace[i]
                if key in live or cap == 0:
                    return int(key not in live) + best(i+1, live)
                if len(live) < cap:
                    return 1 + best(i+1, live | {key})
                return 1 + min(best(i+1, (live-{victim}) | {key}) for victim in live)
            assert sg08.belady(trace, cap)['misses'] == best(0, frozenset())


def test_policy_stream_and_certificate_parity():
    beta = (0,-32,14,0,11,19,22,0,0,0,0)
    for seed in range(12):
        q = sg07.task(930000+seed, 'mixed', 5, 4)
        ref = sg08.trace_run(q, record=True)
        for cap in (0, 1, 6, 12):
            for kind in sg08.ONLINE:
                z = sg08.trace_run(q, kind, beta if kind in ('learned', 'shuffled') else (), cap, True)
                assert z['receipt'] == ref['receipt'] and z['requests'] == ref['requests']
                r = sg08.simulate(ref['requests'], cap, kind, fixed_keys(q),
                                  beta if kind in ('learned','shuffled') else ())
                assert z['cache']['table_misses'] == r['misses']
                assert z['cache']['component_steps'] == r['depth_work']*q['d']
                assert z['cache']['peak_tables'] <= cap


def test_metadata_and_bad_policy():
    for bad in (-1, True, 100001):
        with pytest.raises(TaskError):
            Retention(bad, 'lru', frozenset())
    with pytest.raises(TaskError):
        Retention(1, 'unknown', frozenset())
    with pytest.raises(TaskError):
        Retention(1, 'learned', frozenset(), (0,))
    s = Retention(1, 'lru', frozenset())
    for i in range(50):
        assert s.access((1,(i,)),lambda: i) == i
        assert len(s.entries) == 1
        assert s.access((1,(i,)),lambda: -1) == i
    assert s.hits == 50 and s.misses == 50 and s.evictions == 49
