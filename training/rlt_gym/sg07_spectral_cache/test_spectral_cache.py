from random import Random

import pytest

from datagoblin.training_gym.spectral_cache import CachedDualSession
from datagoblin.training_gym.spectral_dual import DualSearchSession, dual_bound, verify_dual_trace
from datagoblin.training_gym.spectral_routing import TaskError, oracle
from sg07 import initial, task


def test_all_sampled_bounds():
    for seed in range(16):
        q = task(seed, 'mixed', 5, 4)
        ref = DualSearchSession(q, initial(q))
        for table_cap, scalar_cap in ((0, 0), (1, 1), (12, 16)):
            s = CachedDualSession(q, initial(q), table_cap=table_cap, scalar_cap=scalar_cap)
            rng = Random(seed)
            for t in range(5):
                for mask in range(32):
                    e = [rng.randrange(-30, 31) for _ in range(t)]
                    assert s.candidate_bound(t, mask, e) == dual_bound(q, t, mask, e)
                    assert s.remaining_bound(t, mask) == ref.remaining_bound(t, mask)
                    assert len(s.tables) <= table_cap and len(s.scalars) <= scalar_cap


def test_interleaved_replay_and_eviction():
    for seed in range(32):
        q = task(seed + 100, 'rotating', 5, 4)
        init = initial(q)
        s = CachedDualSession(q, init, table_cap=1, scalar_cap=1, transition_cap=10000, proposal_cap=10000)
        ref = DualSearchSession(q, init, transition_cap=10000, proposal_cap=10000)
        rng = Random(seed)
        while s.live:
            t, mask, _ = rng.choice(s.frontier())
            e = [rng.randrange(-10, 11) for _ in range(t)]
            s.propose(t, mask, e)
            ref.propose(t, mask, e)
            assert s.receipt() == ref.receipt()
            for m in range(1 << q['d']):
                s.remaining_bound(t, m)
            assert s.remaining_bound(t, mask) == ref.remaining_bound(t, mask)
            if (t, mask) in s.live:
                s.advance(t, mask)
                ref.advance(t, mask)
            assert s.receipt() == ref.receipt()
        r = s.receipt()
        assert r['objective'] == oracle(q)['objective'] and r['gap'] == 0
        assert verify_dual_trace(q, init, r['events'], transition_cap=10000, proposal_cap=10000) == r
        assert len(s.overlay) <= s.proposals


def test_reject_bad_proposal_and_caps():
    q = {'d': 3, 'T': 1, 'bits': [1,1,1], 'x': [0,1], 'k': [1], 'lambda': 0, 'u': [[0,2,3]]}
    for bad in (-1, True, 100001):
        with pytest.raises(TaskError):
            CachedDualSession(q, [1], table_cap=bad)
    s = CachedDualSession(q, [1], proposal_cap=1, table_cap=0, scalar_cap=0)
    before = s.receipt()
    for bad in ([], [True], [10**10]):
        with pytest.raises(TaskError):
            s.propose(1, 0, bad)
        assert s.receipt() == before
    s.propose(1, 0, [0])
    before = s.receipt()
    with pytest.raises(TaskError):
        s.propose(1, 0, [0])
    assert s.receipt() == before
