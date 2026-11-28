from random import Random

import pytest

import sg07
from datagoblin.training_gym.spectral_bank import BankSession, subset_sums
from datagoblin.training_gym.spectral_cache import CachedDualSession
from datagoblin.training_gym.spectral_dual import dual_bound, verify_dual_trace
from datagoblin.training_gym.spectral_routing import TaskError


def test_exhaustive_bank():
    checks = 0
    for seed in range(20):
        q = sg07.task(940000 + seed, 'mixed', 5, 4)
        if seed % 4 == 0:
            q['k'][0] = 0
        if seed % 5 == 0:
            q['k'][1] = 5
        if seed % 7 == 0:
            q['lambda'] = 0
        initial = sg07.initial(q)
        ref = CachedDualSession(q, initial, scalar_cap=0)
        for split in (False, True):
            s = BankSession(q, initial, split=split, scalar_cap=0, table_cap=1)
            for t in range(q['T'] + 1):
                masks = list(range(1 << q['d']))
                expected = [ref.remaining_bound(t, m) for m in masks]
                assert s.bank.batch(t, masks, s.weak[t]) == expected
                for m in masks:
                    for e in s.etas:
                        assert s.candidate_bound(t, m, e[:t]) == dual_bound(q, t, m, e[:t])
                        checks += 1
                    assert s.remaining_bound(t, m) == expected[m]
            assert s.bank.steps == 6 * q['T'] * q['d']
            with pytest.raises(TaskError):
                s.bank.batch(0, [-1], 0)
    assert checks == 38400


def test_interleaved_replay():
    for seed in range(24):
        q = sg07.task(941000 + seed, 'rotating', 5, 4)
        initial = sg07.initial(q)
        ref = CachedDualSession(q, initial, transition_cap=10000, proposal_cap=10000)
        banks = [BankSession(q, initial, split=split, table_cap=1, scalar_cap=0,
                             transition_cap=10000, proposal_cap=10000) for split in (False, True)]
        rng = Random(seed)
        while ref.live:
            t, m, _ = rng.choice(ref.frontier())
            eta = [rng.randrange(-20, 21) for _ in range(t)]
            ref.propose(t, m, eta)
            for b in banks:
                b.propose(t, m, eta)
                assert b.receipt() == ref.receipt()
            if (t, m) in ref.live:
                ref.advance(t, m)
                for b in banks:
                    b.advance(t, m)
                    assert b.receipt() == ref.receipt()
        assert verify_dual_trace(q, initial, ref.events, transition_cap=10000, proposal_cap=10000) == ref.receipt()


def test_subset_sums_and_integer_range():
    assert subset_sums([2, -3, 7]) == [0, 2, -3, -1, 7, 9, 4, 6]
    q = dict(d=10, T=10, bits=[100]*10, x=[-1000]+[1000]*10,
             k=[10]*10, u=[[-10000]*10 for _ in range(10)], **{'lambda': 100})
    s = BankSession(q, [1023]*10, split=True)
    for m in (0, 1, 1023):
        for e in s.etas:
            assert s.candidate_bound(10, m, e) == dual_bound(q, 10, m, e)
