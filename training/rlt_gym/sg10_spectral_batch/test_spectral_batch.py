from random import Random

import pytest

import sg07
from datagoblin.training_gym.spectral_bank import BankSession
from datagoblin.training_gym.spectral_batch import BatchedBankSession
from datagoblin.training_gym.spectral_dual import verify_dual_trace
from datagoblin.training_gym.spectral_routing import TaskError, oracle

MODES = ('dedup', 'batch', 'hybrid')


def test_quotients():
    checks = 0
    for seed in range(24):
        q = sg07.task(960000 + seed, 'mixed', 5, 4)
        if seed % 3 == 0:
            q['k'][0] = 0
        if seed % 5 == 0:
            q['k'][1] = 5
        s = BatchedBankSession(q, sg07.initial(q), mode='batch')
        for t in range(1, q['T'] + 1):
            for suffix in range(1 << q['d']):
                groups = s._quotient(t, suffix)
                for union, (m, u) in groups.items():
                    actions = [(a, v) for a, v in s.actions[t - 1] if a | suffix == union]
                    assert (m, u) == max(actions, key=lambda a: a[1])
                    checks += 1
                assert set(groups) == {m | suffix for m, _ in s.actions[t - 1]}
    assert checks > 3000


def test_batch_bounds_and_overlay():
    checks = 0
    for seed in range(16):
        q = sg07.task(961000 + seed, 'mixed', 5, 4)
        init = sg07.initial(q)
        ref = BankSession(q, init, split=True, scalar_cap=0)
        for cap in (0, 1, 16):
            for mode in MODES:
                s = BatchedBankSession(q, init, mode=mode, scalar_cap=cap)
                for t in range(q['T'] + 1):
                    masks = list(range(1 << q['d']))
                    # Exercise an accepted-proposal overlay independently of the cache.
                    m = seed % len(masks)
                    s.overlay[t, m] = ref.remaining_bound(t, m) - 1
                    expected = [s.overlay.get((t, m), ref.remaining_bound(t, m)) for m in masks]
                    assert s._bounds(t, masks) == expected
                    assert s._bounds(t, masks[::-1]) == expected[::-1]
                    assert len(s.scalars) <= cap
                    checks += 2 * len(masks)
    assert checks == 46080


def test_interleaved_exact_replay():
    for seed in range(36):
        q = sg07.task(962000 + seed, 'rotating', 5, 4)
        init = sg07.initial(q)
        ref = BankSession(q, init, split=True, transition_cap=10000, proposal_cap=10000)
        arms = [BatchedBankSession(q, init, mode=m, scalar_cap=seed % 3,
                                   table_cap=1, transition_cap=10000, proposal_cap=10000) for m in MODES]
        rng = Random(seed)
        while ref.live:
            t, m, _ = rng.choice(ref.frontier())
            eta = [rng.randrange(-20, 21) for _ in range(t)]
            ref.propose(t, m, eta)
            for arm in arms:
                arm.propose(t, m, eta)
                assert arm.receipt() == ref.receipt()
                assert arm.live == ref.live and arm.seen == ref.seen
            if (t, m) in ref.live:
                ref.advance(t, m)
                for arm in arms:
                    arm.advance(t, m)
                    assert arm.receipt() == ref.receipt()
                    assert arm.live == ref.live and arm.seen == ref.seen
        assert ref.receipt()['gap'] == 0 and ref.incumbent == oracle(q)['objective']
        assert verify_dual_trace(q, init, ref.events, transition_cap=10000, proposal_cap=10000) == ref.receipt()


def test_terminal_ties_and_cap():
    q = dict(d=4, T=1, bits=[1]*4, x=[0, 1], k=[2], u=[[0, 0, 3, 3]], **{'lambda': 0})
    for mode in MODES:
        s = BatchedBankSession(q, [3], mode=mode, transition_cap=1)
        before = s.receipt()
        with pytest.raises(TaskError):
            s.advance(1, 0)
        assert s.receipt() == before
        s = BatchedBankSession(q, [3], mode=mode)
        s.advance(1, 0)
        assert s.schedule == [12] and s.receipt()['gap'] == 0
        with pytest.raises(TaskError):
            s.advance(1, 0)
    # Equal maximum among different legal masks: keep earliest original action.
    q['u'] = [[0, 3, 3, 3]]
    s = BatchedBankSession(q, [3], mode='batch')
    s.advance(1, 0)
    assert s.schedule == [6]


def test_extreme_valid_integer_range():
    q = dict(d=10, T=3, bits=[100]*10, x=[-1000, -999, 999, 1000], k=[0, 5, 10],
             u=[[-10000]*10, [10000]*10, [-10000]*10], **{'lambda': 100})
    s = BatchedBankSession(q, [0, 31, 1023], mode='batch', scalar_cap=1)
    ref = BankSession(q, [0, 31, 1023], split=True, scalar_cap=0)
    for t in range(4):
        masks = list(range(1024))
        assert s._bounds(t, masks) == [ref.remaining_bound(t, m) for m in masks]
