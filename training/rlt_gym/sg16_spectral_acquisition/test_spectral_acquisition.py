from copy import deepcopy
from random import Random

import pytest
import sg07
import sg11
import sg16
from datagoblin.training_gym.spectral_acquisition import Portfolio, replay
from datagoblin.training_gym.spectral_blocks import root_vector
from datagoblin.training_gym.spectral_routing import TaskError, oracle


def test_all_stages_against_exact_oracle():
    checks = 0
    for seed in range(24):
        q = sg07.task(1630000+seed, 'mixed', 5, 4)
        opt = oracle(q)['objective']
        for plan in sg16.PLANS:
            z = sg16.run(q, plan, {}, 0)
            r = z['receipt']
            assert r['objective'] == opt == r['upper']
            assert replay(q, z['initial'], r['records']) == r
            assert all(s['before'] >= s['after'] for s in z['stages'])
            checks += 1
    assert checks == 96


def test_monotone_intersections_and_negative_floor():
    for seed in range(32):
        q = sg07.task(1631000+seed, 'rotating', 5, 4)
        initial = sg11.initial(q)
        p = Portfolio(q, initial)
        opt = oracle(q)['objective']
        p.dual(root_vector(q), recover=False)
        rng = Random(seed)
        for _ in range(6):
            lo, hi = p.lower, p.upper
            p.dual([rng.randrange(-50,51) for _ in range(q['T'])], 2, node_cap=1)
            assert lo <= p.lower <= opt <= p.upper <= hi
        p.block([root_vector(q)], 10000)
        assert p.gap == 0 and p.lower == opt
        assert replay(q, initial, p.records) == p.receipt()
    q = dict(d=3,T=3,bits=[1,1,1],x=[0,1,1,1],k=[1,1,1],
             u=[[0,0,-110],[-110,0,0],[0,-110,0]], **{'lambda':1})
    p = Portfolio(q,[1,2,1]); p.dual([-1,-1,-1],2)
    assert p.lower == p.upper == -2
    assert p.records[-1]['certificate']['bound_num'] == -3


def test_tampered_receipt_and_deadline():
    q = sg07.task(1632000, 'mixed', 5, 4)
    init = sg11.initial(q); p = Portfolio(q, init)
    p.dual(root_vector(q), recover=False)
    before = p.receipt()
    for eta, den in (([0],1),([0]*q['T'],True),([True]*q['T'],1)):
        with pytest.raises(TaskError):
            p.dual(eta,den)
        assert p.receipt() == before
    p.block([root_vector(q)],2048,deadline=lambda:True)
    assert p.records[-1]['receipt']['event_count'] == 0
    r = deepcopy(p.records);r[0]['certificate']['integer_upper'] -= 1
    with pytest.raises(TaskError):
        replay(q,init,r)
    r = deepcopy(p.records);r[1]['receipt']['upper'] -= 1
    with pytest.raises(TaskError):
        replay(q,init,r)
    assert replay(q,init,p.records) == p.receipt()
