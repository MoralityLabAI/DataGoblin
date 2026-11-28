"""RV22-Q: regression for factorized activation and checked admission."""
import copy
import sys
from pathlib import Path
from random import Random
from unittest.mock import patch

import pytest


def module():
    here = Path(__file__).resolve().parent
    scripts = here if (here / 'activation.py').exists() else here.parents[1] / 'scripts' / 'research_v2_2'
    with patch.object(sys, 'path', [str(scripts), *sys.path]):
        import activation as m
    return m


def test_activation_and_admission_are_independent():
    m = module()
    rng = Random(7702)
    d = m.world(rng, 12)
    ds = [m.variant(d,rng) for _ in range(48)]
    reference = m.evaluate(ds,'cold')[1]
    for arm in m.ARMS:
        stats, values = m.evaluate(ds,arm,[100.]+[0.]*8,seed=913)
        assert values == reference
        assert stats['hits']+stats['solves'] == len(ds)
        assert stats['peak_accounted'] <= m.CAP
        if arm == 'trusted_lru':
            assert stats['checks'] == 0 and stats['admissions'] == 1
        else:
            assert stats['checks'] == stats['admissions']
    always, _ = m.evaluate(ds,'always_second')
    sampled, _ = m.evaluate(ds,'sample_second')
    switch, _ = m.evaluate(ds,'switch1')
    assert always['keys'] == 48 and always['checks'] == 1
    assert sampled['keys'] == 12 and sampled['checks'] == 1
    assert sampled['activations'] == 0 and switch['activations'] > 0
    assert switch['hits'] > sampled['hits']
    with pytest.raises(ValueError,match='capacity'):
        m.evaluate(ds,'cost',[100.]+[0.]*8,cap=m.RESERVE-1)
    empty, values = m.evaluate(ds,'cost',[100.]+[0.]*8,cap=m.RESERVE)
    assert values == reference and empty['admissions'] == empty['checks'] == empty['hits'] == 0


def test_alias_hard_evidence_and_forced_counterfactual():
    m = module()
    ds = m.make_stream(7781,'alias',48)
    reference = m.evaluate(ds,'cold')[1]
    with patch.object(m,'canonical',wraps=m.canonical) as canon:
        cost, values = m.evaluate(ds,'cost',[100.]+[0.]*8)
    assert values == reference and canon.call_count == cost['keys'] == 12
    assert cost['activations'] == cost['checks'] == cost['hits'] == 0
    jitter, values = m.evaluate(ds,'jitter1',seed=913)
    assert values == reference and jitter['hits'] > 0 and jitter['checks'] == 1
    machine = m.Machine('sample_second')
    for t,d in enumerate(ds[:10]):
        machine.step(d,t)
    prior = copy.deepcopy(machine.__dict__)
    for mode in ('full','sample'):
        clone = copy.deepcopy(machine)
        values = [clone.step(ds[t],t,forced=mode) for t in range(10,22)]
        assert values == reference[10:22]
    assert machine.m == prior['m'] and machine.cache == prior['cache']
    assert machine.mon.buf == prior['mon'].buf
