"""Public-only actor and synthetic probe isolation regressions."""
import copy
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest

from datagoblin.research_v2_2 import native_probe as m


def xor(reward=12):
    return dict(hypotheses=[{'class': c, 'weight': 1} for c in 'ABBA'],
                tests=[dict(id='x',cost=1,bits=[0,0,1,1]),dict(id='y',cost=1,bits=[0,1,0,1])],
                budget=2, correct_reward=reward, wrong_reward=-16)


def test_exact_residual_and_actor_isolation():
    d = xor()
    x, y = m.examples([d])
    assert x.shape[1] == m.NFEATURES and len(y) == 6
    assert np.allclose(y[:2], 11/16)
    assert np.allclose(y[2:], 0)
    with patch.object(m,'action_values',side_effect=AssertionError('oracle in actor')), \
         patch.object(m,'ExactResearch',side_effect=AssertionError('oracle in actor')):
        one,_ = m.tree(d,'one_step')
        two,_ = m.tree(d,'two_step')
        assert m.policy_value(d,one) == 0
        assert m.policy_value(d,two) == 10
        for depth,width in ((1,29),(3,12)):
            model = m.Head(11,width,depth).fit(x,y)
            policy,_ = m.tree(d,'learned',model)
            assert m.policy_value(d,policy) <= 10


def test_probe_reference_exclusion_and_frozen_fit():
    panels = dict(train=[xor(12)],cal=[xor(11)],native_eval=[xor(10)],fresh=[xor(9)],foundation=[xor(8)])
    native = {m.binary_key(xor(10))[0]}
    receipt = m.run_probe(panels,native)
    assert receipt['train_examples'] == 6
    assert receipt['panels']['native_eval']['means']['exact']['utility'] == 8
    assert len(receipt['panel_key_hashes']) == 5
    bad = copy.deepcopy(panels)
    bad['train'] = bad['native_eval']
    with pytest.raises(ValueError,match='overlap'):
        m.run_probe(bad,native)
    bad = copy.deepcopy(panels)
    bad['foundation'] = bad['fresh']
    with pytest.raises(ValueError,match='overlap'):
        m.run_probe(bad,native)
    with pytest.raises(ValueError,match='binding'):
        m.run_probe(panels,set())
    for name in ('native_eval','fresh','foundation'):
        assert all(Fraction(str(x['regret'])) >= 0 for x in receipt['panels'][name]['means'].values())
