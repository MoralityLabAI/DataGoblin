import copy
import random
from dataclasses import replace

import pytest

from datagoblin.zk_replay import Job, Plan, execute, verify
from datagoblin.zk_schedule import PALLAS_BASE, PALLAS_SCALAR, compile_step, initial, next_state, schedule
from datagoblin.zk_sigma import r1cs_satisfied


def fixture(n, w, seed):
    rng = random.Random(seed)
    jobs = []
    for j in range(n):
        parents = tuple(f'j{i}' for i in range(j) if i == j-1 or rng.random() < .3)
        def vec():
            return tuple(rng.randint(-127,127) for _ in range(w))

        jobs.append(Job(f'j{j}', parents, vec(), tuple(vec() for _ in range(w)), vec(), rng.randint(1,5)))
    return Plan('tenant', f'run-{seed}', 'policy', tuple(jobs))


@pytest.mark.parametrize('n', [1, 3, 8, 32])
@pytest.mark.parametrize('w', [1, 2])
@pytest.mark.parametrize('seed', [3, 11, 47])
def test_matches_oracle(n,w,seed):
    p = fixture(n,w,seed)
    expected = execute(p)
    verify(p, expected)
    items = schedule(p)
    assert all(v['constraints'] == items[0]['constraints'] for v in items)
    for j, v in enumerate(items):
        assert r1cs_satisfied(v, PALLAS_BASE) and r1cs_satisfied(v, PALLAS_SCALAR)
        assert next_state(v)[0] == j+1
        assert next_state(v)[3+j*w:3+(j+1)*w] == expected[j]['trace'][-1]
        if j:
            assert tuple(v['witness'][i] for i in v['state_in']) == next_state(items[j-1])
    assert next_state(items[-1])[3:] == tuple(x for r in expected for x in r['trace'][-1])


@pytest.mark.parametrize('wire', ['counter','context','register','output','selector'])
def test_corruption(wire):
    p = fixture(3,2,3)
    e = schedule(p)[1]
    w = {'counter':e['state_out'][0], 'context':e['state_in'][1], 'register':e['state_in'][3],
         'output':e['state_out'][5], 'selector':len(e['state_in'])+1+16*6}[wire]
    e['witness'][w] += 1
    assert not r1cs_satisfied(e, PALLAS_SCALAR)


def test_shape_and_context_pinning():
    p = fixture(3,2,3)
    original = schedule(p)
    changed = replace(p, run='else')
    with pytest.raises(ValueError):
        compile_step(changed, initial(p))
    other = schedule(changed)
    assert original[0]['constraints'] != other[0]['constraints']
    e = copy.deepcopy(original[0])
    e['constraints'] = other[0]['constraints']
    assert not r1cs_satisfied(e,PALLAS_SCALAR)
    with pytest.raises(ValueError):
        compile_step(p,next_state(original[-1]))


def test_native_bundle():
    from datagoblin.zk_schedule import native_bundle
    p = fixture(8,2,3)
    stmt, advice = native_bundle(p)
    assert stmt['steps'] == len(advice) == 8
    assert all(len(a) == stmt['wires'] and a[0] == '1' for a in advice)
    assert stmt['z0'] == [advice[0][i] for i in stmt['state_in']]
    assert stmt['zn'] == [advice[-1][i] for i in stmt['state_out']]
    for a in advice:
        e = {'witness': list(map(int,a)), 'constraints': [[[i,int(k)] for i,k in expr]
             for row in stmt['constraints'] for expr in row]}
        e['constraints'] = [e['constraints'][i:i+3] for i in range(0,len(e['constraints']),3)]
        assert r1cs_satisfied(e,PALLAS_SCALAR)
