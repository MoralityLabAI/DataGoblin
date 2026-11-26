from dataclasses import replace

import pytest
from affine_bridge import Schema, Step, affine, compile_schema, environment_id, translate, validate
from learned_tie import Graph
from weighted import Rule, V, Z, generate, offset


def fixture():
    return (Schema(False, 6, ()), Schema(True, 3, (0,)), Schema(True, 4, (0, 3))), (
        Step(0, 0, ()), Step(1, 3, (0,)), Step(2, 3, (0, 1)))


@pytest.mark.parametrize('seed', range(24))
def test_source_translation(seed):
    x = generate(92000 + seed, 12)
    result = Graph(x).search('deep')
    env, rows, depth = translate(x, result['selection'])
    assert validate(env, x.goal, rows, depth)[-1] == x.goal
    assert len(rows) == result['cost']


@pytest.mark.parametrize('style', range(4))
def test_offsets(style):
    for k in range(8):
        assert affine(offset(k, style)) == (1, k)
    assert affine(Z) == (0, 0)


def test_fixture():
    rules, steps = fixture()
    assert validate(rules, 10, steps, 4) == (6, 9, 10)
    assert environment_id(rules, 10) != environment_id(rules, 9)


@pytest.mark.parametrize('index, change', [(0, {'binder': 1}), (1, {'binder': 4}), (1, {'binder': -1}),
                                         (1, {'binder': 32768}), (1, {'binder': True}), (2, {'refs': (0, 0)}),
                                         (2, {'refs': (0, 2)}), (2, {'refs': (0, -1)}), (2, {'refs': (0,)}),
                                         (2, {'rule': 3})])
def test_bad_steps(index, change):
    env, steps = fixture()
    rows = list(steps)
    rows[index] = replace(rows[index], **change)
    with pytest.raises(ValueError):
        validate(env, 10, rows, 4)


def test_overflow_and_capacity():
    assert validate((Schema(True, 65535, ()),), 65535, (Step(0, 0, ()),), 1) == (65535,)
    with pytest.raises(ValueError):
        validate((Schema(True, 65535, ()),), 1, (Step(0, 1, ()),), 1)
    env, rows = fixture()
    with pytest.raises(ValueError):
        validate(env, 10, rows, 1)
    with pytest.raises(ValueError):
        validate(env, 8, rows, 4)


def test_unsupported_rule():
    with pytest.raises(ValueError):
        compile_schema(Rule((0, ('a', V, V)), ()))
    with pytest.raises(ValueError):
        compile_schema(Rule((0, Z), ()))
