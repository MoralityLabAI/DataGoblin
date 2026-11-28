"""Bounded exact public evidence planning. No external certificate ingestion."""
from __future__ import annotations

import json
from fractions import Fraction

from ._exact_engine import DP, binary_key, check, decode, enc, replay, to_world

MAX_HYPOTHESES, MAX_TESTS, MAX_BUDGET = 64, 8, 12
TERMINALS = ('A', 'B', 'defer')


def _shape(obj, fields):
    if type(obj) is not dict or set(obj) != set(fields):
        raise ValueError('unexpected record shape')


def _integer(x, lo, hi):
    if type(x) is not int or not lo <= x <= hi:
        raise ValueError('integer outside supported range')
    return x


def validate_world(data):
    """Copy a bounded JSON-shaped deterministic binary-test model before solving."""
    _shape(data, ('hypotheses', 'tests', 'budget', 'correct_reward', 'wrong_reward'))
    hs, ts = data['hypotheses'], data['tests']
    if type(hs) is not list or not 1 <= len(hs) <= MAX_HYPOTHESES:
        raise ValueError('hypothesis bound')
    if type(ts) is not list or len(ts) > MAX_TESTS:
        raise ValueError('test bound')
    budget = _integer(data['budget'], 0, MAX_BUDGET)
    correct = _integer(data['correct_reward'], 0, 1_000_000)
    wrong = _integer(data['wrong_reward'], -1_000_000, correct)
    hypotheses, tests, ids = [], [], set()
    for h in hs:
        _shape(h, ('class', 'weight'))
        if type(h['class']) is not str or h['class'] not in ('A', 'B'):
            raise ValueError('invalid class')
        hypotheses.append({'class': h['class'], 'weight': _integer(h['weight'], 1, 1_000_000)})
    for t in ts:
        _shape(t, ('id', 'cost', 'bits'))
        name = t['id']
        if (type(name) is not str or not 1 <= len(name) <= 128 or not name.isascii()
                or not all(c.isalnum() or c in '_.-' for c in name) or name in TERMINALS or name in ids):
            raise ValueError('invalid test ID')
        ids.add(name)
        bits = t['bits']
        if type(bits) is not list or len(bits) != len(hs):
            raise ValueError('bit vector shape')
        tests.append({'id': name, 'cost': _integer(t['cost'], 1, 65535),
                      'bits': [_integer(b, 0, 1) for b in bits]})
    return dict(hypotheses=hypotheses, tests=tests, budget=budget,
                correct_reward=correct, wrong_reward=wrong)


def _native(tree):
    if len(tree) == 1:
        return {'decision': tree[0]}
    return {'test': tree[0], 'zero': _native(tree[1]), 'one': _native(tree[2])}


def policy_value(data, policy):
    """Check reachable native policy shape, legal splits, path costs and value."""
    d = validate_world(data)
    hs = d['hypotheses']
    ts = {t['id']: t for t in d['tests']}

    def visit(node, indices, budget):
        if type(node) is dict and set(node) == {'decision'} and node['decision'] in TERMINALS:
            a = node['decision']
            return sum(hs[i]['weight']*(0 if a == 'defer' else d['correct_reward'] if a == hs[i]['class']
                       else d['wrong_reward']) for i in indices)
        _shape(node, ('test', 'zero', 'one'))
        name = node['test']
        if type(name) is not str or name not in ts or ts[name]['cost'] > budget:
            raise ValueError('invalid policy test')
        test = ts[name]
        z = [i for i in indices if not test['bits'][i]]
        o = [i for i in indices if test['bits'][i]]
        if not z or not o:
            raise ValueError('constant or repeated policy test')
        cost = test['cost']
        return (visit(node['zero'], z, budget-cost)+visit(node['one'], o, budget-cost)
                -cost*sum(hs[i]['weight'] for i in indices))

    return Fraction(visit(policy, list(range(len(hs))), d['budget']), sum(h['weight'] for h in hs))


class ExactResearch:
    """Cold checked solve by default; optional in-process freeze cache of checked roots.

    Capacity counts serialized keys and values, not RAM. No external cache loads.
    """
    def __init__(self, capacity=0):
        self.capacity = _integer(capacity, 0, 65536)
        self._cache = {}
        self._used = 0

    def solve(self, data):
        d = validate_world(data)
        key, scale = binary_key(d)
        total = sum(h['weight'] for h in d['hypotheses'])
        hit = key in self._cache
        edges = verified = 0
        if hit:
            v, tree = json.loads(self._cache[key])
        else:
            solver = DP(to_world(decode(key)))
            v, tree = solver.run()
            edges, verified = solver.edges, check(solver)
        if replay(d, tree) != v*scale:
            raise ValueError('original-world replay mismatch')
        if not hit:
            payload = enc((v, tree))
            size = len(key)+len(payload)
            if self._used+size <= self.capacity:
                self._cache[key] = payload
                self._used += size
        value = Fraction(v*scale, total)
        return dict(value=f'{value.numerator}/{value.denominator}', policy=_native(tree), cache_hit=hit,
                    solve_edges=edges, verify_edges=verified, retained_payload_bytes=self._used,
                    custody='reference_only', training_admission=False)


def action_values(data):
    """Exact forced-first-action values and one-step controls; costs counted once."""
    d = validate_world(data)
    solver = DP(d)
    full, b = solver.full, d['budget']
    w, a = solver.count(full)
    c, r = d['correct_reward'], d['wrong_reward']
    actions = {}
    for name, v in (('A', a*c+(w-a)*r), ('B', (w-a)*c+a*r), ('defer', 0)):
        actions['stop:'+name] = dict(value=Fraction(v,w), one_step=Fraction(v,w), cost=0, masses=[w])
    for test, mask in zip(solver.ts, solver.ms, strict=True):
        o, cost = full & mask, test['cost']
        if not o or o == full or cost > b:
            continue
        z = full ^ o
        v = solver.solve(z,b-cost)[0]+solver.solve(o,b-cost)[0]-cost*w
        masses = [solver.count(m) for m in (z,o)]
        one_step = sum(max(0, a0*c+(w0-a0)*r, (w0-a0)*c+a0*r) for w0,a0 in masses)-cost*w
        actions['test:'+test['id']] = dict(value=Fraction(v,w), one_step=Fraction(one_step,w),
                                           cost=cost, masses=[m[0] for m in masses])
    optimum, _ = solver.run()
    check(solver)
    if max(x['value'] for x in actions.values()) != Fraction(optimum,w):
        raise ValueError('forced action decomposition mismatch')
    return actions
