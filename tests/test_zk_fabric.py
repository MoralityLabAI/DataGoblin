import copy
import hashlib
import json
import sys
from dataclasses import replace

import pytest

from datagoblin import zk_sigma as z
from datagoblin.zk_fabric import (
    CheckerPolicy, CircuitRegistration, Route, certify_corpus, choose_route, external_verify,
    fit_route_cost, lean_statement, lucid_registration, merkle_root, prove_graph, verify_graph,
)
from datagoblin.zk_replay import Job, Plan, step

CTX = 'ab' * 32


def job(x, w, b, depth):
    return Job('j', (), (x,), ((w,),), (b,), depth)


def output(j):
    h = j.source
    for _ in range(j.steps):
        h = step(j, j.source, h)
    return h


@pytest.mark.parametrize('x,w,b', [(1, 2, 0), (127, 127, 127), (-127, 127, -127), (0, -1, 0), (126, 0, 1)])
@pytest.mark.parametrize('depth', [1, 2, 4])
def test_sigma(x, w, b, depth):
    j = job(x, w, b, depth)
    y = output(j)
    p = z.prove(j, j.source, y, CTX)
    assert z.verify(j, j.source, y, CTX, p)
    assert not z.verify(j, j.source, y, 'ac' * 32, p)
    assert not z.verify(j, j.source, (-y[0] if y[0] else 1,), CTX, p)


@pytest.mark.parametrize('field', ['commitments', 'announcements', 'responses', 'backend'])
def test_proof_tamper(field):
    j = job(5, 2, 1, 2)
    p = z.prove(j, j.source, output(j), CTX)
    if field == 'commitments':
        p[field][0] = '00' * 32
    elif field in ('announcements', 'responses'):
        p[field][0][0] = '00' * 32
    else:
        p[field] = 'nova-compressed'
    assert not z.verify(j, j.source, output(j), CTX, p)


def test_wrong_witness():
    j = job(5, 2, 1, 1)
    with pytest.raises(ValueError):
        z.prove(j, j.source, (0,), CTX)


def test_r1cs_fixed_shape():
    j = job(5, 2, 1, 2)
    a = z.nova_step_export(j, (5,), (12,), 0, CTX)
    b = z.nova_step_export(j, (-127,), (-127,), 31, 'ac' * 32)
    assert a['constraints'] == b['constraints']
    assert z.r1cs_satisfied(a, z.Q) and z.r1cs_satisfied(b, z.Q)
    assert [a['witness'][i] for i in a['state_out'][:3]] == [5, 30, 1]
    a['witness'][a['state_out'][1]] += 1
    assert not z.r1cs_satisfied(a, z.Q)


@pytest.fixture(scope='module')
def graph():
    plan = Plan('tenant', 'run', 'policy', (
        Job('a', (), (3,), ((2,),), (1,), 2),
        Job('b', (), (-8,), ((2,),), (2,), 2),
        Job('c', ('a', 'b'), (0,), ((1,),), (0,), 1),
    ))
    return plan, prove_graph(plan)


def test_graph_and_corpus(graph):
    plan, receipts = graph
    terminal = verify_graph(plan, receipts)
    items = [{'job': 'c', 'root': CTX, 'split': 'TRAIN', 'payload': receipts[-1]['output']}]
    manifest = certify_corpus(plan, receipts, {'c': CTX}, {CTX}, items)
    assert manifest['terminal'] == terminal and manifest['count'] == 1
    assert manifest['root'] == merkle_root(manifest['leaves'])


@pytest.mark.parametrize('attack', ['missing', 'context', 'output', 'parent_order', 'proof_swap'])
def test_graph_attacks(graph, attack):
    plan, original = graph
    receipts = copy.deepcopy(original)
    if attack == 'missing':
        receipts.pop(0)
    elif attack == 'context':
        plan = replace(plan, run='other')
    elif attack == 'output':
        receipts[0]['output'] = (0,)
    elif attack == 'parent_order':
        plan = replace(plan, jobs=(*plan.jobs[:-1], replace(plan.jobs[-1], parents=('b', 'a'))))
    else:
        receipts[0]['proof'] = receipts[1]['proof']
    with pytest.raises(ValueError):
        verify_graph(plan, receipts)


@pytest.mark.parametrize('attack', ['split', 'root', 'payload', 'duplicate', 'allowlist'])
def test_corpus_attacks(graph, attack):
    plan, receipts = graph
    item = {'job': 'c', 'root': CTX, 'split': 'TRAIN', 'payload': receipts[-1]['output']}
    items, roots = [item], {CTX}
    if attack == 'split':
        item['split'] = 'TEST'
    elif attack == 'root':
        item['root'] = 'cd' * 32
    elif attack == 'payload':
        item['payload'] = (99,)
    elif attack == 'duplicate':
        items.append(item.copy())
    else:
        roots.clear()
    with pytest.raises(ValueError):
        certify_corpus(plan, receipts, {'c': CTX}, roots, items)


def test_lean_and_external_gate(tmp_path):
    s = lean_statement(CTX, CTX, CTX, CTX, CTX, ('propext',), CTX)
    with pytest.raises(ValueError):
        lean_statement(CTX, CTX, CTX, CTX, CTX, ('sorryAx',), CTX)
    exe = tmp_path / 'checker'
    exe.write_text(f'#!{sys.executable}\nimport hashlib,json,sys\nr=sys.stdin.buffer.read()\n'
                   'print(json.dumps({"accepted":True,"request_sha256":hashlib.sha256(r).hexdigest()}))\n')
    exe.chmod(0o700)
    key = b'test-key'
    policy = CheckerPolicy('zkpi', s['relation'], hashlib.sha256(key).hexdigest(),
                           hashlib.sha256(exe.read_bytes()).hexdigest())
    # Trusted mock verifies transport only; no Lean or ZK soundness claim.
    assert external_verify(policy, exe, key, s, b'mock-proof')
    assert not external_verify(policy, exe, b'other', s, b'mock-proof')
    assert not external_verify(policy, exe, key, dict(s, allowed_axioms=['sorryAx']), b'mock-proof')
    assert not external_verify(policy, exe, key, dict(s, relation='wrong'), b'mock-proof')
    assert not external_verify(policy, exe, key, s, b'')
    exe.write_text('# changed\n')
    assert not external_verify(policy, exe, key, s, b'mock-proof')


def test_lucid_registration():
    key = b'placeholder-test-key'
    reg = CircuitRegistration('id', 'fixture', '1.0.0', 'plonk', 'bn254', hashlib.sha256(key).hexdigest(), 3)
    assert lucid_registration(reg, key)['proof_system'] == 'plonk'
    for system in ('nova', 'ristretto-sigma-dev-v1', 'groth16'):
        with pytest.raises(ValueError):
            lucid_registration(replace(reg, proof_system=system), key)
    with pytest.raises(ValueError):
        lucid_registration(reg, b'bad-key')


def test_proof_class_and_cost():
    a, b = fit_route_cost(((1, 12), (2, 14), (4, 18), (8, 26)))
    assert (a, b) == (10, 2)
    routes = (Route('replay', 0, 1, 'replay'), Route('sigma', a, b, 'zk-dev'), Route('unavailable', 0, 0, 'other'))
    assert choose_route(routes, 16, 42, 'zk-dev') == 'sigma'
    assert choose_route(routes, 16, 41, 'zk-dev') is None
    assert choose_route(routes, 16, 100, 'zk-production') is None


def test_merkle_length_binding():
    assert merkle_root(['aa'*32, 'bb'*32, 'cc'*32]) != merkle_root(['aa'*32, 'bb'*32, 'cc'*32, 'cc'*32])
