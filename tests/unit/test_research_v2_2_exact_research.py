import copy
import json
from fractions import Fraction

import pytest

from datagoblin.research_v2_2.curriculum_audit import SPLITS, audit_directory, digest
from datagoblin.research_v2_2.exact_research import ExactResearch, action_values, policy_value


def xor(reward=12):
    return dict(hypotheses=[{'class': c, 'weight': 1} for c in 'ABBA'],
                tests=[dict(id='x', cost=1, bits=[0,0,1,1]), dict(id='y', cost=1, bits=[0,1,0,1])],
                budget=2, correct_reward=reward, wrong_reward=-16)


def test_exact_input_and_cache():
    d = xor()
    solver = ExactResearch(8192)
    r = solver.solve(d)
    assert Fraction(r['value']) == policy_value(d,r['policy']) == 10
    assert not r['cache_hit'] and r['verify_edges'] == r['solve_edges'] > 0
    v = copy.deepcopy(d)
    v['hypotheses'].reverse()
    for h in v['hypotheses']:
        h['weight'] *= 3
    for t in v['tests']:
        t['bits'].reverse()
    v['tests'].reverse()
    r = solver.solve(v)
    assert r['cache_hit'] and r['verify_edges'] == 0 and Fraction(r['value']) == 10
    assert r['training_admission'] is False
    assert not solver.solve(xor(11))['cache_hit']
    assert not ExactResearch(0).solve(d)['cache_hit']
    assert ExactResearch(1).solve(d)['retained_payload_bytes'] == 0
    for bad in (dict(d, budget=True), dict(d, budget=13), dict(d, extra=1),
                dict(d, tests=[dict(id='A',cost=1,bits=[0,1,0,1])]),
                dict(d, tests=[dict(id='x',cost=1,bits=[False,1,0,1])]),
                dict(d, tests=d['tests']*2)):
        with pytest.raises(ValueError):
            solver.solve(bad)
    with pytest.raises(ValueError):
        policy_value(d, {'test':'x','zero':{'test':'x','zero':{'decision':'A'},'one':{'decision':'B'}},
                         'one':{'decision':'A'}})
    q = action_values(d)
    assert q['test:x']['value'] == q['test:y']['value'] == 10
    assert q['test:x']['one_step'] == -1 and q['stop:defer']['value'] == 0


def frac(x):
    x = Fraction(x)
    return f'{x.numerator}/{x.denominator}'


def fixture(reward):
    d = xor(reward)
    rid = digest({'kind':'adaptive_evidence','data':d})
    q = action_values(d)
    public = dict(root_id=rid, source={}, packet_id='fixture', family='adaptive_evidence', difficulty='standard',
                  public_case=dict(classification='public_development',kind='adaptive_evidence',
                                   case_id='research-v1-'+rid,data=d),
                  actions=[dict(id=k,kind=k.split(':')[0],cost=v['cost'],outcome_masses=v['masses'])
                           for k,v in q.items()])
    best = reward-2
    rows = []
    for name,v in q.items():
        kind, action = name.split(':')
        if kind == 'stop':
            p = {'decision':action}
        else:
            other = 'y' if action == 'x' else 'x'
            p = {'test':action, 'zero':{'test':other,'zero':{'decision':'A'},'one':{'decision':'B'}},
                 'one':{'test':other,'zero':{'decision':'B'},'one':{'decision':'A'}}}
        rows.append(dict(id=name,optimal_continuation_value=frac(v['value']),regret=frac(best-v['value']),
                         one_step_value=frac(v['one_step']),optimal_policy=p))
    ref = dict(root_id=rid,public_sha256=digest(public),optimal_value=frac(best),actions=rows,
               baselines={'stop':dict(id='stop:defer',value='0/1'), 'one_step':dict(id='stop:defer',value='0/1'),
                          'information_then_stop':dict(id='test:y',value='-1/1')})
    return public,ref


def write_fixture(path, *, tamper=False, overlap=False):
    (path/'public').mkdir(exist_ok=True,parents=True)
    (path/'referee').mkdir(exist_ok=True)
    m = dict(schema_version='distill.research_curriculum.v1',source_kind='synthetic_oracle_only',
             training_admission=False,splits=[])
    for i,split in enumerate(SPLITS):
        p,r = fixture(12 if overlap else 12-i)
        if tamper and i == 2:
            r['actions'][-1]['optimal_continuation_value'] = '99/1'
        pd = dict(schema_version='distill.research_public.v1',split=split,rows=[p])
        rd = dict(schema_version='distill.research_referee.v1',split=split,rows=[r],
                  custody='evaluator_only',training_admission=False)
        for name,doc in (('public',pd),('referee',rd)):
            (path/name/f'{split}.json').write_text(json.dumps(doc))
        m['splits'].append(dict(split=split,roots=1,actions=len(p['actions']),public_sha256=digest(pd),
                                referee_sha256=digest(rd),baseline_regret={k:dict(
                                    mean_regret=frac(Fraction(r['optimal_value'])-Fraction(v['value'])),
                                    oracle_ties=0) for k,v in r['baselines'].items()}))
    m['manifest_id'] = digest(m)
    (path/'manifest.json').write_text(json.dumps(m))


def test_curriculum_bindings_and_exact_grades(tmp_path):
    write_fixture(tmp_path)
    receipt = audit_directory(tmp_path)
    assert receipt['training_admission'] is False
    assert sum(r['actions'] for r in receipt['splits'].values()) == 15
    write_fixture(tmp_path,tamper=True)
    with pytest.raises(ValueError,match='exact action value'):
        audit_directory(tmp_path)
    write_fixture(tmp_path,overlap=True)
    with pytest.raises(ValueError,match='overlap'):
        audit_directory(tmp_path)


def test_native_curriculum_when_checkout_available(tmp_path):
    native = pytest.importorskip('Distill.research_curriculum')
    native.build(tmp_path/'native',counts=(2,2,2))
    receipt = audit_directory(tmp_path/'native')
    assert sum(r['roots'] for r in receipt['splits'].values()) == 6
    assert receipt['training_admission'] is False
