"""Evaluator-only audit of pinned Distill adaptive-evidence curriculum v1."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from ._exact_engine import binary_key
from .exact_research import action_values, policy_value, validate_world

UPSTREAM = '7fa19c90372b13d71676e6fef13819d2a18b9be3'
SPLITS = ('train', 'calibration', 'public_eval')
LIMIT = 1_048_576


def digest(x):
    raw = json.dumps(x, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
    if len(raw) >= LIMIT:
        raise ValueError('canonical byte cap')
    return hashlib.sha256(raw).hexdigest()


def _load(path):
    with path.open('rb') as f:
        raw = f.read(LIMIT)
    if len(raw) >= LIMIT:
        raise ValueError('file byte cap')
    return json.loads(raw)


def _need(flag, msg):
    if not flag:
        raise ValueError(msg)


def _fraction(x):
    _need(type(x) is str and len(x) <= 128, 'fraction type')
    v = Fraction(x)
    _need(x == f'{v.numerator}/{v.denominator}', 'noncanonical fraction')
    return v


def audit_row(public, label):
    _need(public['family'] == 'adaptive_evidence', 'unsupported family')
    case = public['public_case']
    _need(case['classification'] == 'public_development' and case['kind'] == 'adaptive_evidence', 'nonpublic case')
    d = validate_world(case['data'])
    rid = digest({'kind': 'adaptive_evidence', 'data': case['data']})
    _need(public['root_id'] == label['root_id'] == rid and case['case_id'] == 'research-v1-'+rid,
          'root binding')
    _need(label['public_sha256'] == digest(public), 'public digest binding')
    q = action_values(d)
    pa = {x['id']: x for x in public['actions']}
    la = {x['id']: x for x in label['actions']}
    _need(len(pa) == len(public['actions']) and len(la) == len(label['actions']) and pa.keys() == la.keys() == q.keys(),
          'action inventory')
    best = max(x['value'] for x in q.values())
    _need(_fraction(label['optimal_value']) == best, 'root optimum')
    for name, v in q.items():
        p, y = pa[name], la[name]
        kind, action = name.split(':', 1)
        _need(p == dict(id=name, kind=kind, cost=v['cost'], outcome_masses=v['masses']), 'public action metadata')
        _need(_fraction(y['optimal_continuation_value']) == v['value'] and _fraction(y['regret']) == best-v['value']
              and _fraction(y['one_step_value']) == v['one_step'], 'exact action value')
        tree = y['optimal_policy']
        _need(tree.get('test' if kind == 'test' else 'decision') == action, 'forced first action')
        _need(policy_value(d, tree) == v['value'], 'forced policy replay')
    stop = max((k for k in q if k.startswith('stop:')), key=lambda k: (q[k]['value'], k))
    myopic = max(q, key=lambda k: (q[k]['one_step'], k.startswith('stop:'), k))
    tests = [k for k in q if k.startswith('test:')]
    info = max(tests, key=lambda k: (min(q[k]['masses']), -q[k]['cost'], k)) if tests else stop
    for name, action in (('stop', stop), ('one_step', myopic), ('information_then_stop', info)):
        record = label['baselines'][name]
        _need(record['id'] == action and _fraction(record['value']) == q[action]['one_step'], 'baseline value')
    return rid, binary_key(d)[0], len(q)


def audit_directory(path):
    """Verify local documents and exact labels; no authorization to fit on them."""
    path = Path(path)
    manifest = _load(path/'manifest.json')
    _need(manifest['schema_version'] == 'distill.research_curriculum.v1'
          and manifest['source_kind'] == 'synthetic_oracle_only'
          and manifest['training_admission'] is False, 'manifest custody')
    payload = {k: v for k,v in manifest.items() if k != 'manifest_id'}
    _need(manifest['manifest_id'] == digest(payload), 'manifest digest')
    _need([s['split'] for s in manifest['splits']] == list(SPLITS), 'split inventory')
    roots, previous_keys, totals = set(), set(), {}
    for entry in manifest['splits']:
        split = entry['split']
        p = _load(path/'public'/f'{split}.json')
        r = _load(path/'referee'/f'{split}.json')
        _need(p['schema_version'] == 'distill.research_public.v1'
              and r['schema_version'] == 'distill.research_referee.v1'
              and p['split'] == r['split'] == split and r['custody'] == 'evaluator_only'
              and r['training_admission'] is False, 'split custody')
        _need(digest(p) == entry['public_sha256'] and digest(r) == entry['referee_sha256'], 'split digest')
        _need(1 <= len(p['rows']) == len(r['rows']) == entry['roots'] <= 256, 'split size')
        count, keys = 0, set()
        regrets = {k: [] for k in ('stop', 'one_step', 'information_then_stop')}
        for public, label in zip(p['rows'], r['rows'], strict=True):
            rid, key, n = audit_row(public, label)
            _need(rid not in roots and key not in previous_keys, 'root or canonical cross-split overlap')
            roots.add(rid)
            keys.add(key)
            count += n
            for name, values in regrets.items():
                values.append(_fraction(label['optimal_value'])-_fraction(label['baselines'][name]['value']))
        _need(count == entry['actions'], 'action count')
        for name, values in regrets.items():
            m = entry['baseline_regret'][name]
            _need(_fraction(m['mean_regret']) == sum(values, Fraction(0))/len(values)
                  and m['oracle_ties'] == sum(v == 0 for v in values), 'aggregate baseline')
        previous_keys.update(keys)
        totals[split] = dict(roots=len(p['rows']), actions=count, distinct_exact_keys=len(keys))
    return dict(schema='datagoblin.distill_curriculum_audit.v1', reviewed_contract_commit=UPSTREAM,
                manifest_id=manifest['manifest_id'], splits=totals, custody='evaluator_only', training_admission=False)
