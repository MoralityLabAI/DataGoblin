"""RV22-M: bounded selected-policy certificates and lazy admission."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from certificate_dp import MaskDP, replay, root, subset, variants, world

CAP = 8192
HORIZON = 12
COHORTS = ('burst', 'mixed', 'churn')
ARMS = ('cold', 'identity', 'canonical', 'canonical_selected', 'checked_root',
        'checked_root_lazy', 'checked_eager', 'checked_lazy', 'checked_learned')


def encode(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def policy(solver, mask, budget):
    _, action = solver.memo[mask, budget]
    if action in ('A', 'B', 'defer'):
        return (action,)
    j = next(j for j, t in enumerate(solver.tests) if t['id'] == action)
    one = mask & solver.masks[j]
    b = budget - solver.tests[j]['cost']
    return action, policy(solver, mask ^ one, b), policy(solver, one, b)


def solve(d):
    solver = MaskDP(d)
    mask = (1 << len(solver.h)) - 1
    v, _ = solver.solve(mask, d['budget'])
    tree = policy(solver, mask, d['budget'])
    if replay(d, tree) != v:
        raise ValueError('cold replay mismatch')
    return solver, v, tree


def check(solver):
    """Independent integer Bellman check; all legal alternatives are verified."""
    d = solver.d
    c, r = d['correct_reward'], d['wrong_reward']
    tests = sorted(d['tests'], key=lambda t: t['id'])
    masks = [sum(bit << i for i, bit in enumerate(t['bits'])) for t in tests]
    counts, checked = {}, {}
    edges = 0
    for (m, b), record in sorted(solver.memo.items(), key=lambda item: item[0][1]):
        if m not in counts:
            rows = [h for i, h in enumerate(d['hypotheses']) if m >> i & 1]
            counts[m] = (sum(h['weight'] for h in rows), sum(h['weight'] for h in rows if h['class'] == 'A'))
        w, a = counts[m]
        best, action = max((a*c+(w-a)*r, 'A'), ((w-a)*c+a*r, 'B'), (0, 'defer'))
        if best < c*w:
            for t, tm in zip(tests, masks, strict=True):
                o = m & tm
                cost = t['cost']
                if not o or o == m or cost > b:
                    continue
                edges += 1
                if (o, b-cost) not in checked or (m^o, b-cost) not in checked:
                    raise ValueError('missing checked child')
                q = checked[o, b-cost][0]+checked[m^o, b-cost][0]-cost*w
                if q > best:
                    best, action = q, t['id']
        if record != (best, action):
            raise ValueError('invalid Bellman record')
        checked[m, b] = record
    if ((1 << len(d['hypotheses']))-1, d['budget']) not in checked:
        raise ValueError('missing checked root')
    return edges


def selected_records(solver, tree):
    d = solver.d
    records = []

    def walk(m, b, node):
        sub = subset(d, [i for i in range(len(solver.h)) if m >> i & 1])
        sub['budget'] = b
        key, scale = root(sub)
        value = solver.memo[m, b][0]
        if value % scale or replay(sub, node) != value:
            raise ValueError('certificate export mismatch')
        records.append((encode(key), encode((value//scale, node))))
        if len(node) == 3:
            j = next(j for j, t in enumerate(solver.tests) if t['id'] == node[0])
            one = m & solver.masks[j]
            b -= solver.tests[j]['cost']
            walk(m^one, b, node[1])
            walk(one, b, node[2])

    walk((1 << len(solver.h))-1, d['budget'], tree)
    return records


def checked_records(solver, tree):
    edges = check(solver)
    return selected_records(solver, tree), edges


class Store:
    """Strict serialized key + value byte cap; Python overhead is excluded."""
    def __init__(self, cap=CAP):
        self.cap = cap
        self.data = OrderedDict()
        self.bytes = self.peak = self.evictions = 0

    def get(self, key):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        return None

    def put(self, key, value):
        size = len(key)+len(value)
        if size > self.cap:
            return
        if key in self.data:
            old = self.data.pop(key)
            self.bytes -= len(key)+len(old)
        while self.bytes+size > self.cap:
            k, v = self.data.popitem(last=False)
            self.bytes -= len(k)+len(v)
            self.evictions += 1
        self.data[key] = value
        self.bytes += size
        self.peak = max(self.peak, self.bytes)


class History:
    def __init__(self):
        self.data = OrderedDict()

    def step(self, key, t, d):
        # Hashes are routing hints only; full canonical keys authorize reuse.
        k = hashlib.sha256(key).digest()
        count, last = self.data.pop(k, (0, -100))
        gap = min(t-last, HORIZON)
        x = np.array([1., min(count, 6)/6, gap/HORIZON, float(gap == 1),
                      min(len(d['hypotheses']), 24)/24, len(d['tests'])/8, d['budget']/9])
        self.data[k] = (count+1, t)
        if len(self.data) > 128:
            self.data.popitem(last=False)
        return x, count


def stream(seed, kind):
    rng = random.Random(seed)
    projects = []
    for j in range(8):
        d = world(rng, 12 if j%2 == 0 else 16, 6 if j%2 == 0 else 8)
        v = dict(variants(d, rng))
        projects.append([d, v['perm_scale'], v['split_duplicate'], v['conditioned'], v['cost_mutation']])
    if kind == 'burst':
        ids = [i for i in range(5) for _ in range(8)]
    elif kind == 'mixed':
        ids = rng.choices(range(8), weights=[6, 5, 4, 3, 2, 1, 1, 1], k=40)
    else:
        ids = list(range(8))*5
    queries = []
    for t, i in enumerate(ids):
        if kind == 'churn' and rng.random() < .65:
            d = world(rng, 12 if t%2 == 0 else 16, 6 if t%2 == 0 else 8)
        else:
            d = rng.choices(projects[i], weights=[3, 3, 2, 2, 1])[0]
        queries.append(copy.deepcopy(d))
    return queries


def fit():
    x, y = [], []
    for ci, kind in enumerate(COHORTS):
        for j in range(24):
            ds = stream(2026112900+ci*100+j, kind)
            keys = [encode(root(d)[0]) for d in ds]
            hist = History()
            for t, (key, d) in enumerate(zip(keys, ds, strict=True)):
                f, _ = hist.step(key, t, d)
                x.append(f)
                y.append(keys[t+1:t+1+HORIZON].count(key))
    x, y = np.array(x), np.array(y)
    # Seven-coefficient additive value head, fitted on independent streams.
    w = np.linalg.solve(x.T@x+np.eye(x.shape[1]), x.T@y)
    return w, len(x)


def evaluate(ds, arm, w=None, threshold=1., cap=CAP):
    store, hist = Store(cap), History()
    metrics = dict(cpu_ms=0., hits=0, misses=0, admissions=0, verify_edges=0, solve_edges=0,
                   check_ms=0., utility_numerator_checks=0)
    values = []
    for t, d in enumerate(ds):
        start = time.process_time_ns()
        if arm == 'cold':
            key, scale = None, 1
        elif arm == 'identity':
            key, scale = encode(d), 1
        else:
            k, scale = root(d)
            key = encode(k)
        rec = store.get(key) if key is not None else None
        if arm in ('checked_lazy', 'checked_learned', 'checked_root_lazy'):
            x, count = hist.step(key, t, d)
        if rec is not None:
            v, tree = json.loads(rec)
            value = v*scale
            if replay(d, tree) != value:
                raise ValueError('cached replay mismatch')
            metrics['hits'] += 1
        else:
            metrics['misses'] += 1
            solver, value, tree = solve(d)
            metrics['solve_edges'] += solver.work
            if arm in ('identity', 'canonical'):
                if value % scale:
                    raise ValueError('nonintegral normalized value')
                store.put(key, encode((value//scale, tree)))
            elif arm == 'canonical_selected':
                for k, v in reversed(selected_records(solver, tree)):
                    store.put(k, v)
            elif arm == 'checked_root' or (arm == 'checked_root_lazy' and count >= 1):
                begin = time.process_time_ns()
                edges = check(solver)
                if value % scale:
                    raise ValueError('nonintegral normalized value')
                store.put(key, encode((value//scale, tree)))
                metrics['check_ms'] += (time.process_time_ns()-begin)/1e6
                metrics['admissions'] += 1
                metrics['verify_edges'] += edges
            elif (arm == 'checked_eager' or (arm == 'checked_lazy' and count >= 1)
                  or (arm == 'checked_learned' and float(x@w) >= threshold)):
                begin = time.process_time_ns()
                records, edges = checked_records(solver, tree)
                # Root is inserted last, protecting it from its own descendants.
                for k, v in reversed(records):
                    store.put(k, v)
                metrics['check_ms'] += (time.process_time_ns()-begin)/1e6
                metrics['admissions'] += 1
                metrics['verify_edges'] += edges
        metrics['cpu_ms'] += (time.process_time_ns()-start)/1e6
        metrics['utility_numerator_checks'] += 1
        values.append((value, sum(h['weight'] for h in d['hypotheses'])))
    metrics.update(peak_payload_bytes=store.peak, evictions=store.evictions, retained=len(store.data))
    return metrics, values


def campaign(trajectories=16, threshold=1.5, seed=2026114500):
    w, n = fit()
    # Frozen from the separate six-arm development campaign.
    cal = [(0.5, 1085.134206), (1.0, 480.700062), (1.5, 454.727946),
           (2.0, 459.288214), (3.0, 462.716295)]
    rng = random.Random(2026112907)
    report = dict(schema='rv22m.lazy_certificate.v1', cap_payload_bytes=CAP, horizon=HORIZON,
                  training_examples=n, head_weights=w.tolist(), development_calibration_ms=cal,
                  threshold=threshold, evaluation_seed=seed, cohorts={})
    for ci, kind in enumerate(COHORTS):
        rows = {a: [] for a in ARMS}
        for j in range(trajectories):
            ds = stream(seed+ci*100+j, kind)
            order = list(ARMS)
            rng.shuffle(order)
            reference = None
            for arm in order:
                result, values = evaluate(ds, arm, w, threshold)
                if reference is None:
                    reference = values
                if reference != values:
                    raise ValueError('cross-arm exact parity failure')
                rows[arm].append(result)
        report['cohorts'][kind] = {a: {k: float(np.mean([r[k] for r in v])) for k in v[0]} for a, v in rows.items()}
        report['cohorts'][kind]['paired_cpu'] = {}
        boot = np.random.default_rng(900+ci).integers(trajectories, size=(10000, trajectories))
        for a in ARMS[1:]:
            delta = np.array([r['cpu_ms']-c['cpu_ms'] for r, c in zip(rows[a], rows['cold'], strict=True)])
            report['cohorts'][kind]['paired_cpu'][a] = dict(delta_ms=float(delta.mean()),
                ci95=np.quantile(delta[boot].mean(axis=1), [.025, .975]).tolist())
    report['scope'] = 'Synthetic streams; independent mask Bellman check; payload byte cap excludes Python overhead.'
    report['queries_per_arm'] = trajectories*40*len(COHORTS)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trajectories', type=int, default=16)
    p.add_argument('--seed', type=int, default=2026114500)
    args = p.parse_args()
    if args.out.exists() or not 1 <= args.trajectories <= 64:
        raise ValueError('fresh output and bounded trajectories required')
    args.out.write_text(json.dumps(campaign(args.trajectories, seed=args.seed), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
