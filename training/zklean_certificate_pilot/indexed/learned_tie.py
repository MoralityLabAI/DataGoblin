"""Exact A* with learned secondary ordering, on original indexed-rule exercises."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from functools import cache
from heapq import heappop, heappush
from itertools import pairwise

import numpy as np
from weighted import (
    FAMILIES, V, Z, Instance, Rule, enumerate_edges, generate, instantiate_rule,
    key, numeral, offset, one_step, trace, value,
)

CAP = 100_000


def family_shift(seed, limit):
    rng = random.Random(seed)
    rules = [Rule((f, Z), (), False) for f in range(FAMILIES)]
    rules.extend(Rule((f, offset(1, 1)), ((f, V),)) for f in range(FAMILIES))
    for _ in range(18):
        family = rng.randrange(FAMILIES)
        k = rng.randint(1, 7)
        arity = 2 if rng.random() < 0.8 else 1
        premises = tuple((rng.randrange(FAMILIES), offset(rng.randrange(k), rng.randrange(4)))
                         for _ in range(arity))
        rules.append(Rule((family, offset(k, rng.randrange(4))), premises))
    rng.shuffle(rules)
    return Instance(limit, tuple(rules), enumerate_edges(rules, limit), key(rng.randrange(FAMILIES), limit))


class BudgetExceeded(Exception):
    pass


class Graph:
    def __init__(self, x):
        self.x = x
        self.edges = [[] for _ in range(x.n)]
        for i, e in enumerate(x.edges):
            assert all(p < e.head for p in e.premises)
            self.edges[e.head].append((sum(1 << p for p in set(e.premises)), i))
        self.height = []
        self.tree = []
        for i in range(x.n):
            self.height.append(min(1 + max((self.height[p] for p in x.edges[j].premises), default=0)
                                   for _, j in self.edges[i]))
            self.tree.append(min(1 + sum(self.tree[p] for p in x.edges[j].premises)
                                 for _, j in self.edges[i]))

    def lower(self, mask):
        count = bound = 0
        while mask:
            atom = mask.bit_length() - 1
            bound = max(bound, count + self.height[atom])
            count += 1
            mask ^= 1 << atom
        return bound

    def features(self, mask):
        if not mask:
            return [0.] * 10
        atoms = [i for i in range(mask.bit_length()) if mask >> i & 1]
        hh = [self.height[a] for a in atoms]
        tt = [self.tree[a] for a in atoms]
        return [len(atoms), self.lower(mask), max(hh), sum(hh), max(tt), sum(tt),
                sum(math.log1p(t) for t in tt), max(atoms) / self.x.n,
                sum(len(self.edges[a]) for a in atoms), min(hh)]

    def labels(self, masks, cap=30_000):
        states = 0

        @cache
        def residual(m):
            nonlocal states
            states += 1
            if states > cap:
                raise BudgetExceeded
            if not m:
                return 0
            a = m.bit_length() - 1
            rest = m ^ (1 << a)
            return 1 + min(residual(rest | p) for p, _ in self.edges[a])

        try:
            return [(self.features(m), residual(m)) for m in masks]
        finally:
            residual.cache_clear()

    def search(self, mode, model=None, cap=CAP):
        root = 1 << self.x.goal
        best = {root: 0}
        parents = {}
        expanded = generated = 0
        q = []
        scores = {}

        def push(mask, g):
            lb = self.lower(mask)
            if mode == 'deep':
                t = 0.
            else:
                if mask not in scores:
                    feats = self.features(mask)
                    scores[mask] = feats[5] if mode == 'tree' else model(feats) if mask else 0.
                t = g + scores[mask]
            # Learned value never changes the admissible primary key.
            heappush(q, (g + lb, t, -g, mask))

        start = time.perf_counter()
        push(root, 0)
        while q:
            _, _, ng, mask = heappop(q)
            g = -ng
            if g != best[mask]:
                continue
            if mask == 0:
                path = {}
                cur = 0
                while cur != root:
                    prev, j = parents[cur]
                    path[self.x.edges[j].head] = j
                    cur = prev
                assert len(path) == g
                return dict(cost=g, expanded=expanded, discovered=len(best), generated=generated,
                            seconds=time.perf_counter()-start, selection=path)
            expanded += 1
            a = mask.bit_length()-1
            rest = mask ^ (1 << a)
            for prem, j in self.edges[a]:
                generated += 1
                nxt = rest | prem
                if g + 1 >= best.get(nxt, 10**12):
                    continue
                if nxt not in best and len(best) >= cap:
                    return dict(cost=None, expanded=expanded, discovered=len(best), generated=generated,
                                seconds=time.perf_counter()-start)
                best[nxt] = g+1
                parents[nxt] = (mask, j)
                push(nxt, g+1)
        raise ValueError('unreachable goal')


def replay(x, selection):
    established = set()
    for claimed, j in sorted(selection.items()):
        e = x.edges[j]
        rule = x.rules[e.rule]
        assert 0 <= e.witness <= x.limit and (rule.bound or e.witness == 0)
        norm = []
        for f, t in instantiate_rule(rule, e.witness):
            tr = trace(t)
            n = value(t)
            assert tr[0] == t and tr[-1] == numeral(n) and 0 <= n <= x.limit
            assert all(one_step(a, b) for a, b in pairwise(tr))
            norm.append(key(f, n))
        assert norm[-1] == claimed and all(p in established for p in norm[:-1])
        established.add(claimed)
    assert x.goal in established
    return len(established)


def fit(seed, data):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    feats, targets = zip(*data, strict=True)
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    mu = X.mean(0)
    scale = np.maximum(X.std(0), 0.01)
    X = torch.tensor((X-mu)/scale)
    y = torch.tensor(y[:, None])
    net = nn.Sequential(nn.Linear(10, 16), nn.Tanh(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(40):
        for ix in torch.randperm(len(X), generator=rng).split(128):
            loss = nn.functional.smooth_l1_loss(net(X[ix]), y[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        mae = float((net(X)-y).abs().mean())
    w1, b1, w2, b2 = [p.detach().numpy().copy() for p in net.parameters()]

    def estimate(feats):
        z = (np.asarray(feats, dtype=np.float32)-mu)/scale
        return float((w2 @ np.tanh(w1 @ z+b1)+b2)[0])

    return estimate, mae


def experiment(n):
    t0 = time.perf_counter()
    rng = random.Random(7122026)
    data = []
    capped = 0
    for seed in range(70000, 70080):
        g = Graph(generate(seed, 12))
        masks = [1 << g.x.goal]
        for _ in range(24):
            masks.append(sum(1 << a for a in rng.sample(range(g.x.n), rng.randint(1, 5))))
        try:
            data.extend(g.labels(masks))
        except BudgetExceeded:
            capped += 1
    models = {}
    mae = {}
    for seed in [181, 191, 193]:
        models[str(seed)], mae[str(seed)] = fit(seed, data)
    arms = ['deep', 'tree', *models]
    res = dict(training_records=len(data), training_roots=80, capped_training_roots=capped,
               training_mae=mae, parameters=193, train_seconds=time.perf_counter()-t0, cap=CAP, splits={})
    for name, start, size, gen in [('same', 81000, 12, generate), ('size', 82000, 24, generate),
                                   ('new_grammar', 83000, 24, family_shift)]:
        s = {a: dict(solved=0, expanded=0, discovered=0, generated=0, seconds=0., rows=0) for a in arms}
        paired = {a: dict(expanded=0, discovered=0, generated=0, seconds=0.) for a in arms}
        matched = 0
        per_case = []
        for seed in range(start, start+n):
            x = gen(seed, size)
            g = Graph(x)
            case = {}
            for a in arms:
                out = g.search(a if a in ('deep', 'tree') else 'learned', models.get(a))
                if out['cost'] is not None:
                    assert replay(x, out.pop('selection')) == out['cost']
                    s[a]['solved'] += 1
                    s[a]['rows'] += out['cost']
                for k in ('expanded', 'discovered', 'generated', 'seconds'):
                    s[a][k] += out[k]
                case[a] = out
            costs = {o['cost'] for o in case.values() if o['cost'] is not None}
            assert len(costs) <= 1
            if all(o['cost'] is not None for o in case.values()):
                matched += 1
                for a, out in case.items():
                    for k in paired[a]:
                        paired[a][k] += out[k]
            per_case.append({'seed': seed, 'arms': case})
        res['splits'][name] = dict(cases=n, matched=matched, arms=s, paired=paired, per_case=per_case)
        print(name, json.dumps({k:v for k,v in res['splits'][name].items() if k != 'per_case'}), flush=True)
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=int, default=48)
    p.add_argument('--out', default='tie_results.json')
    a = p.parse_args()
    r = experiment(a.cases)
    with open(a.out, 'w', encoding='utf8') as fp:
        json.dump(r, fp, indent=2)
