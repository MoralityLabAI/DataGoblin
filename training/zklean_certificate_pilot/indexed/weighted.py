"""ZKL-07: original indexed-family dependent proof grammar, no Lean import."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from functools import cache, lru_cache
from itertools import pairwise

# Natural terms: Z, S(t), Add(t,u), free binder n. Normal forms are Peano numerals.
Z = ('z',)
V = ('v',)
INF = 10**12
FAMILIES = 2


@cache
def numeral(n):
    assert n >= 0
    return Z if n == 0 else ('s', numeral(n - 1))


def offset(k, style):
    if style == 0:
        t = V
        for _ in range(k):
            t = ('s', t)
        return t
    if style == 1:
        return ('a', V, numeral(k))
    if style == 2:
        return ('a', numeral(k), V)
    return ('a', ('a', V, Z), numeral(k))


@cache
def instantiate(t, n):
    if t == V:
        return numeral(n)
    if t == Z:
        return Z
    return (t[0], *(instantiate(x, n) for x in t[1:]))


def reduce_once(t):
    # Leftmost outermost, one Add iota rewrite under a constructor context.
    if t[0] == 'a':
        x, y = t[1:]
        if x == Z:
            return y
        if x[0] == 's':
            return ('s', ('a', x[1], y))
    for i, child in enumerate(t[1:], 1):
        nxt = reduce_once(child)
        if nxt is not None:
            return (*t[:i], nxt, *t[i + 1:])
    return None


@lru_cache(maxsize=50000)
def trace(t):
    out = [t]
    while (nxt := reduce_once(out[-1])) is not None:
        out.append(nxt)
    return tuple(out)


def value(t):
    # Independent denotational interpreter; rejects uninstantiated expressions.
    if t == Z:
        return 0
    if t[0] == 's' and len(t) == 2:
        return 1 + value(t[1])
    if t[0] == 'a' and len(t) == 3:
        return value(t[1]) + value(t[2])
    raise ValueError('invalid closed Nat term')


def one_step(a, b):
    # Independent checker: exactly one constructor-context Add rewrite anywhere.
    if a[0] == 'a':
        x, y = a[1:]
        if (x == Z and b == y) or (x[0] == 's' and b == ('s', ('a', x[1], y))):
            return True
    if a[0] != b[0] or len(a) != len(b):
        return False
    changed = [i for i in range(1, len(a)) if a[i] != b[i]]
    return len(changed) == 1 and one_step(a[changed[0]], b[changed[0]])


@dataclass(frozen=True)
class Rule:
    # forall n : Nat, P_i(expr_i(n)) -> ... -> P_h(expr_h(n));
    # ground facts have bound=False and a closed index.
    head: tuple
    premises: tuple
    bound: bool = True


@dataclass(frozen=True)
class Edge:
    rule: int
    witness: int
    head: int
    premises: tuple
    structural: int
    iota: int
    reads: int

    @property
    def weight(self):
        return self.structural + self.iota + 2 * self.reads


@dataclass(frozen=True)
class Instance:
    limit: int
    rules: tuple
    edges: tuple
    goal: int

    @property
    def n(self):
        return (self.limit + 1) * FAMILIES


def key(family, index):
    return FAMILIES * index + family


def instantiate_rule(rule, n):
    return tuple((f, instantiate(t, n)) for f, t in (*rule.premises, rule.head))


def enumerate_edges(rules, limit):
    edges = []
    for ri, rule in enumerate(rules):
        for n in range(limit + 1) if rule.bound else (0,):
            types = instantiate_rule(rule, n)
            fs = tuple(key(f, value(t)) for f, t in types)
            h, pre = fs[-1], fs[:-1]
            if h // FAMILIES > limit:
                continue
            assert all(p // FAMILIES < h // FAMILIES for p in pre)
            # Count the grammar's explicit lookup, optional Pi instantiation,
            # premise applications, final typing row, iota rows and premise reads.
            structural = 2 + int(rule.bound) + len(pre)
            iota = sum(len(trace(t)) - 1 for _, t in types)
            edges.append(Edge(ri, n, h, pre, structural, iota, len(pre)))
    return tuple(edges)


def generate(seed, limit, shift=False):
    rng = random.Random(seed)
    rules = [Rule((f, Z), (), False) for f in range(FAMILIES)]
    # Always-solvable, expensive successor rules.
    rules += [Rule((f, offset(1, 1)), ((f, V),)) for f in range(FAMILIES)]
    # Independently sample rule signatures, including redundant derivations.
    for _ in range(12):
        h = rng.randrange(FAMILIES)
        k = rng.randint(1, 5 if not shift else 8)
        arity = rng.choices((1, 2), (4, 3))[0]
        pre = tuple((rng.randrange(FAMILIES), offset(rng.randrange(k), rng.randrange(4)))
                    for _ in range(arity))
        rules.append(Rule((h, offset(k, rng.randrange(4))), pre))
    rng.shuffle(rules)
    return Instance(limit, tuple(rules), enumerate_edges(rules, limit), key(rng.randrange(FAMILIES), limit))


def dp(x, unit=False, rounds=None):
    costs = [INF] * x.n
    by_head = sorted(x.edges, key=lambda e: e.head)
    if rounds is None:
        for e in by_head:
            c = (1 if unit else e.weight) + sum(costs[p] for p in e.premises)
            costs[e.head] = min(costs[e.head], c)
    else:
        for _ in range(rounds):
            nxt = costs.copy()
            for e in x.edges:
                c = (1 if unit else e.weight) + sum(costs[p] for p in e.premises)
                nxt[e.head] = min(nxt[e.head], c)
            costs = nxt
    scores = [(1 if unit else e.weight) + sum(costs[p] for p in e.premises) for e in x.edges]
    return costs, scores


def search(x, scores, budget=512):
    candidates = [[] for _ in range(x.n)]
    for i, e in enumerate(x.edges):
        candidates[e.head].append(i)
    for c in candidates:
        c.sort(key=lambda i: (scores[i], i))
    rows = []
    attempts = 0

    def visit(goal):
        nonlocal attempts
        for i in candidates[goal]:
            if attempts >= budget:
                return None
            attempts += 1
            e = x.edges[i]
            start = len(rows)
            refs = []
            for p in e.premises:
                j = visit(p)
                if j is None:
                    break
                refs.append(j)
            else:
                types = instantiate_rule(x.rules[e.rule], e.witness)
                rows.append((e.rule, e.witness, tuple(refs), tuple(trace(t) for _, t in types)))
                return len(rows) - 1
            del rows[start:]
        return None

    return (rows if visit(x.goal) is not None else None), attempts


def verify(x, rows):
    """Independent replay; never trusts scores, Edge objects or claimed costs."""
    if not rows:
        raise ValueError('empty proof')
    types = []
    uses = [0] * len(rows)
    structural = iota = reads = 0
    for j, (rid, n, refs, conversions) in enumerate(rows):
        if not 0 <= rid < len(x.rules) or not 0 <= n <= x.limit:
            raise ValueError('rule/witness out of bounds')
        rule = x.rules[rid]
        if (not rule.bound and n != 0) or len(refs) != len(rule.premises):
            raise ValueError('bad instantiation')
        expected = instantiate_rule(rule, n)
        if len(conversions) != len(expected):
            raise ValueError('missing conversion')
        normal = []
        for (family, t), tr in zip(expected, conversions, strict=True):
            if not tr or tr[0] != t:
                raise ValueError('wrong conversion start')
            if any(not one_step(a, b) for a, b in pairwise(tr)):
                raise ValueError('invalid iota step')
            v = value(t)
            if tr[-1] != numeral(v) or not 0 <= v <= x.limit:
                raise ValueError('wrong normal form')
            normal.append(key(family, v))
            iota += len(tr) - 1
        for ref, ty in zip(refs, normal[:-1], strict=True):
            if not 0 <= ref < j or types[ref] != ty:
                raise ValueError('bad dependent premise')
            uses[ref] += 1
        types.append(normal[-1])
        structural += 2 + int(rule.bound) + len(refs)
        reads += len(refs)
    if types[-1] != x.goal or uses[-1] != 0 or any(v != 1 for v in uses[:-1]):
        raise ValueError('wrong goal or non-tree proof')
    return structural, iota, reads


def weighted(cost):
    return cost[0] + cost[1] + 2 * cost[2]


def measure(xs, score_fn):
    out = dict(n=len(xs), solved=0, optimal=0, attempts=0, cost_sum=0, optimum_sum=0,
               structural=0, iota=0, reads=0, ratios=[])
    for x, scores in zip(xs, score_fn(xs), strict=True):
        rows, attempts = search(x, scores)
        out['attempts'] += attempts
        if rows is None:
            continue
        c = verify(x, rows)
        opt = dp(x)[0][x.goal]
        cost = weighted(c)
        assert cost >= opt
        out['solved'] += 1
        out['optimal'] += int(cost == opt)
        out['cost_sum'] += cost
        out['optimum_sum'] += opt
        out['structural'] += c[0]
        out['iota'] += c[1]
        out['reads'] += c[2]
        out['ratios'].append(cost / opt)
    ratios = out.pop('ratios')
    out['mean_ratio'] = statistics.mean(ratios) if ratios else None
    return out


def neural_run(train, splits, seeds, epochs):
    import torch
    from torch import nn

    torch.set_num_threads(1)

    def batch(xs):
        n = xs[0].n
        m = max(len(x.edges) for x in xs)
        h, p, features, valid, labels = [], [], [], [], []
        for x in xs:
            e = x.edges
            pad = m - len(e)
            h.append([r.head for r in e] + [0] * pad)
            p.append([list(r.premises) + [n] * (2 - len(r.premises)) for r in e] + [[n, n]] * pad)
            features.append([[math.log1p(r.structural), math.log1p(r.iota),
                              float(r.reads), math.log1p(r.weight)] for r in e] + [[0.] * 4] * pad)
            valid.append([True] * len(e) + [False] * pad)
            labels.append([math.log1p(min(v, 4096)) for v in dp(x)[1]] + [0.] * pad)
        return (torch.tensor(h), torch.tensor(p), torch.tensor(features), torch.tensor(valid),
                torch.tensor(labels), n)

    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            self.width = 12
            self.rank = nn.Sequential(nn.Linear(28, 12), nn.Tanh(), nn.Linear(12, 1))
            self.update = nn.GRUCell(1, 12)

        def forward(self, h, p, features, valid, n, loops):
            b, m = h.shape
            s = torch.zeros(b, n + 1, self.width)
            idx = torch.arange(b)[:, None, None]
            for i in range(loops):
                z = s[idx, p].reshape(b, m, 2 * self.width)
                score = self.rank(torch.cat([z, features], -1)).squeeze(-1)
                if i == loops - 1:
                    return score
                best = torch.full((b, n), 12.).scatter_reduce(
                    1, h, score.masked_fill(~valid, 12.), reduce='amin', include_self=True)
                st = self.update(best.reshape(-1, 1), s[:, :n].reshape(-1, self.width)).reshape(b, n, self.width)
                s = torch.cat([st, torch.zeros(b, 1, self.width)], 1)

    results = {}
    for seed in seeds:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model = Ranker()
        opt = torch.optim.Adam(model.parameters(), lr=.008)
        for _ in range(epochs):
            order = list(range(len(train)))
            rng.shuffle(order)
            for start in range(0, len(order), 16):
                h, p, f, valid, y, n = batch([train[i] for i in order[start:start+16]])
                pred = model(h, p, f, valid, n, 6)
                loss = torch.nn.functional.smooth_l1_loss(pred[valid], y[valid])
                opt.zero_grad()
                loss.backward()
                opt.step()

        def predict(xs, loops, fitted=model):
            with torch.no_grad():
                out = []
                for i in range(0, len(xs), 16):
                    h, p, f, valid, _, n = batch(xs[i:i+16])
                    out.extend(fitted(h, p, f, valid, n, loops).tolist())
                return out

        results[str(seed)] = {name: {str(k): measure(xs, lambda ys, loops=k: predict(ys, loops))
                                     for k in (3, 6, 12)} for name, xs in splits.items()}
    return {'parameters': sum(p.numel() for p in model.parameters()), 'seeds': results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbolic-only', action='store_true')
    p.add_argument('--epochs', type=int, default=12)
    args = p.parse_args()
    splits = {'same': [generate(i, 12) for i in range(10000, 10128)],
              'size': [generate(i, 24) for i in range(20000, 20128)],
              'grammar': [generate(i, 24, True) for i in range(30000, 30128)]}
    res = {'objective': 'structural + iota + 2*premise_reads', 'budget': 512,
           'symbolic': {name: {k: measure(xs, f) for k, f in (
               ('local', lambda ys: [[e.weight for e in y.edges] for y in ys]),
               ('unit_dp', lambda ys: [dp(y, unit=True)[1] for y in ys]),
               ('weighted_dp_2', lambda ys: [dp(y, rounds=2)[1] for y in ys]),
               ('weighted_dp_6', lambda ys: [dp(y, rounds=6)[1] for y in ys]),
               ('weighted_dp_exact', lambda ys: [dp(y)[1] for y in ys]))} for name, xs in splits.items()}}
    if not args.symbolic_only:
        train = [generate(i, 12) for i in range(384)]
        res['neural'] = neural_run(train, splits, [171, 173, 179], args.epochs)
        res['train_roots'] = len(train)
        res['epochs'] = args.epochs
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
