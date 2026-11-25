"""Original finite Horn proof-tree gym. No imported corpus or code."""
import argparse
import json
import math
import random
from dataclasses import dataclass

import torch
from torch import nn

INF = 10**9


@dataclass(frozen=True)
class Instance:
    n: int
    rules: tuple  # (head, tuple of antecedents)
    goal: int


def generate(seed, n):
    rng = random.Random(seed)
    # Independent atoms: shuffle IDs to remove a topological-index shortcut.
    order = list(range(n))
    rng.shuffle(order)
    k = rng.randint(3, n - 2)
    rules = [(order[0], ()), (order[1], ())]
    for j in range(2, k + 1):
        pre = (order[j - 1],)
        if rng.random() < .45:
            pre += (order[rng.randrange(j - 1)],)
        rules.append((order[j], pre))
    for _ in range(3 * n - len(rules)):
        h = rng.randrange(n)
        arity = rng.choices([0, 1, 2], [1, 5, 5])[0]
        rules.append((h, tuple(rng.randrange(n) for _ in range(arity))))
    rng.shuffle(rules)
    return Instance(n, tuple(rules), order[k])


def optimal(x, rounds=None):
    """Minimum unfolded Horn proof-tree row count; positive row cost."""
    costs = [INF] * x.n
    for _ in range(x.n if rounds is None else rounds):
        nxt = costs.copy()
        for h, pre in x.rules:
            nxt[h] = min(nxt[h], 1 + sum(costs[a] for a in pre))
        costs = nxt
    scores = [min(INF, 1 + sum(costs[a] for a in pre)) for _, pre in x.rules]
    return costs, scores


def verify(x, rows):
    """Check local references, rule membership, final public goal, and tree shape."""
    if not rows:
        return False
    heads = []
    used = [0] * len(rows)
    for i, (r, refs) in enumerate(rows):
        if not 0 <= r < len(x.rules):
            return False
        h, pre = x.rules[r]
        if len(refs) != len(pre):
            return False
        for a, j in zip(pre, refs, strict=True):
            if not 0 <= j < i or heads[j] != a:
                return False
            used[j] += 1
        heads.append(h)
    return heads[-1] == x.goal and all(v == 1 for v in used[:-1]) and used[-1] == 0


def search(x, scores, budget=128):
    """Depth-first ranked proof construction with branch cycle avoidance."""
    candidates = [[] for _ in range(x.n)]
    for i, (h, _) in enumerate(x.rules):
        candidates[h].append(i)
    for cand in candidates:
        cand.sort(key=lambda r: (scores[r], r))
    remaining = budget
    attempts = 0
    rows = []

    def visit(a, active):
        nonlocal remaining, attempts
        if a in active:
            return None
        for r in candidates[a]:
            if remaining <= 0:
                return None
            remaining -= 1
            attempts += 1
            start = len(rows)
            refs = []
            for p in x.rules[r][1]:
                j = visit(p, active | {a})
                if j is None:
                    break
                refs.append(j)
            else:
                rows.append((r, tuple(refs)))
                return len(rows) - 1
            del rows[start:]
        return None

    root = visit(x.goal, set())
    if root is None:
        return None, attempts
    assert verify(x, rows)
    return rows, attempts


def batch(xs):
    n = xs[0].n
    h = torch.tensor([[r[0] for r in x.rules] for x in xs])
    p = torch.tensor([[list(r[1]) + [n] * (2 - len(r[1])) for r in x.rules] for x in xs])
    arity = (p != n).sum(-1)
    labels = []
    for x in xs:
        _cs, rs = optimal(x)
        labels.append([math.log1p(min(v, 128)) for v in rs])
    return h, p, arity, torch.tensor(labels, dtype=torch.float32), n


class RecurrentRanker(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.rule = nn.Sequential(nn.Linear(2 * width + 3, width), nn.Tanh(), nn.Linear(width, 1))
        self.update = nn.GRUCell(1, width)
        self.width = width

    def forward(self, h, p, arity, n, steps):
        b, m = h.shape
        s = torch.zeros(b, n + 1, self.width)
        idx = torch.arange(b)[:, None, None]
        ar = torch.nn.functional.one_hot(arity, 3).float()
        for t in range(steps):
            z = s[idx, p].reshape(b, m, 2 * self.width)
            score = self.rule(torch.cat([z, ar], -1)).squeeze(-1)
            if t + 1 == steps:
                return score
            best = torch.full((b, n), 10.0).scatter_reduce(1, h, score, reduce='amin', include_self=True)
            st = self.update(best.reshape(-1, 1), s[:, :n].reshape(-1, self.width)).reshape(b, n, self.width)
            s = torch.cat([st, torch.zeros(b, 1, self.width)], 1)


def measure(xs, score_fn):
    out = {'solved': 0, 'optimal': 0, 'attempts': 0, 'extra_rows': 0, 'n': len(xs)}
    for x, scores in zip(xs, score_fn(xs), strict=True):
        rows, attempts = search(x, scores)
        out['attempts'] += attempts
        if rows is not None:
            cost = optimal(x)[0][x.goal]
            out['solved'] += 1
            out['optimal'] += len(rows) == cost
            out['extra_rows'] += len(rows) - cost
    return out


def scores_heur(xs):
    return [[len(pre) for _, pre in x.rules] for x in xs]


def scores_dp(xs, rounds=None):
    return [optimal(x, rounds)[1] for x in xs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=18)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261125)
    train = [generate(i, 12) for i in range(640)]
    test = [generate(i, 12) for i in range(10000, 10256)]
    shift = [generate(i, 24) for i in range(20000, 20256)]
    # No checkpoint or example data is persisted. Evaluation seeds are public pilot seeds.
    model = RecurrentRanker()
    opt = torch.optim.Adam(model.parameters(), lr=.008)
    rng = random.Random(17)
    for _ in range(args.epochs):
        rng.shuffle(train)
        for start in range(0, len(train), 32):
            h, p, ar, y, n = batch(train[start:start + 32])
            pred = model(h, p, ar, n, 6)
            loss = torch.nn.functional.smooth_l1_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    def ml(xs, steps):
        h, p, ar, _, n = batch(xs)
        with torch.no_grad():
            return model(h, p, ar, n, steps).tolist()
    results = {
        'parameters': sum(p.numel() for p in model.parameters()),
        'train_roots': len(train), 'epochs': args.epochs,
        'search_attempt_budget': 128, 'splits': {},
    }
    for name, xs in [('same_size', test), ('size_shift', shift)]:
        results['splits'][name] = {k: measure(xs, f) for k, f in [
            ('arity', scores_heur), ('dp_2', lambda x: scores_dp(x, 2)), ('dp_5', lambda x: scores_dp(x, 5)),
            ('recurrent_3', lambda x: ml(x, 3)), ('recurrent_6', lambda x: ml(x, 6)),
            ('recurrent_12', lambda x: ml(x, 12)), ('exact_dp', scores_dp)]}
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
