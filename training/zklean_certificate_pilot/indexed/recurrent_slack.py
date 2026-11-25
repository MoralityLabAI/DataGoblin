"""ZKL-14: recurrent A* tie-breaker learning residual slack on reachable states."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from heapq import heappop, heappush

import numpy as np
from learned_tie import CAP, BudgetExceeded, Graph, family_shift, replay
from weighted import generate


class SlackGraph(Graph):
    def features(self, mask):
        g = super().features(mask)
        sequence = []
        remaining = mask
        while remaining:
            a = remaining.bit_length() - 1
            candidates = self.edges[a]
            sequence.append([self.height[a], math.log1p(self.tree[a]), a / self.x.n,
                             math.log1p(len(candidates)),
                             sum(p.bit_count() for p, _ in candidates) / len(candidates),
                             sum((p & mask).bit_count() for p, _ in candidates) / len(candidates)])
            remaining ^= 1 << a
        return g, sequence

    def reachable_masks(self, cap=1000):
        root = 1 << self.x.goal
        q = [(self.lower(root), 0, root)]
        best = {root: 0}
        while q and len(best) < cap:
            _, ng, m = heappop(q)
            if best[m] != -ng:
                continue
            if not m:
                break
            a = m.bit_length() - 1
            for p, _ in self.edges[a]:
                nxt = (m ^ (1 << a)) | p
                if -ng + 1 < best.get(nxt, 10**12):
                    best[nxt] = -ng + 1
                    heappush(q, (-ng + 1 + self.lower(nxt), ng - 1, nxt))
        return sorted(best)


def dataset():
    rng = random.Random(14917)
    data = []
    capped = 0
    for seed in range(71000, 71120):
        g = SlackGraph(generate(seed, 12))
        masks = g.reachable_masks()
        masks = rng.sample(masks, min(32, len(masks)))
        try:
            data.extend((feats, v-feats[0][1]) for feats, v in g.labels(masks))
        except BudgetExceeded:
            capped += 1
    return data, capped


def fit(data, seed, recurrent):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    g, seq, y = [], [], []
    for (glob, s), target in data:
        g.append(glob)
        seq.append(s)
        y.append(target)
    glob = np.asarray(g, dtype=np.float32)
    mu = glob.mean(0)
    scale = np.maximum(glob.std(0), .01)
    seq_all = np.asarray([t for s in seq for t in s], dtype=np.float32)
    smu = seq_all.mean(0)
    sscale = np.maximum(seq_all.std(0), .01)
    longest = max(map(len, seq))
    sdata = np.zeros((len(seq), longest, 6), dtype=np.float32)
    valid = np.zeros((len(seq), longest), dtype=bool)
    for i, s in enumerate(seq):
        if s:
            sdata[i, :len(s)] = (np.asarray(s, dtype=np.float32)-smu)/sscale
            valid[i, :len(s)] = True
    G = torch.tensor((glob-mu)/scale)
    S = torch.tensor(sdata)
    M = torch.tensor(valid)
    Y = torch.tensor(y, dtype=torch.float32)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            if recurrent:
                self.cell = nn.RNNCell(6, 8)
                self.out = nn.Linear(18, 1)
            else:
                self.mlp = nn.Sequential(nn.Linear(10, 16), nn.Tanh(), nn.Linear(16, 1))

        def forward(self, gg, ss, mm):
            if recurrent:
                h = torch.zeros((len(gg), 8))
                for i in range(ss.shape[1]):
                    h = torch.where(mm[:, i, None], self.cell(ss[:, i], h), h)
                raw = self.out(torch.cat((gg, h), 1))
            else:
                raw = self.mlp(gg)
            return nn.functional.softplus(raw).squeeze(1)

    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=.005)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(32):
        for ix in torch.randperm(len(Y), generator=rng).split(128):
            pred = model(G[ix], S[ix], M[ix])
            loss = nn.functional.smooth_l1_loss(pred, Y[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        mae = float((model(G, S, M)-Y).abs().mean())
    params = [p.detach().numpy().copy() for p in model.parameters()]

    def estimate(feats):
        gg, ss = feats
        gg = (np.asarray(gg, dtype=np.float32)-mu)/scale
        if recurrent:
            wi, wh, bi, bh, wo, bo = params
            hh = np.zeros(8, dtype=np.float32)
            for s in ss:
                hh = np.tanh(wi @ ((np.asarray(s, dtype=np.float32)-smu)/sscale)+wh @ hh+bi+bh)
            raw = float((wo @ np.concatenate((gg, hh))+bo)[0])
        else:
            wi, bi, wo, bo = params
            raw = float((wo @ np.tanh(wi @ gg+bi)+bo)[0])
        # Slack is nonnegative; h is admissible and still the primary priority.
        return float(feats[0][1] + np.logaddexp(0, raw))

    return estimate, mae, sum(p.numel() for p in model.parameters())


def run(n=48):
    start = time.perf_counter()
    data, capped = dataset()
    models = {}
    training = {}
    for recurrent in (False, True):
        for seed in (211, 223, 227):
            name = ('rnn' if recurrent else 'mlp')+str(seed)
            models[name], mae, params = fit(data, seed, recurrent)
            training[name] = dict(mae=mae, parameters=params)
    res = dict(records=len(data), roots=120, capped=capped, train_seconds=time.perf_counter()-start,
               training=training, cap=CAP, splits={})
    for name, seed, size, gen in [('same', 85000, 12, generate), ('size', 86000, 24, generate),
                                  ('new_grammar', 87000, 32, family_shift)]:
        names = ['deep', 'tree', *models]
        sums = {a: dict(solved=0, expanded=0, discovered=0, generated=0, seconds=0., rows=0) for a in names}
        common = {a: dict(expanded=0, discovered=0, generated=0, seconds=0.) for a in names}
        matched = 0
        for i in range(seed, seed+n):
            x = gen(i, size)
            base = Graph(x)
            learned = SlackGraph(x)
            runs = {}
            for a in names:
                r = base.search(a) if a in ('deep', 'tree') else learned.search('learned', models[a])
                if r['cost'] is not None:
                    assert replay(x, r.pop('selection')) == r['cost']
                    sums[a]['solved'] += 1
                    sums[a]['rows'] += r['cost']
                for k in ('expanded', 'discovered', 'generated', 'seconds'):
                    sums[a][k] += r[k]
                runs[a] = r
            assert len({r['cost'] for r in runs.values() if r['cost'] is not None}) <= 1
            if all(r['cost'] is not None for r in runs.values()):
                matched += 1
                for a, r in runs.items():
                    for k in common[a]:
                        common[a][k] += r[k]
        res['splits'][name] = dict(cases=n, matched=matched, arms=sums, paired=common)
        print(name, json.dumps(res['splits'][name]), flush=True)
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=int, default=48)
    p.add_argument('--out', default='slack_results.json')
    a = p.parse_args()
    with open(a.out, 'w', encoding='utf8') as fp:
        json.dump(run(a.cases), fp, indent=2)
