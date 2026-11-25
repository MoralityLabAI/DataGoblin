"""ZKL-13: admissible mandatory-atom and disjoint-premise lower bounds."""
from __future__ import annotations

import argparse
import json
from functools import cache

from learned_tie import Graph, family_shift, replay
from weighted import generate


class MandatoryGraph(Graph):
    def __init__(self, x, packing=False):
        super().__init__(x)
        self.packing = packing
        self.mandatory = []
        for a, options in enumerate(self.edges):
            candidates = []
            for mask, _ in options:
                need = 0
                while mask:
                    p = mask.bit_length() - 1
                    need |= self.mandatory[p]
                    mask ^= 1 << p
                candidates.append(need)
            common = candidates[0]
            for candidate in candidates[1:]:
                common &= candidate
            self.mandatory.append(common | (1 << a))
        self.lower = cache(self._lower)

    def _lower(self, mask):
        required = 0
        while mask:
            a = mask.bit_length() - 1
            required |= self.mandatory[a]
            mask ^= 1 << a
        h = super().lower(required)
        if not self.packing:
            return h
        demands = []
        atoms = required
        while atoms:
            a = atoms.bit_length() - 1
            atoms ^= 1 << a
            outside = [p & ~required for p, _ in self.edges[a]]
            demand = min(p.bit_count() for p in outside)
            if demand:
                support = 0
                for p in outside:
                    support |= p
                demands.append((support.bit_count(), -demand, support))
        used = extra = 0
        for _, neg_d, support in sorted(demands):
            if not support & used:
                used |= support
                extra -= neg_d
        return max(h, required.bit_count() + extra)


def experiment(n=48):
    report = {}
    splits = [('same', 81000, 12, generate), ('size', 82000, 24, generate),
              ('new_grammar', 83000, 24, family_shift), ('fresh_larger', 84000, 32, family_shift)]
    for name, start, size, gen in splits:
        arms = ('height', 'mandatory', 'packing')
        metrics = {a: dict(solved=0, expanded=0, discovered=0, generated=0, seconds=0., rows=0) for a in arms}
        paired = {a: dict(expanded=0, discovered=0, generated=0, seconds=0.) for a in arms}
        matched = 0
        for seed in range(start, start+n):
            x = gen(seed, size)
            runs = {}
            for a in arms:
                g = Graph(x) if a == 'height' else MandatoryGraph(x, a == 'packing')
                r = g.search('deep')
                if r['cost'] is not None:
                    assert replay(x, r.pop('selection')) == r['cost']
                    metrics[a]['solved'] += 1
                    metrics[a]['rows'] += r['cost']
                for k in ('expanded', 'discovered', 'generated', 'seconds'):
                    metrics[a][k] += r[k]
                runs[a] = r
                if isinstance(g, MandatoryGraph):
                    g.lower.cache_clear()
            assert len({r['cost'] for r in runs.values() if r['cost'] is not None}) <= 1
            if all(r['cost'] is not None for r in runs.values()):
                matched += 1
                for a, r in runs.items():
                    for k in paired[a]:
                        paired[a][k] += r[k]
        report[name] = dict(cases=n, matched=matched, arms=metrics, paired=paired)
        print(name, json.dumps(report[name]), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=int, default=48)
    p.add_argument('--out', default='mandatory_results.json')
    a = p.parse_args()
    with open(a.out, 'w', encoding='utf8') as fp:
        json.dump(experiment(a.cases), fp, indent=2)
