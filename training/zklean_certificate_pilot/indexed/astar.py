"""ZKL-10: exact bounded A* with an admissible ordered-height lower bound."""
from __future__ import annotations

import json
from heapq import heappop, heappush

from dag import BudgetExceeded, certificate, choices, exact, verify_dag
from weighted import generate

CAP = 100_000


def solve(x, cap=CAP):
    by_head = choices(x)
    masks = [sum(1 << p for p in set(e.premises)) for e in x.edges]
    heights = [0] * x.n
    for atom in range(x.n):
        heights[atom] = min(1 + max((heights[p] for p in x.edges[i].premises), default=0)
                            for i in by_head[atom])

    def lower_bound(mask):
        # A dependency path below atom cannot include an outstanding higher atom.
        higher = bound = 0
        while mask:
            atom = mask.bit_length() - 1
            bound = max(bound, higher + heights[atom])
            higher += 1
            mask ^= 1 << atom
        return bound

    root = 1 << x.goal
    q = [(lower_bound(root), 0, root)]  # prefer larger g for equal f
    best = {root: 0}
    parent = {}
    expanded = 0
    while q:
        _, neg_g, mask = heappop(q)
        g = -neg_g
        if g != best[mask]:
            continue
        if mask == 0:
            selected = {}
            cur = 0
            while cur != root:
                prev, i = parent[cur]
                selected[x.edges[i].head] = i
                cur = prev
            assert len(selected) == g
            return g, selected, expanded, len(best)
        expanded += 1
        head = mask.bit_length() - 1
        remaining = mask ^ (1 << head)
        for i in by_head[head]:
            nxt = remaining | masks[i]
            if g + 1 >= best.get(nxt, 10**12):
                continue
            if nxt not in best and len(best) >= cap:
                raise BudgetExceeded
            best[nxt] = g + 1
            parent[nxt] = (mask, i)
            heappush(q, (g + 1 + lower_bound(nxt), -(g + 1), nxt))
    raise ValueError('unreachable')


def evaluate():
    report = {}
    for name, start, size, shift in [('small', 40000, 12, False), ('size', 50000, 24, False),
                                     ('grammar', 60000, 24, True)]:
        stats = dict(cases=128, solved=0, capped=0, dp_capped=0, rescued=0, matched=0,
                     total_rows=0, paired_astar_discovered=0, paired_dp_states=0,
                     max_discovered=0, expanded=0, discovered=0)
        for seed in range(start, start + 128):
            x = generate(seed, size, shift)
            try:
                optimum, _, states = exact(x)
            except BudgetExceeded:
                optimum = None
                stats['dp_capped'] += 1
            try:
                n, selected, expanded, discovered = solve(x)
            except BudgetExceeded:
                stats['capped'] += 1
                continue
            assert verify_dag(x, certificate(x, selected))[0] == n
            stats['solved'] += 1
            stats['total_rows'] += n
            stats['expanded'] += expanded
            stats['discovered'] += discovered
            stats['max_discovered'] = max(stats['max_discovered'], discovered)
            if optimum is None:
                stats['rescued'] += 1
            else:
                assert optimum == n
                stats['matched'] += 1
                stats['paired_astar_discovered'] += discovered
                stats['paired_dp_states'] += states
        report[name] = stats
    return report


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
