"""ZKL-09: exact shared-DAG optimization for acyclic indexed proof exercises."""
from __future__ import annotations

import json
from functools import cache
from itertools import pairwise

from weighted import dp, generate, instantiate_rule, key, numeral, one_step, trace, value

CAP = 100_000


class BudgetExceeded(Exception):
    pass


def choices(x):
    by_head = [[] for _ in range(x.n)]
    for i, e in enumerate(x.edges):
        assert all(p < e.head for p in e.premises)
        by_head[e.head].append(i)
    return by_head


def exact(x, use_weight=False, cap=CAP):
    by_head = choices(x)
    masks = [sum(1 << p for p in set(e.premises)) for e in x.edges]
    decisions = {}
    seen = 0

    @cache
    def solve(mask):
        nonlocal seen
        seen += 1
        if seen > cap:
            raise BudgetExceeded
        if not mask:
            return 0
        head = mask.bit_length() - 1
        remaining = mask ^ (1 << head)
        candidates = []
        # Duplicate obligations are charged once, even when parents share them.
        for i in by_head[head]:
            edge = x.edges[i]
            next_mask = remaining | masks[i]
            candidates.append(((edge.weight if use_weight else 1) + solve(next_mask), i, next_mask))
        cost, i, next_mask = min(candidates)
        decisions[mask] = (i, next_mask)
        return cost

    mask = 1 << x.goal
    try:
        cost = solve(mask)
        selected = {}
        while mask:
            i, next_mask = decisions[mask]
            selected[x.edges[i].head] = i
            mask = next_mask
        return cost, selected, seen
    finally:
        solve.cache_clear()


def tree_rank(x, unit):
    by_head = choices(x)
    _, scores = dp(x, unit=unit)
    selected = {}

    def visit(head):
        if head in selected:
            return
        i = min(by_head[head], key=lambda i: (scores[i], i))
        selected[head] = i
        for p in x.edges[i].premises:
            visit(p)

    visit(x.goal)
    return selected


def certificate(x, selected):
    positions = {}
    rows = []
    for head, i in sorted(selected.items()):
        e = x.edges[i]
        types = instantiate_rule(x.rules[e.rule], e.witness)
        rows.append((e.rule, e.witness, tuple(positions[p] for p in e.premises),
                     tuple(trace(t) for _, t in types)))
        positions[head] = len(rows) - 1
    return rows


def verify_dag(x, rows):
    if not rows:
        raise ValueError('empty proof')
    heads = []
    used = set()
    total = 0
    for j, (rid, n, refs, conversions) in enumerate(rows):
        if not 0 <= rid < len(x.rules) or not 0 <= n <= x.limit:
            raise ValueError('rule or binder')
        rule = x.rules[rid]
        if (not rule.bound and n != 0) or len(refs) != len(rule.premises):
            raise ValueError('arity or binder')
        types = instantiate_rule(rule, n)
        if len(conversions) != len(types):
            raise ValueError('conversion count')
        normal = []
        steps = 0
        for (family, term), tr in zip(types, conversions, strict=True):
            v = value(term)
            if not 0 <= v <= x.limit or not tr or tr[0] != term or tr[-1] != numeral(v):
                raise ValueError('conversion boundary')
            if any(not one_step(a, b) for a, b in pairwise(tr)):
                raise ValueError('conversion step')
            steps += len(tr) - 1
            normal.append(key(family, v))
        for ref, typ in zip(refs, normal[:-1], strict=True):
            if not 0 <= ref < j or heads[ref] != typ:
                raise ValueError('reference')
            used.add(ref)
        if normal[-1] in heads:
            raise ValueError('duplicate atom')
        heads.append(normal[-1])
        total += 2 + int(rule.bound) + 3 * len(refs) + steps
    if heads[-1] != x.goal or used != set(range(len(rows) - 1)):
        raise ValueError('goal or unused row')
    return len(rows), total


def evaluate():
    report = {}
    for name, start, size, shift in [('small', 40000, 12, False), ('size', 50000, 24, False),
                                     ('grammar', 60000, 24, True)]:
        stats = {m: {'rows': 0, 'proxy': 0, 'step_optimal': 0} for m in
                 ['unit_tree_shared', 'weighted_tree_shared', 'exact_steps', 'exact_proxy']}
        completed = aborted = max_states = 0
        for seed in range(start, start + 128):
            x = generate(seed, size, shift)
            try:
                minimum, steps, count = exact(x)
                proxy_minimum, proxy, count2 = exact(x, True)
            except BudgetExceeded:
                aborted += 1
                continue
            max_states = max(max_states, count, count2)
            arms = {'unit_tree_shared': tree_rank(x, True), 'weighted_tree_shared': tree_rank(x, False),
                    'exact_steps': steps, 'exact_proxy': proxy}
            for name2, selected in arms.items():
                n, w = verify_dag(x, certificate(x, selected))
                assert n >= minimum and w >= proxy_minimum
                if name2 == 'exact_steps':
                    assert n == minimum
                if name2 == 'exact_proxy':
                    assert w == proxy_minimum
                stats[name2]['rows'] += n
                stats[name2]['proxy'] += w
                stats[name2]['step_optimal'] += int(n == minimum)
            completed += 1
        report[name] = {'completed': completed, 'aborted': aborted, 'max_obligation_states': max_states,
                        'arms': stats}
    return report


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
