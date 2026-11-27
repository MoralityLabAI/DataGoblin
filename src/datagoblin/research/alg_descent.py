"""ALG-DESCENT-01: exact, toy polynomial certificate routing (stdlib only)."""
from functools import lru_cache
import json
import random

PRIMES = (2, 3, 5, 7, 11)
ARITIES = (2, 3, 4, 8)


def canonical(poly, p):
    return (isinstance(poly, tuple) and len(poly) > 0 and poly[-1] == 1
            and all(type(c) is int and 0 <= c < p for c in poly))


def mul(a, b, p):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] = (out[i+j] + x*y) % p
    return tuple(out), len(a)*len(b)


def product(polys, p):
    acc, cost = (1,), 0
    for poly in polys:
        acc, w = mul(acc, poly, p)
        cost += w
    return acc, cost


def verify(parent, children, p):
    """Return (valid, coefficient multiply-adds charged to verifier)."""
    if p not in PRIMES or not canonical(parent, p) or len(parent) < 3:
        return False, 0
    if not 2 <= len(children) <= max(ARITIES):
        return False, 0
    if any(not canonical(c, p) or not 2 <= len(c) < len(parent) for c in children):
        return False, 0
    if sum(len(c)-1 for c in children) != len(parent)-1:
        return False, 0
    got, cost = product(children, p)
    return got == parent, cost


def partition(n, k):
    q, r = divmod(n, k)
    return (q+1,)*r + (q,)*(k-r)


def action_cost(sizes):
    degree, cost = 0, 0
    for d in sizes:
        cost += (degree+1)*(d+1)
        degree += d
    return cost


@lru_cache(maxsize=None)
def optimal(n):
    if n <= 1:
        return 0, 0
    return min((action_cost(s) + sum(optimal(d)[0] for d in s), k)
               for k in ARITIES if k <= n for s in (partition(n, k),))


def rollout(factors, p, policy):
    """All policies receive the same public ordered linear-factor witness.

    The task is to route certificates, not discover factors. Meter only
    successful product verification; proposal construction is outside this meter.
    """
    root, _ = product(factors, p)
    work = [(root, factors, 0)]
    cost = nodes = depth = coeffs = 0
    while work:
        parent, fs, level = work.pop()
        depth = max(depth, level)
        n = len(fs)
        if n == 1:
            assert parent == fs[0] and canonical(parent, p) and len(parent) == 2
            continue
        ks = [k for k in ARITIES if k <= n]
        if policy == 'binary':
            k = 2
        elif policy == 'wide':
            k = max(ks)
        elif policy == 'optimal':
            k = optimal(n)[1]
        else:
            raise ValueError(policy)
        sizes = partition(n, k)
        chunks, offset = [], 0
        for d in sizes:
            chunks.append(fs[offset:offset+d])
            offset += d
        children = tuple(product(c, p)[0] for c in chunks)
        ok, paid = verify(parent, children, p)
        assert ok and paid == action_cost(sizes)
        cost += paid
        nodes += 1
        coeffs += sum(map(len, children))
        work.extend((c, f, level+1) for c, f in zip(children, chunks))
    return dict(cost=cost, internal_nodes=nodes, depth=depth, certificate_coeffs=coeffs)


def run():
    rows = []
    for n in (8, 16, 32, 64, 128):
        for p in PRIMES:
            for seed in range(4):
                rng = random.Random(10000*n + 100*p + seed)
                fs = tuple((rng.randrange(p), 1) for _ in range(n))
                results = {a: rollout(fs, p, a) for a in ('binary', 'wide', 'optimal')}
                assert results['optimal']['cost'] == optimal(n)[0]
                assert results['optimal']['cost'] <= min(results[a]['cost'] for a in ('binary', 'wide'))
                rows.append(dict(n=n, p=p, seed=seed, results=results))
    # Exact, independent negative fixtures; preserve degree and monicity where possible.
    assert verify((0, 1, 1), ((0, 1), (1, 1)), 2) == (True, 6)
    assert verify((0, 1, 1), ((1, 1), (1, 1)), 2) == (False, 6)
    assert verify((0, 1, 1), ((0, 1),), 2) == (False, 0)
    assert verify((0, 1, 1), ((0, 1), (2, 1)), 2) == (False, 0)
    assert verify((0, 1, 1), ((0, 1), (1, 1)), 4) == (False, 0)
    return rows


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
