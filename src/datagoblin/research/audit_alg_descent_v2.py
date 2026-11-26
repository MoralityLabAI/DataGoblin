"""Independent convolution audit of ALG-DESCENT-02's public action banks."""
from alg_descent_v2 import Root, run

checks = invalid = hidden = 0
for seed in range(40000, 40010):
    root = Root(seed, 16)
    root.oracle((0, 16))
    for st, (v, i, vals) in root.dp.items():
        p, parent = root.p, root.poly(st)
        for j, (spans, cs, cost, f) in enumerate(root.bank(st)):
            q = [1]
            for c in cs:
                q = [sum(q[t] * c[k-t] for t in range(len(q))
                         if 0 <= k-t < len(c)) % p
                     for k in range(len(q)+len(c)-1)]
            valid = tuple(q) == parent
            assert valid == (vals[j] is not None)
            if valid:
                assert vals[j] == cost + sum(root.oracle(s)[0] for s in spans)
            else:
                invalid += 1
                hidden += int(f[6] == f[7] == 1)
            checks += 1
        assert vals[i] == v == min(v for v in vals if v is not None)
    assert run(root, 'oracle')['work'] == root.oracle((0, root.n))[0]
print(dict(candidates=checks, invalid=invalid, passes_both_fingerprints=hidden))
