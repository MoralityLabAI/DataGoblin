"""ALG-DESCENT-13: stronger admissible bounds, shared verified incumbent."""
import heapq, json, math, random, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
if not (HERE / 'proof_planning.py').exists():
    sys.path.insert(0, str(HERE.parent / 'alg_descent_v11'))
    sys.path.insert(0, str(HERE.parent / 'alg_descent_v12'))
from proof_planning import Graph, Oracle
from anytime import scope, check, completion, BUDGETS
from bounds import Bounder, MODES


def roots(split):
    base, count, ns, fs = ((810000, 12, (10,11,12,13), ('uniform','clustered'))
                           if split == 'iid' else
                           (820000, 24, (16,17,18), ('shared','uniform','clustered')))
    return [Graph(base+i, ns[i%len(ns)], fs[(i+i//len(ns))%len(fs)] if split=='ood'
                  else fs[(i//len(ns)+i)%2]) for i in range(count)]


def search(g, arm):
    start = time.perf_counter_ns()
    bounder = Bounder(g, arm)
    counts = dict(expansions=0, probes=0, bound_calls=0, bound_scans=0,
                  cut_rounds=0, bound_ns=0, ff_calls=0, ff_ns=0, verify_calls=0,
                  verify_steps=0, duplicates=0, pruned=0, reopened=0,
                  peak_open=0, peak_best=0, peak_queue=0)
    bc, fc, best, opened, queue = {}, {}, {g.initial: 0}, {}, []
    trace, snaps = [], []
    serial = 0

    def lower(s):
        if s not in bc:
            t = time.perf_counter_ns()
            cert = bounder.evaluate(s)
            counts['bound_ns'] += time.perf_counter_ns() - t
            bc[s] = cert
            counts['bound_calls'] += 1
            counts['cut_rounds'] += len(cert['cuts'])
            counts['bound_scans'] = bounder.scans
        return bc[s]['lower']

    def ff(s):
        if s not in fc:
            t = time.perf_counter_ns()
            fc[s] = completion(g, s)
            counts['ff_ns'] += time.perf_counter_ns() - t
            counts['ff_calls'] += 1
        return fc[s]

    def verify(p, c):
        counts['verify_calls'] += 1
        counts['verify_steps'] += len(p)
        assert check(g, p) == c

    U, incumbent = ff(g.initial)
    verify(incumbent, U)

    def admit(s, cost, path):
        nonlocal serial
        b = lower(s)
        if cost+b >= U:
            counts['pruned'] += 1
            return
        serial += 1
        opened[s] = (cost, path, b, serial)
        heapq.heappush(queue, (cost+b, b, cost, serial, s))
        counts['peak_open'] = max(counts['peak_open'], len(opened))
        counts['peak_queue'] = max(counts['peak_queue'], len(queue))

    def improve(cost, path):
        nonlocal U, incumbent
        if cost < U:
            verify(path, cost)
            U, incumbent = cost, path
            for s, (gc, _, b, _) in list(opened.items()):
                if gc+b >= U:
                    del opened[s]
                    counts['pruned'] += 1

    admit(g.initial, 0, ())
    counts['peak_best'] = 1
    for cap in BUDGETS:
        while counts['expansions'] < cap and opened:
            while queue:
                _, _, _, k, s = heapq.heappop(queue)
                if s in opened and opened[s][3] == k:
                    break
            else:
                raise AssertionError('lost frontier')
            gc, path, b, _ = opened.pop(s)
            assert gc+b < U
            trace.append(dict(state=s, g=gc, path=path))
            counts['expansions'] += 1
            f, p = ff(s)
            improve(gc+f, path+p)
            counts['probes'] += len(g.rules)
            for t, c, i in g.ready(s):
                new, old = gc+c, best.get(t, math.inf)
                if new >= old:
                    counts['duplicates'] += 1
                    continue
                if old < math.inf and t not in opened:
                    counts['reopened'] += 1
                best[t] = new
                opened.pop(t, None)
                counts['peak_best'] = max(counts['peak_best'], len(best))
                if t & g.goalbit:
                    improve(new, path+(i,))
                else:
                    admit(t, new, path+(i,))
        L = min(U, min((gc+b for gc, _, b, _ in opened.values()), default=U))
        snaps.append(dict(budget=cap, L=L, U=U, proof=incumbent, open=len(opened),
                          certified=L==U, elapsed_ns=time.perf_counter_ns()-start, **counts))
    return dict(scope=scope(g), trace=trace, snapshots=snaps, bounds=bc)


def main():
    rows, refs = [], []
    for split in ('iid','ood'):
        for g in roots(split):
            arms = list(MODES)
            random.Random(g.seed+12).shuffle(arms)
            batch = [dict(split=split, root=g.seed, n=g.n, family=g.family, arm=a,
                          **search(g,a)) for a in arms]
            oracle = Oracle(g)
            opt = oracle.value(g.initial)
            states = oracle.audit()
            refs.append(dict(split=split, root=g.seed, optimum=opt, states=states))
            for row in batch:
                for v in row['snapshots']:
                    assert v['L'] <= opt <= v['U']
                for s, cert in row['bounds'].items():
                    assert cert['lower'] <= oracle.values[s]
                rows.append(dict(optimum=opt, **row))
        print(split, len(rows), flush=True)
    (HERE/'receipt.json').write_text(json.dumps(dict(protocol='d11b750c59596bbb5140291dabd779fd56118b9a',
                                                     runs=rows, reference=refs), sort_keys=True))


if __name__ == '__main__':
    main()
