"""STATE-10: proof transport on a propositional Horn fragment; optional Lean replay."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path

SEED = 2026112516
LINEAGES = 24
BUDGETS = (16, 32, 64, 128, 256)
FAMILIES = ('join', 'nested', 'weaken', 'repair', 'rule_removed', 'fact_changed')
POLICIES = ('cold', 'reuse', 'forward')


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def signature(env, name):
    if name in env['facts']:
        return ((), env['facts'][name])
    return env['rules'].get(name)


def make_env(i):
    depth = (6, 10, 14)[i % 3]
    facts, rules = {}, {}
    for k in 'abc':
        facts['f_' + k] = k + '0'
        for j in range(1, depth + 1):
            end = f'{k}{j}'
            rules[f'r_{k}_{j}'] = ((f'{k}{j-1}',), end)
            for d in range(2):
                rules[f'd_{k}_{j}_{d}'] = ((f'dead_{k}_{j}_{d}',), end)
    return {'facts': facts, 'rules': rules}, [k + str(depth) for k in 'abc'], depth


def task(env, goals, depth, family):
    a, b, c = goals
    facts, rules = dict(env['facts']), dict(env['rules'])
    if family == 'nested':
        rules['j0'] = ((b, c), 'bridge')
        rules['j1'] = ((a, 'bridge'), 'target')
    elif family == 'weaken':
        facts['f_new'] = 'new'
        rules['j0'] = ((a, b, 'new'), 'target')
        rules['irrelevant'] = (('new',), 'unused')
        rules = dict(reversed(list(rules.items())))
    else:
        rules['j0'] = ((a, b), 'target')
    if family in ('repair', 'rule_removed'):
        # Change the signature under the same name, rather than deleting the name.
        j = depth // 2
        rules[f'r_a_{j}'] = (('blocked',), f'a{j}')
        if family == 'repair':
            facts['f_repair'] = 'repair0'
            for k in range(1, 5):
                rules[f'fix_{k}'] = ((f'repair{k-1}',), f'repair{k}')
            rules['fix_5'] = (('repair4',), f'a{j}')
    if family == 'fact_changed':
        facts['f_a'] = 'wrong_fact'
    return {'facts': facts, 'rules': rules}, 'target'


def oracle(env, goal):
    # Independent finite least-fixed-point derivability oracle.
    reached = set(env['facts'].values())
    while True:
        add = {q for p, q in env['rules'].values() if set(p) <= reached}
        if add <= reached:
            return goal in reached
        reached |= add


def verify(env, proof, goal):
    # Independent type checker for explicit Horn proof terms. No search.
    nodes = 0
    def typ(p):
        nonlocal nodes
        nodes += 1
        name, children = p
        sig = signature(env, name)
        if sig is None:
            raise ValueError('unknown local declaration')
        pre, out = sig
        if len(children) != len(pre):
            raise ValueError('arity')
        if tuple(typ(ch) for ch in children) != pre:
            raise ValueError('premise type mismatch')
        return out
    try:
        ok = proof is not None and typ(proof) == goal
    except ValueError:
        ok = False
    return ok, nodes


def support(env, proof):
    found = {}
    def visit(p):
        name, children = p
        found[name] = signature(env, name)
        for c in children:
            visit(c)
    visit(proof)
    return found


def index(env, seed):
    facts = {v: k for k, v in env['facts'].items()}
    by_goal, by_input = defaultdict(list), defaultdict(list)
    for name, (premises, goal) in sorted(env['rules'].items()):
        by_goal[goal].append(name)
        for p in premises:
            by_input[p].append(name)
    rng = random.Random(seed)
    for names in by_goal.values():
        rng.shuffle(names)
    return facts, by_goal, by_input


class Exhausted(Exception):
    pass


def search(env, goal, budget, seed, policy, cache):
    facts, by_goal, by_input = index(env, seed)
    work = {'fact_lookups': 0, 'rule_probes': 0, 'dependency_lookups': 0,
            'cache_hits': 0, 'cache_rejections': 0, 'proof_checks': 0, 'proof_nodes': 0}
    used = 0
    def charge(k):
        nonlocal used
        if used == budget:
            raise Exhausted
        work[k] += 1
        used += 1
    memo = {}
    def solve(g):
        if g in memo:
            return memo[g]
        if policy == 'reuse' and g in cache:
            p, deps = cache[g]
            valid = True
            for name, sig in sorted(deps.items()):
                charge('dependency_lookups')
                if signature(env, name) != sig:
                    valid = False
                    break
            if valid:
                work['cache_hits'] += 1
                memo[g] = p
                return p
            work['cache_rejections'] += 1
        charge('fact_lookups')
        if g in facts:
            memo[g] = (facts[g], ())
            return memo[g]
        for name in by_goal[g]:
            charge('rule_probes')
            pre, _ = env['rules'][name]
            children = []
            for a in pre:
                p = solve(a)
                if p is None:
                    break
                children.append(p)
            if len(children) == len(pre):
                memo[g] = (name, tuple(children))
                return memo[g]
        memo[g] = None
        return None
    def forward():
        proofs, queue, remaining = {}, deque(), {n: len(p) for n, (p, _) in env['rules'].items()}
        for name, a in sorted(env['facts'].items()):
            charge('fact_lookups')
            if a not in proofs:
                proofs[a] = (name, ())
                queue.append(a)
        while queue and goal not in proofs:
            a = queue.popleft()
            for name in by_input[a]:
                charge('rule_probes')
                remaining[name] -= 1
                pre, q = env['rules'][name]
                if remaining[name] == 0 and q not in proofs:
                    proofs[q] = (name, tuple(proofs[v] for v in pre))
                    queue.append(q)
        return proofs.get(goal)
    try:
        p = forward() if policy == 'forward' else solve(goal)
        exhausted = False
    except Exhausted:
        p, exhausted = None, True
    ok = False
    if p is not None:
        ok, nodes = verify(env, p, goal)
        work['proof_checks'] = 1
        work['proof_nodes'] = nodes
        assert ok
    return p, dict(work, used=used, success=ok, exhausted=exhausted)


def term(p):
    name, children = p
    return '(' + ' '.join([name] + [term(c) for c in children]) + ')' if children else name


def lean_decl(name, env, goal, proof=None, rhs=None):
    atoms = set(env['facts'].values()) | {goal}
    for pre, q in env['rules'].values():
        atoms.update((*pre, q))
    def ty(x):
        return 'P_' + x
    args = ['(' + ' '.join(ty(a) for a in sorted(atoms)) + ' : Prop)']
    args += [f'({n} : {ty(a)})' for n, a in sorted(env['facts'].items())]
    args += [f'({n} : ' + ' → '.join(map(ty, (*p, q))) + ')' for n, (p, q) in sorted(env['rules'].items())]
    out_type = ty(goal) if proof is not None else 'True'
    return f'theorem {name}\n  ' + '\n  '.join(args) + f'\n  : {out_type} :=\n  ' + (term(proof) if rhs is None else rhs) + '\n\n'


def run(out):
    t = time.process_time()
    rows, warm, certificates, negatives = [], [], [], []
    positive = ['set_option autoImplicit false\n\n']
    for i in range(LINEAGES):
        env, anchors, depth = make_env(i)
        cache = {}
        for g in anchors:
            p, r = search(env, g, 4096, SEED+i, 'cold', {})
            assert r['success']
            cache[g] = (p, support(env, p))
            warm.append(dict(lineage=i, goal=g, context=digest(env), **r))
            positive.append(lean_decl(f'warm_{i}_{g}', env, g, p))
        for j, family in enumerate(FAMILIES):
            e, g = task(env, anchors, depth, family)
            truth = oracle(e, g)
            assert truth == (family not in ('rule_removed', 'fact_changed'))
            key = f'{i}_{family}'
            a, b, c = [cache[k][0] for k in anchors]
            old = ('j1', (a, ('j0', (b, c)))) if family == 'nested' else ('j0', (a, b, ('f_new', ()))) if family == 'weaken' else ('j0', (a, b))
            old_ok, old_nodes = verify(e, old, g)
            assert old_ok == (family in ('join', 'nested', 'weaken'))
            if not old_ok:
                negatives.append(dict(id=key, context=lean_decl('context_ok', e, g, rhs='True.intro'), invalid=lean_decl('invalid_transport', e, g, old)))
            for policy in POLICIES:
                for budget in BUDGETS:
                    p, r = search(e, g, budget, SEED+i, policy, cache)
                    assert not r['success'] or truth
                    rows.append(dict(id=key, lineage=i, depth=depth, family=family, truth=truth,
                                     context_sha256=digest(e), policy=policy, budget=budget, **r))
                    if policy == 'reuse' and budget == BUDGETS[-1] and truth:
                        assert p is not None
                        positive.append(lean_decl('held_'+key, e, g, p))
                        certificates.append(dict(id=key, goal=g, environment_sha256=digest(e), proof=p,
                                                 proof_sha256=digest(p), support=support(e,p)))
    out.mkdir(parents=True, exist_ok=True)
    (out/'STATE10_POSITIVE.lean').write_text(''.join(positive))
    (out/'STATE10_NEGATIVE.json').write_text(json.dumps(negatives, indent=2))
    summary = []
    for budget in BUDGETS:
        for policy in POLICIES:
            q = [r for r in rows if r['budget']==budget and r['policy']==policy]
            summary.append(dict(budget=budget, policy=policy, positives=sum(r['truth'] for r in q), solved=sum(r['success'] for r in q), negatives=sum(not r['truth'] for r in q), false_accept=sum(r['success'] and not r['truth'] for r in q), work=sum(r['used'] for r in q), fact_lookups=sum(r['fact_lookups'] for r in q), rule_probes=sum(r['rule_probes'] for r in q), dependency_lookups=sum(r['dependency_lookups'] for r in q), cache_hits=sum(r['cache_hits'] for r in q), cache_rejections=sum(r['cache_rejections'] for r in q), final_proof_nodes=sum(r['proof_nodes'] for r in q), exhausted=sum(r['exhausted'] for r in q)))
    receipt = dict(seed=SEED, lineages=LINEAGES, tasks=LINEAGES*len(FAMILIES), budgets=BUDGETS, policies=POLICIES, warm=warm, rows=rows, certificates=certificates, invalid_transports=len(negatives), positive_lean_terms=len(positive)-1, summary=summary, cpu_seconds=time.process_time()-t, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), lean_kernel_verified=False)
    (out/'STATE10_RECEIPTS.json').write_text(json.dumps(receipt, indent=2)+'\n')
    with (out/'STATE10_SUMMARY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    for r in summary:
        print(r['budget'],r['policy'],r['solved'],r['work'],r['cache_hits'],r['cache_rejections'])
    print('warm work',sum(r['used'] for r in warm),'cpu seconds',receipt['cpu_seconds'])
    return receipt


def lean_replay(out, negative_limit):
    lean = shutil.which('lean')
    if lean is None:
        raise RuntimeError('Lean executable unavailable')
    def check(path):
        p = subprocess.run([lean, str(path)], capture_output=True, text=True, timeout=120)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    result = {'version':subprocess.check_output([lean,'--version'],text=True).strip(), 'positive':check(out/'STATE10_POSITIVE.lean'), 'negative':[]}
    assert result['positive']['returncode']==0, result['positive']
    for n in json.loads((out/'STATE10_NEGATIVE.json').read_text())[:negative_limit]:
        path=out/'STATE10_negative_check.lean'
        prefix='set_option autoImplicit false\n\n'+n['context']
        path.write_text(prefix)
        valid=check(path)
        assert valid['returncode']==0,valid
        path.write_text(prefix+n['invalid'])
        invalid=check(path)
        assert invalid['returncode']!=0, n['id']
        assert 'type mismatch' in (invalid['stdout']+invalid['stderr']).lower(),invalid
        result['negative'].append(dict(id=n['id'], context=valid, invalid=invalid))
        path.unlink()
    (out/'STATE10_LEAN_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--lean',action='store_true')
    parser.add_argument('--negative-limit',type=int,default=72)
    args=parser.parse_args()
    run(args.out)
    if args.lean:
        lean_replay(args.out,args.negative_limit)
