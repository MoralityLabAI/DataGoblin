"""Emit original Lean 4 proof terms for externally replaying indexed exercises."""
import argparse

from weighted import dp, generate, search, verify


def nat(t):
    if t[0] == 'z':
        return '0'
    if t[0] == 'v':
        return 'n'
    if t[0] == 's':
        return f'(Nat.succ {nat(t[1])})'
    if t[0] == 'a':
        return f'(zklAdd {nat(t[1])} {nat(t[2])})'
    raise ValueError('unknown Nat term')


def prop(t):
    f, index = t
    return f'{["P", "Q"][f]} {nat(index)}'


def export(x, rows):
    verify(x, rows)
    lines = ['-- Generated original exercise; rules are hypotheses.',
             'def zklAdd : Nat → Nat → Nat',
             '  | 0, y => y',
             '  | n + 1, y => Nat.succ (zklAdd n y)', '',
             'theorem zklExercise (P Q : Nat → Prop)']
    for i, r in enumerate(x.rules):
        signature = ' → '.join(prop(t) for t in (*r.premises, r.head))
        if r.bound:
            signature = '∀ n : Nat, ' + signature
        lines.append(f'    (r{i} : {signature})')
    family = ['P', 'Q'][x.goal % 2]
    lines.append(f'    : {family} {x.goal // 2} :=')
    terms = []
    for rid, n, refs, _ in rows:
        args = ([str(n)] if x.rules[rid].bound else []) + [terms[j] for j in refs]
        terms.append(f'(r{rid} {" ".join(args)})' if args else f'r{rid}')
    lines.append('  ' + terms[-1])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()
    x = generate(args.seed, args.limit)
    rows, _ = search(x, dp(x)[1])
    print(export(x, rows), end='')


if __name__ == '__main__':
    main()
