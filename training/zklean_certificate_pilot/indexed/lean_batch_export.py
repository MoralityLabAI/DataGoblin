"""Lean 4 replay of the exact shared-DAG certificates in ZKL-16."""
import argparse
import hashlib
import json
from pathlib import Path

from affine_bridge import compile_schema, environment_id, validate
from export_nova_batch import CASES, export
from lean_export import prop

BATCH_SHA256 = '2d305d145514b39111e2eb06570f91c4bf3d5a0c8ca66afbb9a1d366ae50fce3'


def lean_source(x, case):
    from affine_bridge import Step

    schemas = tuple(map(compile_schema, x.rules))
    rows = tuple(Step(r['rule'], r['binder'], tuple(r['refs'])) for r in case['steps'])
    assert environment_id(schemas, x.goal) == case['environment_id']
    atoms = validate(schemas, x.goal, rows, 6)
    lines = ['-- Generated public synthetic example; all rule schemas are hypotheses.',
             'def zklAdd : Nat → Nat → Nat',
             '  | 0, y => y',
             '  | n + 1, y => Nat.succ (zklAdd n y)', '',
             'theorem zklExercise (P Q : Nat → Prop)']
    for i, rule in enumerate(x.rules):
        signature = ' → '.join(prop(t) for t in (*rule.premises, rule.head))
        if rule.bound:
            signature = '∀ n : Nat, ' + signature
        lines.append(f'    (r{i} : {signature})')
    family = ['P', 'Q'][x.goal % 2]
    lines.append(f'    : {family} {x.goal // 2} :=')
    for i, (step, atom) in enumerate(zip(rows, atoms, strict=True)):
        family = ['P', 'Q'][atom % 2]
        args = ([str(step.binder)] if schemas[step.rule].bound else [])
        args.extend(f'h{j}' for j in step.refs)
        term = ' '.join([f'r{step.rule}', *args])
        lines.append(f'  let h{i} : {family} {atom // 2} := {term}')
    lines.append(f'  h{len(rows) - 1}')
    lines.append('#print axioms zklExercise')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    batch = export()
    raw = (json.dumps(batch, sort_keys=True, separators=(',', ':')) + '\n').encode()
    assert hashlib.sha256(raw).hexdigest() == BATCH_SHA256
    environments = {f'{split}-{seed}': gen(seed, limit) for split, seed, limit, gen in CASES}
    for i, case in enumerate(batch['cases']):
        path = args.out / f'Proof{i}.lean'
        path.write_text(lean_source(environments[case['root']], case), encoding='utf8')


if __name__ == '__main__':
    main()
