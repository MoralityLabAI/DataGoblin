"""Original affine-rule transport and native reference (not a Nova proof)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from learned_tie import Graph, family_shift, replay
from weighted import generate

ATOM_MAX = 65535
BINDER_MAX = 32767


def uint(x, maximum):
    if type(x) is not int or not 0 <= x <= maximum:
        raise ValueError('unsigned field out of range')
    return x


def affine(t):
    """Return (coefficient of n, constant) by structural recursion."""
    match t:
        case ('z',):
            return 0, 0
        case ('v',):
            return 1, 0
        case ('s', x):
            a, b = affine(x)
            return a, b + 1
        case ('a', x, y):
            a, b = affine(x)
            c, d = affine(y)
            return a + c, b + d
        case _:
            raise ValueError('unsupported index syntax')


@dataclass(frozen=True)
class Schema:
    bound: bool
    head: int
    premises: tuple[int, ...]


@dataclass(frozen=True)
class Step:
    rule: int
    binder: int
    refs: tuple[int, ...]


def compile_schema(rule):
    if type(rule.bound) is not bool or len(rule.premises) > 2:
        raise ValueError('unsupported rule')

    def pack(prop):
        family, term = prop
        a, b = affine(term)
        if a != int(rule.bound):
            raise ValueError('outside affine fragment')
        return uint(2 * b + uint(family, 1), ATOM_MAX)

    return Schema(rule.bound, pack(rule.head), tuple(map(pack, rule.premises)))


def environment_id(schemas, goal):
    """Native transport digest only; Nova uses its own Poseidon commitments."""
    payload = {'grammar': 'zkl-affine-v1', 'goal': goal, 'schemas': [asdict(r) for r in schemas]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate(schemas, goal, steps, memory_depth):
    """Independent bounded-integer replay; accepts only a closed derivation of goal."""
    uint(goal, ATOM_MAX)
    uint(memory_depth, 12)
    if memory_depth == 0 or not steps or len(steps) > 1 << memory_depth or not schemas:
        raise ValueError('capacity')
    for r in schemas:
        if type(r.bound) is not bool or len(r.premises) > 2:
            raise ValueError('schema')
        for a in (r.head, *r.premises):
            uint(a, ATOM_MAX)
    atoms = []
    for s in steps:
        r = schemas[uint(s.rule, len(schemas) - 1)]
        n = uint(s.binder, BINDER_MAX)
        if (not r.bound and n != 0) or len(s.refs) != len(r.premises):
            raise ValueError('arity or ground witness')
        for ref, base in zip(s.refs, r.premises, strict=True):
            if not atoms or atoms[uint(ref, len(atoms) - 1)] != uint(base + 2 * n, ATOM_MAX):
                raise ValueError('premise type mismatch')
        atoms.append(uint(r.head + 2 * n, ATOM_MAX))
    if atoms[-1] != goal:
        raise ValueError('wrong goal')
    return tuple(atoms)


def translate(x, selection):
    """Compile a source-verified shared DAG into the bounded affine transport."""
    replay(x, selection)
    schemas = tuple(map(compile_schema, x.rules))
    positions = {}
    steps = []
    expected = []
    for atom, idx in sorted(selection.items()):
        edge = x.edges[idx]
        steps.append(Step(edge.rule, edge.witness, tuple(positions[p] for p in edge.premises)))
        positions[atom] = len(steps) - 1
        expected.append(atom)
    depth = max(1, (len(steps) - 1).bit_length())
    actual = validate(schemas, x.goal, steps, depth)
    if actual != tuple(expected):
        raise ValueError('source/transport mismatch')
    return schemas, tuple(steps), depth


def experiment():
    results = {}
    for name, start, size, gen in [('small', 90000, 12, generate), ('shift', 91000, 24, family_shift)]:
        counts = dict(cases=128, solved=0, capped=0, steps=0, max_steps=0, schema_rows=0, max_schema_rows=0)
        for seed in range(start, start + 128):
            x = gen(seed, size)
            result = Graph(x).search('deep')
            if result['cost'] is None:
                counts['capped'] += 1
                continue
            schemas, steps, _ = translate(x, result['selection'])
            assert len(steps) == result['cost']
            counts['solved'] += 1
            counts['steps'] += len(steps)
            counts['max_steps'] = max(counts['max_steps'], len(steps))
            counts['schema_rows'] += len(schemas)
            counts['max_schema_rows'] = max(counts['max_schema_rows'], len(schemas))
        results[name] = counts
    return results


if __name__ == '__main__':
    print(json.dumps(experiment(), indent=2))
