"""Deterministic public ZKL-16 transport pilot; no weights or sealed roots."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from affine_bridge import environment_id, translate, validate
from dag import tree_rank
from learned_tie import Graph, family_shift
from weighted import generate

RULE_DEPTH = 5
MEMORY_DEPTH = 6
CASES = (
    ('same', 93000, 12, generate),
    ('same', 93001, 12, generate),
    ('shift', 94000, 24, family_shift),
    ('shift', 94001, 24, family_shift),
)


def export():
    cases = []
    for split, seed, limit, gen in CASES:
        x = gen(seed, limit)
        exact = Graph(x).search('deep')
        if exact['cost'] is None:
            raise RuntimeError('reference search capped')
        arms = {'astar': exact['selection'], 'weighted_tree_shared': tree_rank(x, False)}
        root = f'{split}-{seed}'
        for arm, selection in arms.items():
            schemas, rows, _ = translate(x, selection)
            validate(schemas, x.goal, rows, MEMORY_DEPTH)
            assert len(schemas) <= 1 << RULE_DEPTH
            assert exact['cost'] <= len(rows) <= 1 << MEMORY_DEPTH
            cases.append({'root': root, 'arm': arm, 'goal': x.goal,
                          'exact_rows': exact['cost'], 'environment_id': environment_id(schemas, x.goal),
                          'rules': [asdict(s) for s in schemas], 'steps': [asdict(s) for s in rows]})
    return {'schema': 'zkl-affine-batch-v1', 'rule_depth': RULE_DEPTH,
            'memory_depth': MEMORY_DEPTH, 'cases': cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(export(), sort_keys=True, separators=(',', ':')) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
