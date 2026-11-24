"""Public exact-arithmetic tasks for composing distributed experiment receipts.

python clock_order_gym.py
python clock_order_gym.py --graph graph.json
A graph file is a list [lab_u, lab_v, clock_flag, reported_sign] per edge.
The sign model below is a declared noiseless finite game. It is separate from
the real-valued clock commutator of a deterministic shared core.
"""

import argparse
import json
from fractions import Fraction as Q
from itertools import product
from pathlib import Path


def relaxed(residual, state, alpha):
    return state + alpha * residual(state)


def edge(residual, state, a, b):
    ab = relaxed(residual, relaxed(residual, state, a), b)
    ba = relaxed(residual, relaxed(residual, state, b), a)
    g = residual(state)
    slope_a = (residual(state + a * g) - g) / a
    slope_b = (residual(state + b * g) - g) / b
    q = (ab - ba) / (a * b)
    assert q == slope_a - slope_b
    return q


def infer(reports):
    """Constraint propagation for each rival sign, up to a free sign per connected component."""
    for row in reports:
        if (
            len(row) != 4
            or not all(isinstance(v, str) and v for v in row[:2])
            or row[0] == row[1]
            or row[2] not in (0, 1)
            or row[3] not in (-1, 1)
        ):
            raise ValueError("expected distinct lab names, a binary clock flag and a unit sign")
    fits = []
    for rival in (-1, 1):
        neighbors = {}
        for u, v, flag, y in reports:
            required = y * rival**flag
            neighbors.setdefault(u, []).append((v, required))
            neighbors.setdefault(v, []).append((u, required))
        orientation = {}
        for root in neighbors:
            if root in orientation:
                continue
            orientation[root] = 1
            todo = [root]
            while todo:
                v = todo.pop()
                for w, s in neighbors[v]:
                    if w not in orientation:
                        orientation[w] = orientation[v] * s
                        todo.append(w)
        if all(orientation[u] * orientation[v] * rival**flag == y for u, v, flag, y in reports):
            fits.append(rival)
    return fits


def census(design):
    labs = sorted({v for u, w, _ in design for v in (u, w)})
    transcripts = {-1: set(), 1: set()}
    for rival, bits in product((-1, 1), product((-1, 1), repeat=len(labs))):
        g = dict(zip(labs, bits, strict=True))
        signs = tuple(g[u] * g[v] * rival**flag for u, v, flag in design)
        transcripts[rival].add(signs)
    for signs in set.union(*transcripts.values()):
        reports = [[*e, y] for e, y in zip(design, signs, strict=True)]
        assert set(infer(reports)) == {r for r in transcripts if signs in transcripts[r]}
    ambiguous = len(set.intersection(*transcripts.values()))
    return {"distinct_transcripts": len(set.union(*transcripts.values())), "ambiguous_transcripts": ambiguous}


def demo():
    a, b, c = Q(1, 3), Q(1, 2), Q(2, 3)
    cases = {}
    for name, residual in (("affine", lambda x: 2 * x + 3), ("quadratic", lambda x: x * x)):
        q1, q2, q3 = [edge(residual, Q(1), u, v) for u, v in ((a, b), (b, c), (a, c))]
        assert q1 + q2 == q3
        cases[name] = {
            "q_ab": str(q1),
            "q_bc": str(q2),
            "predicted_q_ac": str(q1 + q2),
            "measured_q_ac": str(q3),
            "estimated_directional_curvature": str(-2 * q1 / (b - a)),
        }
    tree = [("A", "B", 0), ("B", "C", 0)]
    return {
        "scope": "two exact finite development games",
        "clock_edges": cases,
        "lab_signs": {
            "tree": census(tree),
            "even_triangle": census([*tree, ("C", "A", 0)]),
            "odd_triangle": census([*tree, ("C", "A", 1)]),
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path)
    args = p.parse_args()
    print(
        json.dumps(
            {"compatible_mechanisms": infer(json.loads(args.graph.read_text()))} if args.graph else demo(), indent=2
        )
    )
