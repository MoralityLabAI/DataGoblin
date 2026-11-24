"""Exact binary Brier gains for compression before a future researcher receives a side parity."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from fractions import Fraction as Q
from itertools import combinations, pairwise
from pathlib import Path


def risk(code, u, t):
    groups = defaultdict(list)
    for x, message in enumerate(code):
        side = (u & x).bit_count() % 2
        groups[message, side].append((t & x).bit_count() % 2)
    return sum(Q(sum(bits) * (len(bits) - sum(bits)), len(bits) * len(code)) for bits in groups.values())


def gains(code, dirs):
    blank = [0] * len(code)
    return [sum(risk(blank, u, t) - risk(code, u, t) for u in dirs) / len(dirs) for t in dirs]


def audit():
    dirs = range(1, 8)
    all_codes = [tuple(mask >> x & 1 for x in range(8)) for mask in range(2, 256, 2)]
    # Fix the zero world's bit to zero to quotient by complementation. Omit the constant.
    lines = {}
    for code in all_codes:
        no_side = sum(risk([0] * 8, 0, t) - risk(code, 0, t) for t in dirs) / 7
        assert no_side == Q(1, 28)
        g = gains(code, dirs)
        baseline, peak = sum(g) / 7, g[0]
        joint_counts = [len({(label, (u & x).bit_count() % 2) for x, label in enumerate(code)}) for u in dirs]
        assert baseline == Q(sum(joint_counts) - 2 * len(dirs), 4 * len(dirs) ** 2)
        lines.setdefault((baseline, peak - baseline), []).append(code)
    linear = tuple(x & 1 for x in range(8))
    majority = tuple(int(x.bit_count() >= 2) for x in range(8))
    crossing = {Q(0), Q(1)}
    for (a, b), (c, d) in combinations(lines, 2):
        if b != d:
            q = (c - a) / (b - d)
            if 0 < q < 1:
                crossing.add(q)
    segments = []
    points = sorted(crossing)
    for left, right in pairwise(points):
        middle = (left + right) / 2
        h = max(lines, key=lambda line: line[0] + middle * line[1])
        if segments and segments[-1]["intercept_slope"] == list(map(str, h)):
            segments[-1]["right"] = str(right)
            continue
        segments.append(
            {
                "left": str(left),
                "right": str(right),
                "intercept_slope": list(map(str, h)),
                "optimal_code_example_at_interior": lines[h][0],
                "equivalent_value_codes": len(lines[h]),
            }
        )

    def pair(code):
        values = gains(code, dirs)
        return {"uniform_future": str(sum(values) / 7), "first_bit_future": str(values[0])}

    assert [s["intercept_slope"] for s in segments] == [["1/14", "1/56"], ["3/49", "15/98"]]
    assert segments[0]["right"] == segments[1]["left"] == "4/53"

    return {
        "world": "three uniform hidden bits, one uploaded bit, uniform announced nonzero side parity",
        "metric": "expected reduction in Brier loss over the side-only posterior student",
        "nonconstant_codes_up_to_complement": len(all_codes),
        "no_side_uniform_gain_all_codes": "1/28",
        "linear_store_first_bit": pair(linear),
        "majority": pair(majority),
        "target_mixture": "(1-lambda) Uniform(nonzero) + lambda point_mass(first_bit)",
        "upper_envelope_all_codes": segments,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    body = json.dumps(audit(), indent=2) + "\n"
    if args.out:
        args.out.write_text(body, encoding="utf-8")
    print(body, end="")
