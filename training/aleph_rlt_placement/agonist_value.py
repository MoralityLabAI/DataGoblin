"""Exact finite test of competing acquisition rules for verified synthetic data.

Run from any directory with standard-library Python. Masks are linear probes of
three hidden bits. An independent posterior enumerator also handles noisy labels.
"""

from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction as Q
from pathlib import Path

# The third bit is a nuisance. The large synthetic proposer gives the Decoy
# challenge a confidently wrong answer; independent verification returns zero.
BANK = {"decoy": 0, "nuisance": 4, "left": 1, "right": 2}
TARGET = 3


def parity(world, mask):
    return (world & mask).bit_count() % 2


def span(masks):
    points = {0}
    for mask in masks:
        points |= {point ^ mask for point in points}
    return points


def posterior_risk(masks, target, error=Q(0), student="matched"):
    """Integrate held-target Brier loss over all eight worlds and transcripts."""
    risk = Q(0)
    for observed in itertools.product((0, 1), repeat=len(masks)):
        masses = [Q(1, 8)] * 8
        compatible = []
        for world in range(8):
            agrees = [parity(world, m) == y for m, y in zip(masks, observed, strict=True)]
            masses[world] *= (1 - error) ** sum(agrees) * error ** (len(masks) - sum(agrees))
            if all(agrees):
                compatible.append(world)
        mass = sum(masses)
        if not mass:
            continue
        positive = sum(w for h, w in enumerate(masses) if parity(h, target))
        if student == "matched":
            forecast = positive / mass
        elif student == "literal":
            forecast = Q(1, 2)  # fresh target has a separate key; fixed prior
        elif student == "trust":
            forecast = Q(sum(parity(h, target) for h in compatible), len(compatible)) if compatible else Q(1, 2)
        else:
            raise ValueError(f"unknown student {student}")
        risk += positive * (1 - forecast) ** 2 + (mass - positive) * forecast**2
    return risk


def report():
    pairs = list(itertools.combinations(BANK, 2))

    def masks(pair):
        return tuple(BANK[name] for name in pair)

    def disagreement(pair):
        # Hard teacher prediction is one on the decoy and zero on other probes.
        return sum(Q(1) if name == "decoy" else Q(1, 2) for name in pair)

    def posterior_value(pair):
        return Q(1, 4) - posterior_risk(masks(pair), TARGET)

    def uniform_best(score):
        high = max(map(score, pairs))
        return [pair for pair in pairs if score(pair) == high]

    policies = {
        "uniform_budget_two": pairs,
        "teacher_error_count": uniform_best(disagreement),
        "total_version_space_reduction": uniform_best(lambda pair: len(span(masks(pair)))),
        "target_span_lookahead": uniform_best(lambda pair: TARGET in span(masks(pair))),
    }
    by_policy = {
        name: {
            "support": support,
            "expected_teacher_errors": str(sum(map(disagreement, support)) / len(support)),
            "verified_student_brier_gain": str(sum(map(posterior_value, support)) / len(support)),
        }
        for name, support in policies.items()
    }
    # Independent comparison of risk enumeration with the span theorem across
    # every distinct batch of <= 3 nonzero probes and every nonzero target.
    checked = 0
    for length in range(4):
        for probes in itertools.combinations(range(1, 8), length):
            for target in range(1, 8):
                exact = posterior_risk(probes, target)
                assert exact == (0 if target in span(probes) else Q(1, 4))
                checked += 1
    stress = {
        str(error): {
            student: str(Q(1, 4) - posterior_risk((1, 2), TARGET, error, student))
            for student in ("matched", "trust", "literal")
        }
        for error in (Q(0), Q(1, 10), Q(1, 4), Q(1, 2))
    }
    assert posterior_value(("left", "right")) == Q(1, 4)
    assert all(posterior_risk((m,), TARGET) == Q(1, 4) for m in BANK.values())
    assert by_policy["teacher_error_count"]["verified_student_brier_gain"] == "0"
    assert stress["1/4"]["trust"] == "-1/8"
    return {
        "scope": "exact synthetic decision problem; uniform hidden world and uniform ties",
        "target_mask": TARGET,
        "probes": BANK,
        "batch_size": 2,
        "exhaustive_span_cases": checked,
        "policies": by_policy,
        "pair_brier_gain_by_label_error_and_student": stress,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    body = json.dumps(report(), indent=2) + "\n"
    if args.out:
        args.out.write_text(body, encoding="utf-8")
    print(body, end="")
