"""Exact exploratory counterexample for selecting distributed uploads by single-site value."""

import argparse
import itertools
import json
from collections import defaultdict
from fractions import Fraction

SITES = "ABC"
WORLDS = list(itertools.product((0, 1), repeat=3))


def risk(sites, p):
    total = Fraction()
    for weight, target in ((p, lambda w: w[2]), (1 - p, lambda w: w[0] ^ w[1])):
        cells = defaultdict(list)
        for w in WORLDS:
            message = tuple(w[SITES.index(i)] for i in sites)
            cells[message].append(target(w))
        for ys in cells.values():
            yes, n = sum(ys), len(ys)
            total += weight * Fraction(yes * (n - yes), 8 * n)
    return total


def table(p):
    if not 0 <= p <= 1:
        raise ValueError("query probability must be in the unit interval")
    losses = {"".join(s): risk(s, p) for n in range(4) for s in itertools.combinations(SITES, n)}
    for sites, loss in losses.items():
        predicted = Fraction(1, 4) * (1 - p * ("C" in sites) - (1 - p) * ("A" in sites and "B" in sites))
        if predicted != loss:
            raise ValueError("analytic frontier differs from direct enumeration")
    one = min(SITES, key=lambda s: (losses[s], s))
    greedy = min((s for s in losses if len(s) == 2 and one in s), key=lambda s: (losses[s], s))
    best = min((s for s in losses if len(s) == 2), key=lambda s: (losses[s], s))
    return dict(
        query_C_probability=str(p),
        risks={s or "none": str(v) for s, v in losses.items()},
        greedy_two=greedy,
        optimum_two=best,
        greedy_regret=str(losses[greedy] - losses[best]),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--p", type=Fraction, default=Fraction(1, 4))
    print(json.dumps(table(cli.parse_args().p), indent=2))
