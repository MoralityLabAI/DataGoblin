"""Paired root estimates for one-rule experiments."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def signed_p_value(values: np.ndarray, draws: int) -> float:
    """Two-sided, root-level Monte Carlo sign-flip test."""
    generator = np.random.default_rng(2042)
    threshold = abs(values.mean())
    exceed = 0
    for start in range(0, draws, 1000):
        size = min(1000, draws - start)
        signs = 2 * generator.integers(0, 2, size=(size, len(values)), dtype=np.int8) - 1
        exceed += int((np.abs((signs * values).mean(axis=1)) >= threshold - 1e-12).sum())
    return (exceed + 1) / (draws + 1)


def holm(values: dict[str, float]) -> dict[str, float]:
    corrected = {}
    maximum = 0.0
    for index, (name, value) in enumerate(sorted(values.items(), key=lambda pair: pair[1])):
        maximum = max(maximum, min(1.0, (len(values) - index) * value))
        corrected[name] = maximum
    return corrected


def contrasts(rows: list[dict], aliases: dict, protocol: dict) -> dict:
    pairs: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        joint = pairs[(row["root"], row["seed"])]
        if row["curator"] in joint:
            raise ValueError("Duplicate matched benchmark observation")
        joint[row["curator"]] = row["reward"]
    summaries = {}
    for name, actual in aliases.items():
        if name == "good":
            continue
        by_seed: dict[int, list] = defaultdict(list)
        by_root: dict[str, list] = defaultdict(list)
        for (root, seed), pair in pairs.items():
            delta = pair["good"] - pair[actual]
            by_seed[seed].append(delta)
            by_root[root].append(delta)
        values = np.asarray([np.mean(items) for items in by_root.values()])
        result = {
            "good_minus_arm": float(values.mean()),
            "paired_roots": len(values),
            "per_seed_gap": {str(seed): float(np.mean(items)) for seed, items in by_seed.items()},
        }
        if actual == "good":
            result.update(identifiable=False, reason="identical_train_selection")
        else:
            rng = np.random.default_rng(2041)
            samples = rng.choice(values, size=(protocol["bootstrap_draws"], len(values)), replace=True).mean(axis=1)
            result.update(
                identifiable=True, root_bootstrap_95=np.quantile(samples, [0.025, 0.975]).tolist(),
                p_value=signed_p_value(values, protocol["sign_flip_draws"]),
            )
        summaries[name] = result
    keys = [f"without_{name}" for name in protocol["original_difference_rules"]]
    distinct = {}
    for name in keys:
        if summaries[name]["identifiable"]:
            distinct[aliases[name]] = summaries[name]["p_value"]
    adjusted = holm(distinct)
    for name in keys:
        if aliases[name] in adjusted:
            summaries[name]["holm_p_value"] = adjusted[aliases[name]]
    return summaries
