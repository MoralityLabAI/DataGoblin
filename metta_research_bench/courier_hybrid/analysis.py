"""Matched-seed root-clustered inference for courier outcomes."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

PRIMARY = {
    "repair_F_minus_C": {"F": 1, "C": -1},
    "supervision_F_minus_E": {"F": 1, "E": -1},
    "interaction_FE_minus_CB": {"F": 1, "C": -1, "E": -1, "B": 1},
}
EXPLORATORY = {
    "repair_D_minus_A": {"D": 1, "A": -1},
    "repair_E_minus_B": {"E": 1, "B": -1},
    "shortest_C_minus_A": {"C": 1, "A": -1},
    "budget_G_minus_F": {"G": 1, "F": -1},
    "soft_I_minus_F": {"I": 1, "F": -1},
    "early_J_minus_F": {"J": 1, "F": -1},
    "gate_K_minus_F": {"K": 1, "F": -1},
    "obligation_L_minus_F": {"L": 1, "F": -1},
}
ENDPOINTS = ("verified_completion", "moves", "calls", "recurrent_visits")
TOTALS = (
    *ENDPOINTS,
    "unused_calls",
    "invalid_proposals",
    "raw_illegal_top",
    "premature_stops",
    "macro_attempts",
    "macro_commits",
    "completion_macro",
    "stop_repair_completion",
)


def analyze(rows: list[dict], seeds: list[int], draws: int, rng_seed: int) -> dict:
    arms = defaultdict(list)
    roots = defaultdict(dict)
    families = {}
    for row in rows:
        root, arm, seed = row["root"], row["arm"], row["seed"]
        if (arm, seed) in roots[root] or families.get(root, row["family"]) != row["family"]:
            raise ValueError("Duplicate trial or inconsistent family in the paired analysis")
        roots[root][arm, seed] = row
        families[root] = row["family"]
        arms[arm].append(row)
    expected = {(arm, seed) for arm in arms if arm != "H" for seed in seeds} | {("H", None)}
    if not roots or any(set(group) != expected for group in roots.values()):
        raise ValueError("Paired analysis requires every arm and seed on every root")
    outcomes = {}
    for arm, group in sorted(arms.items()):
        totals = {metric: sum(row[metric] for row in group) for metric in TOTALS}
        complete = totals["verified_completion"]
        outcomes[arm] = {
            "attempts": len(group),
            **totals,
            "calls_per_completion": totals["calls"] / complete if complete else None,
            "successful_excess_moves": sum(
                row["moves"] - row["shortest_moves"] for row in group if row["verified_completion"]
            ),
            "failures_by_exit": dict(Counter(row["exit_reason"] for row in group if not row["verified_completion"])),
            "by_family": {
                family: {
                    "completed": sum(r["verified_completion"] for r in group if r["family"] == family),
                    "attempted": sum(r["family"] == family for r in group),
                }
                for family in sorted(set(families.values()))
            },
        }
    contrasts = {}
    for name, weights in (PRIMARY | EXPLORATORY).items():
        if not weights.keys() <= arms.keys():
            if name in PRIMARY:
                raise ValueError(f"Required primary contrast is missing arms: {name}")
            continue
        strata = defaultdict(list)
        seed_deltas = defaultdict(list)
        for root, trials in sorted(roots.items()):
            delta = [
                [sum(coefficient * trials[arm, seed][key] for arm, coefficient in weights.items()) for key in ENDPOINTS]
                for seed in seeds
            ]
            strata[families[root]].append(np.mean(delta, axis=0))
            for seed, effects in zip(seeds, delta, strict=True):
                seed_deltas[seed].append(effects[0])
        rng = np.random.default_rng(rng_seed)
        bootstrap = np.zeros((draws, len(ENDPOINTS)))
        for stratum in strata.values():
            values = np.asarray(stratum)
            index = rng.integers(0, len(values), (draws, len(values)))
            bootstrap += values[index].sum(axis=1)
        intervals = np.quantile(bootstrap / len(roots), [0.025, 0.975], axis=0)
        means = np.mean([effect for group in strata.values() for effect in group], axis=0)
        contrasts[name] = {
            "primary": name in PRIMARY,
            "by_endpoint": {
                key: {"estimate": float(means[i]), "ci95": intervals[:, i].tolist()} for i, key in enumerate(ENDPOINTS)
            },
            "completion_by_seed": {str(seed): float(np.mean(values)) for seed, values in seed_deltas.items()},
        }
    attribution = {}
    for baseline, repaired in (("A", "D"), ("B", "E"), ("C", "F")):
        pairs = [(g[baseline, s], g[repaired, s]) for g in roots.values() for s in seeds]
        attribution[repaired] = {
            "gains": sum(not a["verified_completion"] and b["verified_completion"] for a, b in pairs),
            "regressions": sum(a["verified_completion"] and not b["verified_completion"] for a, b in pairs),
            "gains_with_certified_stop_repair": sum(
                not a["verified_completion"] and b["stop_repair_completion"] for a, b in pairs
            ),
        }
    return {"roots": len(roots), "outcomes": outcomes, "contrasts": contrasts, "attribution": attribution}
