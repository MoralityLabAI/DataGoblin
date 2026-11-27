"""Paired, family-stratified bootstrap for complete research episodes."""

from __future__ import annotations

import numpy as np

BOOTSTRAPS = 10_000
FIELDS = ("primary_score", "capability_score")


def paired_effects(baseline, candidate, *, families, seed=0, replicates=BOOTSTRAPS):
    if type(replicates) is not int or not 1 <= replicates <= 100_000:
        raise ValueError("invalid bootstrap size")
    families = tuple(families)
    if not families or len(set(families)) != len(families):
        raise ValueError("invalid family inventory")
    groups = {family: [] for family in families}
    roots = set()
    for old, new in zip(baseline, candidate, strict=True):
        root = old["root_id"]
        if (root in roots or root != new["root_id"] or old["family"] != new["family"]
                or old["family"] not in groups):
            raise ValueError("duplicate, mismatched or unregistered paired root")
        roots.add(root)
        pair = np.array([[old[f] for f in FIELDS], [new[f] for f in FIELDS]], dtype=float)
        if not np.isfinite(pair).all() or (pair < 0).any() or (pair > 1).any():
            raise ValueError("invalid score")
        groups[old["family"]].append(pair[1] - pair[0])
    if any(not group for group in groups.values()):
        raise ValueError("missing registered family")
    rng = np.random.default_rng(seed)
    boot = np.zeros((replicates, len(FIELDS)))
    means = {}
    for family, group in groups.items():
        x = np.asarray(group)
        means[family] = x.mean(axis=0)
        # Both endpoints share the same sampled roots.
        for start in range(0, replicates, 256):
            stop = min(start + 256, replicates)
            idx = rng.integers(len(x), size=(stop - start, len(x)))
            boot[start:stop] += x[idx].mean(axis=1) / len(groups)
    effect = np.mean(list(means.values()), axis=0)
    return {
        "macro_primary_delta": float(effect[0]),
        "macro_capability_delta": float(effect[1]),
        "family_primary_deltas": {f: float(v[0]) for f, v in means.items()},
        "family_capability_deltas": {f: float(v[1]) for f, v in means.items()},
        "family_denominators": {f: len(v) for f, v in groups.items()},
        "complete_denominator": len(roots),
        "paired_primary_bootstrap_95": np.quantile(boot[:, 0], [.025, .975]).tolist(),
        "paired_capability_bootstrap_95": np.quantile(boot[:, 1], [.025, .975]).tolist(),
        "primary_one_sided_lower_95": float(np.quantile(boot[:, 0], .05)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "interval_status": "descriptive_only" if min(map(len, groups.values())) < 8 else "paired_within_family",
    }


def selective_gate(effect, reference_compute, candidate_compute):
    ref = reference_compute.get("rlt_passes", 0)
    cand = candidate_compute.get("rlt_passes")
    if type(ref) is not int or type(cand) is not int or ref <= 0 or cand < 0:
        raise ValueError("missing or invalid recurrent compute counts")
    savings = 1 - cand / ref
    eligible = effect["interval_status"] == "paired_within_family"
    return {"rlt_pass_savings": savings, "noninferiority_margin": .02,
            "passes": bool(eligible and savings >= .25
                           and effect["macro_primary_delta"] >= -.02
                           and effect["primary_one_sided_lower_95"] > -.02),
            "full_panel_eligible": eligible}
