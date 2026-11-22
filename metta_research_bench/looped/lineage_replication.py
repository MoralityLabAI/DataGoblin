"""Measure the lineage guard on a fresh lineage-only frozen reserve."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from MeTTa.ram_hillclimb import digest

from ..experiment import cluster_interval
from ..taskset import FAMILY_REPLICATION_ROOTS, LINEAGE_REPLICATION_SPLIT
from .ablation_stats import signed_p_value
from .interaction import ARMS, PROTOCOL, factorial_metrics
from .revision_replication import REPLICATION, configure_inference, replication_parser, run_replication

LINEAGE_PROTOCOL = {
    **REPLICATION,
    "split": LINEAGE_REPLICATION_SPLIT,
    "roots": FAMILY_REPLICATION_ROOTS,
    "primary_contrast": "((good - without_lineage) + (without_revision - without_lineage_revision)) / 2",
    "secondary_contrast": PROTOCOL["primary_contrast"],
    "positive_replication_alpha": 0.05,
    "prior_revision_report_sha256": "e85d2f8be56beb454525dc73dcce5c05b6519c228f70e63c86402f03e21a0f37",
}


def lineage_metrics(rows: list[dict], seeds: list[int]) -> dict:
    result = factorial_metrics(rows, seeds)
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row["root"], row["seed"])][row["curator"]] = row["reward"]
    roots = sorted({row["root"] for row in rows})
    rewards = np.array([[[pairs[root, seed][arm] for arm in ARMS] for seed in seeds] for root in roots])
    with_revision = rewards[:, :, 0] - rewards[:, :, 1]
    without_revision = rewards[:, :, 2] - rewards[:, :, 3]
    effects = {
        "mean_over_revision": (with_revision + without_revision) / 2,
        "revision_present": with_revision,
        "revision_absent": without_revision,
    }
    result["lineage_guard_effect"] = {}
    for name, values in effects.items():
        root_means = values.mean(axis=1)
        result["lineage_guard_effect"][name] = {
            "estimate": float(root_means.mean()), "paired_roots": len(roots),
            "root_bootstrap_95": cluster_interval(root_means),
            "p_value": signed_p_value(root_means, LINEAGE_PROTOCOL["sign_flip_draws"]),
            "per_seed": {str(seed): float(values[:, i].mean()) for i, seed in enumerate(seeds)},
        }
    return result


def main() -> int:
    parser = replication_parser(__doc__)
    parser.add_argument("--prior-revision", type=Path, required=True)
    args = parser.parse_args()
    configure_inference()
    raw = (args.prior_revision / "frozen_revision_replication.json").read_bytes()
    if digest(raw) != LINEAGE_PROTOCOL["prior_revision_report_sha256"]:
        raise ValueError("The revision replication report must match the archived digest")
    prior_sources = {row["source_sha256"] for row in json.loads(raw)["revision"]["sources"]}
    report = asyncio.run(run_replication(args, "lineage", LINEAGE_PROTOCOL, lineage_metrics,
                                         extra_excluded=prior_sources))
    print(json.dumps({key: report["lineage"][key] for key in ("lineage_guard_effect", "interaction")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
