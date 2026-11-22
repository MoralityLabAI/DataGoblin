"""Oracle-guided loop-budget frontier for *training-partition* MeTTa curation.

This reports synthetic edit-depth ceilings. It does not run a trained model,
learn a stopping policy, or read any development/test/OOD answer.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .collect import ROOT, sha256_bytes
from .looped_sft import split_for
from .sft_oracle import FAMILIES, OOD_FAMILY, evaluate_subset, exact_optimum, first_state, make_task, replay

DEPTHS = (0, 1, 2, 4, 8, 16)


def depth_frontier(*, roots_per_family: int = 512) -> dict[str, Any]:
    if not 1 <= roots_per_family <= 512:
        raise ValueError("roots_per_family must match generated corpus horizon")
    metrics: dict[str, Any] = {}
    aggregate = defaultdict(Counter)
    lengths = Counter()
    for family in FAMILIES:
        if family == OOD_FAMILY:
            continue
        roots = 0
        counts = defaultdict(Counter)
        for seed in range(roots_per_family):
            if split_for(family, seed) != "train_candidate":
                continue
            task = make_task(family, seed)
            oracle = exact_optimum(task)
            trace = replay(task, first_state(task, seed), oracle["selected"])
            length = len(trace) - 1
            lengths[length] += 1
            roots += 1
            for depth in DEPTHS:
                subset = trace[min(depth, len(trace) - 1)]["selected"]
                state = evaluate_subset(task, subset)
                correct = subset == oracle["selected"]
                counts[depth].update(
                    roots=1, exact=int(correct), feasible=int(state["valid"]),
                    source_tokens=state["total_tokens"], goals_covered=len(state["covered_goals"]),
                )
                aggregate[depth].update(
                    roots=1, exact=int(correct), feasible=int(state["valid"]),
                )
        metrics[family] = {
            "roots": roots,
            "depths": {str(k): dict(counts[k]) for k in DEPTHS},
        }
    total_roots = aggregate[0]["roots"]
    if not total_roots:
        raise ValueError("no training roots sampled")
    for depth in DEPTHS:
        if aggregate[depth]["roots"] != total_roots:
            raise ValueError("inconsistent sample support across depth schedules")
    if aggregate[DEPTHS[-1]]["exact"] != total_roots:
        raise ValueError("oracle-guided correction did not converge under bound")
    return {
        "schema_version": "metta-oracle-depth-frontier-v1",
        "root_horizon_per_family": roots_per_family,
        "train_roots_only": True,
        "learner_model": None,
        "policy": "exact_teacher_canonical_edits",
        "native_hyperon_executed": False,
        "training_ready": False,
        "depths": list(DEPTHS),
        "aggregate": {str(k): dict(aggregate[k]) for k in DEPTHS},
        "families": metrics,
        "canonical_edit_length_histogram": dict(sorted(lengths.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roots-per-family", type=int, default=512)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "metta" / "looped_sft" / "depth_frontier.json")
    args = parser.parse_args()
    report = depth_frontier(roots_per_family=args.roots_per_family)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_sha = sha256_bytes(args.out.read_bytes())
    print(json.dumps({
        "roots": report["aggregate"]["0"]["roots"],
        "depths": {k: {"exact": v["exact"], "feasible": v["feasible"]} for k, v in report["aggregate"].items()},
        "sha256": report_sha,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
