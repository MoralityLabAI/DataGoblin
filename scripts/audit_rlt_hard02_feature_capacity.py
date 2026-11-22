"""Audit whether the registered Mathlib tactic features distinguish near misses.

This is a structural diagnostic. No kernel labels, gold tactic IDs, or
evaluation outcomes are read. The unmodified DataGoblin grounded ranker
expects a first-command, List/Nat lemma, and candidate-position feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.training_gym import rlt_lean_grounded as grounded
from datagoblin.training_gym import rlt_lean_train_rank as rank
from datagoblin.training_gym.rlt_gym.hard02_mathlib import SPLITS, generate

POSITION_COLUMN = 6
SEED = 2026112203


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def inspect(*, seed: int = SEED, roots_per_split: int = 8) -> dict[str, Any]:
    roots = generate(per_split=roots_per_split, seed=seed, matched_menus=True)
    train = [root for root in roots if root.split == "train"]
    words = tuple(sorted(set().union(*(grounded.words(r.statement) for r in train))))[:128]
    results = []
    for root in roots:
        # The proof state's goal is constant within a root, regardless of
        # which candidate is scored. Native trace_state is held out here.
        goal_pp = f"⊢ {root.statement}"
        x = rank.features(goal_pp, root.menu, [], words)
        if x.ndim != 2 or x.shape[0] != 5 or x.shape[1] <= POSITION_COLUMN:
            raise ValueError("unregistered tactic feature matrix shape")
        # Position is the only permissible row-index dependence in this
        # comparison. All remaining columns should capture action semantics.
        without_position = np.delete(x, POSITION_COLUMN, axis=1)
        distinct = len({tuple(np.round(row, 12)) for row in without_position})
        full_distinct = len({tuple(np.round(row, 12)) for row in x})
        results.append({
            "root_id": root.root_id,
            "split": root.split,
            "feature_dimension": int(x.shape[1]),
            "nonposition_distinct_action_signatures": distinct,
            "full_distinct_action_signatures": full_distinct,
            "candidate_lengths_equal": len({len(t.encode("utf-8")) for t in root.menu}) == 1,
            "candidate_first_commands": sorted({t.split()[0] for t in root.menu}),
            "qualified_lemma_hits": sorted({h for tactic in root.menu for h in rank.LEMMAS.findall(tactic)}),
        })
    summary: dict[str, Any] = {}
    for split in SPLITS:
        group = [r for r in results if r["split"] == split]
        summary[split] = {
            "roots": len(group),
            "position_only_roots": sum(r["nonposition_distinct_action_signatures"] == 1 for r in group),
            "semantically_distinguishable_roots": sum(
                r["nonposition_distinct_action_signatures"] > 1 for r in group
            ),
            "equal_candidate_length_roots": sum(r["candidate_lengths_equal"] for r in group),
            "first_command_forms": sorted({name for r in group for name in r["candidate_first_commands"]}),
            "qualified_lemma_names": sorted({name for r in group for name in r["qualified_lemma_hits"]}),
        }
    return {
        "schema": "datagoblin.rlt_hard02_feature_identifiability.v1",
        "evidence_class": "offline_source_only_feature_capacity_audit",
        "source_seed": seed,
        "feature_source_sha256": _sha(Path(rank.__file__).read_bytes()),
        "generator_source_sha256": _sha(Path(__import__(
            "datagoblin.training_gym.rlt_gym.hard02_mathlib", fromlist=["__file__"]
        ).__file__).read_bytes()),
        "position_column": POSITION_COLUMN,
        "vocabulary_size": len(words),
        "all_32_sources_position_only": all(
            r["nonposition_distinct_action_signatures"] == 1 for r in results
        ),
        "by_split": summary,
        "per_root": results,
        "limitations": [
            "synthetic_goal_pretty_print_for_source_only_audit",
            "labels_never_accessed",
            "native_compiler_results_required_separately",
            "equal_nonposition_features_make_candidate_semantic_ranking_unidentifiable",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-split", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = inspect(seed=args.seed, roots_per_split=args.per_split)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError("feature audit receipt must be create-only")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "source_seed", "vocabulary_size", "all_32_sources_position_only", "by_split",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
