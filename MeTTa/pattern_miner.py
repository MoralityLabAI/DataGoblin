"""CPU motif discovery over split-safe synthetic MeTTa curation roots.

This is a Pattern Miner / MOSES-inspired bounded *research control*: it mines
one- and two-fact conjunctions from observable curation structure and scores
their association with independently checked invalid initial selections.
It does not execute Hyperon, use external source material, approve training
data, learn a model, or add presentations to the recurrent SFT corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .collect import ROOT, sha256_bytes
from .looped_sft import root_id, split_for
from .sft_oracle import FAMILIES, OOD_FAMILY, evaluate_subset, first_state, make_task

REPORT_SCHEMA = "metta-train-root-motifs-v1"
DEFAULT_OUT = ROOT / "runs" / "metta" / "pattern_miner"


@dataclass(frozen=True)
class Observation:
    root: str
    family: str
    split: str
    features: tuple[str, ...]
    failures: tuple[str, ...]

    @property
    def invalid(self) -> bool:
        return bool(self.failures)


def structural_features(task: dict[str, Any], selected: list[str]) -> tuple[str, ...]:
    """Build features from public task facts and the observable initial set."""
    items = task["items"]
    by_id = {item["id"]: item for item in items}
    if len(selected) != len(set(selected)) or not set(selected) <= set(by_id):
        raise ValueError("unknown or duplicate selected candidate")
    chosen = [by_id[identity] for identity in selected]
    cost = sum(item["cost"] for item in chosen)
    topics = {topic for item in chosen for topic in item["topics"]}
    all_lineages = [item["lineage"] for item in items]
    all_keys = [item["key"] for item in items]
    features = {
        "pool_ge_9" if len(items) >= 9 else "pool_le_8",
        "goals_ge_4" if len(task["goals"]) >= 4 else "goals_le_3",
        "budget_le_30" if task["token_budget"] <= 30 else "budget_gt_30",
        "selected_ge_3" if len(selected) >= 3 else "selected_le_2",
        "selected_cost_ge_75pct" if 4 * cost >= 3 * task["token_budget"] else "selected_cost_lt_75pct",
        "chosen_topics_ge_3" if len(topics) >= 3 else "chosen_topics_le_2",
        "pool_has_prerequisites" if any(item["requires"] for item in items) else "pool_no_prerequisites",
        "pool_lineage_collisions" if len(set(all_lineages)) < len(all_lineages) else "pool_unique_lineages",
        "pool_key_collisions" if len(set(all_keys)) < len(all_keys) else "pool_unique_keys",
        "pool_ineligible_ge_2" if sum(not item["eligible"] for item in items) >= 2 else "pool_ineligible_lt_2",
        "max_cost_ge_14" if max(item["cost"] for item in items) >= 14 else "max_cost_lt_14",
        "pool_has_newer_epoch" if any(item["epoch"] >= 4 for item in items) else "pool_epoch_lt_4",
        "has_conflict_pairs" if task["conflicts"] else "no_conflict_pairs",
    }
    # These describe the public world, not the hidden optimum or action target.
    return tuple(sorted(features))


def generate_observations(*, roots_per_family: int = 512) -> list[Observation]:
    """Read one initial state per TRAIN root; reserved families never get labels."""
    if not 1 <= roots_per_family <= 512:
        raise ValueError("roots_per_family outside frozen synthetic source horizon")
    rows = []
    for family in FAMILIES:
        if family == OOD_FAMILY:
            continue
        for seed in range(roots_per_family):
            if split_for(family, seed) != "train_candidate":
                continue
            task = make_task(family, seed)
            initial = first_state(task, seed)
            validation = evaluate_subset(task, initial)
            rows.append(Observation(root=root_id(family, seed), family=family,
                                    split="train_candidate",
                                    features=structural_features(task, initial),
                                    failures=tuple(validation["violations"])))
    return rows


def _internal_partition(root: str) -> str:
    # Within training roots only; this internal diagnostic partition cannot
    # license use of the existing development/test/OOD roots.
    bucket = int(hashlib.sha256(("motif-internal-v1/" + root).encode()).hexdigest()[:8], 16) % 100
    return "discovery" if bucket < 70 else "confirmation"


def _smoothed_rate(positive: int, total: int) -> float:
    return (positive + 0.5) / (total + 1)


def _support(rows: list[Observation], motif: tuple[str, ...]) -> list[Observation]:
    expected = set(motif)
    return [row for row in rows if expected.issubset(row.features)]


def mine_motifs(observations: list[Observation], *, min_support: int = 20,
                top_k: int = 24) -> dict[str, Any]:
    """Choose candidate rules on discovery roots; inspect confirmation separately.

    Candidate motifs are conjunctions of at most two observable booleans. They
    are *suggestions* to generate counterfactual tests, not admitted rules.
    """
    if min_support < 2 or top_k < 1:
        raise ValueError("invalid mining support or selection bound")
    roots = [row.root for row in observations]
    if not roots or len(roots) != len(set(roots)):
        raise ValueError("empty roots or repeated source lineage")
    if any(row.split != "train_candidate" or row.family == OOD_FAMILY for row in observations):
        raise ValueError("miner cannot inspect reserved root labels")
    if any(not row.features or len(row.features) != len(set(row.features)) for row in observations):
        raise ValueError("features must be nonempty, unique, observable atoms")
    discovery = [r for r in observations if _internal_partition(r.root) == "discovery"]
    confirmation = [r for r in observations if _internal_partition(r.root) == "confirmation"]
    if not discovery or not confirmation:
        raise ValueError("insufficient training roots for internal partition")
    base_discovery = _smoothed_rate(sum(r.invalid for r in discovery), len(discovery))
    base_confirm = _smoothed_rate(sum(r.invalid for r in confirmation), len(confirmation))
    features = sorted({feature for row in discovery for feature in row.features})
    proposals = []
    for width in (1, 2):
        for motif in itertools.combinations(features, width):
            eligible = _support(discovery, motif)
            if len(eligible) < min_support:
                continue
            positive = sum(row.invalid for row in eligible)
            discovery_lift = _smoothed_rate(positive, len(eligible)) / base_discovery
            if discovery_lift <= 1.0:
                continue
            confirmation_matches = _support(confirmation, motif)
            n_confirm = len(confirmation_matches)
            confirm_positive = sum(row.invalid for row in confirmation_matches)
            confirm_lift = (
                _smoothed_rate(confirm_positive, n_confirm) / base_confirm if n_confirm else None
            )
            # Pair surprisingness compares joint frequency with independent
            # marginals; it is exploratory and has no causal interpretation.
            surprisingness = None
            if width == 2:
                left, right = motif
                p_left = (sum(left in row.features for row in discovery) + 0.5) / (len(discovery) + 1)
                p_right = (sum(right in row.features for row in discovery) + 0.5) / (len(discovery) + 1)
                p_joint = (len(eligible) + 0.5) / (len(discovery) + 1)
                surprisingness = round(math.log2(p_joint / (p_left * p_right)), 5)
            source_stamps = sorted(row.root for row in eligible)
            proposals.append({
                "features": list(motif),
                "width": width,
                "source_lineage_sha256": sha256_bytes("\n".join(source_stamps).encode()),
                "discovery": {"support": len(eligible), "invalid": positive,
                              "lift_invalid": round(discovery_lift, 5)},
                "confirmation": {"support": n_confirm, "invalid": confirm_positive,
                                 "lift_invalid": round(confirm_lift, 5) if confirm_lift is not None else None},
                "pair_surprisingness_bits": surprisingness,
                "internal_confirmed": n_confirm >= max(3, min_support // 3)
                and confirm_lift is not None and confirm_lift > 1.0,
                "candidate_only": True,
            })
    # Discovery-only score selects candidates; confirmation is a diagnostic.
    proposals.sort(key=lambda item: (
        -(item["discovery"]["lift_invalid"] - 1) * math.sqrt(item["discovery"]["support"]),
        -item["discovery"]["support"], item["features"],
    ))
    candidates = proposals[:top_k]
    for i, item in enumerate(candidates):
        item["id"] = f"motif_{i:03d}"
    return {
        "schema_version": REPORT_SCHEMA,
        "method": "bounded_observable_conjunctions_suggested_by_pattern_mining",
        "source": "synthetic_python_oracle_initial_state",
        "source_split": "train_candidate",
        "reserved_root_labels_read": 0,
        "root_count": len(observations),
        "discovery_roots": len(discovery),
        "confirmation_roots": len(confirmation),
        "baseline_invalid_discovery": round(base_discovery, 5),
        "baseline_invalid_confirmation": round(base_confirm, 5),
        "min_discovery_support": min_support,
        "top_k": top_k,
        "num_tested_motifs": len(proposals),
        "candidates": candidates,
        "training_ready": False,
        "native_hyperon_miner_executed": False,
        "new_sft_presentations_created": 0,
        "verifier_authority": "synthetic_python_evaluate_subset",
    }


def render_advisory_metta(report: dict[str, Any]) -> str:
    """MeTTa-shaped *facts only*; require separate native interpreter replay."""
    if report.get("schema_version") != REPORT_SCHEMA or report.get("training_ready") is not False:
        raise ValueError("invalid or released motif report")
    lines = ["; Advisory synthetic motif graph; not executed or promoted by Hyperon"]
    for row in report["candidates"]:
        atom = row["id"]
        lines.append(f"(ProposedMotif {atom})")
        for feat in row["features"]:
            lines.append(f"(MotifFeature {atom} {feat})")
        lines.append(f"(TrainRootSupport {atom} {row['discovery']['support']})")
        lines.append(f"(InternallyConfirmed {atom} {'True' if row['internal_confirmed'] else 'False'})")
    return "\n".join(lines) + "\n"


def make_report(*, roots_per_family: int = 512, min_support: int = 20,
                top_k: int = 24) -> dict[str, Any]:
    report = mine_motifs(generate_observations(roots_per_family=roots_per_family),
                         min_support=min_support, top_k=top_k)
    report["roots_per_family"] = roots_per_family
    report["generation_hashes"] = {
        "miner_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "task_oracle_sha256": sha256_bytes((Path(__file__).parent / "sft_oracle.py").read_bytes()),
        "source_split_sha256": sha256_bytes((Path(__file__).parent / "looped_sft.py").read_bytes()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roots-per-family", type=int, default=512)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to((ROOT / "MeTTa").resolve()):
        parser.error("generated candidate artifacts must remain outside committed MeTTa sources")
    report = make_report(roots_per_family=args.roots_per_family,
                         min_support=args.min_support, top_k=args.top_k)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "motifs.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    (args.out / "motifs.metta").write_text(render_advisory_metta(report), encoding="utf-8")
    print(json.dumps({"roots": report["root_count"], "discovery": report["discovery_roots"],
                      "confirmation": report["confirmation_roots"],
                      "candidates": len(report["candidates"]),
                      "internally_confirmed": sum(x["internal_confirmed"] for x in report["candidates"]),
                      "training_ready": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
