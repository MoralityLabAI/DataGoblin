"""AIRIS-inspired one-fact counterfactual discovery for synthetic MeTTa curation.

Only train_candidate roots from the frozen MeTTa-SFT generator are considered.
Motif proposals use discovery roots; their confirmation labels are masked until
the policy and per-root random controls are frozen. Each experimental world
changes one public fact and is graded by the environment-owned bounded Python
oracle. Nothing here executes native AIRIS/MeTTa or authorizes source admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .collect import ROOT, sha256_bytes
from .looped_sft import root_id, split_for
from .pattern_miner import Observation, _internal_partition, mine_motifs, structural_features
from .sft_oracle import (
    FAMILIES,
    OOD_FAMILY,
    evaluate_subset,
    exact_optimum,
    first_state,
    make_task,
    next_edit,
    render_metta,
    validate_task,
)

SCHEMA = "metta-airis-paired-fact-interventions-v1"
DEFAULT_OUT = ROOT / "runs" / "metta" / "airis_counterfactual"
MECHANISMS = ("budget", "eligibility", "lineage", "dependency", "revision")
TARGETS = {
    "budget": "token_budget",
    "eligibility": "ineligible",
    "lineage": "repeated_lineage",
    "dependency": "missing_dependency",
    "revision": "stale_revision",
}
FEATURE_PRIORITIES = (
    ("selected_cost_ge_75pct", "budget"),
    ("pool_has_prerequisites", "dependency"),
    ("pool_lineage_collisions", "lineage"),
    ("pool_key_collisions", "revision"),
    ("pool_ineligible_ge_2", "eligibility"),
    ("budget_le_30", "budget"),
    ("selected_ge_3", "eligibility"),
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return sha256_bytes(canonical(value).encode("utf-8"))


@dataclass(frozen=True)
class Intervention:
    mechanism: str
    field: str
    item_id: str | None
    before: Any
    after: Any

    @property
    def target(self) -> str:
        return TARGETS[self.mechanism]


def _selected(task: dict, selected: list[str]) -> list[dict]:
    lookup = {item["id"]: item for item in task["items"]}
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= lookup.keys():
        raise ValueError("invalid public initial selection")
    return [lookup[name] for name in sorted(selected)]


def propose_interventions(task: dict, selected: list[str]) -> list[Intervention]:
    """Generate bounded interventions from PUBLIC facts, without verifier calls."""
    validate_task(task)
    chosen = _selected(task, selected)
    proposals: dict[str, Intervention] = {}
    total_cost = sum(x["cost"] for x in chosen)
    budget = task["token_budget"]
    new_budget = total_cost - 1 if total_cost <= budget else total_cost
    if new_budget >= 1 and new_budget != budget:
        proposals["budget"] = Intervention("budget", "token_budget", None, budget, new_budget)

    first = chosen[0]
    proposals["eligibility"] = Intervention(
        "eligibility", "eligible", first["id"], first["eligible"], not first["eligible"]
    )

    for left in chosen:
        for right in chosen:
            if left["id"] != right["id"] and left["lineage"] != right["lineage"]:
                proposals["lineage"] = Intervention(
                    "lineage", "lineage", left["id"], left["lineage"], right["lineage"]
                )
                break
        if "lineage" in proposals:
            break

    by_id = {item["id"]: item for item in task["items"]}
    chosen_ids = set(selected)
    for item in chosen:
        index = int(item["id"][1:])
        missing = [f"r{i}" for i in range(index)
                   if f"r{i}" not in chosen_ids and f"r{i}" not in item["requires"]]
        if missing:
            new_requires = tuple([*item["requires"], missing[0]])
            proposals["dependency"] = Intervention(
                "dependency", "requires", item["id"], tuple(item["requires"]), new_requires
            )
            break
        missing_current = [dep for dep in item["requires"] if dep not in chosen_ids]
        if missing_current:
            new_requires = tuple(dep for dep in item["requires"] if dep != missing_current[0])
            proposals["dependency"] = Intervention(
                "dependency", "requires", item["id"], tuple(item["requires"]), new_requires
            )
            break

    if task["family"] == "revision":
        for item in chosen:
            rivals = [x for x in task["items"]
                      if x["key"] == item["key"] and x["id"] != item["id"] and x["eligible"]]
            if rivals:
                # A newer eligible rival changes the status of an existing
                # selected version. The chooser uses only published epochs.
                rival = rivals[0]
                updated = max(x["epoch"] for x in task["items"]
                              if x["key"] == item["key"] and x["eligible"]) + 1
                if updated != rival["epoch"]:
                    proposals["revision"] = Intervention(
                        "revision", "epoch", rival["id"], rival["epoch"], updated
                    )
                    break

    if any(p.mechanism not in MECHANISMS or p.before == p.after for p in proposals.values()):
        raise ValueError("invalid candidate intervention")
    if any(p.item_id is not None and p.item_id not in by_id for p in proposals.values()):
        raise ValueError("candidate points outside synthetic world")
    return [proposals[name] for name in MECHANISMS if name in proposals]


def _one_fact_difference(before: dict, after: dict) -> list[tuple[str, Any, Any]]:
    """Count semantic leaf changes after ignoring the regenerated MeTTa rendering."""
    b = deepcopy(before)
    a = deepcopy(after)
    b.pop("program", None)
    a.pop("program", None)
    changes: list[tuple[str, Any, Any]] = []
    for key in b:
        if key not in a:
            raise ValueError("unexpected missing task field")
        if key != "items":
            if b[key] != a[key]:
                changes.append((key, b[key], a[key]))
            continue
        if len(b["items"]) != len(a["items"]):
            raise ValueError("changed item-count confound")
        for old_item, new_item in zip(b["items"], a["items"], strict=True):
            for field in old_item:
                if field not in new_item:
                    raise ValueError("removed candidate field")
                if old_item[field] != new_item[field]:
                    if field == "requires":
                        delta = len(set(old_item[field]) ^ set(new_item[field]))
                        if delta != 1:
                            raise ValueError("dependency intervention changed multiple facts")
                    changes.append((f'{old_item["id"]}.{field}', old_item[field], new_item[field]))
            if old_item.keys() != new_item.keys():
                raise ValueError("changed candidate schema")
    if b.keys() != a.keys() or len(changes) != 1:
        raise ValueError("intervention must change exactly one public fact")
    return changes


def apply_intervention(task: dict, intervention: Intervention) -> dict:
    """Apply exactly one public fact and regenerate a structurally identical world."""
    validate_task(task)
    if intervention.mechanism not in TARGETS:
        raise ValueError("unregistered intervention")
    altered = deepcopy(task)
    if intervention.item_id is None:
        if intervention.field != "token_budget" or altered["token_budget"] != intervention.before:
            raise ValueError("stale token-budget intervention")
        altered["token_budget"] = intervention.after
    else:
        item = next((i for i in altered["items"] if i["id"] == intervention.item_id), None)
        if item is None or intervention.field not in ("eligible", "lineage", "requires", "epoch"):
            raise ValueError("unrecognized item/field")
        current = tuple(item["requires"]) if intervention.field == "requires" else item[intervention.field]
        if current != intervention.before:
            raise ValueError("stale item intervention")
        item[intervention.field] = (
            list(intervention.after) if intervention.field == "requires" else intervention.after
        )
    altered["program"] = render_metta(altered)
    validate_task(altered)
    delta = _one_fact_difference(task, altered)
    expected = (
        intervention.field if intervention.item_id is None
        else f"{intervention.item_id}.{intervention.field}"
    )
    if delta[0][0] != expected:
        raise ValueError("changed fact differs from declared intervention")
    return altered


def direct_target(task: dict, selected: list[str], mechanism: str) -> bool:
    """Independent, narrowly scoped Boolean check of the targeted violation."""
    chosen = _selected(task, selected)
    chosen_ids = {item["id"] for item in chosen}
    if mechanism == "budget":
        return sum(item["cost"] for item in chosen) > task["token_budget"]
    if mechanism == "eligibility":
        return any(not item["eligible"] for item in chosen)
    if mechanism == "lineage":
        return len({item["lineage"] for item in chosen}) < len(chosen)
    if mechanism == "dependency":
        return any(not set(item["requires"]) <= chosen_ids for item in chosen)
    if mechanism == "revision":
        if task["family"] != "revision":
            return False
        latest = {}
        for item in task["items"]:
            if item["eligible"]:
                latest[item["key"]] = max(item["epoch"], latest.get(item["key"], 0))
        return any(item["epoch"] < latest.get(item["key"], 0) for item in chosen)
    raise ValueError("unknown verifier mechanism")


def discover_policy(*, roots_per_family: int = 512, min_support: int = 20,
                    top_k: int = 24) -> tuple[list[tuple[str, ...]], dict[str, int]]:
    """Get motifs from discovery labels only; confirmation outcomes stay MASKED."""
    if not 1 <= roots_per_family <= 512:
        raise ValueError("root horizon exceeds frozen synthetic corpus")
    rows = []
    count = Counter()
    for family in FAMILIES:
        if family == OOD_FAMILY:
            continue
        for seed in range(roots_per_family):
            if split_for(family, seed) != "train_candidate":
                continue
            root = root_id(family, seed)
            task = make_task(family, seed)
            current = first_state(task, seed)
            failures = ()
            partition = _internal_partition(root)
            if partition == "discovery":
                failures = tuple(evaluate_subset(task, current)["violations"])
            rows.append(Observation(root=root, family=family, split="train_candidate",
                                    features=structural_features(task, current),
                                    failures=failures))
            count[partition] += 1
    report = mine_motifs(rows, min_support=min_support, top_k=top_k)
    # Do not use the miner's confirmation diagnostics; their supplied labels
    # are intentionally empty and only its discovery-ranked motifs are kept.
    motifs = [tuple(row["features"]) for row in report["candidates"]]
    return motifs, dict(count)


def choose_guided(options: list[Intervention], features: tuple[str, ...],
                  ranked_motifs: list[tuple[str, ...]]) -> Intervention:
    if not options:
        raise ValueError("no intervention can be proposed")
    by_name = {x.mechanism: x for x in options}
    visible = set(features)
    for motif in ranked_motifs:
        if set(motif) <= visible:
            for feature, mechanism in FEATURE_PRIORITIES:
                if feature in motif and mechanism in by_name:
                    return by_name[mechanism]
    for feature, mechanism in FEATURE_PRIORITIES:
        if feature in visible and mechanism in by_name:
            return by_name[mechanism]
    return options[0]


def choose_uniform(options: list[Intervention], *, root: str, seed: int) -> Intervention:
    if not options:
        raise ValueError("no control intervention available")
    key = hashlib.sha256(f"airis-control-v1/{seed}/{root}".encode()).digest()
    return options[int.from_bytes(key[:8], "big") % len(options)]


def _verify_arm(task: dict, selected: list[str], intervention: Intervention,
                baseline_state: dict, baseline_optimal: dict) -> dict[str, Any]:
    changed = apply_intervention(task, intervention)
    state = evaluate_subset(changed, selected)
    target = intervention.target
    if direct_target(task, selected, intervention.mechanism) != (target in baseline_state["violations"]):
        raise ValueError("base verifier disagreement")
    if direct_target(changed, selected, intervention.mechanism) != (target in state["violations"]):
        raise ValueError("counterfactual verifier disagreement")
    optimum = exact_optimum(changed)
    before_edit = next_edit(selected, baseline_optimal["selected"])
    after_edit = next_edit(selected, optimum["selected"])
    return {
        "mechanism": intervention.mechanism,
        "fact": intervention.field if intervention.item_id is None else (
            f"{intervention.item_id}.{intervention.field}"
        ),
        "world_sha256": digest(changed),
        "one_fact_delta": 1,
        "target_violation_changed": (
            (target in baseline_state["violations"]) != (target in state["violations"])
        ),
        "violation_signature_changed": baseline_state["violations"] != state["violations"],
        "oracle_first_edit_changed": before_edit != after_edit,
        "oracle_optimal_selection_changed": baseline_optimal["selected"] != optimum["selected"],
        "oracle_checked_subsets": optimum["checked_subsets"],
        "verification": "bounded_python_subset_oracle_plus_independent_target_predicate",
        "native_metta_executed": False,
        "training_ready": False,
    }


def evaluate_pair(family: str, seed: int, motifs: list[tuple[str, ...]],
                  *, control_seed: int) -> dict[str, Any] | None:
    """Run motif and uniform selectors against exactly the SAME public root."""
    if family not in FAMILIES or family == OOD_FAMILY or split_for(family, seed) != "train_candidate":
        raise ValueError("reserved or out-of-range family root")
    root = root_id(family, seed)
    if _internal_partition(root) != "confirmation":
        raise ValueError("exploration root cannot enter internal confirmation")
    task = make_task(family, seed)
    selected = first_state(task, seed)
    options = propose_interventions(task, selected)
    if not options:
        return None
    features = structural_features(task, selected)
    guided = choose_guided(options, features, motifs)
    uniform = choose_uniform(options, root=root, seed=control_seed)
    before = evaluate_subset(task, selected)
    # Identity sham: proves both arms start from precisely the same world.
    if before != evaluate_subset(deepcopy(task), selected):
        raise ValueError("identity sham violated")
    optimal = exact_optimum(task)
    a = _verify_arm(task, selected, guided, before, optimal)
    b = _verify_arm(task, selected, uniform, before, optimal)
    return {
        "root": root,
        "family": family,
        "split": "train_candidate",
        "internal_partition": "confirmation",
        "source_world_sha256": digest(task),
        "source_lineage_stamp": digest(root),
        "public_features_sha256": digest(features),
        "available_mechanisms": [x.mechanism for x in options],
        "selected_source_state": "independently_sampled_public_initial_selection",
        "control_same_intervention": guided == uniform,
        "guided": a,
        "uniform": b,
        "zero_change_sham_passed": True,
        "training_ready": False,
        "new_sft_presentations": 0,
        "oracle_evaluation_only": True,
        "ticket_custody": "shadow_only_not_admitted",
    }


def _confirmation_cases(roots_per_family: int, limit: int) -> list[tuple[str, int]]:
    if not 1 <= limit <= 512:
        raise ValueError("case limit must be in [1,512]")
    families: dict[str, list[tuple[str, int]]] = {}
    for family in FAMILIES:
        if family == OOD_FAMILY:
            continue
        roots = [(family, seed) for seed in range(roots_per_family)
                 if split_for(family, seed) == "train_candidate"
                 and _internal_partition(root_id(family, seed)) == "confirmation"]
        roots.sort(key=lambda x: digest(root_id(*x)))
        families[family] = roots
    ordered = []
    depth = 0
    while len(ordered) < limit:
        added = False
        for family in families:
            if depth < len(families[family]):
                ordered.append(families[family][depth])
                added = True
                if len(ordered) == limit:
                    break
        if not added:
            break
        depth += 1
    return ordered


def run_experiment(*, roots_per_family: int = 512, case_limit: int = 96,
                   seed: int = 20261122, min_support: int = 20,
                   top_k: int = 24) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    motifs, counts = discover_policy(roots_per_family=roots_per_family,
                                    min_support=min_support, top_k=top_k)
    cases = _confirmation_cases(roots_per_family, case_limit)
    if not cases:
        raise ValueError("no confirmation roots available")
    receipts = []
    for family, number in cases:
        receipt = evaluate_pair(family, number, motifs, control_seed=seed)
        if receipt is not None:
            receipts.append(receipt)
    if not receipts:
        raise ValueError("no viable paired interventions")
    if len({r["root"] for r in receipts}) != len(receipts):
        raise ValueError("duplicated source roots across paired arms")
    positive = Counter()
    mechanism_counts = Counter()
    for row in receipts:
        for arm in ("guided", "uniform"):
            mechanism_counts[(arm, row[arm]["mechanism"])] += 1
            positive[(arm, "target")] += int(row[arm]["target_violation_changed"])
            positive[(arm, "signature")] += int(row[arm]["violation_signature_changed"])
            positive[(arm, "first_edit")] += int(row[arm]["oracle_first_edit_changed"])
        a, b = row["guided"]["target_violation_changed"], row["uniform"]["target_violation_changed"]
        positive[("pair", "guided_only")] += int(a and not b)
        positive[("pair", "uniform_only")] += int(b and not a)
        positive[("pair", "ties")] += int(a == b)
        positive[("pair", "same_intervention")] += int(row["control_same_intervention"])
    n = len(receipts)
    policy_scores = {
        arm: {
            "paired_roots": n,
            "targeted_violation_flips": positive[(arm, "target")],
            "targeted_violation_flip_rate": round(positive[(arm, "target")] / n, 6),
            "full_violation_signature_changes": positive[(arm, "signature")],
            "oracle_next_edit_changes": positive[(arm, "first_edit")],
            "mechanisms": {m: mechanism_counts[(arm, m)] for m in MECHANISMS},
        } for arm in ("guided", "uniform")
    }
    report = {
        "schema_version": SCHEMA,
        "experiment_type": "frozen_train_internal_root_paired_counterfactual",
        "source": "original_synthetic_metta_curation_worlds",
        "source_split": "train_candidate",
        "root_horizon_per_family": roots_per_family,
        "discovery_root_count": counts["discovery"],
        "internal_confirmation_root_count": counts["confirmation"],
        "internal_confirmation_labels_used_for_policy": False,
        "reserved_dev_test_ood_roots_accessed": 0,
        "requested_case_limit": case_limit,
        "actual_paired_cases": n,
        "available_confirmation_roots_selected": len(cases),
        "policy_motifs_frozen": [list(x) for x in motifs],
        "motif_policy_uses_discovery_rank_only": True,
        "control": "uniform_public_candidate_mechanism_same_root",
        "control_seed": seed,
        "paired_policy": policy_scores,
        "paired_outcomes": {
            "guided_only_target_flip": positive[("pair", "guided_only")],
            "uniform_only_target_flip": positive[("pair", "uniform_only")],
            "ties": positive[("pair", "ties")],
            "same_intervention": positive[("pair", "same_intervention")],
            "guided_minus_uniform_target_flip_rate": round(
                (positive[("guided", "target")] - positive[("uniform", "target")]) / n, 6
            ),
        },
        "source_root_receipts_sha256": sha256_bytes(
            "\n".join(canonical(row) for row in receipts).encode("utf-8")
        ),
        "generator_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "base_oracle_sha256": sha256_bytes((Path(__file__).parent / "sft_oracle.py").read_bytes()),
        "policy_miner_sha256": sha256_bytes((Path(__file__).parent / "pattern_miner.py").read_bytes()),
        "one_fact_confounds_checked": True,
        "independent_target_predicate_checked": True,
        "all_zero_change_shams_passed": all(row["zero_change_sham_passed"] for row in receipts),
        "source_rights_or_verifier_admission": "no_admission",
        "new_sft_presentations": 0,
        "training_ready": False,
        "native_hyperon_executed": False,
        "native_airis_executed": False,
        "learned_model_evaluated": False,
        "requires_fresh_steward_holdout": True,
    }
    return report, receipts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roots-per-family", type=int, default=512)
    parser.add_argument("--cases", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to((ROOT / "MeTTa").resolve()):
        parser.error("generated case receipts must stay outside committed sources")
    report, receipts = run_experiment(roots_per_family=args.roots_per_family,
                                     case_limit=args.cases, seed=args.seed,
                                     min_support=args.min_support, top_k=args.top_k)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
    (args.out / "internal_confirmation_only.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in receipts), encoding="utf-8"
    )
    print(json.dumps({
        "paired_roots": report["actual_paired_cases"],
        "discovery_roots": report["discovery_root_count"],
        "confirm_pool_roots": report["internal_confirmation_root_count"],
        "guided_flips": report["paired_policy"]["guided"]["targeted_violation_flips"],
        "uniform_flips": report["paired_policy"]["uniform"]["targeted_violation_flips"],
        "rate_delta": report["paired_outcomes"]["guided_minus_uniform_target_flip_rate"],
        "training_ready": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
