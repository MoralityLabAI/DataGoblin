"""Exact one-rule edits to the original MeTTa selector and matched TRAIN audit."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from MeTTa.ram_hillclimb import FAMILIES, digest, root_name
from MeTTa.ram_trajectory import RepairVerifier
from MeTTa.sft_oracle import first_state, make_task

from .pool import RULES, TRAIN_START, audit, curate_root, rule_hashes

MUTATIONS = {
    "closure": (" Closed ", " $ablateClosure "),
    "eligibility": (" Eligible ", " $ablateEligibility "),
    "budget": (" WithinBudget ", " $ablateBudget "),
    "lineage": (" UniqueLineage ", " $ablateLineage "),
    "revision": (" CurrentRevision ", " $ablateRevision "),
    "dependency": (" DependenciesMet ", " $ablateDependency "),
    "conflict": (" ConflictFree ", " $ablateConflict "),
    "coverage": ("(Offer $id $covered ", "(Offer $id $mentions "),
}


def ablation_rules() -> dict[str, str]:
    rules = {name: path.read_text(encoding="utf-8") for name, path in RULES.items()}
    good = rules["good"]
    for name, (before, after) in MUTATIONS.items():
        if good.count(before) != 1:
            raise ValueError(f"Expected one origin rule clause for {name}")
        rules[f"without_{name}"] = good.replace(before, after, 1)
    return rules


def reference_file(path: Path, expected: str) -> list[dict]:
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"Previous locked run has a different {path.name}")
    return [json.loads(line) for line in data.splitlines()]


def prepare_ablation(reference: Path, report: dict) -> tuple[dict, list[dict], dict]:
    original = report["curation"]
    pool = reference_file(reference / "candidate_rollouts.jsonl", original["candidate_pool_sha256"])
    old_cards = reference_file(reference / "curation_audit.jsonl", original["paired_curation_sha256"])
    rules = ablation_rules()
    hashes = rule_hashes(rules)
    if any(hashes[name] != original["rules_sha256"][name] for name in ("good", "sloppy")):
        raise ValueError("The endpoint rules differ from the previous locked run")
    simulated = {name: [] for name in rules}
    cards = []
    train_end = TRAIN_START + report["protocol"]["train_roots_per_family"]
    for root, previous in zip(pool, old_cards, strict=True):
        family, seed = root["family"], root["seed"]
        if family not in FAMILIES or not TRAIN_START <= seed < train_end:
            raise ValueError("Reference input is outside the registered TRAIN pool")
        world = make_task(family, seed)
        if (root["source_sha256"] != digest(world["program"].encode())
                or root["root"] != root_name(family, seed)
                or root["initial"] != first_state(world, seed)
                or root["root"] != previous["root"] or not previous["matched_quota"]):
            raise ValueError("Reference TRAIN root changed or lacks a common quota")
        candidates = root["rollouts"]
        selected, native_checks = curate_root(world, root, candidates, rules)
        if any(selected[name] != previous["selected"][name] for name in ("good", "sloppy")):
            raise ValueError("Original endpoint selection is not reproducible")
        if None in selected.values():
            raise ValueError("A single-rule ablation failed the matched quota")
        verifier = RepairVerifier(world)
        assessed = {}
        for name, index in selected.items():
            actions = candidates[index]
            if len(actions) != root["action_tokens_per_rollout"]:
                raise ValueError("Ablation changed the original supervised token budget")
            if index not in assessed:
                assessed[index] = audit(verifier, root["initial"], actions)
            simulated[name].append((world, root["initial"], actions))
        cards.append({
            "root": root["root"], "family": family, "source_sha256": root["source_sha256"],
            "selected": selected, "native_checks": native_checks,
            "independent_audit": {name: assessed[index] for name, index in selected.items()},
        })
    sigs = {}
    aliases = {}
    arms = {}
    for name in rules:
        signature = tuple(row["selected"][name] for row in cards)
        aliases[name] = sigs.setdefault(signature, name)
        affected = [row for row in cards if row["selected"][name] != row["selected"]["good"]]
        metrics = [row["independent_audit"][name] for row in cards]
        violations: Counter = Counter()
        for row in metrics:
            violations.update(row["violation_types"])
        arms[name] = {
            "selection_changes": len(affected),
            "changes_by_train_family": dict(Counter(row["family"] for row in affected)),
            "supervised_tokens_per_pass": sum(len(actions) for _, _, actions in simulated[name]),
            **{key: sum(row[key] for row in metrics)
               for key in ("safe_completion", "premature_stop", "unsafe_additions")},
            "violation_types": dict(violations),
        }
    return simulated, cards, {
        "same_pool_sha256": original["candidate_pool_sha256"], "train_roots": len(cards),
        "distinct_candidate_rollouts": sum(row["candidate_count"] for row in pool),
        "rules_sha256": hashes, "equivalent_to": aliases, "arms": arms,
    }
