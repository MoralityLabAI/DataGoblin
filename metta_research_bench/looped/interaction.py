"""Measure the matched lineage and revision interaction on fresh frozen roots."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from MeTTa.ram_hillclimb import digest, install_memory_ceiling
from MeTTa.ram_hillclimb_trajectory import check_memory

from ..experiment import VERIFIERS_VERSION, save_receipts, write_json
from .__main__ import LIMIT, evaluate, revision, summarize
from .ablation import MATCHED_FIELDS, load_model, read_reference
from .ablation_data import MUTATIONS, ablation_rules, prepare_ablation, reference_file
from .ablation_stats import signed_p_value
from .model import LoopedPointer, batch_examples, fit, upstream_revision, weight_hash
from .pool import rule_hashes

JOINT = "without_lineage_revision"
ARMS = ("good", "without_lineage", "without_revision", JOINT)
PROTOCOL = {
    "schema": "looped-lineage-revision-interaction-v1",
    "primary_split": "looped_interaction_frozen", "primary_roots_per_family": 40,
    "conflict_split": "looped_interaction_conflict", "conflict_roots": 32,
    "primary_contrast": "without_lineage_revision - without_lineage - without_revision + good",
    "bootstrap_draws": 2000, "sign_flip_draws": 19999,
}


def factorial_rules() -> dict[str, str]:
    singles = ablation_rules()
    before, after = MUTATIONS["revision"]
    return {**{name: singles[name] for name in ARMS[:-1]},
            JOINT: singles["without_lineage"].replace(before, after, 1)}


def factorial_metrics(rows: list[dict], seeds: list[int]) -> dict:
    by_arm = defaultdict(list)
    pairs = defaultdict(dict)
    families = {}
    for row in rows:
        name, root, seed = row["curator"], row["root"], row["seed"]
        pair = pairs[(root, seed)]
        if name not in ARMS or name in pair or families.setdefault(root, row["family"]) != row["family"]:
            raise ValueError("Invalid factorial observation")
        pair[name] = row["reward"]
        by_arm[name].append(row)
    if set(pairs) != {(root, seed) for root in families for seed in seeds}:
        raise ValueError("Every frozen root must have all paired seeds")
    by_root = defaultdict(list)
    by_seed = defaultdict(list)
    for (root, seed), pair in pairs.items():
        if set(pair) != set(ARMS):
            raise ValueError("Every frozen root and seed must have all four arms")
        delta = pair[JOINT] - pair["without_lineage"] - pair["without_revision"] + pair["good"]
        by_root[root].append(delta)
        by_seed[seed].append(delta)
    root_means = {root: float(np.mean(values)) for root, values in by_root.items()}
    values = np.array(list(root_means.values()))
    rng = np.random.default_rng(2041)
    draws = rng.choice(values, size=(PROTOCOL["bootstrap_draws"], len(values)), replace=True).mean(axis=1)
    arms = {name: summarize(by_arm[name]) for name in ARMS}
    return {
        "arms": arms,
        "interaction": {
            "estimate": float(values.mean()), "paired_roots": len(values),
            "additive_joint_prediction": arms["without_lineage"]["rate"] + arms["without_revision"]["rate"]
            - arms["good"]["rate"],
            "root_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "p_value": signed_p_value(values, PROTOCOL["sign_flip_draws"]),
            "per_seed": {str(seed): float(np.mean(values)) for seed, values in by_seed.items()},
        },
        "family_safe_completions": {
            family: {name: sum(row["reward"] for row in by_arm[name] if row["family"] == family) for name in ARMS}
            for family in sorted(set(families.values()))
        },
    }


def train(args: argparse.Namespace) -> dict:
    commit, dirty = revision()
    if dirty:
        raise ValueError("Commit the interaction protocol before training")
    started = time.perf_counter()
    reference, original_lock = read_reference(args.reference)
    prior_raw = (args.prior_ablation / "locked_ablation.json").read_bytes()
    prior = json.loads(prior_raw)
    prior_frozen_raw = (args.prior_ablation / "frozen_ablation_report.json").read_bytes()
    prior_frozen = json.loads(prior_frozen_raw)
    if (prior_frozen["locked_ablation_sha256"] != digest(prior_raw)
            or prior["reference_lock_sha256"] != digest(original_lock)
            or prior["matched_protocol"] != reference["protocol"]):
        raise ValueError("Reference runs do not share the same training lock and protocol")
    prior_selection = prior["selection"]
    old_cards = reference_file(args.prior_ablation / "ablation_curation_audit.jsonl", prior_selection["receipt_sha256"])
    rules = factorial_rules()
    hashes = rule_hashes(rules)
    if any(hashes[name] != prior_selection["rules_sha256"][name] for name in ARMS[:-1]):
        raise ValueError("Factorial controls differ from the archived rules")
    data, cards, selection = prepare_ablation(args.reference, reference, LIMIT, rules=rules)
    for card, old in zip(cards, old_cards, strict=True):
        if (card["source_sha256"] != old["source_sha256"] or
                any(card["selected"][name] != old["selected"][name] for name in ARMS[:-1])):
            raise ValueError("Factorial controls do not reproduce the earlier training selections")

    args.out.mkdir(parents=True, exist_ok=False)
    args.checkpoints.mkdir(parents=True, exist_ok=False)
    selection["receipt_sha256"] = save_receipts(args.out, "interaction_curation_audit.jsonl", cards)
    old_fits = {(row["curator"], row["seed"]): row for row in prior["fits"]}
    seeds = reference["protocol"]["seeds"]
    batches = batch_examples(data[JOINT])
    fits = []
    for seed in seeds:
        torch.manual_seed(seed)
        initial = LoopedPointer()
        control = old_fits[("good", seed)]
        for name in ARMS[:-1]:
            row = old_fits[(name, seed)]
            if (row["initial_sha256"] != weight_hash(initial)
                    or any(row[field] != control[field] for field in MATCHED_FIELDS)):
                raise ValueError("Archived fits do not match the registered initialization or budget")
            location = "reference_checkpoints" if row["original_checkpoint"] else "prior_checkpoints"
            load_model(getattr(args, location), row)
            fits.append({**row, "location": location})
        model, receipt = fit(initial, batches, seed, reference["protocol"]["optimizer_updates"])
        if any(receipt[field] != control[field] for field in MATCHED_FIELDS):
            raise ValueError("Joint treatment changed the registered training budget")
        filename = f"{JOINT}-{seed}.pt"
        torch.save(model.state_dict(), args.checkpoints / filename)
        fits.append({**receipt, "curator": JOINT, "file": filename, "location": "checkpoints"})

    excluded = set(prior["excluded_source_hashes"]) | set(reference["train_hashes"])
    for split in (prior_frozen["primary"], prior_frozen["conflict_transfer"]):
        excluded.update(row["source_sha256"] for row in split["sources"])
    lock = {
        "commit": commit, "protocol": PROTOCOL, "matched_protocol": reference["protocol"],
        "prior_ablation_lock_sha256": digest(prior_raw), "prior_frozen_sha256": digest(prior_frozen_raw),
        "reference_lock_sha256": digest(original_lock), "excluded_source_hashes": sorted(excluded),
        "architecture": prior["architecture"], "optimizer": prior["optimizer"], "evaluation": prior["evaluation"],
        "selection": selection, "fits": fits, "training_wall_seconds": time.perf_counter() - started,
        "training_peak_mib": check_memory(LIMIT),
    }
    write_json(args.out / "locked_interaction.json", lock)
    return lock


async def frozen(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    raw = (args.out / "locked_interaction.json").read_bytes()
    lock = json.loads(raw)
    commit, dirty = revision()
    if (dirty or lock["commit"] != commit or lock["protocol"] != PROTOCOL
            or lock["selection"]["rules_sha256"] != rule_hashes(factorial_rules())):
        raise ValueError("Interaction source, protocol, or rules changed before frozen evaluation")
    if any(args.out.glob("looped_interaction_*_verifiers_traces.jsonl")):
        raise FileExistsError("The interaction has already opened a frozen split")
    models, expected = {}, {}
    for row in lock["fits"]:
        key = (row["curator"], row["seed"])
        models[key] = load_model(getattr(args, row["location"]), row)
        expected[key] = row["checkpoint_sha256"]

    def analyze(rows: list[dict]) -> dict:
        return factorial_metrics(rows, lock["matched_protocol"]["seeds"])

    excluded = set(lock["excluded_source_hashes"])
    primary, seen = await evaluate(args.out, PROTOCOL["primary_split"], PROTOCOL["primary_roots_per_family"],
                                   models, expected, excluded, analyze=analyze)
    conflict, _ = await evaluate(args.out, PROTOCOL["conflict_split"], PROTOCOL["conflict_roots"],
                                 models, expected, excluded | seen, analyze=analyze)
    report = {
        "locked_interaction_sha256": digest(raw), "primary": primary, "conflict_transfer": conflict,
        "evaluation_peak_mib": check_memory(LIMIT), "evaluation_wall_seconds": time.perf_counter() - started,
    }
    write_json(args.out / "frozen_interaction_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "frozen"))
    for argument in ("reference", "reference-checkpoints", "prior-ablation", "prior-checkpoints", "out", "checkpoints"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    if version("verifiers") != VERIFIERS_VERSION:
        raise RuntimeError(f"Requires verifiers=={VERIFIERS_VERSION}")
    upstream_revision()
    install_memory_ceiling(LIMIT)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.stage == "train":
        result = train(args)
        print(json.dumps({"selection": result["selection"]["arms"], "fits": len(result["fits"])}))
    else:
        result = asyncio.run(frozen(args))
        print(json.dumps({split: result[split]["interaction"] for split in ("primary", "conflict_transfer")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
