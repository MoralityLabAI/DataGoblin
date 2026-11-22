"""Train matched one-rule MeTTa ablations, then open fresh frozen reserves."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from importlib.metadata import version
from pathlib import Path

import torch

from MeTTa.ram_hillclimb import digest, install_memory_ceiling
from MeTTa.ram_hillclimb_trajectory import check_memory

from ..experiment import VERIFIERS_VERSION, save_receipts, write_json
from .__main__ import LIMIT, PROTOCOL_FILE, evaluate, revision, summarize
from .ablation_data import ablation_rules, prepare_ablation
from .ablation_stats import contrasts
from .model import BATCH_SIZE, LoopedPointer, batch_examples, fit, upstream_revision, weight_hash
from .pool import rule_hashes

ABLATION_PROTOCOL = Path(__file__).with_name("ablation_protocol.json")
MATCHED_FIELDS = (
    "initial_sha256", "supervised_examples", "supervised_action_tokens", "optimizer_updates", "parameters",
)


def read_reference(reference: Path) -> tuple[dict, bytes]:
    lock = (reference / "locked_development_report.json").read_bytes()
    report = json.loads(lock)
    protocol = json.loads(PROTOCOL_FILE.read_bytes())
    if (report["pilot"] or report["git_tracked_changes"]
            or report["protocol"] != protocol or report["protocol_sha256"] != digest(protocol)
            or report["upstream_commit"] != upstream_revision()
            or (report["torch"], report["verifiers"], report["hyperon"])
            != (str(torch.__version__), version("verifiers"), version("hyperon"))):
        raise ValueError("Reference run is not the matched full benchmark under this runtime")
    return report, lock


def load_model(directory: Path, receipt: dict) -> LoopedPointer:
    model = LoopedPointer()
    model.load_state_dict(torch.load(directory / receipt["file"], map_location="cpu", weights_only=True))
    if weight_hash(model) != receipt["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash changed for {receipt['file']}")
    return model.eval()


def train(out: Path, checkpoints: Path, reference: Path, original: Path, protocol: dict) -> dict:
    commit, dirty = revision()
    if dirty:
        raise ValueError("Commit the ablation before training")
    started = time.perf_counter()
    report, reference_lock = read_reference(reference)
    datasets, cards, selection = prepare_ablation(reference, report)
    out.mkdir(parents=True, exist_ok=False)
    checkpoints.mkdir(parents=True, exist_ok=False)
    aliases = selection["equivalent_to"]
    selection["receipt_sha256"] = save_receipts(out, "ablation_curation_audit.jsonl", cards)
    write_json(out / "ablation_selection.json", selection)
    new_arms = [
        name for name, representative in aliases.items() if name == representative and name not in ("good", "sloppy")
    ]
    batches = {name: batch_examples(datasets[name]) for name in new_arms}
    receipts = []
    for seed in report["protocol"]["seeds"]:
        torch.manual_seed(seed)
        initial = LoopedPointer()
        endpoints = {row["curator"]: row for row in report["fits"] if row["seed"] == seed}
        for name, row in endpoints.items():
            load_model(original, row)
            if (row["initial_sha256"] != weight_hash(initial)
                    or row["optimizer_updates"] != report["protocol"]["optimizer_updates"]
                    or row["supervised_action_tokens"] != BATCH_SIZE * row["optimizer_updates"]
                    or row["supervised_examples"] != selection["arms"][name]["supervised_tokens_per_pass"]):
                raise ValueError("Original endpoint does not match the ablation budget or initialization")
            receipts.append({**row, "original_checkpoint": True})
        for name in new_arms:
            model, receipt = fit(initial, batches[name], seed, report["protocol"]["optimizer_updates"])
            if any(receipt[field] != endpoints["good"][field] for field in MATCHED_FIELDS):
                raise ValueError(f"Rule {name} changed the registered fit budget")
            filename = f"{name}-{seed}.pt"
            torch.save(model.state_dict(), checkpoints / filename)
            receipts.append({"curator": name, "file": filename, "original_checkpoint": False, **receipt})
    prior_frozen = (reference / "frozen_report.json").read_bytes()
    archived = json.loads(prior_frozen)
    if archived["locked_development_sha256"] != digest(reference_lock):
        raise ValueError("Original frozen report does not match the reference lock")
    forbidden = set(report["train_hashes"])
    for split in (report["development"], archived["primary"], archived["conflict_transfer"]):
        forbidden.update(row["source_sha256"] for row in split["sources"])
    lock = {
        "commit": commit, "protocol": protocol, "protocol_sha256": digest(protocol),
        "reference_lock_sha256": digest(reference_lock), "reference_frozen_sha256": digest(prior_frozen),
        "matched_protocol": report["protocol"], "architecture": report["architecture"],
        "optimizer": report["optimizer"], "evaluation": report["evaluation"], "selection": selection,
        "fits": receipts, "excluded_source_hashes": sorted(forbidden),
        "training_wall_seconds": time.perf_counter() - started, "training_peak_mib": check_memory(LIMIT),
    }
    write_json(out / "locked_ablation.json", lock)
    return lock


async def frozen(out: Path, checkpoints: Path, original: Path, protocol: dict) -> dict:
    raw = (out / "locked_ablation.json").read_bytes()
    lock = json.loads(raw)
    commit, dirty = revision()
    if (dirty or lock["commit"] != commit or lock["protocol_sha256"] != digest(protocol)
            or lock["selection"]["rules_sha256"] != rule_hashes(ablation_rules())):
        raise ValueError("Ablation source or protocol changed before frozen evaluation")
    if (out / "frozen_ablation_report.json").exists() or any(out.glob("looped_ablation_*_verifiers_traces.jsonl")):
        raise FileExistsError("This ablation has already opened a frozen split")
    aliases = lock["selection"]["equivalent_to"]
    models, expected = {}, {}
    for row in lock["fits"]:
        key = (row["curator"], row["seed"])
        models[key] = load_model(original if row["original_checkpoint"] else checkpoints, row)
        expected[key] = row["checkpoint_sha256"]

    def analyze(rows: list[dict]) -> dict:
        arms = {name: [row for row in rows if row["curator"] == actual] for name, actual in aliases.items()}
        families = sorted({row["family"] for row in rows})
        return {
            "arms": {name: summarize(arm) for name, arm in arms.items()},
            "family_safe_completions": {
                name: {family: sum(row["reward"] for row in arm if row["family"] == family)
                       for family in families}
                for name, arm in arms.items()
            },
            "contrasts": contrasts(rows, aliases, protocol),
        }

    excluded = set(lock["excluded_source_hashes"])
    primary, seen = await evaluate(out, protocol["primary_split"], protocol["primary_roots_per_family"],
                                   models, expected, excluded, analyze=analyze)
    transfer, _ = await evaluate(out, protocol["transfer_split"], protocol["transfer_roots"],
                                 models, expected, excluded | seen, analyze=analyze)
    report = {
        "locked_ablation_sha256": digest(raw), "primary": primary, "conflict_transfer": transfer,
        "equivalent_to": aliases, "evaluation_peak_mib": check_memory(LIMIT),
    }
    write_json(out / "frozen_ablation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "frozen"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(ABLATION_PROTOCOL.read_bytes())
    if version("verifiers") != VERIFIERS_VERSION:
        raise RuntimeError(f"Requires verifiers=={VERIFIERS_VERSION}")
    upstream_revision()
    install_memory_ceiling(LIMIT)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.stage == "train":
        result = train(args.out, args.checkpoints, args.reference, args.reference_checkpoints, protocol)
        print(json.dumps({"equivalent_to": result["selection"]["equivalent_to"], "fits": len(result["fits"])}))
    else:
        result = asyncio.run(frozen(args.out, args.checkpoints, args.reference_checkpoints, protocol))
        print(json.dumps(result["primary"]["contrasts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
