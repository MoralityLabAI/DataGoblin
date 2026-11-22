"""Generate, train, freeze, and score the paired Looped Transformer experiment."""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import verifiers.v1 as vf

from MeTTa.ram_hillclimb import digest, install_memory_ceiling
from MeTTa.ram_hillclimb_trajectory import check_memory

from ..experiment import VERIFIERS_VERSION, cluster_interval, save_receipts, score_submission, write_json
from ..taskset import MAX_DECISIONS, BenchConfig
from .model import (
    BATCH_SIZE,
    HEADS,
    HIDDEN,
    LEARNING_RATE,
    LOOPS,
    MAX_PROPOSALS,
    TOKENIZER_ID,
    LoopedPointer,
    batch_examples,
    fit,
    solve,
    upstream_revision,
    weight_hash,
)
from .pool import CURATORS, RULES, build_pool, rule_hashes

PROTOCOL_FILE = Path(__file__).with_name("protocol.json")
LIMIT = 8192


def revision() -> tuple[str, bool]:
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo))
    return commit, dirty


def summarize(rows: list[dict]) -> dict:
    solved = sum(row["reward"] for row in rows)
    calls = sum(row["policy"]["policy_verifier_calls"] for row in rows)
    passes = sum(row["policy"]["passes"] for row in rows)
    return {
        "cases": len(rows), "safe_completions": solved, "rate": solved / len(rows),
        "unsafe_additions": sum(row["grade"]["unsafe_additions"] for row in rows),
        "premature_stops": sum(row["grade"]["premature_stop"] for row in rows),
        "exact_first_edits": sum(row["grade"]["exact_first_edit"] for row in rows),
        "remaining_edits": sum(row["grade"]["remaining_edits"] for row in rows),
        "policy_verifier_calls": calls, "passes": passes,
        "policy_calls_per_success": calls / solved if solved else None,
        "passes_per_success": passes / solved if solved else None,
        "vetoes": sum(row["policy"]["vetoes"] for row in rows),
        "grading_constraint_checks": sum(row["grade"]["grading_constraint_checks"] for row in rows),
    }


def paired_metrics(rows: list[dict]) -> dict:
    arms = {name: summarize([row for row in rows if row["curator"] == name]) for name in CURATORS}
    pairs: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        pairs[(row["root"], row["seed"])][row["curator"]] = row["reward"]
    by_root: dict[str, list] = defaultdict(list)
    by_seed: dict[int, list] = defaultdict(list)
    for (root, seed), pair in pairs.items():
        delta = pair["good"] - pair["sloppy"]
        by_root[root].append(delta)
        by_seed[seed].append(delta)
    values = np.array([np.mean(deltas) for deltas in by_root.values()])
    return {
        "arms": arms, "paired_roots": len(by_root), "paired_reward_gain": float(values.mean()),
        "paired_root_bootstrap_95": cluster_interval(values),
        "seed_results": {
            str(seed): {
                **{name: summarize([row for row in rows if row["seed"] == seed and row["curator"] == name])
                   for name in CURATORS},
                "paired_gain": sum(deltas) / len(deltas),
            }
            for seed, deltas in by_seed.items()
        },
    }


async def evaluate(
    out: Path, split: str, count: int, models: dict, expected: dict, forbidden: set[str],
) -> tuple[dict, set[str]]:
    tasks = list(vf.load_taskset(BenchConfig(split=split, roots_per_family=count)))
    hashes = {task.data.source_sha256 for task in tasks}
    if len(tasks) != len(hashes) or hashes & forbidden:
        raise ValueError("Evaluation source overlap")
    rows = []
    path = out / f"{split}_verifiers_traces.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for (curator, seed), model in models.items():
            for task in tasks:
                actions, policy = solve(model, task.data)
                trace = await score_submission(task, actions, f"looped/{curator}/{seed}")
                trace.info.update(policy=policy, checkpoint_sha256=expected[(curator, seed)])
                stream.write(trace.model_dump_json() + "\n")
                rows.append({
                    "curator": curator, "seed": seed, "root": task.data.id, "family": task.data.family,
                    "source_sha256": task.data.source_sha256, "checkpoint_sha256": expected[(curator, seed)],
                    "grade": trace.info["curation_grade"], "policy": policy, "reward": int(trace.reward),
                })
    return {
        **paired_metrics(rows), "sources": [{"root": task.data.id, "source_sha256": task.data.source_sha256}
                                           for task in tasks],
        "receipts_sha256": save_receipts(out, f"{split}_receipts.jsonl", rows),
        "prime_traces_sha256": digest(path.read_bytes()),
    }, hashes


def prepare(out: Path, protocol: dict) -> tuple[dict, list, dict]:
    pool, cards, datasets, summary = build_pool(protocol["train_roots_per_family"], protocol["rollout_cap_per_root"])
    n = summary["matched_roots"]
    if not n:
        raise ValueError("No matched curation presentations")
    differences = (summary["audits"]["good"]["safe_completion"] - summary["audits"]["sloppy"]["safe_completion"]) / n
    agreement_check = summary["selection_disagreements"] / n
    summary.update(
        safe_curation_gap=differences, selection_disagreement_fraction=agreement_check,
        candidate_pool_sha256=save_receipts(out, "candidate_rollouts.jsonl", pool),
        paired_curation_sha256=save_receipts(out, "curation_audit.jsonl", cards),
    )
    token_totals = {arm["supervised_tokens_per_pass"] for arm in summary["audits"].values()}
    if len(token_totals) != 1:
        raise ValueError("Selected action-token budgets differ")
    write_json(out / "pool_manifest.json", summary)
    return datasets, pool, summary


def manipulation_check(summary: dict, protocol: dict, *, pilot: bool) -> None:
    if not pilot and (summary["matched_roots"] < protocol["minimum_matched_train_roots"]
                      or summary["rollouts"] < protocol["minimum_distinct_rollouts"]):
        raise ValueError("Pool did not meet preregistered scale")
    if (summary["safe_curation_gap"] < protocol["minimum_audited_quality_gap"]
            or summary["selection_disagreement_fraction"] < protocol["minimum_selection_disagreement"]):
        raise ValueError("Independent quality or selection gap is below the preregistered floor")


async def train(out: Path, checkpoints: Path, protocol: dict, *, pilot: bool) -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    checkpoints.mkdir(parents=True, exist_ok=False)
    revision_id, dirty = revision()
    if dirty and not pilot:
        raise ValueError("Commit the run configuration before full training")
    datasets, pool, summary = prepare(out, protocol)
    manipulation_check(summary, protocol, pilot=pilot)
    batches = {name: batch_examples(datasets[name]) for name in CURATORS}
    models, expected, fits = {}, {}, []
    for seed in protocol["seeds"]:
        torch.manual_seed(seed)
        initial = LoopedPointer()
        for name in CURATORS:
            model, receipt = fit(initial, batches[name], seed, protocol["optimizer_updates"])
            filename = f"{name}-{seed}.pt"
            torch.save(model.state_dict(), checkpoints / filename)
            models[(name, seed)] = model
            expected[(name, seed)] = receipt["checkpoint_sha256"]
            fits.append({"curator": name, "file": filename, **receipt})
        if fits[-1]["initial_sha256"] != fits[-2]["initial_sha256"]:
            raise ValueError("Pair did not start from identical weights")
    train_hashes = {row["source_sha256"] for row in pool}
    development, _ = await evaluate(out, "looped_dev", protocol["development_roots_per_family"],
                                    models, expected, train_hashes)
    result = {
        "pilot": pilot, "commit": revision_id, "git_tracked_changes": dirty, "upstream_commit": upstream_revision(),
        "verifiers": version("verifiers"), "torch": torch.__version__, "hyperon": version("hyperon"),
        "protocol": protocol, "protocol_sha256": digest(protocol), "curation": summary,
        "architecture": {"type": "weight-tied LoopedDecoderLM pointer", "tokenizer": TOKENIZER_ID,
                         "hidden": HIDDEN, "heads": HEADS, "loops": LOOPS},
        "optimizer": {"kind": "AdamW", "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE},
        "evaluation": {"decision_cap": MAX_DECISIONS, "proposal_cap": MAX_PROPOSALS, "typed_membrane": True},
        "fits": fits, "train_hashes": sorted(train_hashes), "development": development,
        "peak_mib": check_memory(LIMIT), "train_and_dev_wall_seconds": time.perf_counter() - started,
    }
    write_json(out / "locked_development_report.json", result)
    return result


async def frozen(out: Path, checkpoints: Path, protocol: dict) -> dict:
    raw = (out / "locked_development_report.json").read_bytes()
    prior = json.loads(raw)
    commit, dirty = revision()
    if prior["pilot"] or dirty or prior["commit"] != commit or prior["protocol_sha256"] != digest(protocol):
        raise ValueError("Frozen evaluation requires unchanged committed protocol and non-pilot checkpoints")
    if prior["curation"]["rules_sha256"] != rule_hashes(
        {name: path.read_text(encoding="utf-8") for name, path in RULES.items()}
    ):
        raise ValueError("Curator source changed after development")
    if (out / "frozen_report.json").exists() or (out / "looped_frozen_verifiers_traces.jsonl").exists():
        raise FileExistsError("Frozen reserve was already opened for this run")
    manipulation_check(prior["curation"], protocol, pilot=False)
    models, expected = {}, {}
    for row in prior["fits"]:
        model = LoopedPointer()
        model.load_state_dict(torch.load(checkpoints / row["file"], map_location="cpu", weights_only=True))
        if weight_hash(model) != row["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint differs from the development receipt")
        key = (row["curator"], row["seed"])
        models[key] = model.eval()
        expected[key] = row["checkpoint_sha256"]
    excluded = set(prior["train_hashes"]) | {row["source_sha256"] for row in prior["development"]["sources"]}
    primary, seen = await evaluate(out, "looped_frozen", protocol["frozen_roots_per_family"],
                                   models, expected, excluded)
    conflict, _ = await evaluate(out, "looped_conflict", protocol["frozen_conflict_roots"],
                                 models, expected, excluded | seen)
    unsafe_delta = primary["arms"]["good"]["unsafe_additions"] - primary["arms"]["sloppy"]["unsafe_additions"]
    achieved = (
        primary["paired_reward_gain"] >= protocol["minimum_frozen_gain"]
        and primary["paired_root_bootstrap_95"][0] > 0
        and unsafe_delta <= protocol["max_unsafe_addition_increase"]
    )
    result = {
        "locked_development_sha256": digest(raw), "primary": primary, "conflict_transfer": conflict,
        "minimum_frozen_gain": protocol["minimum_frozen_gain"], "unsafe_addition_increase": unsafe_delta,
        "preregistered_threshold_met": achieved, "peak_mib": check_memory(LIMIT),
    }
    write_json(out / "frozen_report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "train", "frozen"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_FILE.read_bytes())
    if version("verifiers") != VERIFIERS_VERSION:
        raise RuntimeError(f"Requires verifiers=={VERIFIERS_VERSION}")
    upstream_revision()
    install_memory_ceiling(LIMIT)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.stage == "pilot":
        protocol.update(train_roots_per_family=8, rollout_cap_per_root=24,
                        development_roots_per_family=3, optimizer_updates=150, seeds=[1101])
    if args.stage == "frozen":
        report = asyncio.run(frozen(args.out, args.checkpoints, protocol))
        summary = {key: report[key] for key in ("preregistered_threshold_met", "unsafe_addition_increase")}
        summary["primary_gain"] = report["primary"]["paired_reward_gain"]
    else:
        report = asyncio.run(train(args.out, args.checkpoints, protocol, pilot=args.stage == "pilot"))
        summary = {"quality_gap": report["curation"]["safe_curation_gap"],
                   "development_gain": report["development"]["paired_reward_gain"]}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
