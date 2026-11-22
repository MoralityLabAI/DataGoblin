"""Paired CPU training and offline Prime Verifiers scoring."""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from itertools import product
from pathlib import Path

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage

from MeTTa.ram_hillclimb import FAMILIES, canonical, digest, install_memory_ceiling
from MeTTa.ram_hillclimb_trajectory import DEFAULT, check_memory, fit, rollout
from MeTTa.ram_trajectory import training_trajectory
from MeTTa.ram_trajectory_policy import features, make_training_batch

from .curation import curate
from .taskset import BENCHMARK_ID, MAX_DECISIONS, BenchConfig, CurationTask

VERIFIERS_VERSION = "0.4.1"
TRAIN_START = 21000
SEEDS = (1101, 1102)
CONTROL = {**DEFAULT, "avoid_revisits": True, "stop_fpr": 0.10}
ARCHITECTURES = {
    "trm": dict(train_arm="trm", rollout_arm="trm", depth=4),
    "ldt_proxy": dict(train_arm="direct", rollout_arm="ldt_proxy", depth=1),
}
REGIMES = ("chronological", "metta_curated")


async def score_submission(task: CurationTask, actions: list[dict], model_id: str) -> vf.Trace:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig(model=model_id, runtime=vf.SubprocessConfig())),
        task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content=task.data.prompt_text), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content=canonical(actions)), sampled=True),
        ],
        is_completed=True,
        ok=True,
    )
    await task.score(trace)
    return trace


def coverage(rows: list[dict]) -> dict:
    counts = Counter(frame["next_edit"]["op"] for row in rows for frame in row["transitions"])
    return {"action_label_counts": dict(counts), "unique_train_roots": len(rows)}


def shared_stop_batch(trajectories: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    observations = [
        (features(row["world"], frame["selected"])[1], frame["stop_target"] == "STOP")
        for row in trajectories for frame in row["transitions"]
    ]
    vectors, targets = zip(*observations, strict=True)
    return np.stack(vectors), np.asarray(targets)


async def evaluate(
    fitted: list[tuple], split: str, roots_per_family: int, out: Path, excluded_hashes: set[str],
) -> tuple[dict, set[str]]:
    taskset = vf.load_taskset(BenchConfig(split=split, roots_per_family=roots_per_family))
    tasks = list(taskset)
    hashes = {task.data.source_sha256 for task in tasks}
    if len(hashes) != len(tasks) or hashes & excluded_hashes:
        raise ValueError(f"Overlapping or duplicated {split} sources")
    records = []
    trace_path = out / f"{split}_verifiers_traces.jsonl"
    with trace_path.open("w", encoding="utf-8", newline="\n") as stream:
        for architecture, regime, seed, models, receipt in fitted:
            spec = ARCHITECTURES[architecture]
            config = {**CONTROL, "inference_depth": spec["depth"]}
            model_id = f"{architecture}/{regime}/{seed}"
            for task in tasks:
                data = task.data
                attempted = rollout(
                    models, spec["rollout_arm"], config, data.family, data.source_seed, decision_limit=MAX_DECISIONS,
                )
                actions = [event["action"] for event in attempted["events"] if event["receipt"]["accepted"]]
                trace = await score_submission(task, actions, model_id)
                if trace.metrics["solved_and_stopped"] != attempted["metrics"]["solved_and_stopped"]:
                    raise ValueError("Independent Verifiers replay disagreed with simulator STOP")
                trace.info["checkpoint_sha256"] = receipt["checkpoint_sha256"]
                trace.info["policy_metrics"] = attempted["metrics"]
                stream.write(trace.model_dump_json() + "\n")
                records.append({
                    "architecture": architecture, "regime": regime, "seed": seed,
                    "root": data.id, "source_sha256": data.source_sha256, "family": data.family,
                    "checkpoint_sha256": receipt["checkpoint_sha256"],
                    "reward": int(trace.reward), "score": trace.info["curation_grade"],
                    "policy_metrics": attempted["metrics"], "submitted_actions": actions,
                })
    result = analyze(records)
    result.update(
        receipts_sha256=save_receipts(out, f"{split}_receipts.jsonl", records),
        prime_verifiers_traces_sha256=digest(trace_path.read_bytes()),
        source_roots=[{"root": task.data.id, "sha256": task.data.source_sha256} for task in tasks],
    )
    return result, hashes


def summarize(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "verified_safe_completions": sum(row["reward"] for row in rows),
        "solved_and_stopped": sum(row["score"]["solved_and_stopped"] for row in rows),
        "premature_stops": sum(row["score"]["premature_stop"] for row in rows),
        "exact_first_edits": sum(row["score"]["exact_first_edit"] for row in rows),
        "unsafe_additions": sum(row["score"]["unsafe_additions"] for row in rows),
        "remaining_edits": sum(row["score"]["remaining_edits"] for row in rows),
        "policy_verifier_calls": sum(row["policy_metrics"]["verifier_calls"] for row in rows),
        "recurrent_passes": sum(row["policy_metrics"]["action_iterations"] for row in rows),
        "grading_exact_enumerations": sum(row["score"]["grading_exact_enumerations"] for row in rows),
    }


def cluster_interval(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(2041)
    means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return [round(float(bound), 6) for bound in np.quantile(means, [0.025, 0.975])]


def analyze(records: list[dict]) -> dict:
    arms = {}
    deltas = {}
    by_root: dict[str, dict] = {}
    for arch in ARCHITECTURES:
        subset = [row for row in records if row["architecture"] == arch]
        pairs: dict[tuple, dict] = defaultdict(dict)
        for regime in REGIMES:
            rows = [row for row in subset if row["regime"] == regime]
            arms[f"{arch}/{regime}"] = {
                **summarize(rows),
                "family_metrics": {
                    family: summarize([row for row in rows if row["family"] == family])
                    for family in sorted({row["family"] for row in rows})
                },
            }
            for row in rows:
                pairs[(row["root"], row["seed"])][regime] = row["reward"]
        root_deltas: dict[str, list[int]] = defaultdict(list)
        for (root, _), pair in pairs.items():
            root_deltas[root].append(pair["metta_curated"] - pair["chronological"])
        by_root[arch] = {root: sum(values) / len(values) for root, values in root_deltas.items()}
        changes = np.array([change for values in root_deltas.values() for change in values])
        deltas[arch] = {
            "wins": int((changes > 0).sum()), "losses": int((changes < 0).sum()),
            "paired_reward_delta": float(changes.mean()),
            "root_cluster_bootstrap_95_interval": cluster_interval(np.array(list(by_root[arch].values()))),
        }
    interaction = np.array([by_root["ldt_proxy"][root] - trm for root, trm in by_root["trm"].items()])
    return {
        "arms": arms, "curation_effects": deltas,
        "ldt_minus_trm_curation_interaction": round(float(interaction.mean()), 6),
        "interaction_root_cluster_bootstrap_95_interval": cluster_interval(interaction),
    }


def write_json(path: Path, data: dict) -> str:
    raw = (json.dumps(data, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return digest(raw)


def save_receipts(out: Path, name: str, records: list[dict]) -> str:
    data = "".join(canonical(row) + "\n" for row in records).encode()
    (out / name).write_bytes(data)
    return digest(data)


async def run(
    out: Path, *, train_count: int, dev_count: int, frozen_count: int, conflict_count: int,
    steps: int, development_only: bool,
) -> dict:
    if version("verifiers") != VERIFIERS_VERSION:
        raise RuntimeError(f"MeTTa Research Bench requires verifiers=={VERIFIERS_VERSION}")
    if not (1 <= train_count <= 24 and 1 <= dev_count <= 24 and 1 <= frozen_count <= 24
            and 1 <= conflict_count <= 24 and 16 <= steps <= 128):
        raise ValueError("Experiment outside registered CPU limits")
    memory = install_memory_ceiling(8192)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=repo).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo))
    trajectories = [training_trajectory(family, seed)
                    for family, seed in product(FAMILIES, range(TRAIN_START, TRAIN_START + train_count))]
    sources = [{key: row[key] for key in ("root", "source_sha256", "presentation")} for row in trajectories]
    seen = {source["source_sha256"] for source in sources}
    if len(seen) != len(sources):
        raise ValueError("Duplicate training source")
    data, curation_receipts = curate(trajectories)
    stop_x, stop_y = shared_stop_batch(trajectories)
    batches = {
        regime: (*make_training_batch(rows, CONTROL)[:3], stop_x, stop_y)
        for regime, rows in data.items()
    }
    fitted = []
    for (architecture, spec), (regime, rows), seed in product(ARCHITECTURES.items(), data.items(), SEEDS):
        config = {**CONTROL, "inference_depth": spec["depth"]}
        models, receipt = fit(rows, spec["train_arm"], config, seed, steps, 8192, batch=batches[regime])
        fitted.append((architecture, regime, seed, models, receipt))
    matched_budget = (
        "action_supervision_rows", "stop_supervision_rows", "total_optimizer_updates", "action_matmul_macs",
    )
    for arch in ARCHITECTURES:
        for seed in SEEDS:
            baseline, curated = [
                receipt for name, _, fit_seed, _, receipt in fitted if name == arch and fit_seed == seed
            ]
            if any(baseline[field] != curated[field] for field in matched_budget):
                raise ValueError("Paired curation arms must have identical optimizer and row budgets")
            if baseline["component_checkpoints"]["stop"] != curated["component_checkpoints"]["stop"]:
                raise ValueError("Curated edit selection must preserve the seed-paired STOP head")
    development, hashes = await evaluate(fitted, "development", dev_count, out, seen)
    seen.update(hashes)
    report = {
        "benchmark": "MeTTa Research Bench", "taskset_id": BENCHMARK_ID,
        "verifiers_version": version("verifiers"), "verifiers_api": "verifiers.v1.Task.score",
        "hyperon_version": version("hyperon"), "numpy_version": np.__version__,
        "git_commit": git_commit, "git_tracked_changes": dirty, "memory": memory,
        "train_sources": sources, "train_sources_sha256": digest(sources), "steps": steps, "seeds": SEEDS,
        "architectures": ARCHITECTURES,
        "curation_receipts_sha256": save_receipts(out, "training_curation_receipts.jsonl", curation_receipts),
        "training_coverage": {regime: coverage(rows) for regime, rows in data.items()},
        "shared_stop_training": {"rows": len(stop_y), "stop_targets": int(stop_y.sum())},
        "training_generation_exact_enumerations": len(trajectories),
        "fits": [{"architecture": arch, "regime": regime, **receipt}
                 for arch, regime, _, _, receipt in fitted],
        "development": development,
    }
    lock = write_json(out / "development_report.json", report)
    if development_only:
        return report
    frozen, hashes = await evaluate(fitted, "frozen", frozen_count, out, seen)
    conflict, _ = await evaluate(fitted, "conflict", conflict_count, out, seen | hashes)
    result = {
        "benchmark": report["benchmark"], "development_report_sha256": lock,
        "fresh_curation": frozen, "fresh_conflict": conflict, "peak_process_rss_mib": check_memory(8192),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(out / "frozen_report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=20, help="TRAIN roots per curation family")
    parser.add_argument("--dev-count", type=int, default=8, help="development roots per curation family")
    parser.add_argument("--frozen-count", type=int, default=16, help="frozen roots per curation family")
    parser.add_argument("--conflict-count", type=int, default=24, help="frozen conflict roots")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(run(args.out, train_count=args.train_count, dev_count=args.dev_count,
                             frozen_count=args.frozen_count, conflict_count=args.conflict_count,
                             steps=args.steps, development_only=args.development_only))
    print(json.dumps({k: result[k] for k in ("benchmark", "wall_seconds") if k in result}))
    return 0
