"""Run bounded, paired MeTTa-11 CPU trajectory fits and frozen transfer."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .airis02_stop import StopHead
from .airis02_student import MAX_ITEMS
from .ram_hillclimb import (
    DEV_START,
    FAMILIES,
    OOD_FAMILY,
    PUBLIC_TEST_START,
    TRAIN_START,
    canonical,
    digest,
    install_memory_ceiling,
    peak_rss_mib,
    root_name,
)
from .ram_trajectory import (
    STOP,
    RepairVerifier,
    export_trajectories,
    training_trajectory,
    typed_proposal,
    verify_metta_transitions,
)
from .ram_trajectory_policy import ActionPolicy, allocation_features, features, fit_allocator, make_training_batch
from .sft_oracle import edit, first_state, make_task, next_edit

DEPTHS = (2, 4, 6, 8)
ARMS = ("direct", "trm", "ldt_proxy")
OPTIONS = {
    "supervision": ("initial", "trajectory"),
    "stop_fpr": (0.05, 0.10, 0.15, 0.20),
    "feedback": (False, True),
    "avoid_revisits": (False, True),
    "adaptive_depth": (False, True),
    "train_depth": DEPTHS,
    "inference_depth": DEPTHS,
}
DEFAULT = dict(supervision="trajectory", stop_fpr=0.10, feedback=False, avoid_revisits=False,
               adaptive_depth=False, train_depth=4, inference_depth=4)
MAX_DECISIONS = 8
MAX_PROPOSALS_PER_DECISION = 4
SEEDS = (1101, 1102)


def revise(config: dict, proposal: dict) -> dict:
    if not isinstance(proposal, dict) or set(proposal) != {"arm", "field", "value"}:
        raise ValueError("Specify exactly one arm, field and value")
    arm, field, value = proposal["arm"], proposal["field"], proposal["value"]
    if arm not in ARMS or not isinstance(field, str) or field not in OPTIONS:
        raise ValueError("Unknown arm or field")
    if not any(type(value) is type(opt) and value == opt for opt in OPTIONS[field]) or config[field] == value:
        raise ValueError("Proposal must change one registered value")
    if ((arm == "direct" and field in ("train_depth", "inference_depth", "adaptive_depth"))
            or (arm != "ldt_proxy" and field == "feedback")):
        raise ValueError("Field does not apply to this arm")
    return {**config, field: value}


def process_peak_mib() -> float:
    if platform.system() == "Windows":
        import psutil

        return psutil.Process().memory_info().peak_wset / 1048576
    return peak_rss_mib()


def check_memory(limit: int) -> float:
    peak = process_peak_mib()
    if peak > limit:
        raise MemoryError(f"Peak process RSS {peak:.2f} MiB exceeded {limit} MiB")
    return round(peak, 2)


def fit(
    trajectories: list[dict], arm: str, config: dict, seed: int, steps: int, limit: int,
    *, batch: tuple[np.ndarray, ...] | None = None,
) -> tuple:
    start = time.perf_counter()
    cpu = time.process_time()
    x, y, masks, stop_x, stop_y = make_training_batch(trajectories, config) if batch is None else batch
    recurrent = arm != "direct"
    model = ActionPolicy(seed, x.shape[-1], recurrent=recurrent)
    depth = config["train_depth"] if recurrent else 1
    work = model.work_per_update(x, depth)
    budget = steps * model.work_per_update(x, 4)
    updates = budget // work
    losses = model.fit(x, y, masks, steps=updates, lr=0.30, depth=depth)
    stop = StopHead(seed, input_dim=stop_x.shape[1])
    stop_fit = stop.fit(stop_x, stop_y, steps=steps, lr=0.30, max_false_stop_rate=config["stop_fpr"])
    allocator = False
    allocation_fit = None
    if config["adaptive_depth"]:
        continuation_states = np.repeat(stop_x[~stop_y], 2, axis=0)
        allocator, allocation_fit = fit_allocator(model, x, y, masks, continuation_states)
    checkpoint = {
        "action": model.digest(),
        "stop": stop.digest(),
        "allocator": allocator.digest() if isinstance(allocator, StopHead) else allocator,
    }
    receipt = {
        "seed": seed,
        "config": config,
        "checkpoint_sha256": digest(checkpoint),
        "component_checkpoints": checkpoint,
        "parameter_count": sum(p.size for p in model.parameters) + stop.params
        + (allocator.params if isinstance(allocator, StopHead) else 0),
        "action_supervision_rows": len(x),
        "action_target_states": len(y) // 2 if config["supervision"] == "trajectory" else len(trajectories),
        "stop_supervision_rows": len(stop_y),
        "action_updates": updates,
        "stop_updates": steps,
        "allocator_updates": allocation_fit["updates"] if allocation_fit else 0,
        "total_optimizer_updates": updates + steps + (allocation_fit["updates"] if allocation_fit else 0),
        "action_matmul_macs": work * updates,
        "action_matmul_mac_budget": budget,
        "stop_matmul_macs": steps * stop_x.size * 2,
        "action_train_depth": depth,
        "initial_action_loss": losses[0],
        "final_action_loss": losses[-1],
        "final_stop_loss": float(-np.log(np.where(stop_y, stop.probabilities(stop_x),
                                                1 - stop.probabilities(stop_x))).mean()),
        "stop_calibration": stop_fit,
        "adaptive_compute_calibration": allocation_fit,
        "training_wall_seconds": time.perf_counter() - start,
        "training_cpu_seconds": time.process_time() - cpu,
        "peak_process_rss_mib": check_memory(limit),
    }
    return (model, stop, allocator), receipt


def action_scores(
    model: ActionPolicy, allocator: StopHead | bool, x: np.ndarray, stop_x: np.ndarray, allowed: np.ndarray,
    *, depth: int, adaptive: bool,
) -> tuple[np.ndarray, int, int]:
    end = 8 if adaptive else depth
    allocations = 0
    for count, state in enumerate(model.states(x, end), 1):
        if adaptive and count == 2:
            scores = state @ model.wo
            extend = allocator
            if isinstance(allocator, StopHead):
                extend = allocator.should_stop(allocation_features(scores, allowed, stop_x)[None, :])
                allocations += 1
            if not extend:
                return scores, count, allocations
    return state @ model.wo, count, allocations


def rollout(
    models: tuple, arm: str, config: dict, family: str, seed: int,
    *, depth: int | None = None, decision_limit: int = MAX_DECISIONS,
) -> dict:
    if not 1 <= decision_limit <= 10:
        raise ValueError("Outside registered repair decision limits")
    model, stop, allocator = models
    fixed_depth = depth or (1 if arm == "direct" else config["inference_depth"])
    adaptive = depth is None and config["adaptive_depth"]
    task = make_task(family, seed)
    verifier = RepairVerifier(task)
    selected = first_state(task, seed)
    visited = {tuple(selected)}
    optimum = verifier.optimum["selected"]
    first = next_edit(selected, optimum)
    metrics = Counter({key: 0 for key in (
        "solved_and_stopped", "reached_optimum", "premature_stops", "exact_first_edit",
        "unsafe_additions", "new_violation_additions", "inapplicable_actions", "vetoes",
        "action_iterations", "score_evaluations", "retry_score_evaluations", "allocator_calls", "action_readouts",
        "stop_head_calls", "abstentions", "revisits", "checked_refinements", "applied_edits",
        "post_veto_applied_edits", "post_veto_progress_edits", "post_veto_exact_edits",
    )})
    events = []
    metrics["reached_optimum"] = int(selected == optimum)
    for decision in range(decision_limit):
        x, stop_x = features(task, selected)
        metrics["stop_head_calls"] += 1
        if stop.should_stop(stop_x[None, :]):
            receipt = verifier.check(STOP, selected)
            metrics["solved_and_stopped"] = int(receipt["correct_stop"])
            metrics["premature_stops"] = int(not receipt["correct_stop"])
            metrics["exact_first_edit"] += int(decision == 0 and first == STOP)
            events.append({"decision": decision, "selected": selected, "action": STOP, "receipt": receipt})
            break
        actions = [{"op": "remove" if item["id"] in selected else "add", "id": item["id"]}
                   for item in task["items"]]
        allowed = np.arange(MAX_ITEMS) < len(actions)
        if config["avoid_revisits"]:
            allowed[:len(actions)] = [tuple(edit(selected, action)) not in visited for action in actions]
        feedback = scores = None
        accepted = False
        for attempt in range(MAX_PROPOSALS_PER_DECISION):
            if not allowed.any():
                break
            if scores is None or config["feedback"]:
                if feedback is not None:
                    x, _ = features(task, selected, feedback)
                scores, iterations, allocations = action_scores(
                    model, allocator, x, stop_x, allowed, depth=fixed_depth, adaptive=adaptive,
                )
                metrics["action_iterations"] += iterations
                metrics["score_evaluations"] += 1
                metrics["action_readouts"] += 1 + int(adaptive and iterations == 8)
                metrics["retry_score_evaluations"] += int(attempt > 0)
                metrics["allocator_calls"] += allocations
            index = int(np.argmax(np.where(allowed, scores, -np.inf)))
            action = actions[index]
            proposal = typed_proposal(action) if arm == "ldt_proxy" else None
            if proposal is not None:
                receipt = verifier.refine(proposal, selected)
                metrics["checked_refinements"] += 1
            else:
                receipt = verifier.check(action, selected)
            accepted = receipt["accepted"]
            events.append({
                "decision": decision, "attempt": attempt, "selected": selected, "action": action,
                "typed_proposal": proposal, "receipt": receipt,
            })
            if accepted:
                metrics["exact_first_edit"] += int(decision == 0 and action == first)
                metrics["applied_edits"] += 1
                if attempt:
                    metrics["post_veto_applied_edits"] += 1
                    metrics["post_veto_progress_edits"] += int(receipt["remaining_after"] < receipt["remaining_before"])
                    metrics["post_veto_exact_edits"] += int(action == next_edit(selected, optimum))
                if action["op"] == "add":
                    metrics["unsafe_additions"] += int(bool(receipt["result_violations"]))
                    metrics["new_violation_additions"] += int(bool(receipt["new_violations"]))
                selected = receipt["selected_after"]
                metrics["revisits"] += int(tuple(selected) in visited)
                visited.add(tuple(selected))
                metrics["reached_optimum"] |= int(selected == optimum)
                break
            metrics["vetoes"] += int(proposal is not None)
            metrics["inapplicable_actions"] += int(not receipt["structurally_applicable"])
            allowed[index] = False
            feedback = {"action": action, "new_violations": receipt["veto_violations"]}
        if not accepted:
            metrics["abstentions"] += 1
            break
    return {
        "root": root_name(family, seed),
        "source_sha256": digest(task["program"].encode()),
        "family": family,
        "metrics": {
            **metrics, "verifier_calls": verifier.checks, "constraint_checks": verifier.constraint_checks,
            "exact_optimum_calls": 1, "subsets_enumerated": verifier.optimum["checked_subsets"],
            "remaining_edits": len(set(selected) ^ set(optimum)),
        },
        "events": events,
    }


def summarize(rows: list[dict]) -> dict:
    counts: Counter = Counter()
    for row in rows:
        counts.update(row["metrics"])
    solves = counts["solved_and_stopped"]
    return {
        **counts,
        "cases": len(rows),
        "verifier_calls_per_completed_repair": round(counts["verifier_calls"] / solves, 5) if solves else None,
        "iterations_per_completed_repair": round(counts["action_iterations"] / solves, 5) if solves else None,
    }


def selection_key(metrics: dict) -> tuple[int, ...]:
    return (
        metrics["solved_and_stopped"], -metrics["premature_stops"], metrics["exact_first_edit"],
        -metrics["remaining_edits"], -metrics["verifier_calls"], -metrics["action_iterations"],
    )


def improves(prior: dict, candidate: dict) -> tuple[bool, str]:
    safety = ("new_violation_additions", "unsafe_additions", "inapplicable_actions")
    if any(candidate[key] > prior[key] for key in safety):
        return False, "increased_unsafe_or_invalid_additions"
    if selection_key(candidate) <= selection_key(prior):
        return False, "development_score_did_not_improve"
    return True, "development_score_improved"


def run(
    out: Path, *, train_roots: int, dev_roots: int, steps: int, limit: int, proposals: list[dict],
    freeze: bool, expected: Path | None = None, native_replay: bool = False,
) -> dict:
    if not (1 <= train_roots <= 24 and 1 <= dev_roots <= 8 and 16 <= steps <= 256
            and 512 <= limit <= 8192 and isinstance(proposals, list) and len(proposals) <= 8):
        raise ValueError("Experiment outside the registered CPU pilot bounds")
    if freeze != (expected is not None):
        raise ValueError("Frozen transfer requires a prior development report")
    out.mkdir(parents=True, exist_ok=False)
    memory = install_memory_ceiling(limit)
    start = time.perf_counter()
    cpu = time.process_time()
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo))
    train = [training_trajectory(family, seed) for family in FAMILIES
             for seed in range(TRAIN_START, TRAIN_START + train_roots)]
    dev = [(family, seed) for family in FAMILIES for seed in range(DEV_START, DEV_START + dev_roots)]
    source = [{key: row[key] for key in ("root", "source_sha256", "presentation")} for row in train]
    manifest = export_trajectories(train, out)
    native = verify_metta_transitions(train, out) if native_replay else None
    configs = {arm: DEFAULT.copy() for arm in ARMS}
    fits = []
    selections = []
    champions = {}
    development_path = out / "development_receipts.jsonl"

    def experiment(arm: str, config: dict, name: str, *, depth_sweep: bool = False) -> tuple:
        fitted = []
        totals = []
        runs = []
        branches: dict[int, list[dict]] = {depth: [] for depth in DEPTHS} if depth_sweep else {}
        with development_path.open("a", encoding="utf-8") as stream:
            for seed in SEEDS:
                models, receipt = fit(train, arm, config, seed, steps, limit)
                fitted.append((seed, models))
                receipt["training_roots_sha256"] = digest(source)
                receipt["git_commit"] = commit
                began = time.perf_counter()
                current = [rollout(models, arm, config, family, root_seed) for family, root_seed in dev]
                receipt["development"] = summarize(current)
                receipt["family_development"] = {
                    family: summarize([row for row in current if row["family"] == family]) for family in FAMILIES
                }
                receipt["development_wall_seconds"] = time.perf_counter() - began
                receipt["peak_process_rss_mib"] = check_memory(limit)
                runs.append(receipt)
                totals.extend(current)
                for row in current:
                    stream.write(canonical({"experiment": name, "seed": seed, **row}) + "\n")
                for loop_depth, rows in branches.items():
                    sweep = current if loop_depth == config["inference_depth"] else [
                        rollout(models, arm, config, family, root_seed, depth=loop_depth)
                        for family, root_seed in dev
                    ]
                    rows.extend(sweep)
                    if sweep is not current:
                        for row in sweep:
                            stream.write(canonical({"experiment": name, "seed": seed,
                                                    "fixed_depth_probe": loop_depth, **row}) + "\n")
        aggregate = summarize(totals)
        fits.append({"name": name, "arm": arm, "config": config, "runs": runs, "development": aggregate,
                     "fixed_depth_probes": {depth: summarize(rows) for depth, rows in branches.items()}})
        return fitted, aggregate

    for arm in ARMS:
        champions[arm] = experiment(arm, configs[arm], arm, depth_sweep=arm == "trm")
    for i, proposal in enumerate(proposals, 1):
        arm = proposal["arm"]
        config = revise(configs[arm], proposal)
        candidate = experiment(arm, config, f"proposal_{i}")
        previous = champions[arm][1]
        accepted, reason = improves(previous, candidate[1])
        selections.append({
            "experiment": f"proposal_{i}", "proposal": proposal,
            "hypothesis_source": "externally_proposed_and_independently_scored",
            "accepted": accepted, "reason": reason, "prior_selection_key": selection_key(previous),
            "candidate_selection_key": selection_key(candidate[1]),
        })
        if accepted:
            configs[arm] = config
            champions[arm] = candidate
    checkpoint_by_arm = {
        arm: [
            digest({
                "action": action.digest(), "stop": stop.digest(),
                "allocator": alloc.digest() if isinstance(alloc, StopHead) else alloc,
            })
            for _, (action, stop, alloc) in fitted
        ]
        for arm, (fitted, _) in champions.items()
    }
    report = {
        "schema": "metta-11-trajectory-pilot-v1", "git_commit": commit, "git_tracked_changes": dirty,
        "python": platform.python_version(), "numpy": np.__version__, "seeds": SEEDS,
        "train_sources": source, "training_roots_sha256": digest(source),
        "development_root_names": [root_name(*pair) for pair in dev],
        "training_views": manifest,
        "native_transition_replay": {key: native[key] for key in (
            "runtime", "scope", "roots_matched", "transitions_matched",
        )} if native else None,
        "generation_verifier_calls": sum(row["generation_verifier_calls"] for row in train),
        "generation_constraint_checks": sum(row["generation_constraint_checks"] for row in train),
        "generation_exact_optimum_calls": sum(row["generation_exact_optimum_calls"] for row in train),
        "generation_subsets_enumerated": sum(row["generation_subsets_enumerated"] for row in train),
        "memory": memory, "cpu_only": True, "max_decisions": MAX_DECISIONS,
        "max_proposals_per_decision": MAX_PROPOSALS_PER_DECISION,
        "reference_action_steps_at_depth_four": steps, "fits": fits,
        "development_selection": selections,
        "champions": configs,
        "champion_checkpoints": checkpoint_by_arm,
        "development_receipts_sha256": digest(development_path.read_bytes()),
        "conflict_transfer_executed": freeze,
    }
    if freeze:
        frozen = json.loads(expected.read_text(encoding="utf-8"))
        for key in ("champions", "champion_checkpoints", "training_roots_sha256", "development_root_names"):
            if frozen[key] != report[key]:
                raise ValueError(f"Frozen-model reproduction disagrees on {key}")
        if frozen["conflict_transfer_executed"]:
            raise ValueError("Freeze from a development-only report")
        transfer_path = out / "frozen_conflict_receipts.jsonl"
        transfer = {}
        with transfer_path.open("w", encoding="utf-8") as stream:
            for arm, (fitted, _) in champions.items():
                transferred = []
                for i, (seed, model) in enumerate(fitted):
                    for root_seed in range(PUBLIC_TEST_START, PUBLIC_TEST_START + dev_roots):
                        row = rollout(model, arm, configs[arm], OOD_FAMILY, root_seed)
                        transferred.append(row)
                        stream.write(canonical({"arm": arm, "seed": seed,
                                                "checkpoint_sha256": checkpoint_by_arm[arm][i], **row}) + "\n")
                transfer[arm] = summarize(transferred)
        (out / "frozen_conflict_report.json").write_text(
            canonical({"metrics": transfer, "receipts_sha256": digest(transfer_path.read_bytes())}) + "\n",
            encoding="utf-8",
        )
    report["wall_seconds"] = time.perf_counter() - start
    report["process_cpu_seconds"] = time.process_time() - cpu
    report["peak_process_rss_mib"] = check_memory(limit)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-roots", type=int, default=3)
    parser.add_argument("--dev-roots", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--ram-limit-mib", type=int, default=8192)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--frozen-development-report", type=Path)
    parser.add_argument("--native-replay", action="store_true")
    args = parser.parse_args()
    proposals = json.loads(args.proposals.read_text(encoding="utf-8")) if args.proposals else []
    report = run(args.out, train_roots=args.train_roots, dev_roots=args.dev_roots, steps=args.steps,
                 limit=args.ram_limit_mib, proposals=proposals, freeze=args.frozen_development_report is not None,
                 expected=args.frozen_development_report, native_replay=args.native_replay)
    print(json.dumps({key: report[key] for key in ("champions", "wall_seconds", "peak_process_rss_mib")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
