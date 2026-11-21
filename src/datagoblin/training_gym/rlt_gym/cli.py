"""Command interface for bounded rollouts and offline Codex/Opus queues."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured, load_structured


def add_arguments(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="rlt_command", required=True)
    make = commands.add_parser("make-suite", help="Freeze a generated suite with lineage-separated splits")
    make.add_argument("--seed", type=int, default=7)
    make.add_argument("--per-family", type=int, default=4, help="Tasks per family per split")
    make.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run", help="Fit small CPU controls and collect a matched rollout matrix")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    run.add_argument("--modes", nargs="+", choices=("direct", "rlt", "matched_damping"),
                     default=["direct", "rlt", "matched_damping"])
    run.add_argument("--training-modes", nargs="+", choices=("direct", "rlt", "matched_damping"), default=["direct"])
    run.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    run.add_argument("--splits", nargs="+", default=["calibration"],
                     choices=("train", "calibration", "ordinary_test", "ood_test", "sealed_poison"))
    run.add_argument("--readout-site", choices=("endpoint", "carried_answer"), default="carried_answer")
    run.add_argument("--training-depth", type=int, default=4)
    run.add_argument("--terminal-only", action="store_true", help="Score the final visit and retain earlier checks")
    run.add_argument("--max-wall-seconds", type=float, default=120)
    run.add_argument("--omit-controls", action="store_true")
    run.add_argument("--intervention", type=Path, help="Explicit local state-pulse JSON; record with each episode")
    replay = commands.add_parser("replay", help="Recompute native trajectories and checker receipts")
    replay.add_argument("--suite", type=Path, required=True)
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--out", type=Path)
    prepare = commands.add_parser("teacher-prepare", help="Write public request packets for a named teacher")
    prepare.add_argument("--suite", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--provider", choices=("codex", "opus", "local"), default="codex")
    prepare.add_argument("--model", required=True, help="Operator-supplied model/session identity")
    prepare.add_argument("--split", choices=("train", "calibration"), default="train")
    prepare.add_argument("--max-tasks", type=int, default=64)
    prepare.add_argument("--max-visits", type=int, default=4)
    prepare.add_argument("--frontier", type=Path, help="A train-only frontier manifest from this frozen suite")
    repair = commands.add_parser("teacher-repair", help="Ask a teacher to repair checked native training failures")
    repair.add_argument("--suite", type=Path, required=True)
    repair.add_argument("--rollouts", type=Path, required=True)
    repair.add_argument("--out", type=Path, required=True)
    repair.add_argument("--provider", choices=("codex", "opus", "local"), default="codex")
    repair.add_argument("--model", required=True)
    repair.add_argument("--max-tasks", type=int, default=14)
    repair.add_argument("--max-visits", type=int, default=4, help="Total visits including the native failed attempt")
    ingest = commands.add_parser("teacher-import", help="Recheck a teacher's answers against the frozen environment")
    ingest.add_argument("--suite", type=Path, required=True)
    ingest.add_argument("--requests", type=Path, required=True)
    ingest.add_argument("--responses", type=Path, required=True)
    ingest.add_argument("--out", type=Path, required=True)
    continuation = commands.add_parser("teacher-continue", help="Queue unresolved episodes with checked feedback")
    continuation.add_argument("--suite", type=Path, required=True)
    continuation.add_argument("--rollouts", type=Path, required=True)
    continuation.add_argument("--out", type=Path, required=True)
    continuation.add_argument("--max-tasks", type=int)
    export = commands.add_parser("export", help="Export verified train SFT, transitions, outcomes, repairs, and pairs")
    export.add_argument("--suite", type=Path, required=True)
    export.add_argument("--rollouts", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument(
        "--include-controls", action="store_true", help="Explicitly retain labeled engineering controls",
    )
    frontier = commands.add_parser("frontier", help="Balance training failures/disagreements into teacher work")
    frontier.add_argument("--suite", type=Path, required=True)
    frontier.add_argument("--rollouts", type=Path, required=True)
    frontier.add_argument("--out", type=Path, required=True)
    frontier.add_argument("--max-tasks", type=int, default=64)


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    from .runner import mine_frontier, replay_run, run_suite
    from .tasks import load_suite, write_suite
    from .teachers import (
        continue_teacher_batch,
        export_dataset,
        import_teacher_responses,
        prepare_teacher_batch,
        prepare_teacher_repairs,
    )

    if args.rlt_command == "make-suite":
        path = write_suite(args.out, seed=args.seed, per_family=args.per_family)
        manifest, tasks = load_suite(path)
        return {"manifest_path": str(path), "tasks": len(tasks), "manifest_digest": manifest["manifest_digest"]}
    if args.rlt_command == "run":
        result = run_suite(
            args.suite, args.out, seeds=tuple(args.seeds), modes=tuple(args.modes),
            training_modes=tuple(args.training_modes), depths=tuple(args.depths), splits=tuple(args.splits),
            readout_site=args.readout_site, training_depth=args.training_depth,
            verified_stop=not args.terminal_only, max_wall_seconds=args.max_wall_seconds,
            include_controls=not args.omit_controls,
            intervention=load_structured(args.intervention) if args.intervention else None,
        )
        return {key: value for key, value in result.items() if key != "summary"}
    if args.rlt_command == "replay":
        result = replay_run(args.suite, args.run)
        if args.out:
            if args.out.exists():
                raise ValueError("replay output already exists")
            dump_structured(args.out, result)
        return result
    if args.rlt_command == "teacher-prepare":
        task_ids = None
        if args.frontier:
            frontier = load_structured(args.frontier)
            suite = args.suite / "manifest.json" if args.suite.is_dir() else args.suite
            if (frontier.get("schema_version") != "rlt_gym.frontier.v1"
                    or frontier.get("suite_manifest_sha256") != sha256_file(suite)):
                raise ValueError("frontier belongs to a different suite")
            task_ids = frontier["task_ids"]
        return prepare_teacher_batch(
            args.suite, args.out, provider=args.provider, model=args.model, split=args.split,
            max_tasks=args.max_tasks, max_visits=args.max_visits, task_ids=task_ids,
        )
    if args.rlt_command == "teacher-import":
        return import_teacher_responses(args.suite, args.requests, args.responses, args.out)
    if args.rlt_command == "teacher-repair":
        return prepare_teacher_repairs(
            args.suite, args.rollouts, args.out, provider=args.provider, model=args.model,
            max_tasks=args.max_tasks, max_visits=args.max_visits,
        )
    if args.rlt_command == "teacher-continue":
        return continue_teacher_batch(args.suite, args.rollouts, args.out, max_tasks=args.max_tasks)
    if args.rlt_command == "export":
        return export_dataset(args.suite, args.rollouts, args.out, include_controls=args.include_controls)
    if args.rlt_command == "frontier":
        return mine_frontier(args.suite, args.rollouts, args.out, max_tasks=args.max_tasks)
    raise ValueError("unknown RLT gym command")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datagoblin rlt-gym")
    add_arguments(parser)
    try:
        result = dispatch(parser.parse_args(argv))
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (KeyError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"datagoblin rlt-gym: {exc}", file=sys.stderr)
        return 2
