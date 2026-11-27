"""CLI for Research Verifiers v2.1: Controller-Assisted Research and Externalized Persona."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagoblin.research_v2.runner import DEFAULT_REGISTRY

from .controllers import CONTROLLER_MODES
from .runner import compare, run_control, run_endpoint
from .suite import canon, load_suite, write_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--split", choices=("development", "heldout"), default="development")
    build.add_argument("--seed", type=int)
    build.add_argument("--per-family", type=int, default=4)
    build.add_argument("--exclude-registry", type=Path, action="append", default=[])

    check = commands.add_parser("validate")
    check.add_argument("--suite", type=Path, required=True)

    control = commands.add_parser("control")
    control.add_argument("--suite", type=Path, required=True)
    control.add_argument("--out", type=Path, required=True)
    control.add_argument("--policy", choices=("reference", "noop"), default="reference")
    control.add_argument("--max-episodes", type=int, default=8)

    run = commands.add_parser("run")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--binding", type=Path, required=True)
    run.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    run.add_argument("--max-episodes", type=int, default=32)
    run.add_argument("--controller-mode", choices=CONTROLLER_MODES, default="autonomous")
    run.add_argument("--max-turns-per-episode", type=int, default=8)
    run.add_argument("--run-wall-seconds", type=float, default=None)

    cmp_cmd = commands.add_parser("compare")
    cmp_cmd.add_argument("reports", nargs="+", type=Path)
    cmp_cmd.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "build":
        excluded: set[str] = set()
        for path in args.exclude_registry:
            data = json.loads(path.read_text())
            excluded.update(data.get("held_root_ids", []))
            excluded.update(data.get("held_semantic_sha256", []))
        result = write_suite(
            args.out, seed=args.seed, per_family=args.per_family, split=args.split, exclusions=excluded
        )
    elif args.command == "validate":
        cases = load_suite(args.suite)
        result = {"verified_cases": len(cases), "families": sorted({c["family"] for c in cases})}
    elif args.command == "control":
        result = run_control(args.suite, args.out, control=args.policy, max_episodes=args.max_episodes)
    elif args.command == "run":
        result = run_endpoint(
            args.suite,
            args.out,
            model_id=args.model_id,
            binding_path=args.binding,
            registry_path=args.registry,
            max_episodes=args.max_episodes,
            controller_mode=args.controller_mode,
            max_turns_per_episode=args.max_turns_per_episode,
            run_wall_seconds=args.run_wall_seconds,
        )
    else:
        result = compare(args.reports)
        if args.out.exists():
            raise ValueError("comparison output must be fresh")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(canon(result) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
