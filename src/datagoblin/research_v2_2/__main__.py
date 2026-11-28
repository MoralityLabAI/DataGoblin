"""Run the v2.2 controller comparison over validated v2.1 tasks."""

import argparse
import json
from pathlib import Path

from .controllers import CONTROLLER_MODES
from .runner import compare, run_endpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("suite", "out", "registry", "binding"):
        run.add_argument(f"--{name}", required=True, type=Path)
    run.add_argument("--exclusion-registry", type=Path)
    run.add_argument("--model-id", required=True)
    run.add_argument("--controller-mode", choices=CONTROLLER_MODES, required=True)
    run.add_argument("--max-episodes", type=int, default=32)
    run.add_argument("--max-turns", type=int, default=8)
    run.add_argument("--wall-seconds", type=float, default=1800)
    cmp = sub.add_parser("compare")
    cmp.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        result = run_endpoint(
            args.suite, args.out, model_id=args.model_id,
            binding_path=args.binding, registry_path=args.registry,
            controller_mode=args.controller_mode, max_episodes=args.max_episodes,
            max_turns_per_episode=args.max_turns, run_wall_seconds=args.wall_seconds,
            exclusion_registry=args.exclusion_registry,
        )
    else:
        result = compare(args.reports)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
