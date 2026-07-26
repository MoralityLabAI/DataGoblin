#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagoblin.experiments.ldt_shift_challenge import (
    run_shift_candidates,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen candidate setups on the public LDT shift"
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--shift-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--trainer-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    if arguments.model_root.resolve().is_relative_to(repository_root):
        raise SystemExit("model weights must remain outside the repository")
    if arguments.out.resolve().is_relative_to(repository_root):
        raise SystemExit("candidate outputs must remain outside the repository")
    receipt = run_shift_candidates(
        contract_path=arguments.contract,
        manifest_path=arguments.shift_manifest,
        model_root=arguments.model_root,
        trainer_summary_path=arguments.trainer_summary,
        out_dir=arguments.out,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "setup_count": receipt["setup_count"],
                "verifier_targets_seen": receipt[
                    "verifier_targets_seen"
                ],
            },
            sort_keys=True,
        )
    )
