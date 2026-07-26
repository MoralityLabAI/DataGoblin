#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagoblin.experiments.ldt_shadow_succession import (
    evaluate_shadow_succession,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently join LDT shadow succession verifier evidence"
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--succession-manifest", type=Path, required=True)
    parser.add_argument("--shadow-run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    if arguments.out.resolve().is_relative_to(repository_root):
        raise SystemExit("comparison outputs must remain outside the repository")
    report = evaluate_shadow_succession(
        contract_path=arguments.contract,
        succession_manifest_path=arguments.succession_manifest,
        shadow_run_dir=arguments.shadow_run_dir,
        out_dir=arguments.out,
    )
    print(
        json.dumps(
            {
                "succession_passed": report["succession_passed"],
                "round_count": report["summary"]["round_count"],
                "review_queue_count": report["summary"][
                    "review_queue_count"
                ],
            },
            sort_keys=True,
        )
    )
