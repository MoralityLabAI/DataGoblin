#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagoblin.experiments.ldt_shift_challenge import (
    evaluate_shift_challenge,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join immutable LDT shift outputs with sealed verifier labels"
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--shift-manifest", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--verifier-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    if arguments.out.resolve().is_relative_to(repository_root):
        raise SystemExit("comparison outputs must remain outside the repository")
    report = evaluate_shift_challenge(
        contract_path=arguments.contract,
        manifest_path=arguments.shift_manifest,
        candidate_run_dir=arguments.candidate_run_dir,
        verifier_dir=arguments.verifier_dir,
        out_dir=arguments.out,
    )
    print(
        json.dumps(
            {
                "challenge_execution_passed": report[
                    "challenge_execution_passed"
                ],
                "qualified_setups": report["qualified_setups"],
                "policy_lattice_reliance_detected": report[
                    "causal_ablation"
                ]["policy_lattice_reliance_detected"],
                "review_queue_count": report["review_queue_count"],
            },
            sort_keys=True,
        )
    )
