#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagoblin.experiments.ldt_shadow import run_shadow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen LDT as a non-authoritative shadow"
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--shadow-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trainer-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    if arguments.model.resolve().is_relative_to(repository_root):
        raise SystemExit("model weights must remain outside the repository")
    if arguments.out.resolve().is_relative_to(repository_root):
        raise SystemExit("generated shadow runs must remain outside the repository")
    receipt = run_shadow(
        contract_path=arguments.contract,
        manifest_path=arguments.shadow_manifest,
        model_path=arguments.model,
        trainer_summary_path=arguments.trainer_summary,
        out_dir=arguments.out,
    )
    print(json.dumps(receipt, sort_keys=True))
