#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagoblin.experiments.ldt_shift_verifier import (
    seal_shift_verifier_labels,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal independent LDT shift verifier labels"
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--shift-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    if arguments.out.resolve().is_relative_to(repository_root):
        raise SystemExit("sealed verifier labels must remain outside the repository")
    receipt = seal_shift_verifier_labels(
        contract_path=arguments.contract,
        manifest_path=arguments.shift_manifest,
        out_dir=arguments.out,
    )
    print(
        json.dumps(
            {
                "seal_passed": receipt["seal_passed"],
                "pair_count": receipt["pair_count"],
                "array_count": receipt["array_count"],
                "preference_distribution": receipt[
                    "preference_distribution"
                ],
            },
            sort_keys=True,
        )
    )
