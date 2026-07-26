#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagoblin.experiments.ldt_shadow import evaluate_shadow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join LDT shadow recommendations with verifier custody"
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--shadow-manifest", type=Path, required=True)
    parser.add_argument("--pair-recommendations", type=Path, required=True)
    parser.add_argument("--array-recommendations", type=Path, required=True)
    parser.add_argument("--shadow-run-receipt", type=Path, required=True)
    parser.add_argument("--verifier-pair-labels", type=Path, required=True)
    parser.add_argument("--verifier-array-labels", type=Path, required=True)
    parser.add_argument("--supersedes-comparison", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    report = evaluate_shadow(
        contract_path=arguments.contract,
        manifest_path=arguments.shadow_manifest,
        pair_recommendations_path=arguments.pair_recommendations,
        array_recommendations_path=arguments.array_recommendations,
        shadow_run_receipt_path=arguments.shadow_run_receipt,
        verifier_pair_labels_path=arguments.verifier_pair_labels,
        verifier_array_labels_path=arguments.verifier_array_labels,
        out_dir=arguments.out,
        supersedes_comparison_path=arguments.supersedes_comparison,
    )
    print(
        json.dumps(
            {
                "monitoring_passed": report["monitoring_passed"],
                "review_queue_count": report["review_queue_count"],
            },
            sort_keys=True,
        )
    )
