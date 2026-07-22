from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .corpus_projection import build_corpus_projection, build_safe_field_projection
from .environment import (
    compare_adapter_reports,
    evaluate_adapter_setup,
    run_mock_setups_on_dataset,
    run_verifier_demo,
)
from .semantic_review import build_semantic_review_queue, finalize_semantic_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate game-corpus adapter setups in a sealed verifier environment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run deterministic mock setups against a synthetic four-game fixture")
    demo.add_argument("--config", type=Path, default=Path("configs/curation_adapter_verifier.yaml"))
    demo.add_argument("--out", type=Path, default=Path("runs/adapter-verifier-demo"))
    demo.add_argument("--seed", type=int, default=7)

    project = subparsers.add_parser(
        "project-corpus", help="build a bounded content-free projection of approved exports"
    )
    project.add_argument("--config", type=Path, default=Path("configs/curation_adapter_verifier.yaml"))
    project.add_argument("--aicourt-receipt", type=Path, required=True)
    project.add_argument("--aicourt-decisions", type=Path, required=True)
    project.add_argument("--diplomacy-root", type=Path, action="append", required=True)
    project.add_argument("--diplomacy-manifest", type=Path, action="append", required=True)
    project.add_argument("--out", type=Path, required=True)
    project.add_argument("--seed", type=int, default=29)
    project.add_argument("--max-files-per-root", type=int, default=64)
    project.add_argument("--max-records-per-file", type=int, default=64)
    project.add_argument("--max-records-per-cohort", type=int, default=24)
    project.add_argument("--max-record-bytes", type=int, default=1_000_000)

    safe_project = subparsers.add_parser(
        "project-safe-fields", help="build an environment-verified safe-field derivation gym lane"
    )
    safe_project.add_argument("--config", type=Path, default=Path("configs/curation_adapter_verifier.yaml"))
    safe_project.add_argument("--aicourt-receipt", type=Path, required=True)
    safe_project.add_argument("--aicourt-decisions", type=Path, required=True)
    safe_project.add_argument("--diplomacy-root", type=Path, action="append", required=True)
    safe_project.add_argument("--diplomacy-manifest", type=Path, action="append", required=True)
    safe_project.add_argument("--out", type=Path, required=True)
    safe_project.add_argument("--seed", type=int, default=41)
    safe_project.add_argument("--max-files-per-root", type=int, default=64)
    safe_project.add_argument("--max-records-per-file", type=int, default=64)
    safe_project.add_argument("--max-records-per-cohort", type=int, default=16)
    safe_project.add_argument("--max-record-bytes", type=int, default=1_000_000)

    mocks = subparsers.add_parser("mock-setups", help="run deterministic controls on a frozen projected dataset")
    mocks.add_argument("--config", type=Path, required=True)
    mocks.add_argument("--dataset-manifest", type=Path, required=True)
    mocks.add_argument("--steward-labels", type=Path, required=True)
    mocks.add_argument("--out", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="score one adapter setup against stewarded labels")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--dataset-manifest", type=Path, required=True)
    evaluate.add_argument("--steward-labels", type=Path, required=True)
    evaluate.add_argument("--candidate-manifest", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)

    compare = subparsers.add_parser("compare", help="compare two or more adapter evaluation reports")
    compare.add_argument("--report", type=Path, action="append", required=True)
    compare.add_argument("--out", type=Path, required=True)

    review = subparsers.add_parser(
        "build-semantic-review",
        help="build a bounded two-reviewer semantic gate from a safe-field projection",
    )
    review.add_argument("--config", type=Path, default=Path("configs/semantic_review.yaml"))
    review.add_argument("--dataset-manifest", type=Path, required=True)
    review.add_argument("--derivation-ledger", type=Path, required=True)
    review.add_argument("--approved-root", type=Path, action="append", required=True)
    review.add_argument("--out", type=Path, required=True)
    review.add_argument("--seed", type=int, default=73)
    review.add_argument("--max-items", type=int, default=24)
    review.add_argument("--max-audit-items", type=int, default=512)
    review.add_argument("--max-per-source-file", type=int, default=2)
    review.add_argument("--max-record-bytes", type=int, default=1_000_000)

    finalize = subparsers.add_parser(
        "finalize-semantic-review",
        help="validate two blind annotations and emit a content-free agreement report",
    )
    finalize.add_argument("--review-manifest", type=Path, required=True)
    finalize.add_argument("--review-map", type=Path, required=True)
    finalize.add_argument("--annotations", type=Path, action="append", required=True)
    finalize.add_argument("--near-duplicate-audit", type=Path, required=True)
    finalize.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            run_verifier_demo(args.config, args.out, seed=args.seed)
            print(args.out / "comparison.md")
        elif args.command == "project-corpus":
            build_corpus_projection(
                args.config,
                args.aicourt_receipt,
                args.aicourt_decisions,
                args.diplomacy_root,
                args.diplomacy_manifest,
                args.out,
                seed=args.seed,
                max_files_per_root=args.max_files_per_root,
                max_records_per_file=args.max_records_per_file,
                max_records_per_cohort=args.max_records_per_cohort,
                max_record_bytes=args.max_record_bytes,
            )
            print(args.out / "public_dataset" / "dataset_manifest.json")
        elif args.command == "project-safe-fields":
            build_safe_field_projection(
                args.config,
                args.aicourt_receipt,
                args.aicourt_decisions,
                args.diplomacy_root,
                args.diplomacy_manifest,
                args.out,
                seed=args.seed,
                max_files_per_root=args.max_files_per_root,
                max_records_per_file=args.max_records_per_file,
                max_records_per_cohort=args.max_records_per_cohort,
                max_record_bytes=args.max_record_bytes,
            )
            print(args.out / "public_dataset" / "dataset_manifest.json")
        elif args.command == "mock-setups":
            run_mock_setups_on_dataset(
                args.config,
                args.dataset_manifest,
                args.steward_labels,
                args.out,
            )
            print(args.out / "comparison.md")
        elif args.command == "evaluate":
            report = evaluate_adapter_setup(
                args.config,
                args.dataset_manifest,
                args.steward_labels,
                args.candidate_manifest,
                args.predictions,
                args.out,
            )
            print(args.out / "evaluation_report.json")
            return 0 if report["promotion_assessment"]["eligible"] else 3
        elif args.command == "compare":
            compare_adapter_reports(args.report, args.out)
            print(args.out / "comparison.md")
        elif args.command == "build-semantic-review":
            build_semantic_review_queue(
                args.config,
                args.dataset_manifest,
                args.derivation_ledger,
                args.approved_root,
                args.out,
                seed=args.seed,
                max_items=args.max_items,
                max_audit_items=args.max_audit_items,
                max_per_source_file=args.max_per_source_file,
                max_record_bytes=args.max_record_bytes,
            )
            print(args.out / "public_review" / "review_manifest.json")
        else:
            finalize_semantic_review(
                args.review_manifest,
                args.review_map,
                args.annotations,
                args.near_duplicate_audit,
                args.out,
            )
            print(args.out / "review_report.json")
    except (KeyError, ValueError, OSError) as exc:
        print(f"adapter-verifier: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
