from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from datagoblin.data_etiquette.evidence_receipts import EvidenceReceiptStore

from .corpus_projection import build_corpus_projection, build_safe_field_projection
from .drift_benchmark import run_drift_confirmation, run_drift_gym
from .environment import (
    compare_adapter_reports,
    evaluate_adapter_setup,
    run_mock_setups_on_dataset,
    run_verifier_demo,
)
from .evidence_replay import verify_semantic_evidence
from .semantic_review import build_semantic_review_queue, finalize_semantic_review
from .skill_tournament import run_curation_skill_tournament_succession
from .tournament import run_adapter_tournament, run_adapter_tournament_confirmation


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

    drift = subparsers.add_parser(
        "drift-gym",
        help="fit and compare a shape-only learned adapter against the frozen exact-key baseline",
    )
    drift.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preregistrations/game_adapter_schema_drift_v1.yaml"),
    )
    drift.add_argument("--out", type=Path, required=True)
    drift.add_argument("--seed", type=int, default=101)

    confirmation = subparsers.add_parser(
        "drift-confirm",
        help="run the preregistered pilot and fresh-seed schema-drift confirmation",
    )
    confirmation.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preregistrations/game_adapter_schema_drift_v1.yaml"),
    )
    confirmation.add_argument("--out", type=Path, required=True)

    tournament = subparsers.add_parser(
        "tournament",
        help="run normal, compact, and coverage-pressure adapters on one four-game fixture",
    )
    tournament.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preregistrations/game_adapter_tournament_v1.yaml"),
    )
    tournament.add_argument("--out", type=Path, required=True)
    tournament.add_argument("--seed", type=int, default=401)

    tournament_confirmation = subparsers.add_parser(
        "tournament-confirm",
        help="run pilot and different-surface four-game adapter tournaments",
    )
    tournament_confirmation.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preregistrations/game_adapter_tournament_v1.yaml"),
    )
    tournament_confirmation.add_argument("--out", type=Path, required=True)

    succession = subparsers.add_parser(
        "skill-tournament-succession",
        help="run pressure-varied tournaments and emit an LDT curation-taste foundation",
    )
    succession.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/preregistrations/curation_skill_tournament_succession_v1.yaml"
        ),
    )
    succession.add_argument("--skill-before-fingerprint", type=Path, required=True)
    succession.add_argument("--skill-after-fingerprint", type=Path, required=True)
    succession.add_argument("--candidate-validator", type=Path, required=True)
    succession.add_argument("--out", type=Path, required=True)

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
    finalize.add_argument("--evidence-registry", type=Path)
    finalize.add_argument("--evidence-receipts", type=Path)
    finalize.add_argument("--out", type=Path, required=True)

    replay = subparsers.add_parser(
        "verify-semantic-evidence",
        help="run private environment replays and issue claim-scoped evidence receipts",
    )
    replay.add_argument("--challenges", type=Path, required=True)
    replay.add_argument("--review-map", type=Path, required=True)
    replay.add_argument("--derivation-ledger", type=Path, required=True)
    replay.add_argument("--evidence-registry", type=Path, required=True)
    replay.add_argument("--method-specs", type=Path)
    replay.add_argument("--approved-root", type=Path, action="append", required=True)
    replay.add_argument("--aicourt-corpus-root", type=Path, required=True)
    replay.add_argument("--centauri-root", type=Path, required=True)
    replay.add_argument("--they-sing-root", type=Path, required=True)
    replay.add_argument("--diplomacy-engine-root", type=Path, required=True)
    replay.add_argument("--diplomacy-corpus-root", type=Path, action="append", required=True)
    replay.add_argument(
        "--court-engine-snapshot",
        action="append",
        default=[],
        metavar="VERSION=ENGINE_PATH",
    )
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--max-record-bytes", type=int, default=2_000_000)
    replay.add_argument("--timeout-seconds", type=int, default=180)
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
        elif args.command == "drift-gym":
            comparison = run_drift_gym(args.config, args.out, seed=args.seed)
            print(args.out / "comparison.md")
            return (
                0
                if comparison["promotion_assessment"]["eligible_for_next_architecture_gate"]
                else 3
            )
        elif args.command == "drift-confirm":
            confirmation = run_drift_confirmation(args.config, args.out)
            print(args.out / "confirmation_report.json")
            return 0 if confirmation["confirmed_for_next_architecture_gate"] else 3
        elif args.command == "tournament":
            tournament = run_adapter_tournament(
                args.config,
                args.out,
                seed=args.seed,
            )
            print(args.out / "tournament_report.json")
            return 0 if tournament["qualifiers"] else 3
        elif args.command == "tournament-confirm":
            confirmation = run_adapter_tournament_confirmation(args.config, args.out)
            print(args.out / "tournament_confirmation.json")
            return (
                0
                if confirmation["confirmed_for_checkpoint_candidate_admission"]
                else 3
            )
        elif args.command == "skill-tournament-succession":
            succession = run_curation_skill_tournament_succession(
                args.config,
                args.skill_before_fingerprint,
                args.skill_after_fingerprint,
                args.candidate_validator,
                args.out,
            )
            print(args.out / "succession_report.json")
            return 0 if succession["succession_passed"] else 3
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
        elif args.command == "verify-semantic-evidence":
            snapshots = _parse_engine_snapshots(args.court_engine_snapshot)
            verify_semantic_evidence(
                args.challenges,
                args.review_map,
                args.derivation_ledger,
                args.evidence_registry,
                args.approved_root,
                args.out,
                aicourt_corpus_root=args.aicourt_corpus_root,
                centauri_root=args.centauri_root,
                they_sing_root=args.they_sing_root,
                diplomacy_engine_root=args.diplomacy_engine_root,
                diplomacy_corpus_roots=args.diplomacy_corpus_root,
                court_engine_snapshots=snapshots,
                method_specs_dir=args.method_specs,
                max_record_bytes=args.max_record_bytes,
                timeout_seconds=args.timeout_seconds,
            )
            print(args.out / "evidence_replay_report.json")
        else:
            if (args.evidence_registry is None) != (args.evidence_receipts is None):
                raise ValueError("evidence registry and receipt store must be supplied together")
            evidence_store = (
                EvidenceReceiptStore.from_jsonl(args.evidence_registry, args.evidence_receipts)
                if args.evidence_registry is not None
                else None
            )
            finalize_semantic_review(
                args.review_manifest,
                args.review_map,
                args.annotations,
                args.near_duplicate_audit,
                args.out,
                evidence_store=evidence_store,
            )
            print(args.out / "review_report.json")
    except (KeyError, ValueError, OSError) as exc:
        print(f"adapter-verifier: {exc}", file=sys.stderr)
        return 2
    return 0


def _parse_engine_snapshots(values: Sequence[str]) -> dict[str, Path]:
    snapshots: dict[str, Path] = {}
    for value in values:
        version, separator, path = value.partition("=")
        if not separator or not version or not path or version in snapshots:
            raise ValueError("court engine snapshots must be unique VERSION=ENGINE_PATH values")
        snapshots[version] = Path(path)
    return snapshots


if __name__ == "__main__":
    raise SystemExit(main())
