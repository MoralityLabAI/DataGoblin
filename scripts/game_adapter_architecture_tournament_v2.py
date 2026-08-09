from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagoblin.adapter_verifier.architecture_tournament import (
    apply_v15_shadow_sorter,
    audit_architecture_confirmation_packet,
    build_architecture_tournament_fixture,
    evaluate_architecture_development,
    evaluate_architecture_heldout,
    train_architecture_candidate,
    validate_architecture_candidate,
)


def _paths(values: list[Path]) -> list[Path]:
    return [value.resolve() for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the v2 four-game architecture tournament")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", required=True, type=Path)
    generate.add_argument("--out", required=True, type=Path)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--dataset-manifest", required=True, type=Path)
    train.add_argument("--release-manifest", required=True, type=Path)
    train.add_argument("--candidate-id", required=True)
    train.add_argument("--seed", required=True, type=int)
    train.add_argument("--out", required=True, type=Path)

    preflight = subparsers.add_parser("preflight-candidate")
    preflight.add_argument("--config", required=True, type=Path)
    preflight.add_argument("--dataset-manifest", required=True, type=Path)
    preflight.add_argument("--candidate-run", required=True, type=Path)
    preflight.add_argument("--allow-uncapped-test", action="store_true")

    development = subparsers.add_parser("evaluate-development")
    development.add_argument("--config", required=True, type=Path)
    development.add_argument("--dataset-manifest", required=True, type=Path)
    development.add_argument("--custody-manifest", required=True, type=Path)
    development.add_argument("--candidate-run", action="append", required=True, type=Path)
    development.add_argument("--allow-uncapped-test", action="store_true")
    development.add_argument("--out", required=True, type=Path)

    shadow = subparsers.add_parser("shadow-sort")
    shadow.add_argument("--config", required=True, type=Path)
    shadow.add_argument("--development-evaluation", required=True, type=Path)
    shadow.add_argument("--development-arrays", required=True, type=Path)
    shadow.add_argument("--retained-model", required=True, type=Path)
    shadow.add_argument("--out", required=True, type=Path)

    audit = subparsers.add_parser("audit-packet")
    audit.add_argument("--config", required=True, type=Path)
    audit.add_argument("--dataset-manifest", required=True, type=Path)
    audit.add_argument("--development-evaluation", required=True, type=Path)
    audit.add_argument("--development-arrays", required=True, type=Path)
    audit.add_argument("--shadow-predictions", required=True, type=Path)
    audit.add_argument("--shadow-receipt", required=True, type=Path)
    audit.add_argument("--candidate-run", action="append", required=True, type=Path)
    audit.add_argument("--allow-uncapped-test", action="store_true")
    audit.add_argument("--out", required=True, type=Path)

    heldout = subparsers.add_parser("evaluate-heldout")
    heldout.add_argument("--config", required=True, type=Path)
    heldout.add_argument("--dataset-manifest", required=True, type=Path)
    heldout.add_argument("--custody-manifest", required=True, type=Path)
    heldout.add_argument("--development-evaluation", required=True, type=Path)
    heldout.add_argument("--packet-audit", required=True, type=Path)
    heldout.add_argument("--shadow-predictions", required=True, type=Path)
    heldout.add_argument("--candidate-run", action="append", required=True, type=Path)
    heldout.add_argument("--allow-uncapped-test", action="store_true")
    heldout.add_argument("--out", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "generate":
        result = build_architecture_tournament_fixture(args.config.resolve(), args.out.resolve())
    elif args.command == "train":
        result = train_architecture_candidate(
            args.config.resolve(),
            args.dataset_manifest.resolve(),
            args.release_manifest.resolve(),
            args.out.resolve(),
            candidate_id=args.candidate_id,
            seed=args.seed,
        )
    elif args.command == "preflight-candidate":
        result = validate_architecture_candidate(
            args.config.resolve(),
            args.dataset_manifest.resolve(),
            args.candidate_run.resolve(),
            require_resource_receipt=not args.allow_uncapped_test,
        )
    elif args.command == "evaluate-development":
        result = evaluate_architecture_development(
            args.config.resolve(),
            args.dataset_manifest.resolve(),
            args.custody_manifest.resolve(),
            _paths(args.candidate_run),
            args.out.resolve(),
            require_resource_receipts=not args.allow_uncapped_test,
        )
    elif args.command == "shadow-sort":
        result = apply_v15_shadow_sorter(
            args.config.resolve(),
            args.development_evaluation.resolve(),
            args.development_arrays.resolve(),
            args.retained_model.resolve(),
            args.out.resolve(),
        )
    elif args.command == "audit-packet":
        result = audit_architecture_confirmation_packet(
            args.config.resolve(),
            args.dataset_manifest.resolve(),
            args.development_evaluation.resolve(),
            args.development_arrays.resolve(),
            args.shadow_predictions.resolve(),
            args.shadow_receipt.resolve(),
            _paths(args.candidate_run),
            args.out.resolve(),
            require_resource_receipts=not args.allow_uncapped_test,
        )
    else:
        result = evaluate_architecture_heldout(
            args.config.resolve(),
            args.dataset_manifest.resolve(),
            args.custody_manifest.resolve(),
            args.development_evaluation.resolve(),
            args.packet_audit.resolve(),
            args.shadow_predictions.resolve(),
            _paths(args.candidate_run),
            args.out.resolve(),
            require_resource_receipts=not args.allow_uncapped_test,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
