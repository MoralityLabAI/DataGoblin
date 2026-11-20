from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datagoblin.adapters.upstream_registry import UpstreamRegistry, discover_upstreams
from datagoblin.builders.ticket_writer import write_tickets
from datagoblin.config import repository_root
from datagoblin.constitution.validator import ConstitutionalValidator, DecisionKind
from datagoblin.data_etiquette import (
    audit_sorter_candidate,
    build_etiquette_training_dataset,
    evaluate_sorter_predictions,
    run_etiquette_demo,
)
from datagoblin.datasets.exports import (
    export_outcome_prediction_jsonl,
    export_preference_jsonl,
    export_sft_jsonl,
    export_verifier_jsonl,
    read_episodes,
)
from datagoblin.domain.learner import LearnerSnapshot
from datagoblin.domain.recipes import DataRecipe
from datagoblin.domain.research_episode import ResearchEpisode
from datagoblin.experiments.cycle_uncertainty import evaluate_cycle_signal
from datagoblin.experiments.recipe_verdict_dataset import (
    generate_verdict_dataset,
    read_dataset_rows,
    steward_sealed_labels,
)
from datagoblin.experiments.recurrent_baselines import (
    compare_recurrent_reports,
    evaluate_recurrent_baseline_suite,
    trace_recurrent_baseline_suite,
)
from datagoblin.experiments.role_conditioning import run_role_conditioning_ablation
from datagoblin.hashing import sha256_file
from datagoblin.hrm_text.availability import detect_hrm_availability
from datagoblin.hrm_text.instrumented_forward import (
    DeterministicMockRecurrentModel,
    TraceOptions,
    TransformersHRMTextModel,
    run_instrumented_forward,
)
from datagoblin.hrm_text.preflight import HardCaps, run_checkpoint_preflight
from datagoblin.hrm_text.prompting import ROLE_CONDITIONS, build_verdict_prompt, schema_complexity
from datagoblin.ledger.database import ExperimentLedger
from datagoblin.reporting.markdown import experiment_markdown
from datagoblin.research.demo import run_demo
from datagoblin.research.heuristic_backend import HeuristicGoblinBackend
from datagoblin.research.orchestrator import ResearchOrchestrator, emit_builder_tickets
from datagoblin.serialization import dump_structured, load_structured
from datagoblin.training_gym import compare_candidates, evaluate_candidate, run_training_gym_demo


def _default_snapshot_path() -> Path:
    return repository_root() / "examples" / "learner_snapshot.delayed_causality.json"


def _load_snapshot(path: Path) -> LearnerSnapshot:
    return LearnerSnapshot.from_jsonable(load_structured(path))


def _write_json_stdout(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datagoblin")
    sub = parser.add_subparsers(dest="command", required=True)
    from datagoblin.training_gym.rlt_gym.cli import add_arguments as add_rlt_gym_arguments

    add_rlt_gym_arguments(sub.add_parser("rlt-gym", help="Collect checked RLT rollouts and teacher training traces"))
    sub.add_parser("discover-upstreams")
    status = sub.add_parser("upstream-status")
    status.add_argument("--config", type=Path)
    ingest = sub.add_parser("ingest-upstreams")
    ingest.add_argument("--config", type=Path, default=repository_root() / "configs/upstreams.example.yaml")
    ingest.add_argument("--out", type=Path, default=Path("context/external"))
    ingest.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    context = sub.add_parser("make-context")
    context.add_argument("--out", type=Path, default=Path("context/context_bundle.md"))
    context.add_argument("--include-external", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("--recipe", type=Path, required=True)
    propose = sub.add_parser("propose")
    propose.add_argument("--snapshot", type=Path, required=True)
    propose.add_argument("--out", type=Path, required=True)
    tickets = sub.add_parser("emit-tickets")
    tickets.add_argument("--plan", type=Path, required=True)
    tickets.add_argument("--out-dir", type=Path, required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--out", type=Path, default=Path("runs/demo"))
    demo.add_argument("--snapshot", type=Path, default=_default_snapshot_path())
    report = sub.add_parser("report")
    report.add_argument("--experiment-id", required=True)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--ledger", type=Path, default=Path("runs/demo/ledger.sqlite"))
    synth = sub.add_parser("synth-episodes")
    synth.add_argument("--count", type=int, default=32)
    synth.add_argument("--seed", type=int, default=7)
    synth.add_argument("--out", type=Path, required=True)
    for name in ("export-sft", "export-preferences", "export-verifier", "export-outcomes"):
        export = sub.add_parser(name)
        export.add_argument("--input", type=Path, required=True)
        export.add_argument("--out", type=Path, required=True)
    status_hrm = sub.add_parser("hrm-status")
    status_hrm.add_argument("--local-model-path", type=Path)
    status_hrm.add_argument("--model-id")
    verdict_dataset = sub.add_parser("make-verdict-dataset")
    verdict_dataset.add_argument("--recipe", type=Path, default=repository_root() / "examples/recipe.good.yaml")
    verdict_dataset.add_argument("--seed", type=int, default=7)
    verdict_dataset.add_argument("--out", type=Path, required=True)
    steward = sub.add_parser("steward-sealed-labels")
    steward.add_argument("--dataset-manifest", type=Path, required=True)
    steward.add_argument("--out", type=Path, required=True)
    trace = sub.add_parser("trace-hrm")
    trace.add_argument("--dataset-manifest", type=Path, required=True)
    trace.add_argument("--out", type=Path, required=True)
    trace.add_argument("--config", type=Path, default=repository_root() / "configs/hrm_text_trace.example.yaml")
    trace.add_argument("--preregistration", type=Path, default=_default_preregistration_path())
    trace.add_argument(
        "--maximum-examples",
        type=int,
        help="bounded smoke-run limit; omitted means trace every manifest row",
    )
    _add_trace_model_arguments(trace)
    evaluate = sub.add_parser("evaluate-cycle-signal")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--sealed-labels", type=Path, required=True)
    evaluate.add_argument("--calibration", choices=("logistic", "isotonic"), default="logistic")
    evaluate.add_argument("--bootstrap-samples", type=int, default=200)
    evaluate.add_argument("--seed", type=int, default=7)
    roles = sub.add_parser("role-ablation")
    roles.add_argument("--dataset-manifest", type=Path, required=True)
    roles.add_argument("--out", type=Path, required=True)
    roles.add_argument("--config", type=Path, default=repository_root() / "configs/hrm_text_trace.example.yaml")
    roles.add_argument("--maximum-examples", type=int, default=12)
    _add_trace_model_arguments(roles)
    baselines = sub.add_parser("trace-recurrent-baselines")
    baselines.add_argument("--dataset-manifest", type=Path, required=True)
    baselines.add_argument("--out", type=Path, required=True)
    baselines.add_argument("--config", type=Path, default=repository_root() / "configs/hrm_text_trace.example.yaml")
    baselines.add_argument("--sealed-labels", type=Path, required=True)
    baselines.add_argument("--condition", choices=ROLE_CONDITIONS, default="direct")
    baselines.add_argument("--calibration", choices=("logistic", "isotonic"), default="logistic")
    baselines.add_argument("--bootstrap-samples", type=int, default=200)
    baselines.add_argument("--seed", type=int, default=7)
    compare = sub.add_parser("compare-recurrent-baselines")
    compare.add_argument("--primary-run", type=Path, required=True)
    compare.add_argument("--baseline-root", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    preflight = sub.add_parser("hrm-preflight")
    preflight.add_argument("--local-model-path", type=Path, required=True)
    preflight.add_argument("--out", type=Path, required=True)
    preflight.add_argument("--run-id", default="hrm-text-cycle-real-v1-ram4096")
    preflight.add_argument("--ram-mb", type=int, default=4096)
    preflight.add_argument("--cpu-percent", type=int, default=50)
    preflight.add_argument("--io-mb-s", type=int, default=50)
    preflight.add_argument("--wall-clock-seconds", type=int, default=1800)
    gym_demo = sub.add_parser("gym-demo")
    gym_demo.add_argument(
        "--contract",
        type=Path,
        default=repository_root() / "configs/preregistrations/training_gym_v1.yaml",
    )
    gym_demo.add_argument("--recipe", type=Path, default=repository_root() / "examples/recipe.good.yaml")
    gym_demo.add_argument("--seed", type=int, default=7)
    gym_demo.add_argument("--out", type=Path, default=Path("runs/training-gym-demo"))
    gym_evaluate = sub.add_parser("gym-evaluate")
    gym_evaluate.add_argument("--contract", type=Path, required=True)
    gym_evaluate.add_argument("--dataset-manifest", type=Path, required=True)
    gym_evaluate.add_argument("--candidate-manifest", type=Path, required=True)
    gym_evaluate.add_argument("--predictions", type=Path, required=True)
    gym_evaluate.add_argument("--sealed-labels", type=Path)
    gym_evaluate.add_argument("--out", type=Path, required=True)
    gym_compare = sub.add_parser("gym-compare")
    gym_compare.add_argument("--contract", type=Path, required=True)
    gym_compare.add_argument("--report", type=Path, action="append", required=True)
    gym_compare.add_argument("--baseline-candidate-id", required=True)
    gym_compare.add_argument("--out", type=Path, required=True)
    etiquette_demo = sub.add_parser("etiquette-demo")
    etiquette_demo.add_argument(
        "--policy",
        type=Path,
        default=repository_root() / "configs/data_etiquette.yaml",
    )
    etiquette_demo.add_argument("--seed", type=int, default=7)
    etiquette_demo.add_argument("--out", type=Path, default=Path("runs/etiquette-demo"))
    etiquette_evaluate = sub.add_parser("etiquette-evaluate")
    etiquette_evaluate.add_argument("--dataset-manifest", type=Path, required=True)
    etiquette_evaluate.add_argument("--predictions", type=Path, required=True)
    etiquette_evaluate.add_argument("--sealed-labels", type=Path, required=True)
    etiquette_evaluate.add_argument("--out", type=Path, required=True)
    etiquette_build = sub.add_parser("etiquette-build-training-dataset")
    etiquette_build.add_argument(
        "--policy",
        type=Path,
        default=repository_root() / "configs/data_etiquette.yaml",
    )
    etiquette_build.add_argument("--seed", type=int, default=7)
    etiquette_build.add_argument("--out", type=Path, required=True)
    etiquette_audit = sub.add_parser("etiquette-audit-candidate")
    etiquette_audit.add_argument("--evaluation-report", type=Path, required=True)
    etiquette_audit.add_argument("--run-receipt", type=Path, required=True)
    etiquette_audit.add_argument(
        "--policy",
        type=Path,
        default=repository_root() / "configs/data_etiquette.yaml",
    )
    etiquette_audit.add_argument("--out", type=Path, required=True)
    return parser


def _default_preregistration_path() -> Path:
    return repository_root() / "configs/preregistrations/cycle_uncertainty_v1.yaml"


def _add_trace_model_arguments(parser: argparse.ArgumentParser) -> None:
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--mock", action="store_true", help="use deterministic test model; not HRM evidence")
    model.add_argument("--local-model-path", type=Path)
    model.add_argument("--model-id")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="explicitly permit checkpoint-provided Python code; review the source first",
    )
    parser.add_argument(
        "--confirm-hard-cap-wrapper",
        action="store_true",
        help="confirm this process is already under an operator-provided RAM/CPU/IO cap",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--condition", choices=ROLE_CONDITIONS, default="direct")


def _make_context(path: Path, include_external: bool) -> None:
    include = [
        "README.md",
        "docs/research_charter.md",
        "docs/architecture.md",
        "docs/data_recipe_spec.md",
        "docs/builder_protocol.md",
        "configs/constitution.yaml",
    ]
    if include_external:
        include.extend(
            path.relative_to(repository_root()).as_posix()
            for path in sorted((repository_root() / "context/external").rglob("*"))
            if path.is_file() and path.suffix.lower() in {".md", ".json", ".yaml", ".py"}
        )
    lines = ["# DataGoblin context bundle"]
    for relative in include:
        source = repository_root() / relative
        if source.is_file():
            text = source.read_text(encoding="utf-8", errors="replace")[:24000]
            lines.extend(("", "---", "", f"# FILE: {relative}", "", text))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _proposal_payload(snapshot: LearnerSnapshot) -> dict[str, Any]:
    result = ResearchOrchestrator(HeuristicGoblinBackend()).propose(snapshot)
    return {
        "experiment_id": result.experiment_id,
        "scout_findings": result.scout_findings.to_jsonable(),
        "hypotheses": [item.to_jsonable() for item in result.hypotheses],
        "recipe_original": result.original_recipe.to_jsonable(),
        "rat_critique": result.rat_critique.to_jsonable(),
        "repaired_recipe": result.repaired_recipe.to_jsonable(),
        "constitutional_decision": result.constitutional_decision.to_jsonable(),
        "builder_tickets": [item.to_jsonable() for item in result.tickets],
    }


def _synthetic_episodes(count: int, seed: int, path: Path) -> None:
    if count < 1:
        raise ValueError("count must be positive")
    base = _load_snapshot(_default_snapshot_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(count):
            snapshot = replace(base, snapshot_id=f"{base.snapshot_id}-synth-{seed}-{index:04d}")
            result = ResearchOrchestrator(HeuristicGoblinBackend()).propose(snapshot)
            episode = ResearchEpisode(
                episode_id=f"synthetic-{seed}-{index:04d}",
                schema_version="1.0",
                learner_snapshot=snapshot,
                scout_findings=result.scout_findings.to_jsonable(),
                hypothesis_set=result.hypotheses,
                proposed_recipe=result.original_recipe,
                rat_critique=result.rat_critique.to_jsonable(),
                repaired_recipe=result.repaired_recipe,
                constitutional_decision=result.constitutional_decision.to_jsonable(),
                builder_tickets=result.tickets,
                experiment_outcome=None,
                prediction_error={},
                interpretation="No outcome: synthetic planning episode only.",
                next_action="execute_or_abstain",
            )
            handle.write(json.dumps(episode.to_jsonable(), sort_keys=True, separators=(",", ":")) + "\n")


def _trace_options(config_path: Path) -> TraceOptions:
    value = load_structured(config_path)["trace"]
    return TraceOptions(
        selected_token_indices=tuple(int(item) for item in value["selected_token_indices"]),
        logits_tolerance=float(value["logits_tolerance"]),
        output_byte_budget=int(value["output_byte_budget"]),
        store_sampled_activations=bool(value["store_sampled_activations"]),
        activation_byte_budget=int(value["activation_byte_budget"]),
    )


def _trace_model(args: argparse.Namespace) -> Any:
    if args.mock:
        return DeterministicMockRecurrentModel()
    if not args.confirm_hard_cap_wrapper:
        raise ValueError(
            "real HRM execution requires --confirm-hard-cap-wrapper; "
            "DataGoblin does not create or certify an OS sandbox"
        )
    return TransformersHRMTextModel.load(
        local_model_path=args.local_model_path,
        model_id=args.model_id,
        allow_download=args.allow_download,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=int(load_structured(args.config)["model"]["max_new_tokens"]),
        device=getattr(args, "device", "cpu"),
    )


def _trace_verdict_dataset(args: argparse.Namespace) -> None:
    rows = read_dataset_rows(args.dataset_manifest)
    maximum_examples = getattr(args, "maximum_examples", None)
    if maximum_examples is not None:
        if maximum_examples < 1:
            raise ValueError("--maximum-examples must be positive")
        rows = rows[:maximum_examples]
    args.out.mkdir(parents=True, exist_ok=True)
    run_manifest = load_structured(args.dataset_manifest)
    run_manifest["source_manifest_hash"] = sha256_file(args.dataset_manifest)
    run_dataset_dir = args.out / "dataset"
    run_dataset_dir.mkdir(exist_ok=True)
    for split in run_manifest["splits"].values():
        source = args.dataset_manifest.parent / str(split["path"])
        target = run_dataset_dir / source.name
        shutil.copyfile(source, target)
        split["path"] = f"dataset/{target.name}"
    dump_structured(args.out / "dataset_manifest.json", run_manifest)
    shutil.copyfile(args.preregistration, args.out / "preregistration.yaml")
    configured_execution = load_structured(args.config)["execution"]
    dump_structured(
        args.out / "execution_policy_receipt.json",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model_mode": "deterministic_mock" if args.mock else "optional_real_checkpoint",
            "external_hard_cap_wrapper_confirmed": bool(args.confirm_hard_cap_wrapper),
            "declared_limits": configured_execution,
            "download_opt_in": bool(args.allow_download),
            "checkpoint_code_opt_in": bool(getattr(args, "trust_remote_code", False)),
            "device": getattr(args, "device", "cpu"),
            "maximum_examples": maximum_examples,
            "boundary": "operator-provided wrapper; DataGoblin does not certify OS-level enforcement",
        },
    )
    model = _trace_model(args)
    trace_path = args.out / "trace_summary.jsonl"
    options = _trace_options(args.config)
    try:
        with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                recipe = DataRecipe.from_jsonable(row["recipe"])
                prompt = build_verdict_prompt(recipe, condition=args.condition)
                # The prompt contains only the public recipe. Sealed labels do not enter this process.
                label = row.get("authoritative_label")
                summary = run_instrumented_forward(
                    model,
                    prompt,
                    authoritative_decision=label["decision"] if label else None,
                    authoritative_violations=label["violations"] if label else (),
                    schema_complexity=schema_complexity(recipe),
                    artifact_hashes=(str(row["recipe_hash"]),),
                    options=options,
                ).to_jsonable()
                summary.update(
                    {
                        "example_id": row["example_id"],
                        "split": row["split"],
                        "category": row["category"],
                        "role_condition": args.condition,
                        "label_authority": "datagoblin.constitution.ConstitutionalValidator",
                    }
                )
                handle.write(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n")
    finally:
        dump_structured(args.out / "model_cleanup_receipt.json", model.close())
    claim_boundary = "\n".join(
        (
            "# Claim boundary",
            "",
            "- Convergence does not imply correctness.",
            "- This experiment does not establish alignment.",
            "- Recurrent signals may merely track prompt complexity.",
            "- HRM-specific value requires comparison with recurrent baselines.",
            "- A null result is valid.",
            "- Mock-model traces are pipeline checks and are not evidence about HRM-Text.",
            "",
        )
    )
    (args.out / "claim_boundary.md").write_text(claim_boundary, encoding="utf-8")


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "rlt-gym":
        from datagoblin.training_gym.rlt_gym.cli import dispatch as dispatch_rlt_gym

        _write_json_stdout(dispatch_rlt_gym(args))
    elif args.command == "discover-upstreams":
        _write_json_stdout([item.to_jsonable() for item in discover_upstreams(repository_root())])
    elif args.command == "upstream-status":
        if args.config:
            _write_json_stdout(UpstreamRegistry.load(args.config).status())
        else:
            _write_json_stdout([item.to_jsonable() for item in discover_upstreams(repository_root())])
    elif args.command == "ingest-upstreams":
        receipt = UpstreamRegistry.load(args.config).ingest(args.out, mode=args.mode)
        dump_structured(args.out / "ingestion_receipt.json", receipt)
        _write_json_stdout(receipt)
    elif args.command == "make-context":
        _make_context(args.out, args.include_external)
        print(args.out)
    elif args.command == "validate":
        recipe = DataRecipe.from_jsonable(load_structured(args.recipe))
        decision = ConstitutionalValidator().validate(recipe)
        _write_json_stdout(decision.to_jsonable())
        return 0 if decision.decision == DecisionKind.ACCEPT else 2
    elif args.command == "propose":
        payload = _proposal_payload(_load_snapshot(args.snapshot))
        dump_structured(args.out, payload)
        return 0 if payload["constitutional_decision"]["execution_permitted"] else 2
    elif args.command == "emit-tickets":
        plan = load_structured(args.plan)
        recipe = DataRecipe.from_jsonable(plan.get("repaired_recipe", plan))
        experiment_id = str(plan.get("experiment_id", "experiment_from_plan"))
        write_tickets(emit_builder_tickets(experiment_id, recipe), args.out_dir)
    elif args.command == "demo":
        run_demo(_load_snapshot(args.snapshot), args.seed, args.out)
        print(args.out)
    elif args.command == "report":
        with ExperimentLedger(args.ledger) as ledger:
            value = ledger.get("experiments", args.experiment_id)
        if value is None:
            raise KeyError(args.experiment_id)
        from datagoblin.domain.experiments import ExperimentRecord

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(experiment_markdown(ExperimentRecord.from_jsonable(value)), encoding="utf-8")
    elif args.command == "synth-episodes":
        _synthetic_episodes(args.count, args.seed, args.out)
    elif args.command == "hrm-status":
        _write_json_stdout(
            detect_hrm_availability(model_path=args.local_model_path, model_id=args.model_id).to_jsonable()
        )
    elif args.command == "make-verdict-dataset":
        manifest = generate_verdict_dataset(args.recipe, args.out, seed=args.seed)
        _write_json_stdout(manifest.to_jsonable())
    elif args.command == "steward-sealed-labels":
        _write_json_stdout(steward_sealed_labels(args.dataset_manifest, args.out))
    elif args.command == "trace-hrm":
        _trace_verdict_dataset(args)
        print(args.out / "trace_summary.jsonl")
    elif args.command == "evaluate-cycle-signal":
        evaluate_cycle_signal(
            args.run_dir / "trace_summary.jsonl",
            args.run_dir,
            sealed_labels_path=args.sealed_labels,
            calibration_method=args.calibration,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        print(args.run_dir / "cycle_signal_report.md")
    elif args.command == "role-ablation":
        rows = read_dataset_rows(args.dataset_manifest)
        model = _trace_model(args)
        try:
            run_role_conditioning_ablation(
                model,
                rows,
                args.out,
                maximum_examples=args.maximum_examples,
                trace_options=_trace_options(args.config),
            )
        finally:
            dump_structured(args.out / "model_cleanup_receipt.json", model.close())
    elif args.command == "trace-recurrent-baselines":
        trace_recurrent_baseline_suite(
            args.dataset_manifest,
            args.out,
            trace_options=_trace_options(args.config),
            condition=args.condition,
        )
        evaluate_recurrent_baseline_suite(
            args.out,
            sealed_labels_path=args.sealed_labels,
            calibration_method=args.calibration,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        print(args.out / "baseline_manifest.json")
    elif args.command == "compare-recurrent-baselines":
        compare_recurrent_reports(args.primary_run, args.baseline_root, args.out)
        print(args.out)
    elif args.command == "hrm-preflight":
        assessment = run_checkpoint_preflight(
            args.local_model_path,
            args.out,
            run_id=args.run_id,
            caps=HardCaps(args.ram_mb, args.cpu_percent, args.io_mb_s, args.wall_clock_seconds),
        )
        print(args.out / "summary.json")
        return 0 if assessment.load_permitted else 3
    elif args.command == "gym-demo":
        run_training_gym_demo(args.contract, args.recipe, args.out, seed=args.seed)
        print(args.out / "comparison.md")
    elif args.command == "gym-evaluate":
        evaluate_candidate(
            args.contract,
            args.dataset_manifest,
            args.candidate_manifest,
            args.predictions,
            args.out,
            sealed_labels_path=args.sealed_labels,
        )
        print(args.out / "evaluation_report.md")
    elif args.command == "gym-compare":
        compare_candidates(args.contract, args.report, args.baseline_candidate_id, args.out)
        print(args.out / "comparison.md")
    elif args.command == "etiquette-demo":
        run_etiquette_demo(args.policy, args.out, seed=args.seed)
        print(args.out / "sorter_evaluation.json")
    elif args.command == "etiquette-evaluate":
        evaluate_sorter_predictions(
            args.dataset_manifest,
            args.predictions,
            args.sealed_labels,
            args.out,
        )
        print(args.out / "trained_sorter_evaluation.json")
    elif args.command == "etiquette-build-training-dataset":
        build_etiquette_training_dataset(args.policy, args.out, seed=args.seed)
        print(args.out / "dataset_manifest.json")
    elif args.command == "etiquette-audit-candidate":
        decision = audit_sorter_candidate(
            args.evaluation_report,
            args.run_receipt,
            args.policy,
            args.out,
        )
        print(args.out)
        return 0 if decision["promotion_eligible"] else 3
    else:
        episodes = read_episodes(args.input)
        exporters = {
            "export-sft": export_sft_jsonl,
            "export-preferences": export_preference_jsonl,
            "export-verifier": export_verifier_jsonl,
            "export-outcomes": export_outcome_prediction_jsonl,
        }
        exporters[args.command](episodes, args.out)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return dispatch(build_parser().parse_args(argv))
    except (KeyError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"datagoblin: {exc}", file=sys.stderr)
        return 2
