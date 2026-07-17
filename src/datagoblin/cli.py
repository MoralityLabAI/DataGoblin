from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from datagoblin.adapters.upstream_registry import UpstreamRegistry, discover_upstreams
from datagoblin.builders.ticket_writer import write_tickets
from datagoblin.config import repository_root
from datagoblin.constitution.validator import ConstitutionalValidator, DecisionKind
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
from datagoblin.ledger.database import ExperimentLedger
from datagoblin.reporting.markdown import experiment_markdown
from datagoblin.research.demo import run_demo
from datagoblin.research.heuristic_backend import HeuristicGoblinBackend
from datagoblin.research.orchestrator import ResearchOrchestrator, emit_builder_tickets
from datagoblin.serialization import dump_structured, load_structured


def _default_snapshot_path() -> Path:
    return repository_root() / "examples" / "learner_snapshot.delayed_causality.json"


def _load_snapshot(path: Path) -> LearnerSnapshot:
    return LearnerSnapshot.from_jsonable(load_structured(path))


def _write_json_stdout(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datagoblin")
    sub = parser.add_subparsers(dest="command", required=True)
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
    return parser


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


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "discover-upstreams":
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
