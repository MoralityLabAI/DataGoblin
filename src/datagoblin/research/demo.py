from __future__ import annotations

from pathlib import Path
from typing import Any

from datagoblin.builders.template_builder import TemplateBuilder
from datagoblin.builders.ticket_writer import write_tickets
from datagoblin.constitution.charter import Charter
from datagoblin.constitution.validator import DecisionKind
from datagoblin.domain.evidence import EvidenceClaim, EvidenceType
from datagoblin.domain.experiments import ExperimentRecord, LifecycleStatus
from datagoblin.domain.learner import LearnerSnapshot
from datagoblin.domain.research_episode import ResearchEpisode
from datagoblin.harness.runner import run_toy_harness
from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.ids import content_id, utc_now
from datagoblin.ledger.artifact_store import ArtifactStore
from datagoblin.ledger.database import ExperimentLedger
from datagoblin.reporting.markdown import experiment_markdown
from datagoblin.serialization import dump_structured

from .heuristic_backend import HeuristicGoblinBackend
from .membrane import EvidenceVerification, MembraneDecision, ProposalAction, ResearchMembrane, ResearchProposal
from .orchestrator import ResearchOrchestrator
from .state import ResearchState, TransitionAuthority


def run_demo(snapshot: LearnerSnapshot, seed: int, output_root: Path) -> ResearchEpisode:
    output_root.mkdir(parents=True, exist_ok=True)
    result = ResearchOrchestrator(HeuristicGoblinBackend()).propose(snapshot)
    if result.constitutional_decision.decision != DecisionKind.ACCEPT:
        raise RuntimeError("repaired proposal did not pass the constitution")

    dump_structured(output_root / "learner_snapshot.json", snapshot.to_jsonable())
    dump_structured(output_root / "scout_findings.json", result.scout_findings.to_jsonable())
    dump_structured(output_root / "hypotheses.json", [item.to_jsonable() for item in result.hypotheses])
    dump_structured(output_root / "recipe.original.yaml", result.original_recipe.to_jsonable())
    dump_structured(output_root / "rat_critique.json", result.rat_critique.to_jsonable())
    dump_structured(output_root / "recipe.repaired.yaml", result.repaired_recipe.to_jsonable())
    dump_structured(output_root / "constitutional_decision.json", result.constitutional_decision.to_jsonable())
    ticket_paths = write_tickets(result.tickets, output_root / "tickets")

    builder = TemplateBuilder()
    build_results = []
    for ticket in result.tickets:
        build_results.append(builder.build(ticket, output_root / "build" / ticket.ticket_id, seed=seed))
    aggregate_build_receipt = {
        "schema_version": "1.0",
        "experiment_id": result.experiment_id,
        "tickets": [item.receipt for item in build_results],
        "all_accepted": all(item.accepted for item in build_results),
        "network_used": False,
        "subprocess_used": False,
        "claim_boundary": "Trusted template construction only; no evaluation authority is implied.",
    }
    aggregate_build_receipt["receipt_hash"] = sha256_json(aggregate_build_receipt)
    dump_structured(output_root / "build_receipt.json", aggregate_build_receipt)
    if not aggregate_build_receipt["all_accepted"]:
        raise RuntimeError("trusted builder did not accept every ticket")

    membrane = ResearchMembrane()
    state = result.research_state
    membrane_decisions = list(result.membrane_decisions)
    state = _advance(
        membrane,
        state,
        ResearchProposal(
            ProposalAction.REQUEST_EXPERIMENT,
            {"build_receipt_hash": aggregate_build_receipt["receipt_hash"]},
            EvidenceClaim(EvidenceType.SIMULATED, "template_builder", asserted_by="builder"),
            LifecycleStatus.BUILT,
        ),
        TransitionAuthority.BUILDER,
        membrane_decisions,
    )

    harness = run_toy_harness(seed)
    dump_structured(output_root / "train_split_manifest.json", harness.train_manifest)
    dump_structured(output_root / "evaluation_split_manifest.public.json", harness.evaluation_public_manifest)
    dump_structured(
        output_root / "verification_receipts.json",
        [item.to_jsonable() for item in harness.verification_receipts],
    )
    state = _advance(
        membrane,
        state,
        ResearchProposal(
            ProposalAction.REQUEST_EXPERIMENT,
            {"harness": "causal_switches_v1", "execution_completed": True},
            EvidenceClaim(EvidenceType.SIMULATED, "toy_harness", asserted_by="harness"),
            LifecycleStatus.EXECUTED,
        ),
        TransitionAuthority.HARNESS,
        membrane_decisions,
    )
    evaluation_proposal = ResearchProposal(
        ProposalAction.REQUEST_EXPERIMENT,
        {"actual_metrics": harness.actual_metrics, "evaluation_manifest": harness.evaluation_public_manifest},
        EvidenceClaim(
            EvidenceType.INDEPENDENT_HARNESS,
            "causal_switches_independent_verifier",
            tuple(item.receipt_hash for item in harness.verification_receipts),
            asserted_by="harness",
            verified_by="causal_switches_independent_verifier",
        ),
        LifecycleStatus.EVALUATED,
    )
    verification = EvidenceVerification.issue(
        state,
        evaluation_proposal,
        evidence_type=EvidenceType.INDEPENDENT_HARNESS,
        verifier_id="causal_switches_independent_verifier",
        artifact_refs=tuple(item.receipt_hash for item in harness.verification_receipts),
    )
    state = _advance(
        membrane,
        state,
        evaluation_proposal,
        TransitionAuthority.HARNESS,
        membrane_decisions,
        verification,
    )
    dump_structured(
        output_root / "lifecycle.json",
        {"final_state": state, "membrane_decisions": membrane_decisions},
    )
    now = utc_now()
    charter = Charter.load()
    build_files = tuple(path for item in build_results for path in item.outputs)
    source_root = Path(__file__).resolve().parents[1]
    record = ExperimentRecord(
        experiment_id=result.experiment_id,
        schema_version="1.0",
        snapshot_hash=sha256_json(snapshot.to_jsonable()),
        hypothesis_hashes=tuple(sha256_json(item.to_jsonable()) for item in result.hypotheses),
        recipe_hash=sha256_json(result.repaired_recipe.to_jsonable()),
        constitution_hash=charter.hash,
        builder_ticket_hashes=tuple(sha256_file(path) for path in ticket_paths),
        build_artifact_hashes=tuple(sha256_file(path) for path in build_files),
        generator_hash=sha256_file(source_root / "environments" / "causal_switches" / "generator.py"),
        solver_hash=sha256_file(source_root / "environments" / "causal_switches" / "solver.py"),
        verifier_hash=sha256_file(source_root / "environments" / "causal_switches" / "verifier.py"),
        learner_hash=sha256_file(source_root / "learners" / "numpy_linear.py"),
        train_split_manifest=harness.train_manifest,
        evaluation_split_manifest=harness.evaluation_public_manifest,
        predicted_outcomes=tuple(item.to_jsonable() for item in result.repaired_recipe.predicted_outcomes),
        actual_metrics={
            **harness.actual_metrics,
            "baseline_accuracy": harness.baseline_metrics["accuracy"],
            "improved_accuracy": harness.improved_metrics["accuracy"],
        },
        run_status=state.lifecycle_status,
        violations=(),
        resource_use={"cpu_seconds": 0.0, "memory_mb_peak": 0.0, "storage_bytes": 0.0},
        seeds={"public_demo_seed": (seed,), "private_evaluation_seeds": ()},
        errors=(),
        interpretation=harness.interpretation,
        next_experiment_suggestions=(
            "Run the explicit-history-feature ablation under matched compute.",
            "Repeat across a frozen seed ensemble before any stronger interpretation.",
        ),
        evidence_type=EvidenceType.INDEPENDENT_HARNESS,
        claim_boundary=(
            "Toy smoke evidence for causal_switches_v1 only; no claim about real TRM, HRM, LDT, "
            "autonomous science, alignment, or a future 1B model."
        ),
        timestamps={"created": now, "executed": now, "evaluated": now},
    )
    dump_structured(output_root / "experiment_record.json", record.to_jsonable())
    (output_root / "report.md").write_text(experiment_markdown(record), encoding="utf-8")

    midpoint = (
        sum(
            (
                result.repaired_recipe.predicted_outcomes[0].lower,
                result.repaired_recipe.predicted_outcomes[0].upper,
            )
        )
        / 2
    )
    episode = ResearchEpisode(
        episode_id=content_id("episode", {"experiment": result.experiment_id}),
        schema_version="1.0",
        learner_snapshot=snapshot,
        scout_findings=result.scout_findings.to_jsonable(),
        hypothesis_set=result.hypotheses,
        proposed_recipe=result.original_recipe,
        rat_critique=result.rat_critique.to_jsonable(),
        repaired_recipe=result.repaired_recipe,
        constitutional_decision=result.constitutional_decision.to_jsonable(),
        builder_tickets=result.tickets,
        experiment_outcome=record,
        prediction_error={"heldout_accuracy_delta": harness.actual_metrics["heldout_accuracy_delta"] - midpoint},
        interpretation=harness.interpretation,
        next_action="run_history_feature_ablation",
    )
    dump_structured(output_root / "research_episode.json", episode.to_jsonable())

    store = ArtifactStore(output_root / "artifacts")
    artifact_entries: dict[str, Any] = {}
    public_files = [
        path
        for path in output_root.iterdir()
        if path.is_file() and path.name not in {"run_manifest.json", "ledger.sqlite"}
    ]
    for path in public_files:
        media = "application/json" if path.suffix == ".json" else "text/plain"
        reference = store.put_file(path, media_type=media)
        artifact_entries[path.name] = reference.to_jsonable()
    run_manifest = {
        "schema_version": "1.0",
        "run_id": result.experiment_id,
        "seed": seed,
        "artifacts": artifact_entries,
        "tickets": [path.relative_to(output_root).as_posix() for path in ticket_paths],
        "ledger": "ledger.sqlite",
        "claim_boundary": record.claim_boundary,
    }
    run_manifest["manifest_hash"] = sha256_json(run_manifest)
    dump_structured(output_root / "run_manifest.json", run_manifest)

    with ExperimentLedger(output_root / "ledger.sqlite") as ledger:
        ledger.insert("learner_snapshots", snapshot.snapshot_id, snapshot.to_jsonable())
        for hypothesis in result.hypotheses:
            ledger.insert("hypotheses", hypothesis.hypothesis_id, hypothesis.to_jsonable())
        ledger.insert("recipes", result.repaired_recipe.recipe_id, result.repaired_recipe.to_jsonable())
        ledger.insert(
            "constitutional_decisions",
            result.constitutional_decision.decision_receipt_hash,
            result.constitutional_decision.to_jsonable(),
        )
        for ticket in result.tickets:
            ledger.insert("builder_tickets", ticket.ticket_id, ticket.to_jsonable())
        ledger.insert("builds", aggregate_build_receipt["receipt_hash"], aggregate_build_receipt)
        ledger.insert("experiments", record.experiment_id, record.to_jsonable())
        ledger.insert("research_episodes", episode.episode_id, episode.to_jsonable())
        for membrane_decision in membrane_decisions:
            ledger.insert("membrane_decisions", sha256_json(membrane_decision), membrane_decision.to_jsonable())
        for name, reference in artifact_entries.items():
            ledger.insert("artifacts", reference["sha256"], {"name": name, **reference})
    return episode


def _advance(
    membrane: ResearchMembrane,
    state: ResearchState,
    proposal: ResearchProposal,
    authority: TransitionAuthority,
    decisions: list[MembraneDecision],
    verification: EvidenceVerification | None = None,
) -> ResearchState:
    decision = membrane.apply(state, proposal, verification=verification, authority=authority)
    decisions.append(decision)
    if not decision.accepted:
        raise RuntimeError(decision.reason)
    return decision.after
