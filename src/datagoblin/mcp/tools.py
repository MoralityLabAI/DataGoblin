"""Tool table: each handler calls the DataGoblin Python function behind the matching CLI subcommand."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datagoblin.config import repository_root
from datagoblin.ledger.migrations import TABLES
from datagoblin.serialization import load_structured, to_jsonable

from .errors import McpToolError, Sandbox, arg_choice, arg_int, arg_str, read_json

Handler = Callable[[Sandbox, dict[str, Any]], dict[str, Any]]

EXPORT_KINDS = ("sft", "preferences", "verifier", "outcomes")
MAX_FILE_BYTES = 1_000_000
DEFAULT_SNAPSHOT = "examples/learner_snapshot.delayed_causality.json"
DEFAULT_RECIPE = "examples/recipe.good.yaml"
DEFAULT_POLICY = "configs/data_etiquette.yaml"
DEFAULT_CONTRACT = "configs/preregistrations/training_gym_v1.yaml"


def _structured(path: Path) -> dict[str, Any]:
    try:
        return load_structured(path)
    except (ValueError, OSError) as exc:
        raise McpToolError("invalid_input", f"{path.name}: {exc}") from exc


def _default_or(box: Sandbox, args: dict[str, Any], key: str, default: str) -> Path:
    return box.read_path(args.get(key, default), key)


def _files_under(box: Sandbox, directory: Path) -> list[str]:
    return sorted(box.rel(item) for item in directory.rglob("*") if item.is_file())


def _load_snapshot(path: Path) -> Any:
    from datagoblin.domain.learner import LearnerSnapshot

    return LearnerSnapshot.from_jsonable(_structured(path))


def validate_recipe(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.constitution.validator import ConstitutionalValidator, DecisionKind
    from datagoblin.domain.recipes import DataRecipe

    path = box.read_path(arg_str(args, "recipe"), "recipe")
    decision = ConstitutionalValidator().validate(DataRecipe.from_jsonable(_structured(path)))
    payload = decision.to_jsonable()
    return {
        "accepted": decision.decision == DecisionKind.ACCEPT,
        "decision": payload["decision"],
        "rule_ids": sorted({item.rule_id for item in decision.triggered_rules}),
        "execution_permitted": decision.execution_permitted,
        "constitutional_decision": payload,
    }


def propose_plan(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.cli import _proposal_payload
    from datagoblin.serialization import dump_structured

    snapshot = _load_snapshot(box.read_path(args.get("snapshot", DEFAULT_SNAPSHOT), "snapshot"))
    plan = _proposal_payload(snapshot)
    result: dict[str, Any] = {
        "experiment_id": plan["experiment_id"],
        "execution_permitted": plan["constitutional_decision"]["execution_permitted"],
        "plan": plan,
    }
    if args.get("out") is not None:
        out = box.write_path(args["out"], "out")
        dump_structured(out, plan)
        result["written"] = [box.rel(out)]
    return result


def emit_tickets(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.builders.ticket_writer import write_tickets
    from datagoblin.domain.recipes import DataRecipe
    from datagoblin.research.orchestrator import emit_builder_tickets

    plan = _structured(box.read_path(arg_str(args, "plan"), "plan"))
    out_dir = box.write_path(arg_str(args, "out_dir"), "out_dir")
    recipe = DataRecipe.from_jsonable(plan.get("repaired_recipe", plan))
    paths = write_tickets(emit_builder_tickets(str(plan.get("experiment_id", "experiment_from_plan")), recipe), out_dir)
    return {"written": [box.rel(path) for path in paths], "ticket_count": len(paths)}


def run_demo(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.research import demo

    seed = arg_int(args, "seed", 7, 0, 2**31 - 1)
    out = box.write_path(args.get("out", "runs/demo"), "out")
    snapshot = _load_snapshot(box.read_path(args.get("snapshot", DEFAULT_SNAPSHOT), "snapshot"))
    episode = demo.run_demo(snapshot, seed, out)
    return {
        "episode_id": episode.episode_id,
        "experiment_id": episode.experiment_outcome.experiment_id if episode.experiment_outcome else None,
        "run_dir": box.rel(out),
        "written": _files_under(box, out),
    }


def synth_episodes(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.cli import _synthetic_episodes

    count = arg_int(args, "count", 32, 1, 1000)
    seed = arg_int(args, "seed", 7, 0, 2**31 - 1)
    out = box.write_path(arg_str(args, "out"), "out")
    _synthetic_episodes(count, seed, out)
    return {"written": [box.rel(out)], "count": count}


def export_episodes(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.datasets import exports

    kind = arg_choice(args, "kind", EXPORT_KINDS)
    episodes = exports.read_episodes(box.read_path(arg_str(args, "input"), "input"))
    out = box.write_path(arg_str(args, "out"), "out")
    writers = {
        "sft": exports.export_sft_jsonl,
        "preferences": exports.export_preference_jsonl,
        "verifier": exports.export_verifier_jsonl,
        "outcomes": exports.export_outcome_prediction_jsonl,
    }
    writers[kind](episodes, out)
    rows = sum(1 for line in out.read_text(encoding="utf-8").splitlines() if line.strip())
    return {"kind": kind, "episodes": len(episodes), "rows": rows, "written": [box.rel(out)]}


def _open_ledger(box: Sandbox, run_dir_value: Any) -> tuple[Path, sqlite3.Connection]:
    run_dir = box.read_path(run_dir_value, "run_dir")
    ledger = run_dir / "ledger.sqlite"
    if not ledger.is_file():
        raise McpToolError("not_found", f"no ledger.sqlite in {box.rel(run_dir)}")
    connection = sqlite3.connect(f"{ledger.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return run_dir, connection


def _table(args: dict[str, Any]) -> str:
    return arg_choice(args, "table", TABLES)


def report(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    import json

    from datagoblin.domain.experiments import ExperimentRecord
    from datagoblin.reporting.markdown import experiment_markdown

    run_dir, connection = _open_ledger(box, args.get("run_dir"))
    experiment_id = arg_str(args, "experiment_id")
    try:
        row = connection.execute(
            "SELECT payload_json FROM experiments WHERE record_id = ?", (experiment_id,)
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise McpToolError("not_found", f"no experiment {experiment_id} in {box.rel(run_dir)}")
    markdown = experiment_markdown(ExperimentRecord.from_jsonable(json.loads(row["payload_json"])))
    result: dict[str, Any] = {"experiment_id": experiment_id, "markdown": markdown}
    if args.get("out") is not None:
        out = box.write_path(args["out"], "out")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        result["written"] = [box.rel(out)]
    return result


def ledger_summary(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    run_dir, connection = _open_ledger(box, args.get("run_dir"))
    try:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in TABLES
        }
        experiments = [row[0] for row in connection.execute("SELECT record_id FROM experiments ORDER BY created_at")]
    finally:
        connection.close()
    return {"run_dir": box.rel(run_dir), "table_counts": counts, "experiment_ids": experiments}


def ledger_list(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    table = _table(args)
    limit = arg_int(args, "limit", 50, 1, 500)
    _, connection = _open_ledger(box, args.get("run_dir"))
    try:
        rows = connection.execute(
            f"SELECT record_id, content_hash, created_at FROM {table} ORDER BY created_at, record_id LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        connection.close()
    return {"table": table, "records": [dict(row) for row in rows]}


def ledger_get(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    import json

    table = _table(args)
    record_id = arg_str(args, "record_id")
    _, connection = _open_ledger(box, args.get("run_dir"))
    try:
        row = connection.execute(
            f"SELECT payload_json, content_hash FROM {table} WHERE record_id = ?",
            (record_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise McpToolError("not_found", f"no record {record_id} in table {table}")
    return {"table": table, "record_id": record_id, "content_hash": row["content_hash"],
            "record": json.loads(row["payload_json"])}


def run_files(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    run_dir = box.read_path(arg_str(args, "run_dir"), "run_dir")
    if not run_dir.is_dir():
        raise McpToolError("invalid_arguments", "run_dir must be a directory")
    files = [item for item in run_dir.rglob("*") if item.is_file()]
    listing = [{"path": box.rel(item), "bytes": item.stat().st_size} for item in sorted(files)[:500]]
    return {"run_dir": box.rel(run_dir), "files": listing, "truncated": len(files) > 500}


def run_read_file(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    path = box.read_path(arg_str(args, "path"), "path")
    if not path.is_file():
        raise McpToolError("invalid_arguments", "path must be a file")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise McpToolError("too_large", f"{box.rel(path)} exceeds {MAX_FILE_BYTES} bytes")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    result: dict[str, Any] = {"path": box.rel(path), "text": text}
    if path.suffix.lower() == ".json":
        result["json"] = read_json(path)
    return result


def upstream_status(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.adapters.upstream_registry import UpstreamRegistry, discover_upstreams

    if args.get("config") is not None:
        return {"upstreams": UpstreamRegistry.load(box.read_path(args["config"], "config")).status()}
    return {"upstreams": [item.to_jsonable() for item in discover_upstreams(repository_root())]}


def etiquette_classify(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.data_etiquette import DataCollectionRecord, EtiquettePolicy

    values = args.get("records")
    if not isinstance(values, list) or not values or len(values) > 500:
        raise McpToolError("invalid_arguments", "records must be a list of 1 to 500 DataCollectionRecord objects")
    if not all(isinstance(item, dict) for item in values):
        raise McpToolError("invalid_arguments", "each record must be an object")
    policy = EtiquettePolicy(_default_or(box, args, "policy", DEFAULT_POLICY))
    records = [DataCollectionRecord.from_jsonable(item) for item in values]
    decisions = policy.decide_batch(records)
    return {
        "policy_version": policy.version,
        "policy_hash": policy.hash,
        "decisions": [item.to_jsonable() for item in decisions],
    }


def etiquette_demo(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.data_etiquette import run_etiquette_demo

    out = box.write_path(args.get("out", "runs/etiquette-demo"), "out")
    report_value = run_etiquette_demo(
        _default_or(box, args, "policy", DEFAULT_POLICY), out, seed=arg_int(args, "seed", 7, 0, 2**31 - 1)
    )
    return {"report": to_jsonable(report_value), "run_dir": box.rel(out), "written": _files_under(box, out)}


def etiquette_evaluate(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.data_etiquette import evaluate_sorter_predictions

    out = box.write_path(arg_str(args, "out"), "out")
    report_value = evaluate_sorter_predictions(
        box.read_path(arg_str(args, "dataset_manifest"), "dataset_manifest"),
        box.read_path(arg_str(args, "predictions"), "predictions"),
        box.read_path(arg_str(args, "sealed_labels"), "sealed_labels"),
        out,
    )
    return {"report": to_jsonable(report_value), "written": _files_under(box, out)}


def gym_demo(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.training_gym import run_training_gym_demo

    out = box.write_path(args.get("out", "runs/training-gym-demo"), "out")
    comparison = run_training_gym_demo(
        _default_or(box, args, "contract", DEFAULT_CONTRACT),
        _default_or(box, args, "recipe", DEFAULT_RECIPE),
        out,
        seed=arg_int(args, "seed", 7, 0, 2**31 - 1),
    )
    return {"comparison": to_jsonable(comparison), "run_dir": box.rel(out), "written": _files_under(box, out)}


def gym_evaluate(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.training_gym import evaluate_candidate

    out = box.write_path(arg_str(args, "out"), "out")
    sealed = box.read_path(args["sealed_labels"], "sealed_labels") if args.get("sealed_labels") else None
    evaluation = evaluate_candidate(
        box.read_path(arg_str(args, "contract"), "contract"),
        box.read_path(arg_str(args, "dataset_manifest"), "dataset_manifest"),
        box.read_path(arg_str(args, "candidate_manifest"), "candidate_manifest"),
        box.read_path(arg_str(args, "predictions"), "predictions"),
        out,
        sealed_labels_path=sealed,
    )
    return {"evaluation": to_jsonable(evaluation), "written": _files_under(box, out)}


def gym_compare(box: Sandbox, args: dict[str, Any]) -> dict[str, Any]:
    from datagoblin.training_gym import compare_candidates

    reports = args.get("reports")
    if not isinstance(reports, list) or len(reports) < 2:
        raise McpToolError("invalid_arguments", "reports must be a list of at least two evaluation report paths")
    out = box.write_path(arg_str(args, "out"), "out")
    comparison = compare_candidates(
        box.read_path(arg_str(args, "contract"), "contract"),
        [box.read_path(item, "reports[]") for item in reports],
        arg_str(args, "baseline_candidate_id"),
        out,
    )
    return {"comparison": to_jsonable(comparison), "written": _files_under(box, out)}


def _schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


_PATH = {"type": "string"}
_SEED = {"type": "integer", "minimum": 0, "default": 7}


def _tool(name: str, title: str, description: str, schema: dict[str, Any], handler: Handler, *,
          read_only: bool) -> dict[str, Any]:
    return {
        "name": name,
        "title": title,
        "description": description,
        "inputSchema": schema,
        "annotations": {"readOnlyHint": read_only, "destructiveHint": False, "idempotentHint": True,
                        "openWorldHint": False},
        "handler": handler,
    }


TOOLS: tuple[dict[str, Any], ...] = (
    _tool("validate_recipe", "Validate a DataRecipe",
          "Run the constitutional validator on a DataRecipe file (examples/ or runs/). Returns the decision, "
          "execution_permitted and the triggered rule ids (for example evaluation_leakage).",
          _schema({"recipe": _PATH}, ("recipe",)), validate_recipe, read_only=True),
    _tool("propose_plan", "Propose a research plan",
          "Propose hypotheses, a recipe, a RAT critique, a repaired recipe, a constitutional decision and builder "
          "tickets from a learner snapshot (default examples/learner_snapshot.delayed_causality.json). Optional out "
          "writes the plan under runs/.",
          _schema({"snapshot": _PATH, "out": _PATH}), propose_plan, read_only=False),
    _tool("emit_tickets", "Emit builder tickets",
          "Write builder tickets for a plan file (from propose_plan) into a directory under runs/.",
          _schema({"plan": _PATH, "out_dir": _PATH}, ("plan", "out_dir")), emit_tickets, read_only=False),
    _tool("run_demo", "Run the offline research demo",
          "Run the offline toy research cycle (propose, build, harness, ledger) into runs/. Defaults: seed 7, "
          "out runs/demo.",
          _schema({"seed": _SEED, "out": _PATH, "snapshot": _PATH}), run_demo, read_only=False),
    _tool("synth_episodes", "Synthesize planning episodes",
          "Write count synthetic planning episodes (JSONL) under runs/.",
          _schema({"count": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 32}, "seed": _SEED,
                   "out": _PATH}, ("out",)), synth_episodes, read_only=False),
    _tool("export_episodes", "Export training rows from episodes",
          "Export SFT, preference, verifier or outcome-prediction JSONL rows from a research episode file "
          "(research_episode.json or episodes JSONL).",
          _schema({"kind": {"type": "string", "enum": list(EXPORT_KINDS)}, "input": _PATH, "out": _PATH},
                  ("kind", "input", "out")), export_episodes, read_only=False),
    _tool("report", "Render an experiment report",
          "Render the markdown report for an experiment stored in a run directory's ledger. Optional out writes it "
          "under runs/.",
          _schema({"run_dir": _PATH, "experiment_id": {"type": "string"}, "out": _PATH},
                  ("run_dir", "experiment_id")), report, read_only=False),
    _tool("ledger_summary", "Summarize a run ledger",
          "Record counts per ledger table and the experiment ids in a run directory's ledger.sqlite.",
          _schema({"run_dir": _PATH}, ("run_dir",)), ledger_summary, read_only=True),
    _tool("ledger_list", "List ledger records",
          "List record ids, content hashes and timestamps of one ledger table.",
          _schema({"run_dir": _PATH, "table": {"type": "string", "enum": list(TABLES)},
                   "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50}},
                  ("run_dir", "table")), ledger_list, read_only=True),
    _tool("ledger_get", "Get a ledger record",
          "Fetch one ledger record payload (for example the experiments table's experiment record).",
          _schema({"run_dir": _PATH, "table": {"type": "string", "enum": list(TABLES)},
                   "record_id": {"type": "string"}},
                  ("run_dir", "table", "record_id")), ledger_get, read_only=True),
    _tool("run_files", "List files of a run",
          "List the files (path and size) under a run directory.",
          _schema({"run_dir": _PATH}, ("run_dir",)), run_files, read_only=True),
    _tool("run_read_file", "Read a run file",
          "Read one text or JSON file (at most 1 MB) under examples/, runs/ or configs/.",
          _schema({"path": _PATH}, ("path",)), run_read_file, read_only=True),
    _tool("upstream_status", "Upstream repository status",
          "Report which upstream repositories are present on this machine (git state only, no network).",
          _schema({"config": _PATH}), upstream_status, read_only=True),
    _tool("etiquette_classify", "Classify collection records",
          "Apply the data-collection etiquette policy to DataCollectionRecord objects and return each disposition "
          "(REJECT, QUARANTINE, DISCOVERY_ONLY, VERIFIED_TRAIN, EVALUATION_ONLY, SEALED_EVALUATION) with violations.",
          _schema({"records": {"type": "array", "items": {"type": "object"}}, "policy": _PATH}, ("records",)),
          etiquette_classify, read_only=True),
    _tool("etiquette_demo", "Run the etiquette demo",
          "Run the offline etiquette sorter demo (mock sorter) into runs/. Defaults: seed 7, out runs/etiquette-demo.",
          _schema({"seed": _SEED, "out": _PATH, "policy": _PATH}), etiquette_demo, read_only=False),
    _tool("etiquette_evaluate", "Evaluate sorter predictions",
          "Score sorter predictions against a dataset manifest and sealed labels; writes the report under runs/.",
          _schema({"dataset_manifest": _PATH, "predictions": _PATH, "sealed_labels": _PATH, "out": _PATH},
                  ("dataset_manifest", "predictions", "sealed_labels", "out")), etiquette_evaluate, read_only=False),
    _tool("gym_demo", "Run the training-gym demo",
          "Run the offline training-gym demo (frozen contract, mock candidates, comparison) into runs/.",
          _schema({"seed": _SEED, "out": _PATH, "contract": _PATH, "recipe": _PATH}), gym_demo, read_only=False),
    _tool("gym_evaluate", "Evaluate a gym candidate",
          "Evaluate one candidate's predictions under a frozen training-gym contract; writes the report under runs/.",
          _schema({"contract": _PATH, "dataset_manifest": _PATH, "candidate_manifest": _PATH, "predictions": _PATH,
                   "sealed_labels": _PATH, "out": _PATH},
                  ("contract", "dataset_manifest", "candidate_manifest", "predictions", "out")),
          gym_evaluate, read_only=False),
    _tool("gym_compare", "Compare gym candidates",
          "Compare two or more candidate evaluation reports against a baseline candidate id; writes under runs/.",
          _schema({"contract": _PATH, "reports": {"type": "array", "items": _PATH, "minItems": 2},
                   "baseline_candidate_id": {"type": "string"}, "out": _PATH},
                  ("contract", "reports", "baseline_candidate_id", "out")), gym_compare, read_only=False),
)

BY_NAME = {item["name"]: item for item in TOOLS}


def tool_list() -> list[dict[str, Any]]:
    return [{key: value for key, value in item.items() if key != "handler"} for item in TOOLS]
