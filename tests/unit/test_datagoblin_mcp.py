from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from datagoblin.mcp.errors import Sandbox
from datagoblin.mcp.server import PROTOCOL, handle, serve
from datagoblin.mcp.tools import TOOLS

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def box(tmp_path: Path) -> Sandbox:
    shutil.copytree(REPO / "examples", tmp_path / "examples")
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    (tmp_path / "runs").mkdir()
    return Sandbox(tmp_path)


def call(box: Sandbox, name: str, **arguments: Any) -> tuple[dict[str, Any], bool]:
    reply = handle(box, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": name, "arguments": arguments}})
    assert reply is not None
    result = reply["result"]
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
    return result["structuredContent"], result["isError"]


def ok(box: Sandbox, name: str, **arguments: Any) -> dict[str, Any]:
    payload, error = call(box, name, **arguments)
    assert not error, payload
    return payload


def test_initialize_list_ping_and_unknown_method(box: Sandbox) -> None:
    init = handle(box, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init is not None
    assert init["result"]["protocolVersion"] == PROTOCOL == "2025-06-18"
    assert init["result"]["serverInfo"]["name"] == "datagoblin"
    assert init["result"]["instructions"]
    assert handle(box, {"jsonrpc": "2.0", "id": 2, "method": "ping"}) == {"jsonrpc": "2.0", "id": 2, "result": {}}
    assert handle(box, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    unknown = handle(box, {"jsonrpc": "2.0", "id": 3, "method": "nope/nothing"})
    assert unknown is not None and unknown["error"]["code"] == -32601
    listing = handle(box, {"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
    assert listing is not None
    tools = {item["name"]: item for item in listing["result"]["tools"]}
    assert len(tools) == len(TOOLS)
    assert "handler" not in tools["validate_recipe"]
    assert tools["validate_recipe"]["annotations"]["readOnlyHint"] is True
    assert tools["run_demo"]["annotations"]["readOnlyHint"] is False
    assert all(item["inputSchema"]["type"] == "object" for item in tools.values())


def test_serve_frames_lines_and_reports_parse_errors(box: Sandbox) -> None:
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        "not json",
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "",
    ]
    out = io.BytesIO()
    serve(box, io.BytesIO(("\n".join(lines) + "\n").encode()), out)
    frames = [json.loads(line) for line in out.getvalue().decode().splitlines()]
    assert frames[0]["result"] == {}
    assert frames[1]["error"]["code"] == -32700
    assert len(frames) == 2


def test_bad_calls_return_stable_error_codes(box: Sandbox) -> None:
    assert call(box, "no_such_tool") == ({"error": "unknown_tool", "detail": "no tool named no_such_tool"}, True)
    assert call(box, "validate_recipe")[0]["error"] == "invalid_arguments"
    assert call(box, "validate_recipe", recipe="examples/recipe.good.yaml", extra=1)[0]["error"] == "invalid_arguments"
    assert call(box, "validate_recipe", recipe="examples/missing.yaml")[0]["error"] == "not_found"
    assert call(box, "validate_recipe", recipe="../outside.yaml")[0]["error"] == "path_not_allowed"
    assert call(box, "ledger_get", run_dir="examples", table="nope", record_id="x")[0]["error"] == "invalid_arguments"
    reply = handle(box, {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "ping_me"}})
    assert reply is not None and reply["result"]["isError"] is True


def test_writes_are_confined_to_runs(box: Sandbox, tmp_path: Path) -> None:
    for target in ("examples/plan.json", "plan.json", "runs", "../escape.json", str(tmp_path.parent / "x.json")):
        payload, error = call(box, "propose_plan", out=target)
        assert error and payload["error"] == "path_not_allowed", target
    assert not (tmp_path / "examples" / "plan.json").exists()
    assert call(box, "synth_episodes", out="runs/../examples/e.jsonl")[0]["error"] == "path_not_allowed"


def test_validate_recipe_accepts_good_and_flags_leakage(box: Sandbox) -> None:
    good = ok(box, "validate_recipe", recipe="examples/recipe.good.yaml")
    assert good["accepted"] is True and good["rule_ids"] == []
    bad = ok(box, "validate_recipe", recipe="examples/recipe.invalid_eval_leakage.yaml")
    assert bad["accepted"] is False
    assert "evaluation_leakage" in bad["rule_ids"]
    assert bad["execution_permitted"] is False


def test_propose_plan_then_emit_tickets(box: Sandbox) -> None:
    proposal = ok(box, "propose_plan", snapshot="examples/learner_snapshot.delayed_causality.json",
                  out="runs/plan.json")
    assert proposal["execution_permitted"] is True
    assert proposal["plan"]["builder_tickets"]
    assert proposal["written"] == ["runs/plan.json"]
    tickets = ok(box, "emit_tickets", plan="runs/plan.json", out_dir="runs/tickets")
    assert tickets["ticket_count"] == len(proposal["plan"]["builder_tickets"])
    assert all((box.root / path).is_file() for path in tickets["written"])
    assert ok(box, "propose_plan")["experiment_id"] == proposal["experiment_id"]


def test_demo_ledger_report_and_exports(box: Sandbox) -> None:
    demo = ok(box, "run_demo", seed=7, out="runs/demo")
    assert demo["run_dir"] == "runs/demo"
    assert "runs/demo/ledger.sqlite" in demo["written"]
    experiment_id = demo["experiment_id"]

    summary = ok(box, "ledger_summary", run_dir="runs/demo")
    assert summary["experiment_ids"] == [experiment_id]
    assert summary["table_counts"]["research_episodes"] == 1
    listing = ok(box, "ledger_list", run_dir="runs/demo", table="experiments")
    assert [item["record_id"] for item in listing["records"]] == [experiment_id]
    record = ok(box, "ledger_get", run_dir="runs/demo", table="experiments", record_id=experiment_id)
    assert record["record"]["experiment_id"] == experiment_id
    missing = call(box, "ledger_get", run_dir="runs/demo", table="experiments", record_id="nope")
    assert missing[0]["error"] == "not_found"
    assert call(box, "ledger_summary", run_dir="examples")[0]["error"] == "not_found"

    files = ok(box, "run_files", run_dir="runs/demo")
    assert any(item["path"] == "runs/demo/research_episode.json" for item in files["files"])
    text = ok(box, "run_read_file", path="runs/demo/experiment_record.json")
    assert text["json"]["experiment_id"] == experiment_id

    rendered = ok(box, "report", run_dir="runs/demo", experiment_id=experiment_id, out="runs/demo_report.md")
    assert rendered["markdown"].startswith("# DataGoblin experiment report")
    assert (box.root / "runs" / "demo_report.md").read_text(encoding="utf-8") == rendered["markdown"]
    assert call(box, "report", run_dir="runs/demo", experiment_id="nope")[0]["error"] == "not_found"

    for kind in ("sft", "preferences", "verifier", "outcomes"):
        exported = ok(box, "export_episodes", kind=kind, input="runs/demo/research_episode.json",
                      out=f"runs/export_{kind}.jsonl")
        assert exported["episodes"] == 1 and exported["rows"] >= 1
        assert (box.root / f"runs/export_{kind}.jsonl").is_file()


def test_synth_episodes_feed_exports(box: Sandbox) -> None:
    synth = ok(box, "synth_episodes", count=3, seed=5, out="runs/episodes.jsonl")
    assert synth["count"] == 3
    lines = (box.root / "runs" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    exported = ok(box, "export_episodes", kind="sft", input="runs/episodes.jsonl", out="runs/sft.jsonl")
    assert exported["episodes"] == 3
    assert call(box, "synth_episodes", count=0, out="runs/x.jsonl")[0]["error"] == "invalid_arguments"
    assert call(box, "export_episodes", kind="bogus", input="runs/episodes.jsonl", out="runs/y")[0]["error"] == (
        "invalid_arguments"
    )


def test_upstream_status_is_read_only_listing(box: Sandbox) -> None:
    assert isinstance(ok(box, "upstream_status")["upstreams"], list)


def _record(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "item_id": "item-1", "schema_version": "1.0", "content_sha256": "a" * 64, "source_kind": "synthetic",
        "purpose_id": "unit-test", "lawful_basis": "LEGITIMATE_INTEREST", "allowed_uses": ["TRAINING"],
        "data_residency_region": "local", "sensitivity_class": "PUBLIC", "license_status": "PERMISSIVE",
        "consent_status": "NOT_REQUIRED", "personal_data_status": "NONE", "redacted": True,
        "contains_secret": False, "contains_hidden_reasoning": False, "contains_model_weight": False,
        "evaluation_material": False, "sealed_evaluation": False, "model_generated": False, "generator_id": None,
        "evidence_type": "UNKNOWN", "asserted_by": "tester", "verified_by": None,
        "verification_receipt_sha256": None, "verification_subject_sha256": None,
        "verification_custody": "NONE", "provenance_refs": ["ref:one"], "dedup_group": "g1",
        "intended_split": "train", "retention_days": 30, "deletion_requested": False, "public_metadata": {},
    }
    value.update(overrides)
    return value


def test_etiquette_classify_rejects_secrets_and_discovers_unverified(box: Sandbox) -> None:
    result = ok(box, "etiquette_classify", records=[
        _record(item_id="secret", contains_secret=True, dedup_group="g2"),
        _record(item_id="plain"),
    ])
    by_id = {item["item_id"]: item for item in result["decisions"]}
    assert by_id["secret"]["disposition"] == "REJECT"
    assert "secret_or_credential" in by_id["secret"]["violations"]
    assert by_id["plain"]["disposition"] == "DISCOVERY_ONLY"
    assert call(box, "etiquette_classify", records=[])[0]["error"] == "invalid_arguments"
    assert call(box, "etiquette_classify", records=[{"item_id": "x"}])[0]["error"] == "invalid_input"


def test_etiquette_demo_and_evaluate(box: Sandbox) -> None:
    demo = ok(box, "etiquette_demo", seed=7, out="runs/etiquette")
    assert demo["report"]
    assert "runs/etiquette/sorter_evaluation.json" in demo["written"]
    built = ok(box, "etiquette_demo", seed=8, out="runs/etiquette8")
    assert built["run_dir"] == "runs/etiquette8"
    assert call(box, "etiquette_evaluate", dataset_manifest="runs/none.json", predictions="runs/p.json",
                sealed_labels="runs/s.json", out="runs/ev")[0]["error"] == "not_found"


def test_gym_demo_evaluate_and_compare(box: Sandbox) -> None:
    demo = ok(box, "gym_demo", seed=7, out="runs/gym")
    assert "runs/gym/comparison.md" in demo["written"]
    reports = sorted(path for path in demo["written"] if path.endswith("evaluation_report.json"))
    assert len(reports) >= 2
    first = json.loads((box.root / reports[0]).read_text(encoding="utf-8"))
    compared = ok(box, "gym_compare", contract="runs/gym/frozen_contract.yaml", reports=reports,
                  baseline_candidate_id=first["candidate_id"], out="runs/gym_compare")
    assert "runs/gym_compare/comparison.md" in compared["written"]
    assert call(box, "gym_compare", contract="runs/gym/frozen_contract.yaml", reports=reports[:1],
                baseline_candidate_id="x", out="runs/gc")[0]["error"] == "invalid_arguments"
    missing = call(box, "gym_evaluate", contract="runs/gym/frozen_contract.yaml",
                   dataset_manifest="runs/gym/public_dataset/dataset_manifest.json",
                   candidate_manifest="runs/none.json", predictions="runs/none.jsonl", out="runs/ge")
    assert missing[1] and missing[0]["error"] == "not_found"
