from __future__ import annotations

import io
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from datagoblin.hillclimb import bench
from datagoblin.hillclimb.agonist_worker import first_json_object
from datagoblin.hillclimb.curate import curate
from datagoblin.hillclimb.lanes import HillclimbError, Registry, call_reason, is_loopback_url
from datagoblin.hillclimb.ledger import Ledger, gate_verdict
from datagoblin.hillclimb.server import handle, serve
from datagoblin.hillclimb.service import Service

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def registry() -> Registry:
    return Registry.load()


@pytest.fixture
def service(tmp_path: Path) -> Service:
    return Service(root=ROOT, state_dir=tmp_path / "state", max_tier="loopback")


def _gate(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"direction": "up", "min_delta": 0.04, "selection_split": "dev", "patience": 3,
                            "plateau": 0, "incumbent": 0.5, "results": [{"value": 0.6, "split": "dev"}]}
    args.update(overrides)
    return gate_verdict(**args)


def test_gate_rules() -> None:
    assert _gate()["verdict"] == "promote"
    assert _gate(incumbent=None)["verdict"] == "baseline"
    assert _gate(results=[{"value": 0.53, "split": "dev"}])["verdict"] == "replicate"
    assert _gate(results=[{"value": 0.53, "split": "dev"}, {"value": 0.53, "split": "dev"}])["verdict"] == "reject"
    assert _gate(results=[{"value": 0.6, "split": "calibration"}])["verdict"] == "invalid"
    assert _gate(results=[])["reason"] == "no_result"
    assert _gate(direction="down", results=[{"value": 0.4, "split": "dev"}])["verdict"] == "promote"
    stopped = _gate(results=[{"value": 0.5, "split": "dev"}], plateau=2)
    assert stopped["verdict"] == "reject" and stopped["lane_stopped"] is True
    assert _gate(min_delta=0.03, incumbent=0.92, results=[{"value": 0.95, "split": "dev"}])["verdict"] == "promote"


def test_registry_lanes_are_climbable(registry: Registry) -> None:
    assert set(registry.lanes) == {"self_adapter", "looped_modules", "agonist_evals", "agonist_rollouts",
                                   "curation", "research_loop"}
    for lane in registry.lanes.values():
        assert lane.recipes
        if lane.metric is not None:
            assert any(lane.metric.name in registry.recipes[r].produces for r in lane.recipes)
    assert {r.tier for r in registry.recipes.values()} == {"offline", "loopback", "gpu"}


def test_call_reason_order(registry: Registry) -> None:
    send = registry.recipes["agonist_send"]
    lane = registry.lanes["agonist_evals"]
    good = {"requests": "run:r0001", "base_url": "http://127.0.0.1:18090/v1", "model": "inkling-2-mini"}

    def reason(params: dict[str, Any], status: str = "open", tier: str = "loopback") -> str:
        return call_reason(lane_status=status, lane_recipes=lane.recipes, recipe=send, max_tier=tier, params=params)

    assert reason(good) == "ok"
    assert reason(good, status="plateaued") == "lane_not_open"
    assert reason(good, tier="offline") == "tier_not_allowed"
    assert reason({**good, "shell": "rm"}) == "unknown_param"
    assert reason({k: v for k, v in good.items() if k != "model"}) == "missing_param"
    assert reason({**good, "base_url": "http://10.0.0.5:18090/v1"}) == "non_loopback_url"
    assert reason({**good, "max_tokens": 10}) == "bad_param"
    assert not is_loopback_url("http://127.0.0.1/v1")
    rollout = registry.recipes["rollout_capture"]
    assert call_reason(lane_status="open", lane_recipes=lane.recipes, recipe=rollout, max_tier="gpu",
                       params={}) == "recipe_not_in_lane"


def test_bench_splits_are_disjoint_and_gold_matches_rules(registry: Registry, tmp_path: Path) -> None:
    views = {}
    for split in bench.SPLITS:
        tasks, gold = bench.make_tasks(split, 3, 150, registry)
        views[split] = {task["view_sha256"] for task in tasks}
        assert {task["type"] for task in tasks} == set(bench.TYPES)
        for task, answer in zip(tasks, gold, strict=True):
            assert bench.score_answer(task["type"], answer["gold"], bench.target_for(task["type"], answer["gold"]))
    assert not views["train"] & views["dev"] and not views["train"] & views["test"] and not views["dev"] & views["test"]
    manifest = bench.make("train", 3, 40, tmp_path / "bench", registry)
    assert manifest["count"] == 40
    again = bench.make("train", 3, 40, tmp_path / "bench2", registry)
    assert again["tasks_sha256"] == manifest["tasks_sha256"]
    sft = bench.export_sft(tmp_path / "bench", tmp_path / "sft")
    assert sft["rows"] == 40 and sft["by_evidence_class"] == {"synthetic_exact_oracle": 40}
    rows = [json.loads(line) for line in (tmp_path / "sft" / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
    # bonsai-gym's load_examples needs an id and messages ending in one assistant turn.
    assert all(row["id"].endswith("/exact") and row["messages"][-1]["role"] == "assistant" for row in rows)
    assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]
    bench.make("dev", 3, 20, tmp_path / "dev", registry)
    with pytest.raises(ValueError, match="train-split"):
        bench.export_sft(tmp_path / "dev", tmp_path / "sft-dev")


def test_ledger_detects_tampering(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    ledger.append("proposal", {"move_id": "m0001", "lane": "curation", "hypothesis": "h", "plan": ["curate"]})
    ledger.append("decision", {"move_id": "m0001", "verdict": "explored"})
    assert len(ledger.read()) == 2
    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"curation"', '"looped_modules"')
    ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(HillclimbError, match="ledger_corrupt"):
        ledger.read()


def test_curate_custody(tmp_path: Path) -> None:
    rows = [
        {"root": "a", "split": "train", "x": 1},
        {"root": "a", "split": "train", "x": 1, "training_ready": True},
        {"root": "a", "split": "train", "x": 2},
        {"root": "a", "split": "train", "x": 3},
        {"root": "b", "split": "train", "x": 1},
        {"root": "b", "split": "dev", "x": 9},
        {"root": "c", "split": "dev", "x": 1},
        {"split": "train", "x": 5},
    ]
    source = tmp_path / "in.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = curate([source], tmp_path / "out", max_per_root=2)
    assert manifest["admitted"] == {"rows": 2, "roots": 1, "families": {"unknown": 2}}
    assert manifest["rejected"] == {"duplicate": 1, "missing_custody_fields": 1, "non_train_split": 2,
                                    "root_cap": 1, "root_split_collision": 1}
    assert manifest["leakage_roots"] == 1
    admitted = [json.loads(line) for line in (tmp_path / "out" / "admitted.jsonl").read_text().splitlines()]
    assert all(row["training_ready"] is False for row in admitted)


def test_service_moves_through_three_lanes(service: Service) -> None:
    move = service.call("propose_move", {"lane": "agonist_evals", "plan": ["agonist_demo", "agonist_audit"],
                                         "hypothesis": "Fixture replies verify and set a baseline."})["move_id"]
    demo = service.call("run_recipe", {"move_id": move, "recipe": "agonist_demo"})
    assert demo["status"] == "ok", demo["stderr_tail"]
    audit = service.call("run_recipe", {"move_id": move, "recipe": "agonist_audit", "params": {
        "requests": f"{demo['out']}/queue/teacher_requests.jsonl",
        "replies": f"{demo['out']}/queue/offline_fixture_replies.jsonl"}})
    assert audit["status"] == "ok" and audit["produces_lane_metric"]
    with pytest.raises(HillclimbError, match="not_evidence"):
        service.call("record_result", {"move_id": move, "run_id": audit["run_id"]})
    assert service.call("abandon_move", {"move_id": move, "reason": "fixture wiring only"})["verdict"] == "invalid"

    move = service.call("propose_move", {"lane": "curation", "plan": ["curate"],
                                         "hypothesis": "Audited candidates give two train roots."})["move_id"]
    run = service.call("run_recipe", {"move_id": move, "recipe": "curate",
                                      "params": {"inputs": [f"{audit['out']}/train_candidates.jsonl"]}})
    assert service.call("record_result", {"move_id": move, "run_id": run["run_id"]})["value"] == 2.0
    assert service.call("decide", {"move_id": move})["verdict"] == "baseline"

    move = service.call("propose_move", {"lane": "looped_modules", "plan": ["rlt_make_suite", "rlt_run"],
                                         "hypothesis": "One-depth RLT controls on a tiny suite set an incumbent."})
    suite = service.call("run_recipe", {"move_id": move["move_id"], "recipe": "rlt_make_suite",
                                        "params": {"per_family": 1}})
    fit = service.call("run_recipe", {"move_id": move["move_id"], "recipe": "rlt_run", "params": {
        "suite": suite["out"], "seeds": [7], "depths": [1], "max_wall_seconds": 60}})
    assert fit["status"] == "ok", fit["stderr_tail"]
    result = service.call("record_result", {"move_id": move["move_id"], "run_id": fit["run_id"]})
    assert 0.0 <= result["value"] <= 1.0 and result["split"] == "calibration"
    service.call("decide", {"move_id": move["move_id"]})

    status = service.call("ledger_status", {})
    assert status["records"] == 13
    assert {lane["id"]: lane["promotions"] for lane in status["lanes"]}["curation"] == 1


def test_service_refusals(service: Service, tmp_path: Path) -> None:
    move = service.call("propose_move", {"lane": "self_adapter", "plan": ["bench_make"],
                                         "hypothesis": "Checks that refusals carry stable codes."})["move_id"]
    with pytest.raises(HillclimbError, match="move_already_open"):
        service.call("propose_move", {"lane": "self_adapter", "plan": ["bench_make"], "hypothesis": "a second move"})
    with pytest.raises(HillclimbError, match="tier_not_allowed"):
        service.call("run_recipe", {"move_id": move, "recipe": "adaptergym_launch",
                                    "params": {"plan": "plans/bdb_compare.json", "tags": "suite"}})
    with pytest.raises(HillclimbError, match="bad_param: bench_make: parameter 'split' \\(given 'test'\\)"):
        service.call("run_recipe", {"move_id": move, "recipe": "bench_make", "params": {"split": "test"}})
    with pytest.raises(HillclimbError, match="missing_param: bench_make: parameter 'split'"):
        service.call("run_recipe", {"move_id": move, "recipe": "bench_make", "params": {"count": 10}})
    with pytest.raises(HillclimbError, match="bad_path"):
        service.call("run_recipe", {"move_id": move, "recipe": "export_driver_sft",
                                    "params": {"tasks": str(tmp_path.parent)}})
    with pytest.raises(HillclimbError, match="bad_arguments"):
        service.call("decide", {"move_id": move, "verdict": "promote"})
    made = service.call("run_recipe", {"move_id": move, "recipe": "bench_make", "params": {"split": "dev",
                                                                                         "count": 10}})
    with pytest.raises(HillclimbError, match="no_metric"):
        service.call("record_result", {"move_id": move, "run_id": made["run_id"]})
    assert service.call("abandon_move", {"move_id": move, "reason": "test"})["verdict"] == "invalid"
    read_only = Service(root=ROOT, state_dir=service.state_dir, read_only=True)
    assert {tool["name"] for tool in read_only.tools()} == {"suite_status", "ledger_status", "list_recipes"}
    with pytest.raises(HillclimbError, match="read_only"):
        read_only.call("propose_move", {"lane": "curation", "plan": ["curate"], "hypothesis": "not allowed here"})


class _FakeModel(BaseHTTPRequestHandler):
    replies: ClassVar[list[str]] = []

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send({"data": [{"id": "fake-27b"}]})

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            self._send({"tokens": list(range(len(request["content"].split())))})
            return
        content = type(self).replies.pop(0) if type(self).replies else _answer(request["messages"])
        self._send({"model": "fake-27b", "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def _answer(messages: list[dict[str, str]]) -> str:
    """Answer gate tasks with the real rule and every other bench type wrongly."""
    system, view = messages[0]["content"], json.loads(messages[1]["content"])
    kind = next(k for k, text in bench.RULES.items() if text == system)
    if kind == "gate_verdict":
        gate = view["lane_gate"]
        gold = gate_verdict(direction=gate["direction"], min_delta=gate["min_delta"],
                            selection_split=gate["selection_split"], patience=gate["patience"],
                            plateau=view["plateau"], incumbent=view["incumbent"], results=view["results"])
        answer: dict[str, Any] = {"verdict": gold["verdict"], "lane_stopped": gold["lane_stopped"]}
    else:
        answer = {"legal": False, "reason": "nonsense"} if kind == "proposal_legality" else {"lane": "nowhere"}
    return "<think>check the rules</think>" + json.dumps(answer)


@pytest.fixture
def fake_model() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeModel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    _FakeModel.replies = []


def test_agonist_send_and_audit_through_service(service: Service, fake_model: str) -> None:
    move = service.call("propose_move", {"lane": "agonist_evals", "plan": ["agonist_stage", "agonist_send",
                                                                           "agonist_audit"],
                                         "hypothesis": "A bigger local model builds world pairs."})["move_id"]
    staged = service.call("run_recipe", {"move_id": move, "recipe": "agonist_stage"})
    requests = Path(service.state_dir / "moves" / move / f"{staged['run_id']}-agonist_stage" / "out" /
                    "teacher_requests.jsonl")
    first = json.loads(requests.read_text(encoding="utf-8").splitlines()[0])
    world = {"setting": "Fictional reef", "scenario": "A crew weighs a dive.", "surface_cue": "green",
             "state": {"available": 4, "required": 3, "safety_approved": True, "source_fresh": True}}
    flipped = {**world, "state": {**world["state"], "source_fresh": False}}
    good = {"request_id": "ignored", "teacher_model": "x", "world_a": world, "world_b": flipped,
            "intervention": {"field": "source_fresh", "from": True, "to": False}, "design_note": "One flip."}
    _FakeModel.replies = ["Here you go:\n```json\n" + json.dumps(good) + "\n```",
                          json.dumps({k: v for k, v in good.items() if k != "design_note"}), "no json at all"]
    sent = service.call("run_recipe", {"move_id": move, "recipe": "agonist_send", "params": {
        "requests": f"{staged['out']}/teacher_requests.jsonl", "base_url": fake_model, "model": "fake-27b"}})
    assert sent["status"] == "ok", sent["stderr_tail"]
    audit = service.call("run_recipe", {"move_id": move, "recipe": "agonist_audit", "params": {
        "requests": f"{staged['out']}/teacher_requests.jsonl", "replies": f"{sent['out']}/teacher_replies.jsonl"}})
    result = service.call("record_result", {"move_id": move, "run_id": audit["run_id"]})
    assert result["value"] == pytest.approx(1 / 3)
    replies = (service.state_dir / "moves" / move / f"{sent['run_id']}-agonist_send" / "out" /
               "teacher_replies.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(replies[0])["request_id"] == first["request_id"]


def test_driver_bench_run_scores_exactly(service: Service, fake_model: str, registry: Registry) -> None:
    move = service.call("propose_move", {"lane": "self_adapter", "plan": ["bench_make", "driver_bench"],
                                         "hypothesis": "The fake driver answers only gate tasks correctly."})["move_id"]
    made = service.call("run_recipe", {"move_id": move, "recipe": "bench_make", "params": {"split": "dev",
                                                                                         "count": 30}})
    run = service.call("run_recipe", {"move_id": move, "recipe": "driver_bench", "params": {
        "tasks": made["out"], "base_url": fake_model, "model": "fake-27b", "limit": 30}})
    assert run["status"] == "ok", run["stderr_tail"]
    result = service.call("record_result", {"move_id": move, "run_id": run["run_id"]})
    tasks, _ = bench.make_tasks("dev", 0, 30, registry)
    assert result["value"] == pytest.approx(sum(t["type"] == "gate_verdict" for t in tasks) / 30)
    assert service.call("decide", {"move_id": move})["verdict"] == "baseline"


def test_recipes_are_tools_with_exact_schemas(service: Service, tmp_path: Path) -> None:
    tools = {tool["name"]: tool for tool in service.tools()}
    assert "run_recipe" not in tools and "adaptergym_launch" not in tools  # unlisted; gpu is above loopback
    assert {"driver_bench", "agonist_send", "rollout_capture", "rlt_run", "curate"} <= set(tools)
    schema = tools["rlt_run"]["inputSchema"]
    assert schema["required"] == ["move_id", "suite"] and schema["additionalProperties"] is False
    assert schema["properties"]["depths"] == {"type": "array", "items": {"type": "integer", "enum": [1, 2, 4, 8, 16]},
                                              "minItems": 1, "uniqueItems": True, "maxItems": 5,
                                              "default": [1, 2, 4, 8]}
    assert tools["bench_make"]["inputSchema"]["properties"]["split"]["enum"] == ["train", "dev"]
    offline = Service(root=ROOT, state_dir=tmp_path / "offline", max_tier="offline")
    assert "agonist_send" not in {tool["name"] for tool in offline.tools()}
    move = service.call("propose_move", {"lane": "self_adapter", "plan": ["bench_make"],
                                         "hypothesis": "A recipe runs when called by its own tool name."})["move_id"]
    made = service.call("bench_make", {"move_id": move, "split": "dev", "count": 10})
    assert made["status"] == "ok" and made["out"] == "run:r0001"
    with pytest.raises(HillclimbError, match="bad_arguments: bench_make needs move_id"):
        service.call("bench_make", {"split": "dev"})


def test_first_json_object() -> None:
    assert first_json_object('<think>{"no": 1}</think> text {"a": {"b": "}"}} tail') == {"a": {"b": "}"}}
    assert first_json_object("nothing") is None


def test_mcp_framing(service: Service) -> None:
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "decide", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 4, "method": "nope"},
    ]
    reader = io.BytesIO(("\n".join(json.dumps(m) for m in lines) + "\nnot json\n").encode())
    writer = io.BytesIO()
    serve(service, reader, writer)
    replies = [json.loads(line) for line in writer.getvalue().decode().splitlines()]
    assert [r.get("id") for r in replies] == [1, 2, 3, 4, None]
    assert replies[0]["result"]["serverInfo"]["name"] == "datagoblin-hillclimb"
    runnable = [r for r in service.registry.recipes.values() if service.runnable(r)]
    assert len(replies[1]["result"]["tools"]) == 7 + len(runnable)
    assert replies[2]["result"]["isError"] is True
    assert replies[2]["result"]["structuredContent"]["error"] == "bad_arguments"
    assert replies[3]["error"]["code"] == -32601
    assert handle(service, {"jsonrpc": "2.0", "method": "notifications/cancelled"}) is None
