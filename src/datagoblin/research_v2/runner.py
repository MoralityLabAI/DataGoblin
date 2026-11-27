"""Checkpoint-neutral rollout receipts and paired, complete-denominator reports."""

from __future__ import annotations

import json
import math
import platform
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import VERSION
from .environment import LIMITS, ResearchSession
from .suite import FAMILIES, canon, digest, implementation_digest, load_suite, reference_actions

COMPARISON = {
    "schema": "research-v2-comparison-1",
    "context_tokens": 2048,
    "max_new_tokens_per_turn": 512,
    "max_turns": 8,
    "max_requests": 64,
    "temperature": 0,
    "seed": 0,
    "history": "task-plus-complete-current-observation",
    "selection": "balanced-family-prefix",
    "environment_limits": LIMITS,
}
DEFAULT_REGISTRY = Path(__file__).resolve().parents[3] / "configs/research_v2/models.json"


def registry(path: Path = DEFAULT_REGISTRY) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "datagoblin-research-v2-model-registry-1" or type(data.get("models")) is not list:
        raise ValueError("unsupported model registry")
    result = {row["id"]: row for row in data["models"]}
    if len(result) != len(data["models"]):
        raise ValueError("duplicate registry identity")
    for row in result.values():
        if not row.get("revision") and not row.get("weights_sha256"):
            raise ValueError("checkpoint needs an immutable revision or weight SHA-256")
    return result


def register_checkpoint(entry_path: Path, registry_path: Path, out: Path) -> dict[str, Any]:
    entry = json.loads(entry_path.read_text())
    required = {
        "id",
        "model_id",
        "weights_sha256",
        "precision",
        "context_length",
        "protocol",
        "template",
        "training_positions",
        "training_index_sha256",
        "parameters",
        "recurrence",
    }
    if not isinstance(entry, dict) or not required <= entry.keys():
        raise ValueError("future checkpoint is missing required identity fields")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,95}", entry["id"]):
        raise ValueError("invalid checkpoint ID")
    for key in ("weights_sha256", "training_index_sha256"):
        if type(entry[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", entry[key]):
            raise ValueError("SHA-256 identity required")
    for key in ("parameters", "context_length", "training_positions"):
        if type(entry[key]) is not int or entry[key] < (0 if key == "training_positions" else 1):
            raise ValueError("invalid checkpoint count")
    if entry["protocol"] not in {"raw-completion", "chat-completions", "prefix-lm-completion"}:
        raise ValueError("unknown checkpoint protocol")
    if entry["id"] in registry(registry_path):
        raise ValueError("checkpoint IDs are immutable; register a new variant")
    if out.exists():
        raise ValueError("registry output must be fresh")
    data = json.loads(registry_path.read_text())
    data["models"].append({"revision": None, "endpoint_status": "unbound", "display_name": entry["id"], **entry})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return {"registered": entry["id"], "registry_sha256": digest(data), "registry": str(out)}


def select_cases(cases: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if type(maximum) is not int or not 8 <= maximum <= 128 or maximum % len(FAMILIES):
        raise ValueError("max_episodes must be a multiple of eight in 8..128")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_family[case["family"]].append(case)
    if any(not by_family[f] for f in FAMILIES):
        raise ValueError("every task family must be represented")
    count = min(maximum // len(FAMILIES), min(len(by_family[f]) for f in FAMILIES))
    return [by_family[family][i] for i in range(count) for family in FAMILIES]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty result set")
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        score = row.get("primary_score")
        if type(score) not in {int, float} or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("nonfinite or unbounded score")
        families[row["family"]].append(row)
    summary = {}
    for family, group in sorted(families.items()):
        summary[family] = {
            "cases": len(group),
            "mean_score": sum(r["primary_score"] for r in group) / len(group),
            "successes": sum(r["success"] is True for r in group),
            "integrity_passes": sum(r["integrity_pass"] is True for r in group),
            "completed": sum(r["completed"] is True for r in group),
            "format_or_tool_errors": sum(r["cost"]["errors"] for r in group),
        }
    return {
        "cases": len(rows),
        "families": summary,
        "macro_score": sum(s["mean_score"] for s in summary.values()) / len(summary),
        "success_rate": sum(r["success"] is True for r in rows) / len(rows),
    }


def _start(out: Path) -> None:
    if out.exists():
        raise ValueError("run output must be fresh")
    out.mkdir(parents=True, mode=0o700)


def _finish(out: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    row_file = out / "results.jsonl"
    row_file.write_text("".join(canon(row) + "\n" for row in rows))
    import hashlib

    report = {
        "schema": VERSION,
        **metadata,
        "summary": aggregate(rows),
        "results_sha256": hashlib.sha256(row_file.read_bytes()).hexdigest(),
        "task_ids": [r["task_id"] for r in rows],
        "implementation_sha256": implementation_digest(),
        "experiment_runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.system(),
        },
        "artifact_files": {
            str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((out / "artifacts").glob("*/*.json"))
        },
    }
    report["report_sha256"] = digest(report)
    (out / "report.json").write_text(canon(report) + "\n")
    return report


def run_control(suite_path: Path, out: Path, *, control: str = "reference", max_episodes: int = 8) -> dict[str, Any]:
    cases = select_cases(load_suite(suite_path), max_episodes)
    if any(c["split"] != "development" for c in cases) or control not in {"reference", "noop"}:
        raise ValueError("offline controls require development tasks and a registered control")
    _start(out)
    rows = []
    for case in cases:
        session = ResearchSession(case)
        try:
            if control == "reference":
                for action in reference_actions(case):
                    session.step_json(canon(action))
                    if session.done:
                        break
        except (ValueError, RuntimeError, OSError):
            # A broken control is retained as an incomplete zero, including all later cases.
            session.done = True
        session.save_artifacts(out / "artifacts")
        rows.append(session.result())
    manifest = json.loads((suite_path / "manifest.json").read_text())
    return _finish(
        out,
        rows,
        {
            "candidate_id": f"control/{control}",
            "role": "control",
            "model_evaluated": False,
            "suite_sha256": manifest["suite_sha256"],
            "comparison": COMPARISON,
            "comparison_sha256": digest(COMPARISON),
            "inference_requests": 0,
            "checkpoint_rank_eligible": False,
        },
    )


def run_endpoint(
    suite_path: Path,
    out: Path,
    *,
    model_id: str,
    binding_path: Path,
    registry_path: Path = DEFAULT_REGISTRY,
    max_episodes: int = 8,
) -> dict[str, Any]:
    from .transport import request_action, validate_binding

    entries = registry(registry_path)
    if model_id not in entries:
        raise ValueError("unknown checkpoint; add its immutable entry to a versioned registry")
    entry = entries[model_id]
    binding = validate_binding(entry, json.loads(binding_path.read_text()))
    cases = select_cases(load_suite(suite_path), max_episodes)
    manifest = json.loads((suite_path / "manifest.json").read_text())
    _start(out)
    started = time.monotonic()
    rows, requests, usage, failures = [], 0, {"prompt_tokens": 0, "completion_tokens": 0}, 0
    max_requests = min(COMPARISON["max_requests"], binding["resource_caps"]["max_requests"] // 2)
    run_seconds = min(900, binding["resource_caps"]["wall_seconds"])
    comparison = {
        **COMPARISON,
        "effective_max_actions": max_requests,
        "effective_max_http_requests": 2 * max_requests,
        "effective_run_seconds": run_seconds,
    }
    deadline = started + run_seconds
    http_requests, inference_requests, successful_responses, unknown_usage_calls = 0, 0, 0, 0
    traces_path = out / "transcript.jsonl"
    with traces_path.open("w") as traces:
        for case in cases:
            session = ResearchSession(case, deadline=deadline)
            observation = session.observation(status="ready")
            stop_reason = "turn_limit"
            for _ in range(COMPARISON["max_turns"]):
                remaining = session.remaining_seconds()
                if requests >= max_requests or remaining <= 0:
                    stop_reason = "run_budget_exhausted"
                    break
                prompt = case["prompt"] + "\nCurrent observation:\n" + canon(observation) + "\nAction:\n"
                # Every model gets this same bounded external state. Earlier turns stay in the receipt.
                requests += 1
                try:
                    response = request_action(
                        entry,
                        binding,
                        prompt,
                        max_tokens=COMPARISON["max_new_tokens_per_turn"],
                        timeout_seconds=min(30.0, remaining),
                    )
                    http_requests += response.get("http_requests", 2)
                    inference_requests += response.get("inference_requests", 1)
                    successful_responses += 1
                    for field in usage:
                        usage[field] += response["usage"][field]
                    observation = session.step_json(response["text"])
                    record = {
                        "task_id": case["id"],
                        "request": requests,
                        "prompt_sha256": digest(prompt),
                        "response": response,
                        "observation": observation,
                    }
                    traces.write(canon(record) + "\n")
                    traces.flush()
                    if session.done:
                        stop_reason = "session_closed"
                        break
                except (ValueError, OSError, RuntimeError, TimeoutError) as error:
                    failures += 1
                    stop_reason = type(error).__name__
                    receipt = getattr(error, "receipt", {})
                    http_requests += receipt.get("http_requests", 0)
                    inference_requests += receipt.get("inference_requests", 0)
                    known_usage = receipt.get("usage")
                    if isinstance(known_usage, dict) and all(
                        type(known_usage.get(field)) is int and known_usage[field] >= 0 for field in usage
                    ):
                        for field in usage:
                            usage[field] += known_usage[field]
                    elif receipt.get("inference_requests", 0):
                        unknown_usage_calls += receipt["inference_requests"]
                    traces.write(
                        canon(
                            {
                                "task_id": case["id"],
                                "request": requests,
                                "error": type(error).__name__,
                                "transport_receipt": receipt,
                            }
                        )
                        + "\n"
                    )
                    traces.flush()
                    break
            session.save_artifacts(out / "artifacts")
            result = session.result()
            result["rollout_stop_reason"] = stop_reason
            # The core digest remains a separate, unchanged envelope.
            core_hash = result.pop("result_sha256")
            result["core_result_sha256"] = core_hash
            result["result_sha256"] = digest(result)
            rows.append(result)
    identity = {"registry_entry": entry, "binding": binding}
    return _finish(
        out,
        rows,
        {
            "candidate_id": model_id,
            "role": "checkpoint",
            "model_evaluated": successful_responses > 0,
            "suite_sha256": manifest["suite_sha256"],
            "comparison": comparison,
            "comparison_sha256": digest(comparison),
            "identity_sha256": digest(identity),
            "identity": identity,
            "identity_authority": "operator_attested; response model ID checked",
            "attempted_actions": requests,
            "http_requests": http_requests,
            "inference_requests": inference_requests,
            "successful_responses": successful_responses,
            "inference_requests_with_unknown_usage": unknown_usage_calls,
            "max_completion_tokens_reserved": inference_requests * COMPARISON["max_new_tokens_per_turn"],
            "usage_scope": "Reported totals include failed requests; unreported usage is tracked separately.",
            "request_failures": failures,
            "usage": usage,
            "checkpoint_rank_eligible": failures == 0 and all(r["completed"] for r in rows),
            "wall_seconds": round(time.monotonic() - started, 6),
        },
    )


def read_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import hashlib

    from .learners import validate_artifact

    path = path / "report.json" if path.is_dir() else path
    report = json.loads(path.read_text())
    if digest({k: v for k, v in report.items() if k != "report_sha256"}) != report.get("report_sha256"):
        raise ValueError("report digest mismatch")
    if report.get("schema") != VERSION or digest(report["comparison"]) != report["comparison_sha256"]:
        raise ValueError("unsupported report or mismatched comparison contract")
    if report.get("role") == "checkpoint":
        identity = report.get("identity", {})
        if (
            digest(identity) != report.get("identity_sha256")
            or identity.get("registry_entry", {}).get("id") != report["candidate_id"]
            or identity.get("binding", {}).get("registry_id") != report["candidate_id"]
        ):
            raise ValueError("checkpoint identity binding mismatch")
    data = (path.parent / "results.jsonl").read_bytes()
    if hashlib.sha256(data).hexdigest() != report["results_sha256"]:
        raise ValueError("result bytes changed")
    rows = [json.loads(line) for line in data.splitlines()]
    if [r["task_id"] for r in rows] != report["task_ids"] or len(set(report["task_ids"])) != len(rows):
        raise ValueError("result membership changed")
    if len({r["root_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate result roots")
    for row in rows:
        if digest({k: v for k, v in row.items() if k != "result_sha256"}) != row.get("result_sha256"):
            raise ValueError("row digest mismatch")
        if "core_result_sha256" in row:
            core = {
                k: v for k, v in row.items() if k not in {"result_sha256", "core_result_sha256", "rollout_stop_reason"}
            }
            if digest(core) != row["core_result_sha256"]:
                raise ValueError("core result binding mismatch")
    if aggregate(rows) != report["summary"]:
        raise ValueError("aggregate does not replay")
    observed_hashes = []
    for name, expected in report.get("artifact_files", {}).items():
        if not re.fullmatch(r"artifacts/[0-9a-f]{16}/a[1-4]\.json", name):
            raise ValueError("unsafe artifact path")
        artifact_path = path.parent / name
        if (
            artifact_path.is_symlink()
            or not artifact_path.resolve().is_relative_to(path.parent.resolve())
            or artifact_path.stat().st_size > 100001
        ):
            raise ValueError("invalid artifact file")
        raw = artifact_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("artifact file digest mismatch")
        artifact = json.loads(raw)
        validate_artifact(artifact)
        observed_hashes.append(artifact["artifact_sha256"])
    expected_hashes = sorted(h for row in rows for h in row.get("artifact_hashes", []))
    if sorted(observed_hashes) != expected_hashes:
        raise ValueError("artifact inventory does not match episode receipts")
    return report, rows


def compare(paths: list[Path]) -> dict[str, Any]:
    if not 2 <= len(paths) <= 32:
        raise ValueError("compare requires 2..32 reports")
    loaded = [read_report(path) for path in paths]
    baseline, base_rows = loaded[0]
    for report, _ in loaded[1:]:
        if any(
            report.get(k) != baseline.get(k)
            for k in (
                "suite_sha256",
                "implementation_sha256",
                "comparison_sha256",
                "task_ids",
                "role",
                "experiment_runtime",
            )
        ):
            raise ValueError("reports require identical tasks, code, budget, split and role")
    if len({r[0]["candidate_id"] for r in loaded}) != len(loaded):
        raise ValueError("duplicate candidate identity; use a separate ID for each checkpoint/export")
    comparisons = []
    for report, rows in loaded[1:]:
        grouped = {f: [] for f in FAMILIES}
        for old, new in zip(base_rows, rows, strict=True):
            if old["root_id"] != new["root_id"] or old["family"] != new["family"]:
                raise ValueError("paired root mismatch")
            grouped[new["family"]].append(new["primary_score"] - old["primary_score"])
        per_family = {f: sum(v) / len(v) for f, v in grouped.items()}
        rng = np.random.default_rng(0)
        bootstrap = [
            sum(float(np.mean(rng.choice(v, size=len(v), replace=True))) for v in grouped.values()) / len(grouped)
            for _ in range(2000)
        ]
        comparisons.append(
            {
                "candidate_id": report["candidate_id"],
                "reference_id": baseline["candidate_id"],
                "macro_delta": sum(per_family.values()) / len(per_family),
                "family_deltas": per_family,
                "paired_root_bootstrap_95": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
                "interval_status": "descriptive_only" if len(rows) < 32 else "paired_within_family",
                "complete_denominator": len(rows),
                "rank_eligible": report["checkpoint_rank_eligible"],
            }
        )
    return {
        "schema": "research-v2-paired-comparison-1",
        "suite_sha256": baseline["suite_sha256"],
        "candidates": [{"id": r["candidate_id"], "summary": r["summary"]} for r, _ in loaded],
        "paired": comparisons,
        "winner": None,
        "scope": "Frozen task comparison; public development and failed runs remain explicitly labeled.",
    }
