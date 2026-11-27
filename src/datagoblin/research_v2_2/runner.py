"""Research Verifiers v2.2 runner with per-episode action budgets & controller accounting."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.research_v2.runner import DEFAULT_REGISTRY, registry, select_cases
from datagoblin.research_v2.transport import request_action, validate_binding
from datagoblin.research_v2_1.environment import LIMITS, ResearchSessionV21
from datagoblin.research_v2_1.suite import FAMILIES, canon, digest, load_suite, reference_actions

from . import VERSION, implementation_digest
from .controllers import CONTROLLER_MODES, ResearchController
from .paired import paired_effects, selective_gate

COMPARISON_V22 = {
    "schema": "research-v2.2-comparison-1",
    "context_tokens": 2048,
    "max_new_tokens_per_turn": 512,
    "max_turns_per_episode": 8,
    "per_episode_action_budget": True,
    "temperature": 0,
    "seed": 0,
    "history": "task-plus-persistent-compact-state",
    "selection": "balanced-family-prefix",
    "environment_limits": LIMITS,
}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty result set")
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for field in ("primary_score", "capability_score"):
            score = row.get(field)
            if type(score) not in {int, float} or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"nonfinite or unbounded {field}")
        families[row["family"]].append(row)
    summary = {}
    for family, group in sorted(families.items()):
        summary[family] = {
            "cases": len(group),
            "mean_score": sum(r["primary_score"] for r in group) / len(group),
            "mean_capability_score": sum(r["capability_score"] for r in group) / len(group),
            "successes": sum(r["success"] is True for r in group),
            "integrity_passes": sum(r["integrity_pass"] is True for r in group),
            "completed": sum(r["completed"] is True for r in group),
            "format_or_tool_errors": sum(r["cost"]["errors"] for r in group),
        }
    return {
        "cases": len(rows),
        "families": summary,
        "macro_score": sum(s["mean_score"] for s in summary.values()) / len(summary),
        "macro_capability_score": sum(s["mean_capability_score"] for s in summary.values()) / len(summary),
        "success_rate": sum(r["success"] is True for r in rows) / len(rows),
    }


def _start(out: Path) -> None:
    if out.exists():
        raise ValueError("run output must be fresh")
    out.mkdir(parents=True, mode=0o700)


def _finish(out: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    row_file = out / "results.jsonl"
    row_file.write_text("".join(canon(row) + "\n" for row in rows))
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
        session = ResearchSessionV21(case)
        try:
            if control == "reference":
                for action in reference_actions(case):
                    session.step_json(canon(action))
                    if session.done:
                        break
        except (ValueError, RuntimeError, OSError):
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
            "controller_mode": "control",
            "model_evaluated": False,
            "suite_sha256": manifest["suite_sha256"],
            "comparison": COMPARISON_V22,
            "comparison_sha256": digest(COMPARISON_V22),
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
    controller_mode: str = "autonomous",
    max_turns_per_episode: int = 8,
    run_wall_seconds: float | None = None,
    exclusion_registry: Path | None = None,
) -> dict[str, Any]:
    if controller_mode not in CONTROLLER_MODES:
        raise ValueError(f"unknown controller mode: {controller_mode}")
    entries = registry(registry_path)
    if model_id not in entries:
        raise ValueError("unknown checkpoint; add its immutable entry to a versioned registry")
    entry = entries[model_id]
    binding = validate_binding(entry, json.loads(binding_path.read_text()))
    cases = select_cases(load_suite(suite_path), max_episodes)
    manifest = json.loads((suite_path / "manifest.json").read_text())
    from .gates import validate_panel

    validate_panel(cases, suite_path, exclusion_registry)
    controller = ResearchController(controller_mode)

    _start(out)
    started = time.monotonic()
    per_ep_turns = min(LIMITS["turns"], max(1, int(max_turns_per_episode)))
    max_actions_total = min(len(cases) * per_ep_turns, binding["resource_caps"]["max_requests"] // 2)
    run_seconds = min(
        float(binding["resource_caps"]["wall_seconds"]),
        float(run_wall_seconds) if run_wall_seconds is not None else 1800.0,
    )
    comparison = {
        **COMPARISON_V22,
        "max_turns_per_episode": per_ep_turns,
        "effective_max_actions": max_actions_total,
        "effective_max_http_requests": 2 * max_actions_total,
        "effective_run_seconds": run_seconds,
    }
    deadline = started + run_seconds

    rows: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    requests = 0
    http_requests = 0
    inference_requests = 0
    successful_responses = 0
    failures = 0
    unknown_usage_calls = 0
    interventions = 0
    skipped_model_turns = 0
    externalized_persona_interventions = 0
    intervention_reasons: Counter[str] = Counter()

    traces_path = out / "transcript.jsonl"
    with traces_path.open("w") as traces:
        for case in cases:
            session = ResearchSessionV21(case, deadline=deadline)
            observation = session.observation(status="ready")
            stop_reason = "turn_limit"
            ep_interventions = 0
            for turn_idx in range(per_ep_turns):
                remaining = session.remaining_seconds()
                if remaining <= 0:
                    stop_reason = "run_budget_exhausted"
                    break

                # Pre-turn controller check (e.g., stop immediately after verified check or matched learner pair)
                pre = controller.pre_turn_decision(
                    case=case,
                    state=session.compact_state,
                    turn_index=turn_idx,
                    remaining=session._remaining(),
                )
                if pre.skip_model_call and pre.executed_text is not None:
                    interventions += 1
                    ep_interventions += 1
                    skipped_model_turns += 1
                    if pre.reason:
                        intervention_reasons[pre.reason] += 1
                    if pre.externalized_persona_used:
                        externalized_persona_interventions += 1
                    observation = session.step_json(pre.executed_text)
                    traces.write(
                        canon(
                            {
                                "task_id": case["id"],
                                "turn": turn_idx + 1,
                                "controller_mode": controller_mode,
                                "intervened": True,
                                "skip_model_call": True,
                                "intervention_reason": pre.reason,
                                "externalized_persona_used": pre.externalized_persona_used,
                                "executed_action_sha256": digest(pre.executed_text),
                                "observation": observation,
                            }
                        )
                        + "\n"
                    )
                    traces.flush()
                    if session.done:
                        stop_reason = "session_closed"
                        break
                    continue

                if requests >= max_actions_total:
                    stop_reason = "run_budget_exhausted"
                    break
                prompt = case["prompt"] + "\nCurrent observation:\n" + canon(observation) + "\nAction:\n"
                requests += 1
                try:
                    response = request_action(
                        entry,
                        binding,
                        prompt,
                        max_tokens=COMPARISON_V22["max_new_tokens_per_turn"],
                        timeout_seconds=min(30.0, remaining),
                    )
                    http_requests += response.get("http_requests", 2)
                    inference_requests += response.get("inference_requests", 1)
                    successful_responses += 1
                    for field in usage:
                        usage[field] += response["usage"][field]
                    post = controller.post_model_decision(
                        case=case,
                        state=session.compact_state,
                        turn_index=turn_idx,
                        raw_text=response["text"],
                        artifacts=session._artifacts,
                        remaining=session._remaining(),
                    )
                    exec_text = post.executed_text if post.executed_text is not None else response["text"]
                    if post.intervened:
                        interventions += 1
                        ep_interventions += 1
                        if post.reason:
                            intervention_reasons[post.reason] += 1
                        if post.externalized_persona_used:
                            externalized_persona_interventions += 1
                    observation = session.step_json(exec_text)
                    traces.write(
                        canon(
                            {
                                "task_id": case["id"],
                                "request": requests,
                                "turn": turn_idx + 1,
                                "prompt_sha256": digest(prompt),
                                "controller_mode": controller_mode,
                                "intervened": post.intervened,
                                "intervention_reason": post.reason,
                                "externalized_persona_used": post.externalized_persona_used,
                                "raw_model_action_sha256": digest(response["text"]),
                                "executed_action_sha256": digest(exec_text),
                                "response": response,
                                "observation": observation,
                            }
                        )
                        + "\n"
                    )
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
                                "turn": turn_idx + 1,
                                "controller_mode": controller_mode,
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
            result["controller_mode"] = controller_mode
            result["controller_interventions"] = ep_interventions
            core_hash = result.pop("result_sha256")
            result["core_result_sha256"] = core_hash
            result["result_sha256"] = digest(result)
            rows.append(result)

    identity = {"registry_entry": entry, "binding": binding, "controller": controller.metadata()}
    candidate_id = f"{model_id}@{controller_mode}"
    return _finish(
        out,
        rows,
        {
            "candidate_id": candidate_id,
            "base_model_id": model_id,
            "controller_mode": controller_mode,
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
            "request_failures": failures,
            "usage": usage,
            "controller_accounting": {
                "controller_mode": controller_mode,
                "total_interventions": interventions,
                "skipped_model_turns": skipped_model_turns,
                "externalized_persona_interventions": externalized_persona_interventions,
                "episodes_with_intervention": sum(1 for r in rows if r["controller_interventions"] > 0),
                "interventions_by_reason": dict(sorted(intervention_reasons.items())),
            },
            "checkpoint_rank_eligible": False,
            "controller_compute": controller.accounting(),
            "wall_seconds": round(time.monotonic() - started, 6),
        },
    )


def read_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from datagoblin.research_v2.learners import validate_artifact

    path = path / "report.json" if path.is_dir() else path
    report = json.loads(path.read_text())
    if digest({k: v for k, v in report.items() if k != "report_sha256"}) != report.get("report_sha256"):
        raise ValueError("report digest mismatch")
    if report.get("schema") != VERSION or digest(report["comparison"]) != report["comparison_sha256"]:
        raise ValueError("unsupported v2.2 report or mismatched comparison contract")
    data = (path.parent / "results.jsonl").read_bytes()
    if hashlib.sha256(data).hexdigest() != report["results_sha256"]:
        raise ValueError("result bytes changed")
    rows = [json.loads(line) for line in data.splitlines()]
    if [r["task_id"] for r in rows] != report["task_ids"] or len(set(report["task_ids"])) != len(rows):
        raise ValueError("result membership changed")
    for row in rows:
        if digest({k: v for k, v in row.items() if k != "result_sha256"}) != row.get("result_sha256"):
            raise ValueError("row digest mismatch")
    if aggregate(rows) != report["summary"]:
        raise ValueError("aggregate does not replay")
    for name, expected in report.get("artifact_files", {}).items():
        artifact_path = path.parent / name
        raw = artifact_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("artifact file digest mismatch")
        validate_artifact(json.loads(raw))
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
                "task_ids",
                "role",
                "experiment_runtime",
                "comparison_sha256",
                "base_model_id",
            )
        ):
            raise ValueError("reports require identical tasks, code, budgets, actor and role")
        if report.get("identity", {}).get("binding") != baseline.get("identity", {}).get("binding"):
            raise ValueError("actor binding mismatch")
    if len({r[0]["candidate_id"] for r in loaded}) != len(loaded):
        raise ValueError("duplicate candidate identity")
    comparisons = []
    for report, rows in loaded[1:]:
        effects = paired_effects(base_rows, rows, families=FAMILIES)
        comparison = {
            "candidate_id": report["candidate_id"],
            "reference_id": baseline["candidate_id"],
            "controller_mode": report.get("controller_mode"),
            **effects,
            "rank_eligible": False,
        }
        if (baseline.get("controller_mode") == "rlt_state_correction"
                and report.get("controller_mode") == "selective_rlt"):
            comparison["selective_gate"] = selective_gate(
                effects, baseline.get("controller_compute", {}),
                report.get("controller_compute", {}),
            )
        comparisons.append(comparison)
    return {
        "schema": "research-v2.2-paired-comparison-1",
        "suite_sha256": baseline["suite_sha256"],
        "candidates": [
            {
                "id": r["candidate_id"],
                "controller_mode": r.get("controller_mode"),
                "summary": r["summary"],
                "controller_accounting": r.get("controller_accounting"),
                "controller_compute": r.get("controller_compute"),
            }
            for r, _ in loaded
        ],
        "paired": comparisons,
        "winner": None,
        "scope": (
            "Research Verifiers v2.2 paired comparison preserving strict primary_score "
            "alongside intermediate capability_score."
        ),
    }
