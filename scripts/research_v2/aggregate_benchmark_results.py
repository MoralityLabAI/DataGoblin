"""Aggregate verified Research Verifiers v2 held-panel runs, telemetry, and paired comparisons.

Reads immutable run reports via datagoblin.research_v2.runner.read_report(),
runs datagoblin.research_v2.runner.compare() across all paired checkpoint runs,
extracts concrete failure cases and learner-improvement metrics, and merges
explicit blocked-model records into one auditable benchmark ledger JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datagoblin.research_v2.runner import compare, read_report
from datagoblin.research_v2.suite import FAMILIES, LEARNERS, digest


def _summarize_reply_safely(raw_text: str) -> dict[str, Any]:
    """Summarize a model reply without leaking held-panel prompt/answer payloads."""
    stripped = raw_text.strip()
    summary: dict[str, Any] = {
        "char_len": len(raw_text),
        "reply_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16],
        "valid_json": False,
    }
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            summary["valid_json"] = True
            summary["action"] = obj.get("action")
            if isinstance(obj.get("config"), dict):
                summary["config_keys"] = sorted(str(k) for k in obj["config"])
            if isinstance(obj.get("answer"), dict):
                summary["answer_keys"] = sorted(str(k) for k in obj["answer"])
            elif obj.get("answer") is not None:
                summary["answer_type"] = type(obj["answer"]).__name__
    except Exception:
        pass
    return summary


def _summarize_run(run_dir: Path, meta_dir: Path | None) -> dict[str, Any]:
    report, rows = read_report(run_dir)
    telemetry = {}
    if meta_dir is not None and (meta_dir / "telemetry.json").is_file():
        telemetry = json.loads((meta_dir / "telemetry.json").read_text(encoding="utf-8"))

    # Parse transcript for error codes, timeouts, truncations, invalid JSON, abstentions
    transcript_path = run_dir / "transcript.jsonl"
    error_codes: Counter[str] = Counter()
    obs_errors: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    check_passed_count = 0
    check_failed_count = 0
    failure_samples: list[dict[str, Any]] = []
    if transcript_path.is_file():
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "error" in rec:
                rcpt = rec.get("transport_receipt") or {}
                code = str(rcpt.get("error_code") or rec["error"])
                error_codes[code] += 1
                if len(failure_samples) < 8:
                    failure_samples.append({
                        "task_id": rec["task_id"],
                        "request": rec["request"],
                        "kind": "transport_error",
                        "error_code": code,
                        "latency_seconds": round(float(rcpt.get("latency_seconds", 0.0)), 4),
                        "prompt_tokens": rcpt.get("prompt_tokens"),
                        "usage": rcpt.get("usage"),
                    })
            else:
                obs = rec.get("observation") or {}
                reply_meta = _summarize_reply_safely(str(rec.get("response", {}).get("text", "")))
                if reply_meta.get("action"):
                    action_counts[str(reply_meta["action"])] += 1
                if obs.get("check_passed") is True:
                    check_passed_count += 1
                elif obs.get("check_passed") is False:
                    check_failed_count += 1
                if "error" in obs:
                    err_msg = str(obs["error"])
                    obs_errors[err_msg] += 1
                    if len(failure_samples) < 12:
                        failure_samples.append({
                            "task_id": rec["task_id"],
                            "request": rec["request"],
                            "kind": "session_action_error",
                            "error": err_msg,
                            "reply_summary": reply_meta,
                        })
                elif obs.get("check_passed") is True and len(failure_samples) < 14:
                    failure_samples.append({
                        "task_id": rec["task_id"],
                        "request": rec["request"],
                        "kind": "check_passed_without_subsequent_submit",
                        "check_passed": True,
                        "reply_summary": reply_meta,
                    })

    # Learner metrics & verifier calls across rows
    stop_reasons = Counter(r.get("rollout_stop_reason", "unknown") for r in rows)
    total_turns = sum(r["cost"]["turns"] for r in rows)
    total_experiments = sum(r["cost"]["experiments"] for r in rows)
    total_checks = sum(r["cost"]["checks"] for r in rows)
    total_opt_updates = sum(r["cost"]["optimizer_updates_reserved"] for r in rows)
    total_cell_calls = sum(r["cost"]["measured_cell_calls"] for r in rows)
    total_errors = sum(r["cost"]["errors"] for r in rows)
    abstentions = sum(
        1
        for r in rows
        if not r["completed"]
        and r["cost"]["turns"] == 0
        and r.get("rollout_stop_reason") == "run_budget_exhausted"
    )

    learner_deltas = []
    weights_changed_count = 0
    for r in rows:
        if r["family"] in LEARNERS and r.get("metrics"):
            m = r["metrics"]
            if "delta_from_initialization" in m:
                learner_deltas.append(float(m["delta_from_initialization"]))
            if m.get("weights_changed") is True:
                weights_changed_count += 1

    identity = report.get("identity") or {}
    reg_entry = identity.get("registry_entry") or {}
    binding = identity.get("binding") or {}

    return {
        "candidate_id": report["candidate_id"],
        "role": report["role"],
        "model_evaluated": report["model_evaluated"],
        "checkpoint_rank_eligible": report["checkpoint_rank_eligible"],
        "report_sha256": report["report_sha256"],
        "results_sha256": report["results_sha256"],
        "suite_sha256": report["suite_sha256"],
        "implementation_sha256": report["implementation_sha256"],
        "comparison_sha256": report["comparison_sha256"],
        "checkpoint_metadata": {
            "model_id": reg_entry.get("model_id"),
            "revision": reg_entry.get("revision"),
            "weights_sha256": binding.get("weights_sha256") or reg_entry.get("weights_sha256"),
            "tokenizer_sha256": binding.get("tokenizer_sha256"),
            "precision": reg_entry.get("precision"),
            "protocol": reg_entry.get("protocol"),
            "parameters": reg_entry.get("parameters"),
            "expanded_export_parameters": reg_entry.get("expanded_export_parameters"),
            "recurrence": reg_entry.get("recurrence"),
            "thinking_mode": binding.get("thinking_mode"),
            "hrm_condition": binding.get("hrm_condition"),
            "cap_wrapper_receipt_sha256": binding.get("cap_wrapper_receipt_sha256"),
        },
        "summary": report["summary"],
        "rollout_accounting": {
            "attempted_actions": report.get("attempted_actions", 0),
            "http_requests": report.get("http_requests", 0),
            "inference_requests": report.get("inference_requests", 0),
            "successful_responses": report.get("successful_responses", 0),
            "request_failures": report.get("request_failures", 0),
            "unknown_usage_calls": report.get("inference_requests_with_unknown_usage", 0),
            "stop_reasons": dict(sorted(stop_reasons.items())),
            "action_counts": dict(sorted(action_counts.items())),
            "verifier_checks_passed": check_passed_count,
            "verifier_checks_failed": check_failed_count,
            "transport_error_codes": dict(sorted(error_codes.items())),
            "session_error_messages": dict(sorted(obs_errors.items())),
            "total_turns": total_turns,
            "total_experiments": total_experiments,
            "total_verifier_checks": total_checks,
            "total_optimizer_updates_reserved": total_opt_updates,
            "total_measured_cell_calls": total_cell_calls,
            "total_format_or_tool_errors": total_errors,
            "budget_exhausted_unattempted_cases": abstentions,
            "learner_submitted_deltas": learner_deltas,
            "learner_mean_delta_from_init": (
                round(sum(learner_deltas) / len(learner_deltas), 6) if learner_deltas else None
            ),
            "learner_weights_changed_submissions": weights_changed_count,
            "persisted_artifact_count": len(report.get("artifact_files", {})),
        },
        "resources": {
            "wall_seconds": report.get("wall_seconds", 0.0),
            "prompt_tokens": (report.get("usage") or {}).get("prompt_tokens", 0),
            "completion_tokens": (report.get("usage") or {}).get("completion_tokens", 0),
            "device": telemetry.get("device"),
            "cpu_seconds": telemetry.get("cpu_seconds"),
            "gpu_seconds": telemetry.get("gpu_seconds"),
            "peak_ram_mib": telemetry.get("peak_ram_mib"),
            "peak_vram_mib": telemetry.get("peak_vram_mib"),
            "load_seconds": telemetry.get("load_seconds"),
            "parity_checks": telemetry.get("parity_checks"),
        },
        "concrete_failure_samples": failure_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-run", type=Path, action="append", default=[])
    parser.add_argument("--dev-run", type=Path, action="append", default=[])
    parser.add_argument("--control-run", type=Path, action="append", default=[])
    parser.add_argument("--blocked-json", type=Path, default=None)
    parser.add_argument("--persona-report", type=Path, default=None)
    parser.add_argument("--dataset-receipt", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def meta_for(run_dir: Path) -> Path | None:
        m = run_dir.parent / f".meta-{run_dir.name}"
        return m if m.is_dir() else None

    held_summaries = [_summarize_run(p, meta_for(p)) for p in args.held_run]
    dev_summaries = [_summarize_run(p, meta_for(p)) for p in args.dev_run]
    control_summaries = [_summarize_run(p, None) for p in args.control_run]

    paired_held = compare(args.held_run) if len(args.held_run) >= 2 else None
    paired_dev = compare(args.dev_run) if len(args.dev_run) >= 2 else None
    blocked = (
        json.loads(args.blocked_json.read_text(encoding="utf-8"))
        if args.blocked_json and args.blocked_json.is_file()
        else []
    )
    persona = (
        json.loads(args.persona_report.read_text(encoding="utf-8"))
        if args.persona_report and args.persona_report.is_file()
        else None
    )
    dataset = (
        json.loads(args.dataset_receipt.read_text(encoding="utf-8"))
        if args.dataset_receipt and args.dataset_receipt.is_file()
        else None
    )

    ledger = {
        "schema": "datagoblin-research-v2-benchmark-ledger-1",
        "families": list(FAMILIES),
        "held_panel_runs": held_summaries,
        "held_panel_paired_comparison": paired_held,
        "dev_fixture_runs": dev_summaries,
        "dev_fixture_paired_comparison": paired_dev,
        "deterministic_controls": control_summaries,
        "blocked_or_deferred_registry_entries": blocked,
        "jinn_beast_persona_experiment": persona,
        "specialist_candidate_dataset_receipt": dataset,
    }
    ledger["ledger_sha256"] = digest(ledger)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.out.write_bytes(raw)
    print(json.dumps({
        "out": str(args.out),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "held_runs": len(held_summaries),
        "dev_runs": len(dev_summaries),
        "blocked": len(blocked),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
