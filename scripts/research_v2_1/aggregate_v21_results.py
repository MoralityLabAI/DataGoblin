"""Aggregate Research Verifiers v2.1 runs, paired comparisons, Jinn/Beast disentanglement, and Baby Goblin exp."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datagoblin.research_v2.suite import FAMILIES, digest
from datagoblin.research_v2_1.runner import compare, read_report


def _summarize_v21_run(run_dir: Path, meta_dir: Path | None) -> dict[str, Any]:
    report, rows = read_report(run_dir)
    telemetry = {}
    if meta_dir is not None and (meta_dir / "telemetry.json").is_file():
        telemetry = json.loads((meta_dir / "telemetry.json").read_text(encoding="utf-8"))

    stop_reasons = Counter(r.get("rollout_stop_reason", "unknown") for r in rows)
    milestones_agg = Counter()
    for r in rows:
        for k, v in (r.get("milestones") or {}).items():
            if v is True:
                milestones_agg[k] += 1

    learner_deltas = [
        float(r["metrics"]["delta_from_initialization"])
        for r in rows
        if isinstance(r.get("metrics"), dict) and "delta_from_initialization" in r["metrics"]
    ]
    return {
        "candidate_id": report["candidate_id"],
        "base_model_id": report.get("base_model_id"),
        "controller_mode": report.get("controller_mode"),
        "role": report["role"],
        "checkpoint_rank_eligible": report["checkpoint_rank_eligible"],
        "report_sha256": report["report_sha256"],
        "results_sha256": report["results_sha256"],
        "suite_sha256": report["suite_sha256"],
        "implementation_sha256": report["implementation_sha256"],
        "comparison_sha256": report["comparison_sha256"],
        "summary": report["summary"],
        "controller_accounting": report.get("controller_accounting"),
        "rollout_accounting": {
            "attempted_actions": report.get("attempted_actions", 0),
            "inference_requests": report.get("inference_requests", 0),
            "successful_responses": report.get("successful_responses", 0),
            "request_failures": report.get("request_failures", 0),
            "stop_reasons": dict(sorted(stop_reasons.items())),
            "total_turns": sum(r["cost"]["turns"] for r in rows),
            "total_experiments": sum(r["cost"]["experiments"] for r in rows),
            "total_checks": sum(r["cost"]["checks"] for r in rows),
            "total_errors": sum(r["cost"]["errors"] for r in rows),
            "milestones_achieved_cases": dict(sorted(milestones_agg.items())),
            "learner_mean_delta_from_init": (
                round(sum(learner_deltas) / len(learner_deltas), 6) if learner_deltas else None
            ),
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
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prospective-run", type=Path, action="append", default=[])
    parser.add_argument("--dev-run", type=Path, action="append", default=[])
    parser.add_argument("--control-run", type=Path, action="append", default=[])
    parser.add_argument("--jinn-disentangle", type=Path, default=None)
    parser.add_argument("--baby-goblin-exp", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def meta_for(run_dir: Path) -> Path | None:
        m = run_dir.parent / f".meta-{run_dir.name}"
        return m if m.is_dir() else None

    prospective = [_summarize_v21_run(p, meta_for(p)) for p in args.prospective_run]
    dev = [_summarize_v21_run(p, meta_for(p)) for p in args.dev_run]
    controls = [_summarize_v21_run(p, None) for p in args.control_run]

    paired_prospective = compare(args.prospective_run) if len(args.prospective_run) >= 2 else None
    paired_dev = compare(args.dev_run) if len(args.dev_run) >= 2 else None
    jinn = (
        json.loads(args.jinn_disentangle.read_text(encoding="utf-8"))
        if args.jinn_disentangle and args.jinn_disentangle.is_file()
        else None
    )
    bg_exp = (
        json.loads(args.baby_goblin_exp.read_text(encoding="utf-8"))
        if args.baby_goblin_exp and args.baby_goblin_exp.is_file()
        else None
    )

    ledger = {
        "schema": "datagoblin-research-v2.1-benchmark-ledger-1",
        "historical_baseline_pr93_commit": "d4aa46c3ff68f03395cc0edadd5faae05782cd84",
        "families": list(FAMILIES),
        "prospective_confirmation_runs": prospective,
        "prospective_paired_comparison": paired_prospective,
        "dev_investigation_runs": dev,
        "dev_paired_comparison": paired_dev,
        "deterministic_controls": controls,
        "jinn_beast_disentanglement": jinn,
        "baby_goblin_early_specialist_experiment": bg_exp,
    }
    ledger["ledger_sha256"] = digest(ledger)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.out.write_bytes(raw)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "prospective_runs": len(prospective),
                "dev_runs": len(dev),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
