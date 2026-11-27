#!/usr/bin/env python3
"""Deterministic, development-only Semaev research-claim audit.

The grader checks calibrated source-scope judgments and required missing obligations.
It does not verify new mathematical proofs or approve ECDSA security claims.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ALLOWED_STATUS = {"scope_limited", "conditional", "unresolved", "pilot_only"}
ALLOWED_CHECKS = {"formal_scope_audit", "lean_proof", "exact_certificate_check", "model_cost_audit"}


def load_tasks(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "development_only_unsealed":
        raise ValueError("This runner only accepts publicly labeled development tasks")
    tasks = {row["id"]: row for row in data["tasks"]}
    if len(tasks) != len(data["tasks"]):
        raise ValueError("Duplicate task ID")
    return tasks


def evaluate(tasks: dict[str, dict], responses: list[dict]) -> dict:
    per_task = []
    seen = set()
    for response in responses:
        tid = response.get("task_id")
        if tid not in tasks or tid in seen:
            raise ValueError(f"unknown or duplicate task ID: {tid}")
        seen.add(tid)
        task = tasks[tid]
        status = response.get("claim_status")
        if status not in ALLOWED_STATUS:
            raise ValueError(f"invalid claim_status on {tid}")
        obligations = response.get("missing_obligations", [])
        evidence = response.get("evidence_paths", [])
        check = response.get("next_check_kind")
        confidence = response.get("confidence")
        if not isinstance(obligations, list) or not all(isinstance(v, str) for v in obligations):
            raise ValueError(f"invalid obligations on {tid}")
        if not isinstance(evidence, list) or not all(isinstance(v, str) for v in evidence):
            raise ValueError(f"invalid evidence paths on {tid}")
        if len(set(obligations)) != len(obligations):
            raise ValueError(f"repeated obligation on {tid}")
        if check not in ALLOWED_CHECKS:
            raise ValueError(f"invalid next check on {tid}")
        if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not 0 <= confidence <= 1:
            raise ValueError(f"confidence must be in [0,1] on {tid}")
        required = set(task["required_obligations"])
        offered = set(obligations)
        missing_coverage = len(required & offered) / len(required)
        extra = len(offered - required)
        precision = len(required & offered) / len(offered) if offered else 0.0
        status_correct = status == task["gold_status"]
        evidence_credit = len(set(evidence) & set(task["source_paths"])) / len(task["source_paths"])
        check_credit = check == task["next_check_kind"]
        score = (
            0.25 * float(status_correct)
            + 0.25 * missing_coverage
            + 0.15 * precision
            + 0.20 * evidence_credit
            + 0.15 * float(check_credit)
        )
        # Accidental invention of missing obligations should lose credit.
        score = max(0.0, score - 0.05 * extra)
        per_task.append({
            "task_id": tid, "score": round(score, 6), "scope_correct": status_correct,
            "obligation_recall": round(missing_coverage, 6),
            "obligation_precision": round(precision, 6),
            "referenced_source_fraction": round(evidence_credit, 6),
            "next_check_correct": check_credit,
            "confidence": confidence,
            "status_brier": round((float(confidence) - float(status_correct)) ** 2, 6),
        })
    per_task.sort(key=lambda row: row["task_id"])
    if not per_task:
        raise ValueError("No responses to grade")
    aggregate = defaultdict(float)
    for row in per_task:
        for key in ("score", "status_brier", "obligation_recall", "obligation_precision"):
            aggregate[key] += row[key]
    return {
        "status": "development_only_unsealed",
        "scoring_scope": "claim-scope and obligation recognition; no new proof verification",
        "tasks_expected": len(tasks), "tasks_graded": len(per_task),
        "mean_score": round(aggregate["score"] / len(per_task), 6),
        "mean_status_brier": round(aggregate["status_brier"] / len(per_task), 6),
        "mean_obligation_recall": round(aggregate["obligation_recall"] / len(per_task), 6),
        "mean_obligation_precision": round(aggregate["obligation_precision"] / len(per_task), 6),
        "per_task": per_task,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=Path(__file__).with_name("tasks_dev.json"))
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tasks = load_tasks(args.tasks)
    responses = [json.loads(line) for line in args.responses.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = evaluate(tasks, responses)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"graded {report['tasks_graded']} / {report['tasks_expected']} development tasks; "
        f"score {report['mean_score']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
