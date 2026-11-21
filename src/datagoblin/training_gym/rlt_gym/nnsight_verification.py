"""Join NNsight mechanistic receipts to independently verified model completions.

This module checks the identity and task split of an operator-collected set of
completed answers. It cannot attest that a completion was generated under a
particular intervention; only an integrated generation capture can do that.
No rows produced here are eligible for student training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datagoblin.serialization import canonical_json

SCHEMAS = frozenset({"rlt_gym.nnsight_loop_probe.v1", "rlt_gym.nnsight_message_bank.v1"})
MAX_CANDIDATES = 16
MAX_LINE_BYTES = 131_072


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_probe_report(report: dict[str, Any], task: dict[str, Any]) -> None:
    if report.get("schema_version") not in SCHEMAS:
        raise ValueError("unknown mechanistic report schema")
    if report.get("task_digest") != task.get("task_digest"):
        raise ValueError("probe/task digest mismatch")
    if task.get("split") == "sealed_poison":
        raise ValueError("sealed poison tasks remain outside the interpretation lane")
    if report.get("verifier_status") != "unverified" or report.get("student_training_eligible") is not False:
        raise ValueError("probe receipt must be unverified and training-ineligible")
    expected_hash = report.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("probe receipt SHA-256 is required")
    canon = {key: value for key, value in report.items() if key != "sha256"}
    if _digest(canon) != expected_hash:
        raise ValueError("probe receipt hash mismatch")
    arms = report.get("arms")
    if not isinstance(arms, dict) or not {"observe", "sham", "lesion", "random"}.issubset(arms):
        raise ValueError("required mechanistic controls are missing")
    if set(arms) - {"observe", "sham", "lesion", "random", "donor_patch"}:
        raise ValueError("unknown probe arm")
    for name, arm in arms.items():
        if not isinstance(arm, dict):
            raise ValueError("invalid mechanistic control arm")
        plan = arm.get("plan")
        if (not isinstance(plan, dict) or plan.get("model_id") != report["model_id"]
                or plan.get("revision") != report["revision"]
                or plan.get("task_digest") != task["task_digest"] or plan.get("kind") != name
                or arm.get("verifier_status") != "unverified"
                or arm.get("student_training_eligible") is not False):
            raise ValueError("unbound mechanistic control arm")


def join_verified_candidates(
    report: dict[str, Any], candidates: list[dict[str, Any]], task: dict[str, Any],
    verify_fn: Callable[[dict[str, Any], Any], dict[str, Any]],
) -> dict[str, Any]:
    """Recheck each completed answer against the real environment verifier.

    A report's source can still be operator-declared. The join reports verdicts
    about supplied answers and never claims intervention-conditioned generation.
    """
    validate_probe_report(report, task)
    arms = report["arms"]
    if len(candidates) != len(arms) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError("exactly one completed response per probed arm is required")
    checked: dict[str, dict[str, Any]] = {}
    for row in candidates:
        if not isinstance(row, dict):
            raise ValueError("response must be a JSON object")
        name = row.get("arm")
        if not isinstance(name, str) or name not in arms or name in checked:
            raise ValueError("unknown or duplicate arm response")
        source = arms[name]
        if (row.get("task_id") != task.get("task_id")
                or row.get("task_digest") != task["task_digest"]
                or row.get("probe_report_sha256") != report["sha256"]
                or row.get("model_id") != report["model_id"]
                or row.get("revision") != report["revision"]
                or row.get("prompt_sha256") != source.get("prompt_sha256")):
            raise ValueError("completion provenance differs from frozen probe receipt")
        if "candidate" not in row:
            raise ValueError("completion is missing candidate JSON")
        verdict = verify_fn(task, row["candidate"])
        if (not isinstance(verdict, dict) or type(verdict.get("passed")) is not bool
                or verdict.get("task_digest") != task["task_digest"]):
            raise RuntimeError("environment verifier returned invalid receipt")
        checked[name] = {
            "candidate_digest": _digest(row["candidate"]),
            "verification": verdict,
            "model_execution_attested": False,
        }
    if set(checked) != set(arms):
        raise ValueError("incomplete arm coverage")
    sham_passed = checked["sham"]["verification"]["passed"]
    return {
        "schema_version": "rlt_gym.nnsight_verifier_join.v1",
        "probe_report_sha256": report["sha256"],
        "task_id": task["task_id"],
        "task_digest": task["task_digest"],
        "split": task["split"],
        "arms": checked,
        "matched_controls": {
            name: {
                "passed": val["verification"]["passed"],
                "differs_from_sham": val["verification"]["passed"] != sham_passed,
            }
            for name, val in checked.items()
        },
        "model_execution_attested": False,
        "causal_effect_verified": False,
        "student_training_eligible": False,
        "claim_boundary": (
            "The environment verifier checked supplied candidate JSON. "
            "This join does not establish that candidates were generated under the recorded interventions."
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            raise ValueError(f"oversized candidate row {index}")
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"candidate row {index} is not an object")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from .tasks import load_suite, verify

    manifest = args.suite / "manifest.json" if args.suite.is_dir() else args.suite
    _, all_tasks = load_suite(manifest)
    tasks = [task for task in all_tasks if task["task_id"] == args.task_id]
    if len(tasks) != 1:
        raise ValueError("task_id not present in frozen suite")
    report = json.loads(args.probe.read_text(encoding="utf-8"))
    result = join_verified_candidates(report, _read_jsonl(args.candidates), tasks[0], verify)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        output.write(canonical_json(result) + "\n")
    print(canonical_json({"result": str(args.out), "verifier_checked_arms": len(result["arms"]),
                          "model_execution_attested": False}))


if __name__ == "__main__":
    main()
