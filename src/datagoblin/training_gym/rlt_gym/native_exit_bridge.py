"""Import BPD's audited pretrained TRM verifier-exit receipts for RLT evaluation.

All ingested rows remain DEVELOPMENT-ONLY and training-ineligible. The
observable is the actual Sudoku verifier; no model hidden state is imported.
The source BPD instrument tests native learned recurrence, not Inkling's
Attention Residual readers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file
from datagoblin.serialization import canonical_json, dump_structured

MODEL_SHA256 = "932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
SRC_SCHEMA = "bpd.native_trm_verifier_exit.v1"
AUDIT_SCHEMA = "bpd.native_trm_verifier_exit_audit.v1"


def _source(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("native exit source exceeds bound")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("native exit source must be a JSON object")
    return value


def _sudoku_valid(question: str, candidate: str) -> bool:
    if not isinstance(candidate, str) or len(candidate) != 81:
        return False
    if not isinstance(question, str) or len(question) != 81:
        return False
    if any(q not in "123456789." for q in question) or any(v not in "123456789" for v in candidate):
        return False
    if any(q != "." and q != candidate[i] for i, q in enumerate(question)):
        return False
    for row in range(9):
        if {candidate[row * 9 + col] for col in range(9)} != set("123456789"):
            return False
    for col in range(9):
        if {candidate[row * 9 + col] for row in range(9)} != set("123456789"):
            return False
    for row in (0, 3, 6):
        for col in (0, 3, 6):
            group = {candidate[(row + i) * 9 + col + j] for i in range(3) for j in range(3)}
            if group != set("123456789"):
                return False
    return True


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(canonical_json(row) + "\n")


def import_native_exit(source_path: Path, audit_path: Path, out: Path) -> dict[str, Any]:
    """Recheck the archived source/audit hashes, custody, and every verifier label."""
    source_sha = sha256_file(source_path)
    source = _source(source_path)
    audit = _source(audit_path)
    if (
        source.get("schema_version") != SRC_SCHEMA
        or source.get("model") != "gaoxin492"
        or source.get("model_sha256") != MODEL_SHA256
        or source.get("evidence_class") != "native_pretrained_development"
        or source.get("split") != "development"
        or source.get("student_training_eligible") is not False
        or source.get("check_uses_planted_solution") is not False
    ):
        raise ValueError("unsupported checkpoint/provenance/split custody")
    if (
        audit.get("schema_version") != AUDIT_SCHEMA
        or audit.get("receipt_sha256") != source_sha
        or audit.get("checkpoint_sha256") != MODEL_SHA256
        or audit.get("root_and_checker_replay_passed") is not True
        or audit.get("source_roots") != source.get("roots")
    ):
        raise ValueError("independent audit missing or bound to different evidence")
    roots = source.get("roots")
    if type(roots) is not int or not 1 <= roots <= 64:
        raise ValueError("invalid source-root count")
    if len(source.get("records", [])) != roots:
        raise ValueError("incomplete root roster")
    if source.get("max_outer_steps") != 2:
        raise ValueError("expected exactly two outer steps")
    seen: set[str] = set()
    transitions: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    first_success = 0
    for item in source["records"]:
        question, root = item.get("question"), item.get("root_id")
        if not isinstance(root, str) or not root or root in seen:
            raise ValueError("duplicate or invalid root identity")
        seen.add(root)
        if not isinstance(question, str) or len(question) != 81 or question.count(".") != source["holes"]:
            raise ValueError("malformed Sudoku source world")
        task_digest = _digest({"question": question})
        if item.get("task_digest") != task_digest or item.get("model_sha256") != MODEL_SHA256:
            raise ValueError("task/checkpoint binding changed")
        policies = item.get("policies", {})
        if not isinstance(policies, dict) or set(policies) != {"fixed_2", "verifier_exit"}:
            raise ValueError("incomplete policy comparison")
        fixed, adaptive = policies["fixed_2"], policies["verifier_exit"]
        fsteps, asteps = fixed["trace"], adaptive["trace"]
        if len(fsteps) != 2 or len(asteps) not in (1, 2):
            raise ValueError("incomplete fixed/early-exit trace")
        if [r["candidate"] for r in fsteps[:len(asteps)]] != [r["candidate"] for r in asteps]:
            raise ValueError("adaptive prefix differs from fixed recurrence")
        verified = []
        for index, step in enumerate(fsteps):
            if step.get("outer_step") != index + 1:
                raise ValueError("noncontiguous recurrence visit")
            passes = _sudoku_valid(question, step.get("candidate"))
            if step.get("sudoku_checker_passed") is not passes:
                raise ValueError("verifier outcome does not replay")
            verified.append(passes)
            transitions.append({
                "schema_version": "rlt_gym.native_exit_transition.v1",
                "root_id": root, "task_digest": task_digest, "family": "sudoku9",
                "split": "development", "teacher_kind": "native_trm",
                "model_sha256": MODEL_SHA256, "outer_step": index + 1,
                "input": {"question": question, "candidate": step["candidate"]},
                "checker_passed": passes,
                "target": {"exit_decision": "stop" if passes else "continue"},
                "next_step_available": index == 0,
                "mean_blank_logit_margin": step.get("mean_blank_logit_margin"),
                "min_blank_logit_margin": step.get("min_blank_logit_margin"),
                "training_eligible": False,
                "training_ineligible_reason": "source_split_development",
            })
        if verified[0] != (len(asteps) == 1):
            raise ValueError("adaptive stop/continue decision inconsistent")
        first_success += int(verified[0])
        episodes.append({
            "schema_version": "rlt_gym.native_exit_episode.v1",
            "root_id": root, "task_digest": task_digest, "family": "sudoku9",
            "split": "development", "model_sha256": MODEL_SHA256,
            "first_verified_pass": 1 if verified[0] else 2 if verified[1] else None,
            "fixed_correct": verified[1],
            "adaptive_correct": verified[len(asteps) - 1],
            "fixed_visits": 2, "adaptive_visits": len(asteps),
            "training_eligible": False,
        })
    if first_success != audit["depth_one_verified"]:
        raise ValueError("first-depth answer rate differs from independent audit")
    if sum(ep["fixed_correct"] for ep in episodes) != audit["fixed_solved"]:
        raise ValueError("fixed-depth answer rate differs from audit")
    if sum(ep["adaptive_correct"] for ep in episodes) != audit["adaptive_solved"]:
        raise ValueError("early-exit answer rate differs from audit")
    if sum(ep["adaptive_visits"] for ep in episodes) != audit["adaptive_steps"]:
        raise ValueError("early-exit compute differs from independent audit")
    if out.exists():
        raise ValueError("fresh output path required")
    out.mkdir(parents=True)
    _write_jsonl(out / "depth_transitions.jsonl", transitions)
    _write_jsonl(out / "episodes.jsonl", episodes)
    manifest = {
        "schema_version": "rlt_gym.native_exit_bridge.v1",
        "evidence_class": "descriptive_pretrained_development_only",
        "split": "development", "training_eligible": False,
        "model_sha256": MODEL_SHA256,
        "source_sha256": source_sha, "source_audit_sha256": sha256_file(audit_path),
        "root_count": roots, "transition_count": len(transitions),
        "first_pass_verified": first_success,
        "fixed_2_correct": audit["fixed_solved"],
        "verifier_exit_correct": audit["adaptive_solved"],
        "fixed_2_visits": audit["fixed_steps"],
        "verifier_exit_visits": audit["adaptive_steps"],
        "step_fraction_saved": audit["step_fraction_saved"],
        "measured_native_fraction_saved": audit["measured_native_fraction_saved"],
        "files": {
            name: sha256_file(out / name)
            for name in ("depth_transitions.jsonl", "episodes.jsonl")
        },
        "claim_boundary": (
            "Observations are real pretrained TRM outcomes rechecked by task verifier. "
            "All development roots and labels are excluded from student training. "
            "This is a task-specific verifier-gated exit, not learned RLT controller or Inkling routing."
        ),
    }
    dump_structured(out / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = import_native_exit(args.input, args.audit, args.out)
    print(json.dumps({k: value[k] for k in (
        "root_count", "first_pass_verified", "verifier_exit_correct",
        "verifier_exit_visits", "training_eligible",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
