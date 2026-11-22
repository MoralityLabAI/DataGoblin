#!/usr/bin/env python3
"""Replay public MeTTa candidate queries directly in pinned Hyperon.

Direct query evidence is a separate observation from assertion-pass receipts.
It is not an independent semantic oracle or an approval for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from corpus import ROOT
from native_assertions import scan_eval_sites
from public_root_splits import build_plan, read_jsonl

PACKAGE_VERSION = "0.2.10"
MAX_PROGRAM_BYTES = 32_768
MAX_STDOUT_BYTES = 65_536


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def select_public_cases(
    packets: list[dict[str, Any]], root_assignments: dict[str, str], limit: int
) -> list[dict[str, Any]]:
    """Round-robin whole roots, drawing one case per root before repeats."""
    if limit < 1:
        raise ValueError("limit must be positive")
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for packet in packets:
        sid = packet["metadata"]["lineage_root"]
        if root_assignments.get(sid) not in {
            "train_candidate", "dev_public", "test_public"
        }:
            raise ValueError("packet is missing its approved public diagnostic partition")
        by_root[sid].append(packet)
    for rows in by_root.values():
        rows.sort(key=lambda row: (row["metadata"]["evaluation_ordinal"], row["id"]))
    ordered = sorted(
        by_root,
        key=lambda sid: (
            ("train_candidate", "dev_public", "test_public").index(root_assignments[sid]),
            sid,
        ),
    )
    selected: list[dict[str, Any]] = []
    for index in range(max(map(len, by_root.values()), default=0)):
        for sid in ordered:
            rows = by_root[sid]
            if index < len(rows):
                selected.append(rows[index])
                if len(selected) == limit:
                    return selected
    return selected


def make_script(packet: dict[str, Any]) -> tuple[bytes, int]:
    payload = json.loads(packet["input"])
    prefix = payload["source_prefix"]
    query = payload["query"]
    if not isinstance(prefix, str) or not isinstance(query, str):
        raise ValueError("Expected a textual source prefix and query")
    prior_evaluations = len(scan_eval_sites(prefix))
    if prior_evaluations != packet["metadata"]["evaluation_ordinal"]:
        raise ValueError("Source-prefix evaluation ordinal differs from pinned receipt")
    script = (prefix + "\n!" + query + "\n").encode("utf-8")
    if len(script) > MAX_PROGRAM_BYTES:
        raise ValueError("Replay program exceeds byte budget")
    return script, prior_evaluations


def classify_native_output(
    exit_code: int | None,
    timed_out: bool,
    stdout: bytes,
    stderr: bytes,
    previous_evaluations: int,
) -> tuple[str, str | None]:
    """Require one output line per preceding evaluation and the added query."""
    if timed_out:
        return "timeout", None
    if exit_code != 0:
        return "exit_nonzero", None
    if len(stdout) > MAX_STDOUT_BYTES:
        return "stdout_over_budget", None
    if stderr:
        return "stderr_nonempty", None
    lines = stdout.decode("utf-8", "replace").splitlines()
    if len(lines) != previous_evaluations + 1:
        return "ambiguous_output_alignment", None
    if any(line.lstrip().startswith("[(Error ") for line in lines):
        return "native_error_atom", lines[-1]
    return "direct_query_observed", lines[-1]


def execute_script(script: bytes, timeout_sec: int) -> dict[str, Any]:
    """Run on an isolated temporary source path with a restricted environment."""
    env = {
        key: value for key, value in os.environ.items()
        if key in {"PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "PYTHONPATH"}
    }
    env["PYTHONNOUSERSITE"] = "1"
    start = time.monotonic()
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="metta05-") as directory:
        path = Path(directory) / "query.metta"
        path.write_bytes(script)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "hyperon.metta", str(path)],
                cwd=directory, env=env, capture_output=True,
                timeout=timeout_sec, check=False,
            )
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout, stderr, exit_code = exc.stdout or b"", exc.stderr or b"", None
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "wall_ms": int((time.monotonic() - start) * 1000),
        "stdout": stdout,
        "stderr": stderr,
    }


def run_pilot(
    packets_path: Path,
    exceptions_path: Path,
    out: Path,
    report_path: Path,
    max_packets: int = 24,
    timeout_sec: int = 8,
    *,
    audit_plan_path: Path | None = None,
) -> dict[str, Any]:
    if not 1 <= max_packets <= 187 or not 1 <= timeout_sec <= 30:
        raise ValueError("Pilot limits must be within fixed safe bounds")
    observed_version = version("hyperon")
    if observed_version != PACKAGE_VERSION:
        raise RuntimeError(f"Expected hyperon=={PACKAGE_VERSION}, got {observed_version}")
    with tempfile.TemporaryDirectory(prefix="metta05-audit-") as directory:
        audit_path = audit_plan_path or Path(directory) / "audit.json"
        plan = build_plan(packets_path, exceptions_path, audit_path)
        packets = read_jsonl(packets_path)
        selected = select_public_cases(packets, plan["root_assignments"], max_packets)
        observations: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for case in selected:
            script, prior = make_script(case)
            execution = execute_script(script, timeout_sec)
            status, output = classify_native_output(
                execution["exit_code"], execution["timed_out"], execution["stdout"],
                execution["stderr"], prior,
            )
            counts[status] += 1
            m = case["metadata"]
            obs = {
                "schema_version": "metta-direct-query-observation-v1",
                "candidate_id": case["id"],
                "lineage_root": m["lineage_root"],
                "partition": plan["root_assignments"][m["lineage_root"]],
                "assertion_type": m["assertion_type"],
                "evaluation_ordinal": m["evaluation_ordinal"],
                "source_blob_sha1": m["source_blob_sha1"],
                "source_context_sha256": m["source_context_sha256"],
                "answer_text_sha256": m["answer_text_sha256"],
                "script_sha256": sha256(script),
                "runtime": {
                    "package": "hyperon", "version": observed_version,
                    "python": platform.python_version(), "system": platform.system(),
                },
                "run_mode": "direct_query_from_source_prefix",
                "previous_evaluations": prior,
                "exit_code": execution["exit_code"],
                "timed_out": execution["timed_out"],
                "wall_ms": execution["wall_ms"],
                "stdout_bytes": len(execution["stdout"]),
                "stderr_bytes": len(execution["stderr"]),
                "stdout_sha256": sha256(execution["stdout"]),
                "stderr_sha256": sha256(execution["stderr"]),
                "direct_output_preview": output[:4096] if output else None,
                "direct_output_sha256": sha256(output.encode()) if output else None,
                "direct_observation": status,
                "assertion_pass_receipt_is_separate": True,
                "semantic_agreement_verified": False,
                "independent_semantic_oracle": False,
                "training_ready": False,
                "sealed_eval_eligible": False,
            }
            observations.append(obs)
            print(
                f"{case['id'][:12]}: {status} prior={prior}"
                f" ms={execution['wall_ms']}", file=sys.stderr,
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = "".join(
        json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"
        for rec in observations
    )
    out.write_text(raw_rows, encoding="utf-8")
    report = {
        "schema_version": "metta-direct-query-pilot-v1",
        "candidate_jsonl_sha256": sha256(packets_path.read_bytes()),
        "exception_jsonl_sha256": sha256(exceptions_path.read_bytes()),
        "selected_packet_ids_sha256": sha256(
            "\n".join(row["candidate_id"] for row in observations).encode()
        ),
        "observation_jsonl_sha256": sha256(raw_rows.encode()),
        "selected_cases": len(observations),
        "selected_source_roots": len({x["lineage_root"] for x in observations}),
        "observed_by_status": dict(sorted(counts.items())),
        "runtime_package": "hyperon",
        "runtime_version": observed_version,
        "run_mode": "direct_query_from_source_prefix",
        "semantic_agreement_verified": False,
        "independent_semantic_oracle": False,
        "training_ready": False,
        "sealed_eval_eligible": False,
        "scope": "native_replay_diagnostics_only",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packets", type=Path,
        default=ROOT / "training_candidates/hyperon_source_query_pairs.jsonl",
    )
    parser.add_argument(
        "--exceptions", type=Path,
        default=ROOT / "training_candidates/hyperon_query_exceptions.jsonl",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/metta/direct_query_pilot/observations.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("runs/metta/direct_query_pilot/report.json"))
    parser.add_argument("--max-packets", type=int, default=24)
    parser.add_argument("--timeout-sec", type=int, default=8)
    args = parser.parse_args()
    result = run_pilot(
        args.packets, args.exceptions, args.out, args.report,
        args.max_packets, args.timeout_sec,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
