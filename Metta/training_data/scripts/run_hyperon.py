#!/usr/bin/env python3
"""Run bounded upstream Hyperon scripts and record process-level execution receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from corpus import ROOT, safe_file, validate_corpus

SOURCE_REPO = "trueagi-io/hyperon-experimental"
SCOPE = "python/tests/scripts/"
PACKAGE_VERSION = "0.2.10"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_suite(root: Path, out: Path, report: Path, max_files: int = 20, timeout_sec: int = 12) -> dict:
    rows, _, errors = validate_corpus(root)
    if errors:
        raise ValueError("Source validation failed: " + "; ".join(errors))
    observed_version = version("hyperon")
    if observed_version != PACKAGE_VERSION:
        raise RuntimeError(f"Unexpected Hyperon version {observed_version}; require {PACKAGE_VERSION}")
    selected = [
        row for row in sorted(rows, key=lambda item: item["id"])
        if row["source_repo"] == SOURCE_REPO
        and row["source_path"].startswith(SCOPE)
        and row["source_kind"] == "upstream_test"
        and not row["imports_detected"]
        and row["embedded_test_assertion_count"] > 0
    ][:max_files]
    if not selected:
        raise RuntimeError("No independently runnable source-test candidates")
    env = {
        key: value for key, value in os.environ.items()
        if key in {"PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "PYTHONPATH"}
    }
    env["PYTHONNOUSERSITE"] = "1"
    receipts = []
    for row in selected:
        source = safe_file(root, row["raw_path"])
        command = [sys.executable, "-m", "hyperon.metta", str(source)]
        start = time.monotonic()
        timed_out = False
        try:
            process = subprocess.run(
                command,
                cwd=source.parent,
                env=env,
                capture_output=True,
                timeout=timeout_sec,
                check=False,
            )
            stdout = process.stdout
            stderr = process.stderr
            exit_code = process.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            exit_code = None
        elapsed_ms = int((time.monotonic() - start) * 1000)
        stdout_text = stdout.decode("utf-8", "replace")
        stderr_text = stderr.decode("utf-8", "replace")
        error_markers = sum(
            "Error" in line or "Assertion failed" in line
            for line in stdout_text.splitlines() + stderr_text.splitlines()
        )
        receipts.append({
            "schema_version": "1.0",
            "source_id": row["id"],
            "source_repo": row["source_repo"],
            "source_commit": row["source_commit"],
            "source_git_blob_sha1": row["upstream_blob_sha1"],
            "source_license_spdx": row["license_spdx"],
            "task_family": row["task_family"],
            "runtime": {
                "package": "hyperon",
                "version": observed_version,
                "python": platform.python_version(),
                "system": platform.system(),
                "executable": "python -m hyperon.metta",
            },
            "environment": "github_actions_ephemeral_runner",
            "run_mode": "single_unmodified_source_file",
            "timed_out": timed_out,
            "timeout_sec": timeout_sec,
            "exit_code": exit_code,
            "wall_ms": elapsed_ms,
            "stdout_sha256": sha256(stdout),
            "stderr_sha256": sha256(stderr),
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_preview": stdout_text[:16_384],
            "stderr_preview": stderr_text[:16_384],
            "source_assertions_lexically_detected": row["embedded_test_assertion_count"],
            "diagnostic_error_marker_lines": error_markers,
            "process_observation": "timeout" if timed_out else "exit_zero" if exit_code == 0 else "exit_nonzero",
            "assertions_independently_verified": 0,
            "verification_status": "not_verified",
            "model_training_eligible": False,
            "rollout_steps": None,
        })
        print(f"{row['id']}: code={exit_code} timeout={timed_out} ms={elapsed_ms} errors={error_markers}")
    outcome = {
        "schema_version": "1.0",
        "source_repo": SOURCE_REPO,
        "runtime_package": "hyperon",
        "runtime_package_version": observed_version,
        "attempted_sources": len(receipts),
        "process_exit_zero": sum(x["process_observation"] == "exit_zero" for x in receipts),
        "process_exit_nonzero": sum(x["process_observation"] == "exit_nonzero" for x in receipts),
        "process_timeouts": sum(x["process_observation"] == "timeout" for x in receipts),
        "detected_error_marker_lines": sum(x["diagnostic_error_marker_lines"] for x in receipts),
        "assertions_independently_verified": 0,
        "verified_training_rollouts": 0,
        "label": "execution_receipts_only",
        "note": "Process exit and text diagnostics do not establish assertion correctness or usable SFT labels.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in receipts),
        encoding="utf-8",
    )
    report.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "runtime_receipts/hyperon_0_2_10.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/hyperon_0_2_10.json")
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--timeout-sec", type=int, default=12)
    args = parser.parse_args()
    if args.max_files < 1 or args.timeout_sec < 1:
        parser.error("Resource limits must be positive")
    print(json.dumps(
        run_suite(args.root, args.out, args.report, args.max_files, args.timeout_sec),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
