#!/usr/bin/env python3
"""MeTTa-08: compare a pinned PeTTa runtime, Hyperon and the finite Python oracle.

This is a public synthetic diagnostic, never a release gate for source SFT or
a sealed confirmation set. All runtime and parse failures become explicit rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from finite_oracle_pilot import make_world, native_run
from finite_semantic_oracle import (
    UnsupportedSemantics,
    parse_one,
    result_signature,
)

PETTA_COMMIT = "ee5cacb6a6ecdd554a1c773aaf47e9a0f8a5af0f"
IMAGE_RE = re.compile(r"^[a-z0-9][a-z0-9./_-]*@sha256:[0-9a-f]{64}$")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_runtime_inputs(repo: Path, image: str) -> None:
    if not IMAGE_RE.fullmatch(image):
        raise ValueError("PeTTa OCI image must be resolved to an immutable sha256 digest")
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    if result.stdout.strip() != PETTA_COMMIT:
        raise ValueError("PeTTa checkout differs from pinned upstream revision")
    if not (repo / "src/main.pl").is_file() or not (repo / "run.sh").is_file():
        raise ValueError("Missing pinned PeTTa source or launcher")


def container_command(
    repo: Path, image: str, source_path: Path | None = None
) -> list[str]:
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--memory", "1024m", "--cpus", "2", "--pids-limit", "128",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "-e", "PETTA_STACK_LIMIT=256m", "-e", "PETTA_TABLE_LIMIT=256m",
        "-v", f"{repo.resolve()}:/petta:ro",
    ]
    if source_path is not None:
        command += ["-v", f"{source_path.resolve()}:/input/world.metta:ro"]
    return [*command, image]


def get_petta_version(repo: Path, image: str) -> str:
    result = subprocess.run(
        [*container_command(repo, image), "swipl", "--version"],
        capture_output=True, text=True, timeout=60, check=True,
    )
    text = result.stdout.strip()
    match = re.search(r"version\s+(\d+)\.(\d+)\.(\d+)", text, re.IGNORECASE)
    if not match or tuple(map(int, match.groups())) < (9, 3, 0):
        raise RuntimeError(f"PeTTa requires SWI-Prolog 9.3+: {text}")
    return text


def run_petta(
    program: str, repo: Path, image: str, timeout_sec: int
) -> dict[str, Any]:
    raw = program.encode("utf-8")
    if len(raw) > 16_384:
        raise ValueError("Finite world exceeds source byte cap")
    with tempfile.TemporaryDirectory(prefix="metta08-world-") as tmp:
        source = Path(tmp) / "world.metta"
        source.write_bytes(raw)
        try:
            outcome = subprocess.run(
                [*container_command(repo, image, source), "sh", "/petta/run.sh", "/input/world.metta", "--silent"],
                capture_output=True, check=False, timeout=timeout_sec,
            )
            return {
                "stdout": outcome.stdout, "stderr": outcome.stderr,
                "exit_code": outcome.returncode, "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": exc.stdout or b"", "stderr": exc.stderr or b"",
                "exit_code": None, "timed_out": True,
            }


def parse_petta_output(line: str) -> list[Any]:
    """PeTTa swrite prints a MeTTa expression of result values: () or (a b)."""
    parsed = parse_one(line.strip())
    if not isinstance(parsed, tuple):
        raise UnsupportedSemantics("Expected PeTTa swrite list expression")
    return list(parsed)


def assess_petta(world: dict[str, Any], run: dict[str, Any]) -> list[dict[str, Any]]:
    lines = run["stdout"].decode("utf-8", "replace").splitlines()
    clean = (
        run["exit_code"] == 0
        and run["timed_out"] is False
        and len(run["stdout"]) <= 65_536
        and len(lines) == len(world["queries"])
    )
    results: list[dict[str, Any]] = []
    for ix, reference in enumerate(world["references"]):
        signature: dict[str, int] | None = None
        if clean:
            try:
                observed = parse_petta_output(lines[ix])
                signature = result_signature(observed)
                status = (
                    "agree" if signature == result_signature(reference)
                    else "mismatch"
                )
            except UnsupportedSemantics:
                status = "unsupported_output"
        else:
            status = "runtime_or_alignment_failure"
        results.append({
            "status": status,
            "signature": signature,
            "output_sha256": sha256(lines[ix].encode()) if clean else None,
        })
    return results


def assess_hyperon(world: dict[str, Any], run: dict[str, Any]) -> list[dict[str, Any]]:
    from finite_semantic_oracle import parse_result_line

    lines = run["stdout"].decode("utf-8", "replace").splitlines()
    clean = (
        run["exit_code"] == 0 and run["timed_out"] is False
        and not run["stderr"] and len(lines) == len(world["queries"])
        and len(run["stdout"]) <= 65_536
    )
    results: list[dict[str, Any]] = []
    for ix, reference in enumerate(world["references"]):
        signature: dict[str, int] | None = None
        if clean:
            try:
                signature = result_signature(parse_result_line(lines[ix]))
                status = (
                    "agree" if signature == result_signature(reference)
                    else "mismatch"
                )
            except UnsupportedSemantics:
                status = "unsupported_output"
        else:
            status = "runtime_or_alignment_failure"
        results.append({
            "status": status, "signature": signature,
            "output_sha256": sha256(lines[ix].encode()) if clean else None,
        })
    return results


def run_suite(
    repo: Path, image: str, out: Path, report_path: Path, *,
    roots: int = 12, timeout_sec: int = 20,
) -> dict[str, Any]:
    if not 1 <= roots <= 32 or not 1 <= timeout_sec <= 60:
        raise ValueError("Resource limits outside the registered pilot")
    validate_runtime_inputs(repo, image)
    swi_version = get_petta_version(repo, image)
    runtime_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for seed in range(roots):
        world = make_world(seed)
        pe = run_petta(world["program"], repo, image, timeout_sec)
        hy = native_run(world["program"], timeout_sec)
        pe_results = assess_petta(world, pe)
        hy_results = assess_hyperon(world, hy)
        for ix, (pe_row, hy_row) in enumerate(
            zip(pe_results, hy_results, strict=True)
        ):
            same = (
                pe_row["signature"] == hy_row["signature"]
                if pe_row["signature"] is not None and hy_row["signature"] is not None
                else None
            )
            counts[f"petta_{pe_row['status']}"] += 1
            counts[f"hyperon_{hy_row['status']}"] += 1
            if same is not None:
                counts["pe_hy_agree" if same else "pe_hy_mismatch"] += 1
            runtime_rows.append({
                "schema_version": "metta08-petta-differential-row-v1",
                "root": world["root"], "query_index": ix,
                "query_sha256": sha256(world["queries"][ix].encode()),
                "source_sha256": world["source_sha256"],
                "reference_signature_sha256": sha256(
                    json.dumps(
                        result_signature(world["references"][ix]), sort_keys=True
                    ).encode()
                ),
                "petta_status": pe_row["status"],
                "hyperon_status": hy_row["status"],
                "petta_hyperon_signature_agree": same,
                "petta_output_sha256": pe_row["output_sha256"],
                "hyperon_output_sha256": hy_row["output_sha256"],
                "petta_stdout_sha256": sha256(pe["stdout"]),
                "petta_stderr_sha256": sha256(pe["stderr"]),
                "hyperon_stdout_sha256": sha256(hy["stdout"]),
                "hyperon_stderr_sha256": sha256(hy["stderr"]),
                "petta_exit_code": pe["exit_code"],
                "hyperon_exit_code": hy["exit_code"],
                "petta_timed_out": pe["timed_out"],
                "hyperon_timed_out": hy["timed_out"],
                "independent_reference": "finite_ground_match_v1",
                "petta_commit": PETTA_COMMIT,
                "petta_oci_image": image,
                "swi_prolog_version": swi_version,
                "hyperon_version": "0.2.10",
                "public_synthetic_only": True,
                "training_ready": False,
                "sealed_eval_eligible": False,
            })
    raw_rows = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in runtime_rows
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(raw_rows, encoding="utf-8")
    report = {
        "schema_version": "metta08-petta-differential-v1",
        "source_roots": roots,
        "queries": len(runtime_rows),
        "counts": dict(sorted(counts.items())),
        "observations_sha256": sha256(raw_rows.encode()),
        "petta_commit": PETTA_COMMIT,
        "petta_oci_image": image,
        "swi_prolog_version": swi_version,
        "hyperon_version": "0.2.10",
        "public_diagnostic": True,
        "input_family": "finite_ground_match",
        "type_probe_in_scope": False,
        "training_ready": False,
        "sealed_eval_eligible": False,
        "model_evaluation_executed": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--petta-repo", type=Path, required=True)
    parser.add_argument("--petta-image", required=True)
    parser.add_argument("--roots", type=int, default=12)
    parser.add_argument("--timeout-sec", type=int, default=20)
    parser.add_argument(
        "--out", type=Path,
        default=Path("runs/metta/petta_diff/observations.jsonl"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("runs/metta/petta_diff/report.json"),
    )
    args = parser.parse_args()
    report = run_suite(
        args.petta_repo, args.petta_image, args.out, args.report,
        roots=args.roots, timeout_sec=args.timeout_sec,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
