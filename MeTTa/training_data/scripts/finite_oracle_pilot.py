#!/usr/bin/env python3
"""Cross-check fresh public finite MeTTa worlds using a Python reference oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

from finite_semantic_oracle import (
    UnsupportedSemantics,
    compare_native,
    finite_match,
    parse_one,
    render,
    result_signature,
    typed_plus_reference,
)

RUNTIME_VERSION = "0.2.10"
PUBLIC_GENERATOR = "metta06-finite-ground-v1"
MAX_SOURCE_BYTES = 16_384


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_world(seed: int) -> dict[str, Any]:
    """Deterministic novel public roots; targets are recomputed by Python."""
    if not 0 <= seed < 128:
        raise ValueError("Seed outside finite generation scope")
    rng = random.Random(f"{PUBLIC_GENERATOR}/{seed}")
    facts: list[str] = []
    names = [f"u{j:02d}_{seed:03d}" for j in range(9)]
    for index, name in enumerate(names):
        group = rng.choice(("alpha", "beta", "gamma"))
        facts.append(f"(Role {name} {group})")
        neighbor = name if index % 3 == 0 else names[(index + 1) % len(names)]
        facts.append(f"(Edge {name} {neighbor})")
        tag = "good" if rng.randrange(3) != 0 else "stale"
        facts.append(f"(Node (Pair {name} v{index:02d}) {tag})")
    queries = [
        "(match &self (Role $who alpha) $who)",
        "(match &self (Edge $x $x) $x)",
        "(match &self (Node (Pair $left $right) good) (Found $right $left))",
        "(match &self (Role $who absent) $who)",
    ]
    parsed_facts = [parse_one(fact) for fact in facts]
    references = [
        finite_match(parsed_facts, parse_one(query)) for query in queries
    ]
    program = "\n".join(
        ["; Public MeTTa-06 finite-ground diagnostic", *facts, *(f"!{q}" for q in queries)]
    ) + "\n"
    if len(program.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("Generated source exceeds input cap")
    return {
        "root": f"{PUBLIC_GENERATOR}:{seed:03d}",
        "seed": seed,
        "program": program,
        "queries": queries,
        "references": references,
        "source_sha256": sha256(program.encode()),
    }


def native_run(program: str, timeout_sec: int) -> dict[str, Any]:
    if not 1 <= timeout_sec <= 30:
        raise ValueError("Timeout must be bounded")
    env = {
        key: value for key, value in os.environ.items()
        if key in {"PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "PYTHONPATH"}
    }
    env["PYTHONNOUSERSITE"] = "1"
    start = time.monotonic()
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="metta06-") as directory:
        source = Path(directory) / "world.metta"
        source.write_text(program, encoding="utf-8")
        try:
            run = subprocess.run(
                [sys.executable, "-m", "hyperon.metta", str(source)],
                cwd=directory,
                env=env,
                capture_output=True,
                timeout=timeout_sec,
                check=False,
            )
            stdout, stderr, exit_code = run.stdout, run.stderr, run.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout, stderr, exit_code = exc.stdout or b"", exc.stderr or b"", None
    return {
        "elapsed_ms": int((time.monotonic() - start) * 1000),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }


def classify_queries(world: dict[str, Any], run: dict[str, Any]) -> list[dict[str, Any]]:
    lines = run["stdout"].decode("utf-8", "replace").splitlines()
    n = len(world["queries"])
    clean = (
        not run["timed_out"]
        and run["exit_code"] == 0
        and not run["stderr"]
        and len(run["stdout"]) <= 65_536
        and len(lines) == n
    )
    rows: list[dict[str, Any]] = []
    for index, (query, reference) in enumerate(
        zip(world["queries"], world["references"], strict=True)
    ):
        native_line = lines[index] if clean else None
        comparison = None
        if native_line is not None:
            try:
                comparison = compare_native(reference, native_line)
                status = "agree" if comparison["agreement"] else "mismatch"
            except UnsupportedSemantics:
                status = "unsupported_native_output"
        else:
            status = "runtime_or_alignment_failure"
        rows.append({
            "schema_version": "metta06-independent-finite-oracle-row-v1",
            "root": world["root"],
            "source_sha256": world["source_sha256"],
            "query_index": index,
            "query_sha256": sha256(query.encode()),
            "reference_sha256": sha256(
                json.dumps(result_signature(reference), sort_keys=True).encode()
            ),
            "reference_result_count": len(reference),
            "native_line_sha256": sha256(native_line.encode()) if native_line else None,
            "comparison": comparison,
            "status": status,
            "native_exit_code": run["exit_code"],
            "native_timed_out": run["timed_out"],
            "native_stdout_sha256": sha256(run["stdout"]),
            "native_stderr_sha256": sha256(run["stderr"]),
            "native_stdout_bytes": len(run["stdout"]),
            "native_stderr_bytes": len(run["stderr"]),
            "native_elapsed_ms": run["elapsed_ms"],
            "runtime": f"hyperon=={RUNTIME_VERSION}",
            "oracle": "independent_python_finite_ground_match_v1",
            "public_generator": PUBLIC_GENERATOR,
            "source_kind": "synthetic_public_diagnostic",
            "training_ready": False,
            "sealed_eval_eligible": False,
            "upstream_author_targets_used": False,
        })
    return rows


def typed_error_probe(timeout_sec: int) -> dict[str, Any]:
    """Observe one bounded negative type case; treat mismatch as data."""
    query = ("+", "5", '"S"')
    expected = typed_plus_reference(query)
    program = '!(pragma! type-check auto)\n!(+ 5 "S")\n'
    proc = native_run(program, timeout_sec)
    lines = proc["stdout"].decode("utf-8", "replace").splitlines()
    comparison = None
    status = "runtime_or_alignment_failure"
    if proc["exit_code"] == 0 and not proc["timed_out"] and not proc["stderr"] and lines:
        try:
            comparison = compare_native(expected, lines[-1])
            status = "agree" if comparison["agreement"] else "mismatch"
        except UnsupportedSemantics:
            status = "unsupported_native_output"
    return {
        "schema_version": "metta06-independent-finite-oracle-row-v1",
        "root": f"{PUBLIC_GENERATOR}:typed-error-plus-arg2",
        "source_sha256": sha256(program.encode()),
        "query_index": 0,
        "query_sha256": sha256(render(query).encode()),
        "reference_sha256": sha256(
            json.dumps(result_signature(expected), sort_keys=True).encode()
        ),
        "reference_result_count": len(expected),
        "comparison": comparison,
        "status": status,
        "native_stdout_sha256": sha256(proc["stdout"]),
        "native_stderr_sha256": sha256(proc["stderr"]),
        "native_exit_code": proc["exit_code"],
        "native_timed_out": proc["timed_out"],
        "native_elapsed_ms": proc["elapsed_ms"],
        "runtime": f"hyperon=={RUNTIME_VERSION}",
        "oracle": "independent_python_ground_number_type_rule_v1",
        "public_generator": PUBLIC_GENERATOR,
        "source_kind": "synthetic_public_type_probe",
        "training_ready": False,
        "sealed_eval_eligible": False,
        "upstream_author_targets_used": False,
    }


def run_suite(out: Path, report_path: Path, *, roots: int = 32, timeout_sec: int = 8) -> dict[str, Any]:
    if not 1 <= roots <= 128:
        raise ValueError("Roots outside bounded public suite")
    observed = version("hyperon")
    if observed != RUNTIME_VERSION:
        raise RuntimeError(f"Expected hyperon=={RUNTIME_VERSION}, got {observed}")
    rows: list[dict[str, Any]] = []
    for seed in range(roots):
        world = make_world(seed)
        run = native_run(world["program"], timeout_sec)
        rows.extend(classify_queries(world, run))
    rows.append(typed_error_probe(timeout_sec))
    counts = Counter(row["status"] for row in rows)
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(raw, encoding="utf-8")
    report = {
        "schema_version": "metta06-finite-oracle-pilot-v1",
        "root_count": roots,
        "ground_match_query_count": roots * 4,
        "typed_error_probe_count": 1,
        "observations": len(rows),
        "outcome_counts": dict(sorted(counts.items())),
        "native_version": observed,
        "python": platform.python_version(),
        "public_generator": PUBLIC_GENERATOR,
        "observation_sha256": sha256(raw.encode()),
        "independent_reference_scope": "finite_ground_match_and_one_typed_add_probe",
        "unbounded_metta_semantics_verified": False,
        "native_semantics_for_source_candidates_verified": False,
        "source_training_ready": False,
        "training_ready": False,
        "sealed_eval_eligible": False,
        "all_targets_public": True,
        "root_disjoint_public_diagnostics_only": True,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--timeout-sec", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("runs/metta/finite_oracle/observations.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("runs/metta/finite_oracle/report.json"))
    args = parser.parse_args()
    result = run_suite(args.out, args.report, roots=args.roots, timeout_sec=args.timeout_sec)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
