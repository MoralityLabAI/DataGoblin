"""Replay bounded MeTTa hill-climb proposals in pinned native Hyperon."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .ram_hillclimb import metta_proposal

PACKAGE_VERSION = "0.2.10"


def replay(
    steps_path: Path,
    out: Path,
    *,
    max_cases: int = 8,
    timeout_sec: int = 8,
) -> dict[str, Any]:
    if not 1 <= max_cases <= 8 or not 1 <= timeout_sec <= 20:
        raise ValueError("Native replay must stay within fixed CPU limits")
    if version("hyperon") != PACKAGE_VERSION:
        raise RuntimeError("Native proposal replay requires hyperon==0.2.10")
    rows = [
        json.loads(line)
        for line in steps_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len(rows) > max_cases:
        raise ValueError("Native proposal replay row budget exceeded")
    env = {
        k: v for k, v in os.environ.items()
        if k in {"PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "PYTHONPATH"}
    }
    env["PYTHONNOUSERSITE"] = "1"
    receipts: list[dict[str, Any]] = []
    for row in rows:
        if (
            row.get("schema_version") != "metta-ram10-hillclimb-step-v1"
            or row.get("training_ready") is not False
        ):
            raise ValueError("Unrecognized or released teacher trace")
        proposal = row["teacher_proposal"]
        source = row["metta_candidate_program"]
        if not isinstance(source, str) or len(source.encode()) > 4096:
            raise ValueError("MeTTa proposal source exceeded byte budget")
        # Exact host re-rendering ensures imported GPT text cannot become
        # arbitrary runtime code through a stored source field.
        ram_limit = int(source.splitlines()[1].split(" n")[-1].split(")")[0])
        if (
            metta_proposal(proposal, ram_limit) != source
            or hashlib.sha256(source.encode()).hexdigest() != row["metta_candidate_sha256"]
        ):
            raise ValueError("Proposal bytes differ from the host-owned grammar")
        arm, field = proposal["arm"], proposal["field"]
        value = str(proposal["value"]).replace(".", "_")
        if isinstance(proposal["value"], float):
            value = "v" + value
        else:
            value = "n" + value
        expected = f"[(Proposal {arm} {field} {value})]"
        with tempfile.TemporaryDirectory(prefix="ram-metta10-") as folder:
            program = Path(folder) / "proposal.metta"
            program.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "hyperon.metta", str(program)],
                cwd=folder, env=env, capture_output=True,
                timeout=timeout_sec, check=False,
            )
        matched = result.returncode == 0 and result.stdout.decode("utf-8", "replace").strip() == expected
        receipts.append({
            "schema_version": "metta-ram10-native-proposal-receipt-v1",
            "turn": row["turn"],
            "arm": arm,
            "proposal_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
            "expected_native_rendering": expected,
            "observed_native_rendering": result.stdout.decode("utf-8", "replace").strip()[:256],
            "exit_code": result.returncode,
            "native_matches": matched,
            "runtime": "hyperon==0.2.10",
            "source_training_ready": False,
        })
    summary = {
        "schema_version": "metta-ram10-native-proposal-report-v1",
        "attempts": len(receipts),
        "native_matches": sum(row["native_matches"] for row in receipts),
        "all_match": all(row["native_matches"] for row in receipts),
        "runtime": "hyperon==0.2.10",
        "scope": "proposal_fact_query_parsing_only",
        "training_ready": False,
        "receipts": receipts,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.steps, args.out)
    print(json.dumps({
        "attempts": result["attempts"],
        "native_matches": result["native_matches"],
        "training_ready": False,
    }, sort_keys=True))
    return 0 if result["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
