"""Independently audit the retained RLT-HARD-02D native Lean source evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from datagoblin.training_gym.rlt_gym.external_checked_exit import digest
from scripts.audit_rlt_hard02 import ALLOWED_AXIOMS, Reader

MATHLIB_SHA = "809c3fb3b5c8f5d7dace56e200b426187516535a"
SCHEMA = "datagoblin.rlt_hard02d_strict_list_qa.v1"
SOURCE = re.compile(r"^source_roots/strict_list/MathlibList_(\d{2})_[0-9a-f]{20}\.lean$")


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def audit(path: Path) -> dict:
    reader = Reader(path)
    try:
        report = reader.json("report.json")
        if report.get("schema") != SCHEMA:
            raise ValueError("wrong strict List native evidence schema")
        env = report.get("environment", {})
        if report.get("upstream_commit") != MATHLIB_SHA or env.get("mathlib_commit") != MATHLIB_SHA:
            raise ValueError("Mathlib source revision differs")
        if "4.12.0" not in env.get("lean_version", ""):
            raise ValueError("unregistered Lean toolchain version")
        hashes = {}
        for section in ("files_sha256", "source_sha256", "certified_source_sha256"):
            entries = report.get(section)
            if not isinstance(entries, dict):
                raise ValueError("missing file SHA256 manifest")
            for name, expected in entries.items():
                if name in hashes or _sha(reader.read(name)) != expected:
                    raise ValueError(f"missing or tampered source/proof hash: {name}")
                hashes[name] = expected
        rows = reader.lines("verified_roots.jsonl")
        if len(rows) != report.get("roots") or len(rows) == 0:
            raise ValueError("incomplete native proof-root ledger")
        if len({row["root_id"] for row in rows}) != len(rows):
            raise ValueError("duplicated native source identity")
        seen_indices = set()
        verifier_shas = set()
        passed_candidates = 0
        passed_roots = 0
        expected_positive = 0
        qa_ok = True
        for row in rows:
            public = row["public_task"]
            if row["root_id"] != public["source_identity"] or row["split"] != "ordinary_test":
                raise ValueError("native source custody differs")
            source_file = public["source_file"]
            match = SOURCE.fullmatch(source_file)
            if match is None or source_file not in hashes:
                raise ValueError("unregistered native source file")
            index = int(match.group(1))
            if index in seen_indices:
                raise ValueError("duplicate proof-root index")
            seen_indices.add(index)
            expected = index % 4 != 3
            if row["expected_valid"] is not expected:
                raise ValueError("registered intended witness flag differs")
            expected_positive += int(expected)
            if "⊢" not in row["goal_state"].get("goal_pp", ""):
                raise ValueError("missing true Lean open proof state")
            menu = row["menu"]
            if (
                len(menu) != 5
                or len(set(menu)) != 5
                or len({len(script.encode("utf-8")) for script in menu}) != 1
                or public["menu_sha256"] != digest(menu)
            ):
                raise ValueError("source candidate menu was modified")
            attempts = row["candidate_outcomes"]
            if len(attempts) != 5 or {r["candidate_id"] for r in attempts} != set(range(5)):
                raise ValueError("missing native tactic checks")
            certified = []
            for attempt in attempts:
                cid = attempt["candidate_id"]
                receipt = attempt["receipt"]
                if receipt.get("task_digest") != digest(public):
                    raise ValueError("proof receipt task identity differs")
                if receipt.get("candidate_digest") != digest({"id": cid}):
                    raise ValueError("proof receipt action identity differs")
                if type(receipt.get("passed")) is not bool:
                    raise ValueError("invalid native outcome type")
                verifier = receipt.get("verifier_sha256")
                if not isinstance(verifier, str) or not re.fullmatch("[0-9a-f]{64}", verifier):
                    raise ValueError("invalid native verifier checksum")
                verifier_shas.add(verifier)
                if receipt["passed"]:
                    details = receipt["evidence"]
                    if not (
                        receipt.get("code") == "kernel_certified"
                        and details.get("kernel_invoked") is True
                        and details.get("kernel_ok") is True
                        and isinstance(details.get("axioms"), list)
                        and set(details["axioms"]) <= ALLOWED_AXIOMS
                    ):
                        raise ValueError("unverified or disallowed Lean proof")
                    certified.append(cid)
                    passed_candidates += 1
            if certified != row["kernel_valid_candidates"]:
                raise ValueError("observed native candidate labels differ from source ledger")
            eligible = len(certified) == int(expected)
            if type(row["contract_passed"]) is not bool or row["contract_passed"] != eligible:
                raise ValueError("native contract flag disagrees with verifier evidence")
            qa_ok &= eligible
            passed_roots += int(bool(certified))
        if seen_indices != set(range(report["roots"])) or len(verifier_shas) != 1:
            raise ValueError("native source order or verifier revision drift")
        if (
            report["actual_kernel_positive_roots"] != passed_roots
            or report["actual_kernel_negative_roots"] != len(rows) - passed_roots
            or report["expected_positive_roots"] != expected_positive
            or report["expected_negative_roots"] != len(rows) - expected_positive
            or report["independent_candidate_replays"] != 5 * len(rows)
            or report["quality_gate_passed"] is not qa_ok
            or len(report["certified_source_sha256"]) != passed_candidates
        ):
            raise ValueError("native result report contradicts kernel receipts")
        return {
            "schema": "datagoblin.rlt_hard02d_independent_verifier.v1",
            "source_roots": len(rows),
            "file_sha256_checks": len(hashes),
            "native_candidate_receipts": 5 * len(rows),
            "independent_native_replays": report["independent_candidate_replays"],
            "kernel_positive_roots": passed_roots,
            "kernel_negative_roots": len(rows) - passed_roots,
            "native_quality_gate_passed": qa_ok,
            "source_and_kernel_integrity_passed": True,
            "sealed_confirmation_opened": False,
        }
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.evidence)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError("audit output must be create-only")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    if result["native_quality_gate_passed"] is not True:
        raise SystemExit("registered List negative-control quality gate failed")


if __name__ == "__main__":
    main()
