"""Reaudit frozen RLT-LIVE-01 archives and assess stop-policy identifiability.

Evidence is immutable and the reader does not fit on evaluation roots. This is
an archive-only control, not a substitute for fresh native tool evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

SPLITS = ("train", "calibration", "ordinary_test", "ood_test")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_zip(path: Path, kind: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            item.startswith("/") or ".." in Path(item).parts for item in names
        ):
            raise ValueError("unsafe or duplicated ZIP member")
        report = json.loads(archive.read("report.json"))
        expected = report.get("file_sha256", report.get("files_sha256"))
        if not isinstance(expected, dict):
            raise ValueError("missing archive evidence manifest")
        for name, checksum in expected.items():
            if _sha(archive.read(name)) != checksum:
                raise ValueError(f"altered evidence member: {name}")
        proofs = report.get("certified_proof_source_sha256", {})
        for name, checksum in proofs.items():
            if _sha(archive.read(name)) != checksum:
                raise ValueError(f"altered certified Lean proof: {name}")
        episodes = [json.loads(line) for line in archive.read("source_episodes.jsonl").splitlines()]
        verified = [json.loads(line) for line in archive.read("verified_rows.jsonl").splitlines()]
        if len(episodes) != len(verified) or not episodes:
            raise ValueError("incomplete source/verification roots")
        ep_ids = {e["root_id"] for e in episodes}
        row_ids = {v["root_id"] for v in verified}
        if len(ep_ids) != len(episodes) or ep_ids != row_ids:
            raise ValueError("non-bijective source/verified roots")
        for episode in episodes:
            if episode["kind"] != kind or episode["split"] not in SPLITS:
                raise ValueError("wrong substrate or source split")
            if episode["verifier_sha256"] != report["verifier_sha256"]:
                raise ValueError("verifier revision drift")
            if episode["source_revision"] != report["source_revision_sha256"]:
                raise ValueError("source revision drift")
            if len(episode["attempts"]) != episode["planned_visits"]:
                raise ValueError("incomplete attempted sequence")
            for i, attempt in enumerate(episode["attempts"]):
                verification = attempt["verification"]
                if attempt["visit"] != i or type(verification["passed"]) is not bool:
                    raise ValueError("invalid candidate sequence")
                if verification["verifier_sha256"] != report["verifier_sha256"]:
                    raise ValueError("candidate checked by another verifier")
                if verification["task_digest"] != episode["task_digest"]:
                    raise ValueError("candidate evaluated on another task")
                if verification["passed"]:
                    evidence = verification["evidence"]
                    if kind == "lean" and not (
                        evidence.get("kernel_invoked") and evidence.get("kernel_ok")
                        and evidence.get("axioms") == []
                    ):
                        raise ValueError("success without audited Lean kernel")
                    if kind == "mcp" and not evidence.get("tools_executed"):
                        raise ValueError("success without actual MCP effects")
        root_by_id = {e["root_id"]: e for e in episodes}
        for row in verified:
            if row["kind"] != kind or row["split"] not in SPLITS:
                raise ValueError("verified row split/kind drift")
            if row["training_eligible"] != (row["split"] == "train"):
                raise ValueError("TRAIN custody violation")
            root = root_by_id[row["root_id"]]
            first = next((a["visit"] for a in root["attempts"] if a["verification"]["passed"]), None)
            if first != row["first_verified_visit"]:
                raise ValueError("first success disagrees with independent replay")
    return report, episodes, verified


def _policy(episodes: list[dict[str, Any]], limit: int, stop_on_success: bool) -> dict[str, int]:
    checked = 0
    successes = 0
    for e in episodes:
        did_pass = False
        for attempt in e["attempts"][:limit]:
            checked += 1
            if attempt["verification"]["passed"]:
                did_pass = True
                if stop_on_success:
                    break
        successes += int(did_pass)
    return {"certified": successes, "roots": len(episodes), "attempts": checked}


def audit(lean_zip: Path, mcp_zip: Path) -> dict[str, Any]:
    outcome: dict[str, Any] = {"evidence_class": "offline_native_receipt_replay_audit", "substrates": {}}
    for kind, archive in (("lean", lean_zip), ("mcp", mcp_zip)):
        report, episodes, verified = _read_zip(archive, kind)
        first_failures = [v for v in verified if v["split"] == "train" and not v["first_passed"]]
        positives = sum(v["future_rescue"] for v in first_failures)
        negatives = len(first_failures) - positives
        exposed = 0
        if kind == "mcp":
            for ep in episodes:
                public = ep["public_task"]
                if "expected_symbol_name" in public or "hint_module" in public:
                    exposed += 1
        outcome["substrates"][kind] = {
            "roots": len(episodes),
            "train": sum(e["split"] == "train" for e in episodes),
            "calibration": sum(e["split"] == "calibration" for e in episodes),
            "held_out": sum(e["split"] in ("ordinary_test", "ood_test") for e in episodes),
            "depth_1": _policy(episodes, 1, stop_on_success=True),
            "depth_2": _policy(episodes, 2, stop_on_success=True),
            "depth_3_full": _policy(episodes, 3, stop_on_success=False),
            "depth_3_stop_on_certified": _policy(episodes, 3, stop_on_success=True),
            "train_first_failure_continue_labels": {"rescuable": positives, "unrescuable": negatives},
            "continuation_identifiable": positives > 0 and negatives > 0,
            "public_symbol_or_module_exposure": exposed,
            "source_revision": report["source_revision_sha256"],
            "verifier_sha256": report["verifier_sha256"],
        }
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lean_archive", type=Path)
    parser.add_argument("mcp_archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.lean_archive, args.mcp_archive)
    if args.output:
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
