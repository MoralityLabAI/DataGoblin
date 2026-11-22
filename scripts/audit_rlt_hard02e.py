"""Independent checksum/receipt audit for RLT-HARD-02E native List stopping.

This process reads frozen GitHub Actions artifacts. It cannot ask Lean to
prove a theorem, modify fitting weights, or re-label test examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from datagoblin.training_gym.rlt_gym.external_checked_exit import digest

SCHEMA = "datagoblin.rlt_hard02e_verified_list_stop.v1"
UPSTREAM_SHA = "809c3fb3b5c8f5d7dace56e200b426187516535a"
SPLITS = ("train", "calibration", "ordinary_test", "ood_test")
ARMS = ("content_learned", "legacy_learned", "fixed_full",
        "random_full", "fixed_one", "suffix_length_rule")
AXIOMS = {"propext", "Quot.sound", "Classical.choice"}
SOURCE = re.compile(r"^source_roots/(train|calibration|ordinary_test|ood_test)/"
                    r"MathlibListE_(\d{2})_[0-9a-f]{20}\.lean$")


class Reader:
    def __init__(self, path: Path):
        self.path = path
        self.zip = ZipFile(path) if path.is_file() else None
        if self.zip:
            entries = self.zip.namelist()
            if len(set(entries)) != len(entries) or any(
                key.startswith("/") or ".." in Path(key).parts for key in entries
            ):
                raise ValueError("unsafe or duplicate evidence archive member")

    def read(self, name: str) -> bytes:
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("unsafe evidence member")
        return self.zip.read(name) if self.zip else (self.path / name).read_bytes()

    def json(self, name: str):
        return json.loads(self.read(name))

    def lines(self, name: str):
        return [json.loads(line) for line in self.read(name).splitlines() if line.strip()]

    def close(self):
        if self.zip:
            self.zip.close()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _verify(receipt: dict, public_task: dict, candidate: int, verifier: str) -> bool:
    if (
        receipt.get("task_digest") != digest(public_task)
        or receipt.get("candidate_digest") != digest({"id": candidate})
        or receipt.get("verifier_sha256") != verifier
    ):
        raise ValueError("untrusted candidate, source or verifier SHA in kernel receipt")
    if type(receipt.get("passed")) is not bool:
        raise ValueError("untyped kernel verification outcome")
    if receipt["passed"]:
        evidence = receipt.get("evidence", {})
        if not (
            receipt.get("code") == "kernel_certified"
            and evidence.get("kernel_invoked") is True
            and evidence.get("kernel_ok") is True
            and isinstance(evidence.get("axioms"), list)
            and set(evidence["axioms"]) <= AXIOMS
        ):
            raise ValueError("claimed success lacks independent Lean kernel/axiom receipt")
    return receipt["passed"]


def audit(path: Path) -> dict:
    reader = Reader(path)
    try:
        report = reader.json("report.json")
        if report.get("schema") != SCHEMA:
            raise ValueError("unregistered RLT-HARD-02E evidence schema")
        upstream = report.get("upstream", {})
        if upstream.get("mathlib_commit") != UPSTREAM_SHA or "4.12.0" not in upstream.get("lean_version", ""):
            raise ValueError("wrong Mathlib source or Lean toolchain")
        expected_verifier = report.get("verifier_sha256")
        if not isinstance(expected_verifier, str) or not re.fullmatch("[0-9a-f]{64}", expected_verifier):
            raise ValueError("invalid verifier identity")
        manifests = {}
        for field in ("file_sha256", "source_sha256", "certified_source_sha256"):
            entries = report.get(field)
            if not isinstance(entries, dict):
                raise ValueError("missing immutable source/proof digest manifest")
            for name, sha in entries.items():
                if name in manifests or not isinstance(sha, str) or _sha(reader.read(name)) != sha:
                    raise ValueError(f"missing, duplicate or corrupted file hash: {name}")
                manifests[name] = sha
        weights = reader.json("weights.json")
        if weights.get("train_only") is not True:
            raise ValueError("model training custody must be TRAIN only")
        if report["model_sha256"] != _sha(reader.read("weights.json")):
            raise ValueError("model checkpoint identity changed")
        thresholds = report["calibration"]["thresholds"]
        if weights["calibration_thresholds"] != thresholds:
            raise ValueError("CAL-frozen threshold differs from published weights")
        if not all(0 <= v <= 1 for v in thresholds.values()):
            raise ValueError("invalid CAL threshold")
        if not all(isinstance(w, (int, float)) and math.isfinite(w)
                   for name in ("content_weights", "legacy_weights")
                   for w in weights[name]):
            raise ValueError("nonfinite frozen policy weights")
        source = reader.lines("sources.jsonl")
        root_map = {r["root_id"]: r for r in source}
        if len(source) != report["source_roots"] or len(root_map) != len(source):
            raise ValueError("source-root cardinality or identity collision")
        if len({r["public_task"]["statement"] for r in source}) != len(source):
            raise ValueError("theorem statement appears in multiple splits")
        if len({r["public_task"]["source_file"] for r in source}) != len(source):
            raise ValueError("duplicate source lineage")
        for s in SPLITS:
            if sum(r["split"] == s for r in source) != report["roots_per_split"]:
                raise ValueError("source population leaks across splits")
        for row in source:
            root_id = row["root_id"]
            public = row["public_task"]
            match = SOURCE.fullmatch(public["source_file"])
            if (
                match is None or match.group(1) != row["split"]
                or public["source_identity"] != root_id
                or public.get("import") != "Mathlib"
                or public.get("upstream_module") != "Mathlib/Data/List/Basic.lean"
                or "⊢" not in row["goal_state"].get("goal_pp", "")
                or public.get("candidate_menu_sha256") != digest(row["menu"])
            ):
                raise ValueError("source lineage, goal state or menu custody drift")
            if (
                len(row["menu"]) != 5
                or len(set(row["menu"])) != 5
                or len({len(t.encode("utf-8")) for t in row["menu"]}) != 1
            ):
                raise ValueError("candidate length matching / uniqueness drift")
        quality = reader.lines("full_menu_quality.jsonl")
        by_quality = {row["root_id"]: row for row in quality}
        if len(quality) != len(source) or set(by_quality) != set(root_map):
            raise ValueError("incomplete frozen native menu labels")
        checked = 0
        qa_by_split = {}
        for split in SPLITS:
            cohort = [r for r in quality if r["split"] == split]
            accepted = 0
            exhausted = 0
            valid_contracts = 0
            for row in cohort:
                root = root_map[row["root_id"]]
                if row["registered_witness"] != root["test_only_expected_witness"]:
                    raise ValueError("test-only oracle source flag changed")
                outcomes = row["attempts"]
                if len(outcomes) != 5 or {a["candidate_id"] for a in outcomes} != set(range(5)):
                    raise ValueError("full-menu native verifier has skipped actions")
                verified = []
                for attempt in outcomes:
                    checked += 1
                    if _verify(attempt["receipt"], root["public_task"],
                               attempt["candidate_id"], expected_verifier):
                        verified.append(attempt["candidate_id"])
                accepted += bool(verified)
                exhausted += not bool(verified)
                expected = int(root["test_only_expected_witness"])
                good = len(verified) == expected
                valid_contracts += good
                if row["native_contract_passed"] != good:
                    raise ValueError("native QA flag disagrees with the kernel")
            qa_by_split[split] = {
                "roots": len(cohort), "actual_native_positive": accepted,
                "actually_exhausted": exhausted, "registered_menu_contract_passed": valid_contracts,
            }
        if qa_by_split != report["source_quality_by_split"]:
            raise ValueError("reported train/test negative labels differ from native receipts")
        if report["evaluator_only_complete_menu_checks"] != checked:
            raise ValueError("evaluator did not check every menu action")
        evaluated = reader.lines("receipts.jsonl")
        index = {(r["root_id"], r["arm"]): r for r in evaluated}
        if len(index) != len(evaluated):
            raise ValueError("duplicated evaluation arm/root identity")
        per_split = {}
        false_stops = {}
        scored_checks = 0
        for split in ("ordinary_test", "ood_test"):
            roots = [r["root_id"] for r in source if r["split"] == split]
            summary = {}
            for arm in ARMS:
                rows = [index[(rid, arm)] for rid in roots]
                n_certified = 0
                n_checks = 0
                secs = 0
                early = 0
                for row in rows:
                    if row["split"] != split:
                        raise ValueError("evaluation arm leaked source split")
                    reference = {
                        x["candidate_id"]: x["receipt"] for x in by_quality[row["root_id"]]["attempts"]
                    }
                    attempts = row["attempts"]
                    ids = [a["candidate_id"] for a in attempts]
                    if len(ids) > 5 or len(set(ids)) != len(ids):
                        raise ValueError("invalid evaluation action budget or repeated candidate")
                    for attempt in attempts:
                        scored_checks += 1
                        cid = attempt["candidate_id"]
                        _verify(attempt["receipt"], root_map[row["root_id"]]["public_task"],
                                cid, expected_verifier)
                        if attempt["receipt"] != reference[cid]:
                            raise ValueError("evaluation receipt differs from evaluator-only oracle")
                    success = bool(attempts and attempts[-1]["receipt"]["passed"])
                    if row["certified"] != success:
                        raise ValueError("reported success disagrees with kernel evidence")
                    n_certified += int(success)
                    n_checks += len(attempts)
                    secs += row["candidate_wall_seconds"]
                    early += int(row["stopped_before_budget"])
                report_row = report["development"][split][arm]
                if (n_certified != report_row["certified"]
                        or n_checks != report_row["candidate_attempts"]
                        or early != report_row["early_negative_stops"]
                        or abs(secs - report_row["candidate_wall_seconds"]) > 0.0001):
                    raise ValueError("native solve/cost summary mismatches source receipt ledger")
                summary[arm] = {"certified": n_certified, "checks": n_checks}
            per_split[split] = summary
            false_stops[split] = {
                arm: sum(
                    bool(index[(root_id, "fixed_full")]["certified"]
                         and not index[(root_id, arm)]["certified"])
                    for root_id in roots
                ) for arm in ("content_learned", "legacy_learned", "suffix_length_rule")
            }
        if false_stops != report["false_stops_against_full_search"]:
            raise ValueError("false STOP count does not match actual kernel receipts")
        replay_counts = report["independent_replays_by_learned_arm"]
        if any(
            replay_counts.get(arm) != sum(
                len(x["attempts"]) for x in evaluated if x["arm"] == arm
            ) for arm in ("content_learned", "legacy_learned")
        ):
            raise ValueError("independent learned kernel replay count is incomplete")
        return {
            "schema": "datagoblin.rlt_hard02e_independent_audit.v1",
            "environment_verified": True, "checksum_members_verified": len(manifests),
            "source_roots": len(source),
            "complete_menu_native_checks": checked,
            "development_native_checks": scored_checks,
            "quality_by_split": qa_by_split,
            "policy_metrics": per_split,
            "false_stops": false_stops,
            "independent_learned_replays": replay_counts,
            "registered_menu_quality_gate_passed": report["quality_gate_passed"],
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
    output = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError("independent audit must be create-only")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output)
    if result["registered_menu_quality_gate_passed"] is not True:
        raise SystemExit("registered negative-menu corpus quality failed")


if __name__ == "__main__":
    main()
