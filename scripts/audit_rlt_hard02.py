"""Audit RLT-HARD-02A native Mathlib evidence without fitting or executing Lean.

Usage: python scripts/audit_rlt_hard02.py archive.zip --out audit.json
The audit checks source provenance, digest manifests, per-arm kernel receipts,
source-root custody, attempted actions and false-stop penalties.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

SPLITS = ("train", "calibration", "ordinary_test", "ood_test")
ARMS = ("learned", "fixed_full", "random_full", "trivial_first")
ALLOWED_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Reader:
    def __init__(self, path: Path):
        self.path = path
        self.archive = ZipFile(path) if path.is_file() else None
        if self.archive:
            members = self.archive.namelist()
            if len(set(members)) != len(members) or any(
                item.startswith("/") or ".." in Path(item).parts for item in members
            ):
                raise ValueError("unsafe or duplicate evidence ZIP members")

    def read(self, name: str) -> bytes:
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("invalid evidence member")
        return self.archive.read(name) if self.archive else (self.path / name).read_bytes()

    def json(self, name: str):
        return json.loads(self.read(name))

    def lines(self, name: str):
        return [json.loads(s) for s in self.read(name).splitlines() if s.strip()]

    def close(self):
        if self.archive:
            self.archive.close()




def expected_menu_exhaustion(source: dict) -> bool:
    """Parse the preregistered negative-control index from generated source."""
    import re

    filename = source["public_task"]["source_file"]
    match = re.search(r"/MathlibRoot_(\d{2})_[0-9a-f]{20}\.lean$", filename)
    if match is None:
        raise ValueError("unknown registered source root name")
    return int(match.group(1)) % 4 == 3



def length_first_control(source: dict, exhaustive: dict) -> dict:
    """Score a one-check length heuristic using only executed native receipts.

    A candidate absent from the exhaustive arm is marked unobserved rather
    than being assigned a synthetic verification result.
    """
    menu = source["menu"]
    choice = max(range(len(menu)), key=lambda i: (len(menu[i].encode("utf-8")), -i))
    tested = {row["candidate_id"]: row["receipt"] for row in exhaustive["attempts"]}
    if choice not in tested:
        return {"candidate_id": choice, "observed": False, "certified": None}
    receipt = tested[choice]
    return {
        "candidate_id": choice,
        "observed": True,
        "certified": bool(receipt["passed"]),
        "kernel_verified": bool(
            receipt["passed"] and receipt["code"] == "kernel_certified"
            and receipt["evidence"].get("kernel_ok") is True
        ),
        "length": len(menu[choice].encode("utf-8")),
        "largest_other_length": max(
            len(action.encode("utf-8")) for index, action in enumerate(menu) if index != choice
        ),
    }


def audit(path: Path) -> dict:
    reader = Reader(path)
    try:
        report = reader.json("report.json")
        if report.get("schema") != "datagoblin.rlt_hard02_mathlib_first_dev.v1":
            raise ValueError("wrong native Mathlib evidence schema")
        env = report.get("environment", {})
        if env.get("mathlib_commit") != "809c3fb3b5c8f5d7dace56e200b426187516535a":
            raise ValueError("Mathlib source pin drift")
        if "4.12.0" not in env.get("lean_version", ""):
            raise ValueError("Lean toolchain mismatch")
        manifests = {}
        for key in ("files_sha256", "source_sha256", "certified_source_sha256"):
            value = report.get(key)
            if not isinstance(value, dict):
                raise ValueError("missing evidence digest manifest")
            manifests.update(value)
        for name, expected in manifests.items():
            if _digest(reader.read(name)) != expected:
                raise ValueError(f"evidence integrity failure: {name}")
        sources = reader.lines("source_records.jsonl")
        ledger = reader.lines("native_receipts.jsonl")
        if len(sources) != report["source_roots"] or not sources:
            raise ValueError("source population mismatch")
        by_id = {r["root_id"]: r for r in sources}
        if len(by_id) != len(sources):
            raise ValueError("duplicate source root")
        if len({r["public_task"]["statement"] for r in sources}) != len(sources):
            raise ValueError("duplicate theorem statement")
        for split in SPLITS:
            group = [s for s in sources if s["split"] == split]
            if len(group) != report["per_split"]:
                raise ValueError("source split cardinality drift")
            if any(s["public_task"]["upstream_module"] != report["split_upstream_files"][split]
                   for s in group):
                raise ValueError("unexpected source lineage")
            if any("⊢" not in s["goal_state"]["goal_pp"] for s in group):
                raise ValueError("missing native Lean open-goal trace")
        provenance = [r["public_task"]["source_file"] for r in sources]
        if len(set(provenance)) != len(provenance):
            raise ValueError("shared source file across roots")
        weights = reader.json("weights.json")
        if weights.get("training_splits") != ["train"]:
            raise ValueError("training-custody exposure")
        if weights["calibration_threshold"] != report["calibration"]["threshold"]:
            raise ValueError("calibration changed after evaluation")
        paired = report.get("paired_feature_ablation") is True
        arms = (*ARMS, "legacy_grounded") if paired else ARMS
        if paired and (
            not isinstance(weights.get("legacy_weights"), list)
            or weights.get("legacy_CAL_threshold") != report["calibration"]["legacy_grounded_threshold"]
            or report.get("feature_profile") != "lexical_hash_v1"
        ):
            raise ValueError("missing TRAIN/CAL matched legacy feature ablation")
        by_slice = Counter()
        native_receipt_count = 0
        for row in ledger:
            root_id = row["root_id"]
            if root_id not in by_id or row["split"] != by_id[root_id]["split"]:
                raise ValueError("receipt belongs to another source")
            arm = row["arm"]
            if arm in ("full_menu_fit", "full_menu_cal"):
                expected = "train" if arm == "full_menu_fit" else "calibration"
                if row["split"] != expected or len(row["attempts"]) != 5:
                    raise ValueError("untrusted training / CAL candidate coverage")
            elif arm in arms:
                if row["split"] not in ("ordinary_test", "ood_test"):
                    raise ValueError("test arm leaked into TRAIN/CAL")
            else:
                raise ValueError("unregistered evaluation policy")
            ids = [a["candidate_id"] for a in row["attempts"]]
            if len(ids) != len(set(ids)) or not all(0 <= x < 5 for x in ids):
                raise ValueError("duplicate or unregistered native candidate ID")
            for candidate in row["attempts"]:
                native_receipt_count += 1
                receipt = candidate["receipt"]
                if type(receipt.get("passed")) is not bool:
                    raise ValueError("untyped native verification outcome")
                if receipt["passed"] and not (
                    receipt.get("code") == "kernel_certified"
                    and receipt["evidence"].get("kernel_invoked")
                    and receipt["evidence"].get("kernel_ok")
                    and isinstance(receipt["evidence"].get("axioms"), list)
                    and set(receipt["evidence"]["axioms"]) <= ALLOWED_AXIOMS
                ):
                    raise ValueError("success without allowlisted independent kernel proof")
            by_slice[(row["split"], arm)] += 1
        for split in SPLITS:
            expected_arm = "full_menu_fit" if split == "train" else "full_menu_cal"
            if split in ("train", "calibration"):
                if by_slice[(split, expected_arm)] != report["per_split"]:
                    raise ValueError("incomplete labeled training / CAL population")
            else:
                for arm in arms:
                    if by_slice[(split, arm)] != report["per_split"]:
                        raise ValueError("missing paired development policy")
        by_row = {(x["root_id"], x["arm"]): x for x in ledger}
        checked = {}
        length_control = {}
        false_stops = {}
        legacy_false_stops = {}
        negative_control_audit = {}
        for split in SPLITS:
            root_ids = [s["root_id"] for s in sources if s["split"] == split]
            arm = "full_menu_fit" if split == "train" else (
                "full_menu_cal" if split == "calibration" else "fixed_full"
            )
            expected = {root_id for root_id in root_ids if expected_menu_exhaustion(by_id[root_id])}
            observed = {
                root_id for root_id in root_ids
                if not any(a["receipt"]["passed"] for a in by_row[(root_id, arm)]["attempts"])
            }
            negative_control_audit[split] = {
                "expected_exhausted": len(expected),
                "kernel_exhausted": len(observed),
                "expected_but_certified": sorted(expected - observed),
                "unexpectedly_exhausted": sorted(observed - expected),
                "contract_passed": expected == observed,
            }
        for split in ("ordinary_test", "ood_test"):
            root_ids = [x["root_id"] for x in sources if x["split"] == split]
            false_stop = 0
            checks = {}
            for arm in arms:
                rows = [by_row[(root_id, arm)] for root_id in root_ids]
                success = sum(bool(r["attempts"] and r["attempts"][-1]["receipt"]["passed"]) for r in rows)
                calls = sum(len(r["attempts"]) for r in rows)
                in_report = report["development_results"][split][arm]
                if success != in_report["kernel_certified"] or calls != in_report["attempts"]:
                    raise ValueError(f"aggregate mismatches native receipt ledger: {split}/{arm}")
                checks[arm] = {"certified": success, "checks": calls}
            for root_id in root_ids:
                learned_attempts = by_row[(root_id, "learned")]["attempts"]
                exhaustive_attempts = by_row[(root_id, "fixed_full")]["attempts"]
                if (exhaustive_attempts and exhaustive_attempts[-1]["receipt"]["passed"]
                        and not (learned_attempts and learned_attempts[-1]["receipt"]["passed"])):
                    false_stop += 1
            checked[split] = checks
            length_rows = [
                length_first_control(by_id[root_id], by_row[(root_id, "fixed_full")])
                for root_id in root_ids
            ]
            length_control[split] = {
                "roots": len(length_rows),
                "observed": sum(r["observed"] for r in length_rows),
                "kernel_certified": sum(r["kernel_verified"] for r in length_rows if r["observed"]),
                "counterfactual_checks": len(length_rows),
                "min_length_margin": min(
                    r["length"] - r["largest_other_length"]
                    for r in length_rows if r["observed"]
                ) if any(r["observed"] for r in length_rows) else None,
            }
            false_stops[split] = false_stop
            if paired:
                legacy_false_stops[split] = sum(
                    bool(
                        by_row[(root_id, "fixed_full")]["attempts"]
                        and by_row[(root_id, "fixed_full")]["attempts"][-1]["receipt"]["passed"]
                        and not (
                            by_row[(root_id, "legacy_grounded")]["attempts"]
                            and by_row[(root_id, "legacy_grounded")]["attempts"][-1]["receipt"]["passed"]
                        )
                    ) for root_id in root_ids
                )
        return {
            "evidence_verified": True,
            "integrity_manifest_members": len(manifests),
            "source_roots": len(sources),
            "kernel_receipts_checked": native_receipt_count,
            "split_metrics": checked,
            "posthoc_length_first_control": length_control,
            "false_stops_vs_full_search": false_stops,
            "legacy_grounded_false_stops_vs_full_search": legacy_false_stops if paired else None,
            "paired_feature_ablation": paired,
            "negative_control_audit": negative_control_audit,
            "registered_exhaustion_contract_passed": all(
                item["contract_passed"] for item in negative_control_audit.values()
            ),
            "requires_manual_hardness_review": True,
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
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
