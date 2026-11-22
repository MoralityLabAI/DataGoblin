"""Audit Bonsai-8B Q1 native Lean tactic-selection receipts and provenance.

The verifier consumes the collected source artifact and never invokes the
model or Lean. It rejects out-of-menu candidate IDs, invalid kernel certificates,
mismatched SHA-256 manifests, and incomplete negative-menu controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from datagoblin.training_gym.rlt_gym.external_checked_exit import digest

SCHEMA = "datagoblin.rlt_hard02f_oxibonsai_q1_native.v1"
MODEL_SHA = "284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54"
LEAN_SHA = "809c3fb3b5c8f5d7dace56e200b426187516535a"
OXI_SHA = "1d10c48e14754ff6999158f8fa13c7c92b40b0dc"
ARMS = ("oxibonsai_q1", "fixed_full", "random_full", "suffix_length_rule")
ALLOWED_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}
SRC = re.compile(r"^source_roots/(ordinary_test|ood_test)/MathlibListE_(\d{2})_[0-9a-f]{20}\.lean$")
HEX = re.compile(r"^[0-9a-f]{64}$")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Reader:
    def __init__(self, path: Path):
        self.path = path
        self.zip = ZipFile(path) if path.is_file() else None
        if self.zip:
            names = self.zip.namelist()
            if len(names) != len(set(names)) or any(
                path.startswith("/") or ".." in Path(path).parts for path in names
            ):
                raise ValueError("duplicate or unsafe evidence archive member")

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


def _kernel(receipt: dict, task: dict, candidate: int) -> bool:
    if (
        receipt.get("task_digest") != digest(task)
        or receipt.get("candidate_digest") != digest({"id": candidate})
        or not HEX.fullmatch(str(receipt.get("verifier_sha256", "")))
    ):
        raise ValueError("source theorem or candidate/checker receipt identity drift")
    if type(receipt.get("passed")) is not bool:
        raise ValueError("native kernel proof outcome must be boolean")
    if receipt["passed"]:
        detail = receipt.get("evidence", {})
        if not (
            receipt.get("code") == "kernel_certified"
            and detail.get("kernel_invoked") is True
            and detail.get("kernel_ok") is True
            and isinstance(detail.get("axioms"), list)
            and set(detail["axioms"]) <= ALLOWED_AXIOMS
        ):
            raise ValueError("untrusted accepted Lean source/axiom certificate")
    return receipt["passed"]


def _summary(rows: list[dict]) -> dict:
    return {
        "roots": len(rows),
        "kernel_certified": sum(bool(row["certified"]) for row in rows),
        "candidate_checks": sum(len(row["attempts"]) for row in rows),
        "native_candidate_wall_seconds": round(sum(row["native_candidate_wall_seconds"] for row in rows), 6),
        "model_calls": sum(row["model_calls"] for row in rows),
        "model_wall_seconds": round(sum(row["model_wall_seconds"] for row in rows), 6),
    }


def audit(path: Path) -> dict:
    reader = Reader(path)
    try:
        report = reader.json("report.json")
        if report.get("schema") != SCHEMA or report.get("upstream_mathlib_commit") != LEAN_SHA:
            raise ValueError("native Bonsai/Lean evidence schema or Lean source pin mismatch")
        if report.get("oxibonsai_repo_commit") != OXI_SHA:
            raise ValueError("OxiBonsai upstream inference engine source pin mismatch")
        binding = reader.json("model_binding.json")
        if binding != report.get("model_identity"):
            raise ValueError("independent model binding differs from reported runner")
        if binding.get("model_sha256") != MODEL_SHA or binding.get("model_id") != "prism-ml/Bonsai-8B-gguf":
            raise ValueError("published Bonsai-8B GGUF checkpoint identity drift")
        for field in ("tokenizer_sha256", "runtime_binary_sha256"):
            if not HEX.fullmatch(str(binding.get(field, ""))):
                raise ValueError("unhashed Bonsai runtime or tokenizer")
        if binding.get("upstream_oxibonsai_sha256_commit") != OXI_SHA:
            raise ValueError("unregistered source provenance")
        if binding.get("transport") not in ("stdio", "http"):
            raise ValueError("unsupported OxiBonsai inference transport")
        if binding["transport"] == "http" and binding.get("server_checkpoint_attestation") != (
            "self_report_only_requires_operator_pinned_startup_command"
        ):
            raise ValueError("untrue claim of HTTP server's checkpoint provenance")
        filenames = {}
        for key in ("artifact_sha256", "native_source_sha256", "certified_source_sha256"):
            items = report.get(key)
            if not isinstance(items, dict):
                raise ValueError("missing evidence integrity manifest")
            for name, sha in items.items():
                if name in filenames or _hash(reader.read(name)) != sha:
                    raise ValueError(f"missing or tampered Bonsai artifact SHA: {name}")
                filenames[name] = sha
        source = reader.lines("evaluator_full_menu.jsonl")
        if not source or len(source) != report.get("source_roots"):
            raise ValueError("unregistered native source-root population")
        if report.get("source_split") not in ("ood_test", "ordinary_test"):
            raise ValueError("unregistered source split or protected sealed data")
        by_source = {r["root_id"]: r for r in source}
        if len(by_source) != len(source):
            raise ValueError("duplicate public source root identity")
        if len({x["public_task"]["statement"] for x in source}) != len(source):
            raise ValueError("duplicate theorem statement")
        source_idx = set()
        qa_checked = 0
        for row in source:
            task = row["public_task"]
            name = task["source_file"]
            m = SRC.fullmatch(name)
            if (
                m is None or name not in filenames
                or row["split"] != report["source_split"]
                or m.group(1) != row["split"]
                or task["source_identity"] != row["root_id"]
                or task["import"] != "Mathlib"
                or task["candidate_menu_sha256"] != digest(row["menu"])
                or "⊢" not in row["goal_state"]["goal_pp"]
            ):
                raise ValueError("source goal/candidate provenance contract changed")
            idx = int(m.group(2))
            source_idx.add(idx)
            if row["registered_expected_witness"] is not (idx % 4 != 3):
                raise ValueError("changed registered test-only expected menu flag")
            if len(row["menu"]) != 5 or len({len(t.encode()) for t in row["menu"]}) != 1:
                raise ValueError("non-uniform source candidate menu")
            qa = row["attempts"]
            if len(qa) != 5 or {x["candidate_id"] for x in qa} != set(range(5)):
                raise ValueError("missing native full-menu QA outcomes")
            hits = 0
            for row_check in qa:
                qa_checked += 1
                hits += int(_kernel(row_check["receipt"], task, row_check["candidate_id"]))
            if hits != int(row["registered_expected_witness"]):
                raise ValueError("source oracle menu label does not agree with kernel")
        if source_idx != set(range(report["source_roots"])) or report["native_qa_checks"] != qa_checked:
            raise ValueError("candidate/native root registry incomplete")
        rows = reader.lines("native_policy_receipts.jsonl")
        pairs = {(r["root_id"], r["arm"]): r for r in rows}
        if len(pairs) != len(rows) or set(pairs) != {
            (rid, arm) for rid in by_source for arm in ARMS
        }:
            raise ValueError("missing source-matched Bonsai/control arm")
        policy_summary = {}
        evaluated = 0
        model_proposals = 0
        for arm in ARMS:
            cohort = [pairs[(rid, arm)] for rid in by_source]
            for row in cohort:
                if row["split"] != report["source_split"]:
                    raise ValueError("protected source split leaked")
                task = by_source[row["root_id"]]["public_task"]
                ref = {x["candidate_id"]: x["receipt"]
                       for x in by_source[row["root_id"]]["attempts"]}
                ids = [x["candidate_id"] for x in row["attempts"]]
                if len(ids) > report["per_root_attempt_budget"] or len(set(ids)) != len(ids):
                    raise ValueError("repeated candidate or exceeded policy check budget")
                for check in row["attempts"]:
                    evaluated += 1
                    candidate = check["candidate_id"]
                    _kernel(check["receipt"], task, candidate)
                    if ref[candidate] != check["receipt"]:
                        raise ValueError("online Bonsai proof disagrees with frozen native QA")
                success = bool(row["attempts"] and row["attempts"][-1]["receipt"]["passed"])
                if row["certified"] is not success:
                    raise ValueError("scored proof lacks accepted kernel source")
                if row["candidate_checks"] != len(ids):
                    raise ValueError("native candidate-check cost changed")
                if arm == "oxibonsai_q1":
                    model_proposals += len(row["model_proposals"])
                    if row["model_calls"] != len(row["model_proposals"]):
                        raise ValueError("claimed model-call count differs from actual proposals")
                    actual = [p["candidate_id"] for p in row["model_proposals"] if "candidate_id" in p]
                    if actual != ids:
                        raise ValueError("selected model candidate differs from kernel outcomes")
                    for p in row["model_proposals"]:
                        if not (HEX.fullmatch(str(p.get("request_sha256", "")))
                                and HEX.fullmatch(str(p.get("response_sha256", "")))):
                            raise ValueError("unhashed model prompt or response")
                        if p["choice"] is not None and not 1 <= p["choice"] <= 6:
                            raise ValueError("unregistered model choice")
                else:
                    if row["model_calls"] != 0:
                        raise ValueError("oracle-only control charged a model invocation")
            summary = _summary(cohort)
            if summary != report["arms"][arm]:
                raise ValueError("published Bonsai/Lean scorecard differs from receipts")
            policy_summary[arm] = summary
        if report["actual_model_invocations"] != model_proposals:
            raise ValueError("missing or fabricated actual Bonsai inference calls")
        if report["independent_bonsai_action_replays"] != policy_summary["oxibonsai_q1"]["candidate_checks"]:
            raise ValueError("Bonsai action kernel replays incomplete")
        false_stop = sum(
            pairs[(rid, "fixed_full")]["certified"] and not pairs[(rid, "oxibonsai_q1")]["certified"]
            for rid in by_source
        )
        if report["false_model_stops_against_fixed"] != false_stop:
            raise ValueError("model STOP errors disagree with the full-search control")
        return {
            "schema": "datagoblin.rlt_hard02f_independent_audit.v1",
            "native_kernel_source_verified": True,
            "model_checkpoint_sha256_valid": True,
            "script_and_proof_file_hashes": len(filenames),
            "native_full_menu_evaluator_checks": qa_checked,
            "native_policy_checks": evaluated,
            "actual_model_inference_calls": model_proposals,
            "live_model_response_hashes_present": True,
            "source_roots": len(source),
            "arms": policy_summary,
            "false_stops_vs_full": false_stop,
            "server_checkpoint_verified_by": binding["server_checkpoint_attestation"],
            "sealed_confirmation_opened": False,
        }
    finally:
        reader.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = audit(args.evidence)
    packet = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError("Bonsai source/verification audit file must be create-only")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(packet)
    print(packet)


if __name__ == "__main__":
    main()
