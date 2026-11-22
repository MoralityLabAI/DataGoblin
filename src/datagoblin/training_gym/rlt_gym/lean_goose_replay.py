"""Replay first-turn Inky tactic lists against Lean without new inference."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file

from .external_checked_exit import digest
from .lean_goose import AXIOMS, ROOTS, SCHEMA, probe, tactic_plan, write_row
from .live_lean_kernel import LeanNativeSession

REPLAY_SCHEMA = "datagoblin.lean_goose_plan_replay.v1"


def _first_turns(archive: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads((archive / "SHA256.json").read_text(encoding="utf-8"))
    source = archive / "transitions.jsonl"
    if sha256_file(source) != manifest.get(source.name):
        raise ValueError("live transcript does not match its manifest")
    expected = {digest(asdict(root)): root for root in ROOTS}
    revision = digest([asdict(root) for root in ROOTS])
    rows = {}
    for raw in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        if row["visit"] != 0:
            continue
        key = row["root_id"]
        root = expected.get(key)
        if root is None or key in rows or row["schema"] != SCHEMA or row["source_revision_sha256"] != revision:
            raise ValueError("unexpected or duplicated first-turn source identity")
        if (row["split"], row["statement"], row["lineage"]) != (root.split, root.statement, root.lineage):
            raise ValueError("first-turn root custody mismatch")
        if row["proof_before"] or not row["goals_before"]:
            raise ValueError("first-turn proof state does not start at the registered root")
        model = row["model"]
        request = {"prompt": row["prompt"], "max_tokens": model.get("max_tokens")}
        response = {k: v for k, v in model.items() if k not in {"request_sha256", "response_sha256"}}
        if model.get("prompt_token_parity") is True and (
            digest(request) != model.get("request_sha256") or digest(response) != model.get("response_sha256")
            or model["provider"]["usage"]["input_tokens"] != model.get("preflight_prompt_tokens")
        ):
            raise ValueError("live model request or response checksum did not match")
        rows[key] = row
    if rows.keys() != expected.keys():
        raise ValueError("replay requires every registered live root")
    return rows


def run(lean: Path, archive: Path, out: Path) -> dict[str, Any]:
    lean = lean.resolve(strict=True)
    version = subprocess.check_output([str(lean), "--version"], encoding="utf-8", timeout=15).strip()
    if not version.startswith("Lean (version 4.12.0,"):
        raise ValueError("the plan replay requires native Lean 4.12.0")
    rows = _first_turns(archive)
    out.mkdir(parents=True)
    for name in ("transitions.jsonl", "episodes.jsonl", "sft.jsonl"):
        (out / name).touch()
    summaries = []
    for key, row in rows.items():
        model = row["model"]
        plan = tactic_plan(model.get("answer", "")) if model.get("prompt_token_parity") is True else []
        native = out / "native" / row["lineage"]
        native.mkdir(parents=True)
        certificate = None
        status = "no_admissible_first_turn_plan"
        checked = 0
        for checked, action in enumerate(plan, 1):
            receipt = probe(lean, row["statement"], plan[:checked], native / f"Step{checked}.lean")
            status = receipt["status"]
            write_row(out / "transitions.jsonl", {
                "schema": REPLAY_SCHEMA, "root_id": key, "split": row["split"], "step": checked,
                "source_model_receipt_sha256": model["response_sha256"], "action": action, "native": receipt,
            })
            if status != "open":
                break
        if status == "closed":
            session = LeanNativeSession(lean, native, {1: row["statement"]})
            compiled = session.run(1, "\n".join(plan[:checked]))
            status = "certification_failed"
            if compiled["ok"]:
                sid = compiled["state"]["sid"]
                audit = session.kernel(sid)
                if audit["ok"] and set(audit["axioms"]) <= AXIOMS:
                    proof = out / "certified_proofs" / f"{row['lineage']}.lean"
                    proof.parent.mkdir(exist_ok=True)
                    shutil.copyfile(session.proof_by_sid[sid][0], proof)
                    certificate = {**audit, "path": proof.relative_to(out).as_posix()}
                    status = "kernel_certified"
        training = certificate is not None and row["split"] == "train"
        if training:
            write_row(out / "sft.jsonl", {
                "schema": REPLAY_SCHEMA, "root_id": key, "lineage": row["lineage"], "split": row["split"],
                "source_model_receipt_sha256": model["response_sha256"], "certificate": certificate,
                "messages": [
                    {"role": "user", "content": "Propose a Lean 4 tactic script, one tactic per line.\n"
                     f"Theorem: {row['statement']}"},
                    {"role": "assistant", "content": "\n".join(plan[:checked])},
                ],
            })
        summary = {"root_id": key, "lineage": row["lineage"], "split": row["split"], "status": status,
                   "prefix_checks": checked, "certificate": certificate, "sft_rows": int(training)}
        summaries.append(summary)
        write_row(out / "episodes.jsonl", summary)
    report = {"schema": REPLAY_SCHEMA, "new_model_requests": 0,
              "driver_sha256": sha256_file(Path(__file__)),
              "source_transitions_sha256": sha256_file(archive / "transitions.jsonl"),
              "lean": version, "lean_binary_sha256": sha256_file(lean), "episodes": summaries,
              "kernel_certified": sum(row["status"] == "kernel_certified" for row in summaries),
              "sft_rows": sum(row["sft_rows"] for row in summaries)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    hashes = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lean", "archive", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.lean, args.archive, args.out)
    print(json.dumps({"kernel_certified": result["kernel_certified"], "sft_rows": result["sft_rows"]}))


if __name__ == "__main__":
    main()
