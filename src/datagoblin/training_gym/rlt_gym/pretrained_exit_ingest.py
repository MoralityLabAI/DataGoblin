"""Strictly import BPD pretrained TRM TRAIN-only exit data into the RLT gym.

Accept a directory with BPD exporter files or a GitHub Actions artifact ZIP
containing datagoblin_train/. Reverify every Sudoku answer and source hash,
preserve root lineage, and reject evaluation/cross-split material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

from datagoblin.serialization import canonical_json, dump_structured

MODEL_SHA = "932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
FILES = ("manifest.json", "controller_sft.jsonl", "outcomes.jsonl", "repairs.jsonl")
MAX_FILE_BYTES = 16 * 1024 * 1024
FEATURES = (
    "mean_blank_logit_margin", "min_blank_logit_margin",
    "p10_blank_logit_margin", "median_blank_logit_margin",
    "mean_blank_entropy", "mean_blank_top_probability",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any) -> str:
    return _sha(canonical_json(value).encode())


def _source(path: Path) -> dict[str, bytes]:
    """Read bounded named members without extracting or trusting ZIP paths."""
    if path.is_dir():
        result = {}
        for name in FILES:
            file = path / name
            if file.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("oversized corpus file")
            result[name] = file.read_bytes()
        return result
    if not zipfile.is_zipfile(path):
        raise ValueError("expected BPD training directory or Actions ZIP")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        prefix = "datagoblin_train/" if "datagoblin_train/manifest.json" in names else ""
        expected = {prefix + name for name in FILES}
        if not expected <= names:
            raise ValueError("missing expected BPD training files in archive")
        result = {}
        for name in FILES:
            info = archive.getinfo(prefix + name)
            if info.is_dir() or info.file_size > MAX_FILE_BYTES:
                raise ValueError("invalid or oversized training member")
            data = archive.read(info)
            if len(data) != info.file_size:
                raise ValueError("ZIP file size mismatch")
            result[name] = data
    return result


def _rows(data: bytes) -> list[dict[str, Any]]:
    rows = []
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("training row must be JSON object")
            rows.append(item)
    return rows


def _sudoku(q: str, answer: str) -> bool:
    if not isinstance(q, str) or not isinstance(answer, str) or len(q) != 81 or len(answer) != 81:
        return False
    if any(v not in "123456789." for v in q) or any(v not in "123456789" for v in answer):
        return False
    if any(c != "." and c != answer[i] for i, c in enumerate(q)):
        return False
    for r in range(9):
        if {answer[r * 9 + c] for c in range(9)} != set("123456789"):
            return False
    for c in range(9):
        if {answer[r * 9 + c] for r in range(9)} != set("123456789"):
            return False
    for r in (0, 3, 6):
        for c in (0, 3, 6):
            if {answer[(r + i) * 9 + c + j] for i in range(3) for j in range(3)} != set(
                "123456789"
            ):
                return False
    return True


def _base(row: dict[str, Any], *, kind: str, source_sha: str) -> tuple[str, str]:
    if (
        row.get("kind") != kind
        or row.get("split") != "train"
        or row.get("training_eligible") is not True
        or row.get("model_sha256") != MODEL_SHA
        or row.get("family") != "sudoku9"
        or row.get("teacher_hidden_state_access") is not False
        or row.get("source_sha256") != source_sha
    ):
        raise ValueError("source, checkpoint or TRAIN custody invalid")
    root = row.get("root_id")
    task = row.get("task_digest")
    if type(root) is not str or not root.startswith("train-sudoku-") or type(task) is not str:
        raise ValueError("training root/task identity invalid")
    return root, task


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def ingest(source: Path, out: Path) -> dict[str, Any]:
    raw = _source(source)
    manifest = json.loads(raw["manifest.json"])
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "rlt_gym.pretrained_trm_train_export.v1"
        or manifest.get("train_only") is not True
        or manifest.get("development_rows_exported") != 0
        or manifest.get("model_sha256") != MODEL_SHA
    ):
        raise ValueError("invalid BPD TRAIN export manifest")
    for name in FILES[1:]:
        if _sha(raw[name]) != manifest.get("files_sha256", {}).get(name):
            raise ValueError(f"corrupt BPD corpus: {name}")
    sft = _rows(raw["controller_sft.jsonl"])
    outcomes = _rows(raw["outcomes.jsonl"])
    preferences = _rows(raw["repairs.jsonl"])
    if len(sft) != manifest["training_root_count"] or len(preferences) != manifest["verified_repair_pairs"]:
        raise ValueError("manifest counts differ from source rows")
    if len(outcomes) != len(sft):
        raise ValueError("outcome training matrix incomplete")
    source_sha = manifest["source_train_sha256"]
    if type(source_sha) is not str or len(source_sha) != 64:
        raise ValueError("source TRAIN digest missing")
    by_root: dict[str, dict[str, Any]] = {}
    records = []
    for example in sft:
        if example.get("schema_version") != "rlt_gym.controller_sft.v1":
            raise ValueError("invalid SFT schema")
        root, task = _base(example, kind="verified_stop_decision", source_sha=source_sha)
        if root in by_root:
            raise ValueError("duplicate controller task root")
        inp, target = example.get("input"), example.get("target")
        if not isinstance(inp, dict) or not isinstance(target, dict):
            raise ValueError("incomplete controller input or target")
        question = inp.get("task_question")
        candidate = inp.get("first_candidate")
        if not isinstance(question, str) or _digest({"question": question}) != task:
            raise ValueError("task question does not match digest")
        truth = _sudoku(question, candidate)
        if example.get("verification", {}).get("first_pass_verified") is not truth:
            raise ValueError("first-step Sudoku verifier disagreement")
        if target.get("decision") != ("stop" if truth else "continue"):
            raise ValueError("unverified stop/continue target")
        names, values = inp.get("logits_only_feature_names"), inp.get("logits_only_features")
        if (
            names != list(FEATURES) or not isinstance(values, list)
            or len(values) != len(FEATURES)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
        ):
            raise ValueError("feature selection changed")
        by_root[root] = {"question": question, "first": candidate, "verified": truth, "task": task}
        records.append({
            "schema_version": "rlt_gym.pretrained_exit_example.v1",
            "kind": "verified_stop_supervision", "split": "train",
            "task_id": root, "root_id": root, "task_digest": task,
            "model_sha256": MODEL_SHA, "training_eligible": True,
            "context": {
                "task": {"family": "sudoku9", "question": question},
                "candidate": candidate,
                "logits_only_features": dict(zip(FEATURES, values, strict=True)),
            },
            "target": {"action": target["decision"]},
            "verification": {"first_pass_verified": truth},
        })
    outcomes_seen: set[str] = set()
    for outcome in outcomes:
        if outcome.get("schema_version") != "rlt_gym.controller_outcome.v1":
            raise ValueError("wrong future-outcome schema")
        root, task = _base(outcome, kind="future_verifier_outcome", source_sha=source_sha)
        if root not in by_root or root in outcomes_seen or by_root[root]["task"] != task:
            raise ValueError("missing, repeated or mismatched outcome root")
        outcomes_seen.add(root)
        state = by_root[root]
        future = _sudoku(state["question"], outcome.get("observed_second_candidate"))
        if (
            outcome.get("first_pass_verified") is not state["verified"]
            or outcome.get("second_pass_verified") is not future
            or outcome.get("second_pass_rescue") is not (future and not state["verified"])
        ):
            raise ValueError("outcome verifier labels do not replay")
        state["future"] = future
        state["second"] = outcome["observed_second_candidate"]
    if outcomes_seen != set(by_root):
        raise ValueError("missing future outcomes")
    repairs_seen = set()
    repairs = []
    for example in preferences:
        if example.get("schema_version") != "rlt_gym.controller_preference.v1":
            raise ValueError("invalid repair preference schema")
        root, task = _base(example, kind="verified_repair", source_sha=source_sha)
        if root not in by_root or root in repairs_seen or task != by_root[root]["task"]:
            raise ValueError("missing or duplicated repair root")
        state = by_root[root]
        if (
            state["verified"] or not state["future"]
            or example.get("rejected", {}).get("candidate") != state["first"]
            or example.get("chosen", {}).get("candidate") != state["second"]
            or example.get("verification") != {"chosen": True, "rejected": False}
        ):
            raise ValueError("unverified repair preference")
        repairs_seen.add(root)
        repairs.append({
            "schema_version": "rlt_gym.pretrained_exit_preference.v1",
            "kind": "verified_repair", "split": "train",
            "root_id": root, "task_digest": task, "training_eligible": True,
            "chosen": {"candidate": state["second"]},
            "rejected": {"candidate": state["first"]},
            "verification": {"chosen": True, "rejected": False},
        })
    if repairs_seen != {root for root, state in by_root.items() if not state["verified"] and state["future"]}:
        raise ValueError("missing verified positive repair examples")
    if out.exists():
        raise ValueError("output directory already exists")
    out.mkdir(parents=True)
    _write(out / "controller_train.jsonl", records)
    _write(out / "repair_preferences.jsonl", repairs)
    report = {
        "schema_version": "rlt_gym.pretrained_exit_ingest.v1",
        "source_manifest_sha256": _sha(raw["manifest.json"]),
        "source_model_sha256": MODEL_SHA,
        "train_only": True, "calibration_rows": 0, "development_rows": 0,
        "training_roots": len(records), "verified_repairs": len(repairs),
        "first_pass_correct": sum(state["verified"] for state in by_root.values()),
        "second_pass_correct": sum(state["future"] for state in by_root.values()),
        "files_sha256": {
            name: _sha((out / name).read_bytes())
            for name in ("controller_train.jsonl", "repair_preferences.jsonl")
        },
        "claim_boundary": (
            "Verified real pretrained 7M TRM STOP/CONTINUE labels from independent TRAIN "
            "Sudoku roots. No native Inkling cross-loop reader, held-out labels or hidden CoT."
        ),
    }
    dump_structured(out / "manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(ingest(args.source, args.out), sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"pretrained exit ingest: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
