#!/usr/bin/env python3
"""Provenance and custody checks for the MeTTa source corpus; stdlib only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LICENSE_ALLOWLIST = frozenset({"MIT", "Apache-2.0"})
SOURCE_KINDS = frozenset({"upstream_test", "negative_test", "example", "library", "benchmark"})
REQUIRED = frozenset({
    "schema_version", "id", "raw_path", "source_repo", "source_commit",
    "source_path", "source_url", "upstream_blob_sha1", "license_spdx",
    "source_kind", "task_family", "runtime_hint", "utf8_bytes", "line_count",
    "embedded_test_assertion_count", "imports_detected", "verification_status",
    "verification_receipt", "rollout_status", "training_status", "split",
    "provenance_status", "model_token_count",
})
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
IMPORT_RE = re.compile(r"\(\s*(?:import!|include|load-file)(?=[\s)])")
ASSERT_RE = re.compile(r"!\s*\(\s*(?:assertEqualToResult|assertEqual|assertEqualMsg|test)\b")


def safe_file(root: Path, relative: str) -> Path:
    """Reject path traversal and absolute paths, including Windows separators."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("unsafe path")
    value = PurePosixPath(relative)
    if value.is_absolute() or any(part in {"..", "."} for part in value.parts):
        raise ValueError("unsafe path")
    base = root.resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError("unsafe path")
    return target


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def code_without_comment_lines(content: str) -> str:
    return "\n".join(line for line in content.split("\n") if not line.lstrip().startswith(";"))


def embedded_assertions(content: str) -> int:
    return len(ASSERT_RE.findall(code_without_comment_lines(content)))


def imports_present(content: str) -> bool:
    return bool(IMPORT_RE.search(code_without_comment_lines(content)))


def validate_corpus(root: Path = ROOT) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    root = root.resolve()
    annotations = sorted((root / "annotations").rglob("*.json"))
    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    blob_ids: dict[str, str] = {}
    total_bytes = 0
    total_assertions = 0
    imported = 0
    for annotation in annotations:
        label = annotation.relative_to(root).as_posix()
        try:
            row = json.loads(annotation.read_text(encoding="utf-8"))
            if not isinstance(row, dict):
                raise ValueError("annotation must be a JSON object")
            absent = sorted(REQUIRED.difference(row))
            if absent:
                raise ValueError(f"missing fields: {', '.join(absent)}")
            slug = row["source_repo"].replace("/", "--")
            path = row["source_path"]
            expected_id = f"{slug}:{path}"
            if row["id"] != expected_id or row["raw_path"] != f"raw/{slug}/{path}":
                raise ValueError("source ID or raw path disagrees with source repo/path")
            if label != f"annotations/{slug}/{path}.json":
                raise ValueError("annotation file path does not match source")
            if not re.fullmatch(r"[^/]+/[^/]+", row["source_repo"]):
                raise ValueError("invalid source_repo")
            if not isinstance(path, str) or not path.endswith(".metta"):
                raise ValueError("source path is not a .metta file")
            if not SHA1_RE.fullmatch(str(row["source_commit"])):
                raise ValueError("missing pinned 40-character source commit")
            if not SHA1_RE.fullmatch(str(row["upstream_blob_sha1"])):
                raise ValueError("invalid source Git blob SHA")
            expected_url = f"https://github.com/{row['source_repo']}/blob/{row['source_commit']}/{path}"
            if row["source_url"] != expected_url:
                raise ValueError("unmatched source URL")
            if row["id"] in ids:
                raise ValueError("duplicate source ID")
            ids.add(row["id"])
            if row["license_spdx"] not in LICENSE_ALLOWLIST:
                raise ValueError("license is not on the approved collection allowlist")
            if not (root / "licenses" / slug / "LICENSE").is_file():
                raise ValueError("missing upstream license copy")
            if row["source_kind"] not in SOURCE_KINDS or not row["task_family"]:
                raise ValueError("missing or unsupported training source labels")
            if row["provenance_status"] != "pinned_upstream_snapshot":
                raise ValueError("source provenance is not pinned")
            if row["verification_status"] not in {"not_run", "passed", "failed", "blocked"}:
                raise ValueError("invalid verification status")
            if row["verification_status"] == "not_run" and row["verification_receipt"] is not None:
                raise ValueError("unexecuted source claims an execution receipt")
            if row["verification_status"] in {"passed", "failed"}:
                if not row["verification_receipt"]:
                    raise ValueError("executed source lacks a verification receipt")
                receipt = safe_file(root, row["verification_receipt"])
                if not receipt.is_file():
                    raise ValueError("verification receipt is missing")
            if row["rollout_status"] != "none":
                raise ValueError("rollout records require a separate verified rollout dataset")
            if row["training_status"] not in {"source_only_pending_review", "source_pretraining_approved"}:
                raise ValueError("source files cannot be silently promoted to rollout/SFT labels")
            if row["split"] != "unassigned":
                raise ValueError("split assignment requires a separate lineage-frozen split manifest")
            raw = safe_file(root, row["raw_path"]).read_bytes()
            text = raw.decode("utf-8")
            if not raw:
                raise ValueError("empty source file")
            digest = git_blob_sha1(raw)
            if digest != row["upstream_blob_sha1"]:
                raise ValueError("upstream Git blob SHA mismatch")
            if digest in blob_ids:
                raise ValueError(f"duplicate source blob of {blob_ids[digest]}")
            blob_ids[digest] = row["id"]
            if len(raw) != row["utf8_bytes"]:
                raise ValueError("source UTF-8 byte count mismatch")
            if len(text.split("\n")) != row["line_count"]:
                raise ValueError("line count mismatch")
            count = embedded_assertions(text)
            if count != row["embedded_test_assertion_count"]:
                raise ValueError("embedded assertion count mismatch")
            imports = imports_present(text)
            if imports != row["imports_detected"]:
                raise ValueError("external import heuristic mismatch")
            if row["model_token_count"] is not None and (
                not isinstance(row["model_token_count"], int) or row["model_token_count"] < 1
            ):
                raise ValueError("invalid model token count")
            total_bytes += len(raw)
            total_assertions += count
            imported += int(imports)
            rows.append(row)
        except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            issues.append(f"{label}: {exc}")
    if not annotations:
        issues.append("no annotated source files found")
    summary = {
        "source_files": len(rows),
        "repos": len({row["source_repo"] for row in rows}),
        "source_bytes": total_bytes,
        "embedded_assertions_unverified": total_assertions,
        "source_files_with_imports": imported,
        "runtime_passed": sum(row["verification_status"] == "passed" for row in rows),
        "verified_rollouts": 0,
        "model_token_count": None,
        "family_counts": dict(sorted(Counter(row["task_family"] for row in rows).items())),
        "errors": len(issues),
    }
    return rows, summary, issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    _, summary, issues = validate_corpus(args.root)
    if args.json:
        print(json.dumps({"summary": summary, "issues": issues}, indent=2, sort_keys=True))
    else:
        print(f"MeTTa: {summary['source_files']} files, {summary['source_bytes']} bytes, "
              f"{summary['repos']} repos, {summary['errors']} errors")
        for issue in issues:
            print(f"ERROR {issue}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
