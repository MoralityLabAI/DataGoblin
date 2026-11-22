#!/usr/bin/env python3
"""Audit public MeTTa Git-tree metadata without importing unreviewed source data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from corpus import ROOT

TREE_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PERMISSIVE = frozenset({"MIT", "Apache-2.0"})


def audit_index(index: Path, report_path: Path | None = None) -> dict[str, Any]:
    files = sorted(index.glob("*.json"))
    if not files:
        raise ValueError("Empty discovery inventory")
    seen_repos: set[str] = set()
    sha_bytes: dict[str, int] = {}
    sha_repos: dict[str, set[str]] = {}
    by_license: dict[str, dict[str, int]] = {}
    per_repo: list[dict[str, Any]] = []
    gross_bytes = 0
    gross_files = 0
    bulk_bytes = 0
    bulk_files = 0
    permissive_small_candidate_bytes = 0
    permissive_small_candidate_files = 0
    for file_path in files:
        entry = json.loads(file_path.read_text(encoding="utf-8"))
        if entry.get("schema_version") != "metta-public-tree-inventory-v1":
            raise ValueError(f"Unexpected index schema: {file_path}")
        repo = entry["repository"]
        if not re.fullmatch(r"[^/]+/[^/]+", repo) or repo in seen_repos:
            raise ValueError(f"Invalid or repeated repository: {repo}")
        seen_repos.add(repo)
        if file_path.name != repo.replace("/", "--") + ".json":
            raise ValueError("Filename does not match repository identity")
        if not TREE_SHA_RE.fullmatch(entry.get("commit", "")):
            raise ValueError(f"Missing pinned upstream commit: {repo}")
        if entry["source_admission"] != "discovery_only" or entry["tokenizer_count"] is not None:
            raise ValueError(f"Discovery accidentally promoted to training data: {repo}")
        source_files = entry["files"]
        if not isinstance(source_files, list):
            raise ValueError(f"File inventory must be a list: {repo}")
        # File order is presentation-only: JS and Python sort mixed-case paths differently.
        # Identity, duplicate-path checks, source bytes, and Git blob checks remain strict.
        license_id = entry.get("license_spdx") or "UNKNOWN"
        seen_paths: set[str] = set()
        repo_bytes = 0
        repo_bulk_bytes = 0
        repo_bulk_files = 0
        for item in source_files:
            path = item["path"]
            size = item["bytes"]
            sha = item["git_blob_sha1"]
            if (
                not isinstance(path, str)
                or path in seen_paths
                or not path.endswith(".metta")
                or path.startswith("/")
                or ".." in path.split("/")
                or not isinstance(size, int)
                or size < 0
                or not TREE_SHA_RE.fullmatch(sha)
            ):
                raise ValueError(f"Malformed source-file inventory: {repo}:{path}")
            seen_paths.add(path)
            if sha in sha_bytes and sha_bytes[sha] != size:
                raise ValueError(f"Contradictory Git blob byte length: {sha}")
            sha_bytes[sha] = size
            sha_repos.setdefault(sha, set()).add(repo)
            bulk = item["kind_guess"] == "bulk_or_generated"
            if item["kind_guess"] not in {"source_or_test", "bulk_or_generated"}:
                raise ValueError(f"Unrecognized file kind: {repo}:{path}")
            flags = item["review_flags"]
            if not isinstance(flags, list):
                raise ValueError(f"Missing review flags: {repo}:{path}")
            if bulk and "dataset_and_generated_content_review" not in flags:
                raise ValueError(f"Missing bulk-data review: {repo}:{path}")
            if license_id in {"UNKNOWN", "NOASSERTION"} and "missing_spdx_license" not in flags:
                raise ValueError(f"Unlicensed file has no quarantine flag: {repo}:{path}")
            repo_bytes += size
            gross_files += 1
            gross_bytes += size
            if bulk:
                repo_bulk_bytes += size
                repo_bulk_files += 1
                bulk_bytes += size
                bulk_files += 1
            if license_id in PERMISSIVE and not flags and 0 < size <= 32768:
                permissive_small_candidate_bytes += size
                permissive_small_candidate_files += 1
        declared = entry["summary"]
        if (
            declared["source_file_count"] != len(source_files)
            or declared["source_bytes"] != repo_bytes
            or declared["bulk_bytes"] != repo_bulk_bytes
            or declared["bulk_files"] != repo_bulk_files
        ):
            raise ValueError("Repository inventory summary mismatch: " + repo)
        slot = by_license.setdefault(license_id, {"repositories": 0, "files": 0, "bytes": 0})
        slot["repositories"] += 1
        slot["files"] += len(source_files)
        slot["bytes"] += repo_bytes
        per_repo.append({
            "repository": repo,
            "commit": entry["commit"],
            "license_spdx": license_id,
            "metta_files": len(source_files),
            "metta_bytes": repo_bytes,
            "bulk_or_generated_bytes": repo_bulk_bytes,
            "admission": "discovery_only",
        })
    cross_repo_duplicated_shas = sum(len(owners) > 1 for owners in sha_repos.values())
    report = {
        "schema_version": "metta-public-discovery-audit-v1",
        "indexed_repositories": len(seen_repos),
        "indexed_metta_files": gross_files,
        "gross_git_source_bytes": gross_bytes,
        "distinct_git_blobs": len(sha_bytes),
        "deduplicated_git_blob_bytes": sum(sha_bytes.values()),
        "duplicated_git_blob_occurrences": gross_files - len(sha_bytes),
        "git_blobs_seen_across_repositories": cross_repo_duplicated_shas,
        "bulk_or_generated_files": bulk_files,
        "bulk_or_generated_bytes": bulk_bytes,
        "small_permissively_licensed_metadata_candidates": permissive_small_candidate_files,
        "small_candidate_bytes": permissive_small_candidate_bytes,
        "token_count": None,
        "license_buckets": dict(sorted(by_license.items())),
        "by_repo": sorted(per_repo, key=lambda x: (-x["metta_bytes"], x["repository"])),
        "training_ready": False,
        "discovery_only": True,
        "notes": (
            "Git-tree byte totals are measured, tokenizer counts are not. "
            "Bulk graph exports, generated corpora, unknown licenses, and derived content "
            "require file-specific provenance and rights review. Repeated blob SHAs "
            "across forks and copies are deduplicated by exact Git bytes."
        ),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "discovery_index")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/public_discovery.json")
    args = parser.parse_args()
    audit = audit_index(args.index, args.out)
    summary = {
        key: audit[key] for key in (
            "indexed_repositories", "indexed_metta_files", "gross_git_source_bytes",
            "distinct_git_blobs", "deduplicated_git_blob_bytes",
            "git_blobs_seen_across_repositories", "bulk_or_generated_bytes",
            "small_permissively_licensed_metadata_candidates", "training_ready"
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
