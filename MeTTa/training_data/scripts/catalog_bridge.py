#!/usr/bin/env python3
"""Join annotated public MeTTa source intake to the frozen curation catalog, opt-in."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from corpus import ROOT, validate_corpus

FROZEN_CATALOG = ROOT.parent / "source_catalog.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bridge(
    root: Path = ROOT,
    original_path: Path = FROZEN_CATALOG,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_rows, summary, errors = validate_corpus(root)
    if errors:
        raise ValueError("Source intake failed provenance validation: " + "; ".join(errors))
    frozen = json.loads(original_path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != "metta-source-catalog-v1":
        raise ValueError("Unexpected frozen MeTTa source catalog schema")
    sources = {entry["source_id"]: dict(entry) for entry in frozen["sources"]}
    repo_to_id = {entry["repository"]: entry["source_id"] for entry in frozen["sources"]}
    catalog_files = [dict(entry) for entry in frozen["files"]]
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    by_blob: dict[str, set[str]] = {}
    for entry in catalog_files:
        source = sources[entry["source_id"]]
        key = (source["repository"], entry["path"])
        if key in by_key:
            raise ValueError("Duplicate root in frozen catalog: " + str(key))
        by_key[key] = entry
        by_blob.setdefault(entry["git_blob_sha"], set()).add(f"{source['repository']}:{entry['path']}")
    counts: Counter[str] = Counter()
    duplicates = []
    new_roots = 0
    for row in sorted(source_rows, key=lambda item: item["id"]):
        repo = row["source_repo"]
        path = row["source_path"]
        key = (repo, path)
        if key in by_key:
            previous = by_key[key]
            if previous["git_blob_sha"] != row["upstream_blob_sha1"]:
                raise ValueError(f"Same source path has conflicting blob identity: {repo}:{path}")
            duplicates.append({"source": f"{repo}:{path}", "sha1": row["upstream_blob_sha1"]})
            continue
        sid = repo_to_id.get(repo)
        if sid is None:
            sid = "expanded_" + repo.replace("/", "__").replace("-", "_")
            if sid in sources:
                raise ValueError("Conflicting deterministic source identifier: " + sid)
            repo_to_id[repo] = sid
            sources[sid] = {
                "source_id": sid,
                "repository": repo,
                "commit": row["source_commit"],
                "license": row["license_spdx"],
                "license_url": (
                    f"https://github.com/{repo}/blob/{row['source_commit']}/LICENSE"
                ),
                "default_partition": "discovery",
                "fetch_mode": "readonly_pinned_vendored_source",
            }
        if sources[sid]["commit"] != row["source_commit"]:
            raise ValueError(f"Unexpected mixed Git source revisions: {repo}")
        matching_blobs = by_blob.get(row["upstream_blob_sha1"], set())
        flags = ["source_only_pending_review", "runtime_assertions_not_verified"]
        if row["imports_detected"]:
            flags.append("runtime_dependent_imports")
        if matching_blobs:
            flags.append("duplicate_content_other_root")
        entry = {
            "source_id": sid,
            "path": path,
            "git_blob_sha": row["upstream_blob_sha1"],
            "bytes": row["utf8_bytes"],
            "category": row["task_family"],
            "canonical_root": f"{sid}:{path}",
            "proposed_split": "unassigned_until_root_dedup",
            "review_flags": flags,
            "eligibility": "quarantine",
            "training_ready": False,
            "source_intake": "MeTTa/training_data",
        }
        by_key[key] = entry
        by_blob.setdefault(row["upstream_blob_sha1"], set()).add(f"{repo}:{path}")
        catalog_files.append(entry)
        counts[repo] += 1
        new_roots += 1
    catalog_files.sort(key=lambda entry: (entry["source_id"], entry["path"]))
    catalog = {
        "schema_version": "metta-source-catalog-v1",
        "snapshot_date": "2026-11-22",
        "notice": (
            "Opt-in expanded source inventory. The frozen reference source catalog "
            "is unchanged. New upstream source files are quarantine-only and "
            "cannot authorize supervised training or evaluation labels."
        ),
        "sources": sorted(sources.values(), key=lambda item: item["source_id"]),
        "files": catalog_files,
        "training_ready": False,
        "source_intake_validation": {
            "source_files": summary["source_files"],
            "source_bytes": summary["source_bytes"],
            "source_repositories": summary["repos"],
            "dedup_level": "repo/path/git_blob_sha1",
            "new_roots": new_roots,
            "overlapping_exact_catalog_roots": len(duplicates),
        },
    }
    report = {
        "schema_version": "metta-catalog-bridge-v1",
        "frozen_catalog_sha256": digest(original_path),
        "frozen_files": len(frozen["files"]),
        "annotated_source_files": summary["source_files"],
        "overlapping_exact_file_roots": len(duplicates),
        "new_quarantined_file_roots": new_roots,
        "merged_file_roots": len(catalog_files),
        "new_by_repo": dict(sorted(counts.items())),
        "training_ready": False,
        "split_status": "unassigned_until_root_dedup",
        "label_custody": "source_discovery_only",
        "note": "No source-derived assertion has independent interpreter+oracle verification.",
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(catalog, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        report["merged_catalog_sha256"] = digest(output_path)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return catalog, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--original", type=Path, default=FROZEN_CATALOG)
    parser.add_argument("--out", type=Path, default=ROOT / "exports/expanded_source_catalog.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/catalog_bridge.json")
    args = parser.parse_args()
    _, report = build_bridge(args.root, args.original, args.out, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
