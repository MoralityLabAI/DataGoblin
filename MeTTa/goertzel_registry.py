"""Audit a read-only, pinned Git-tree inventory of Goertzel ecosystem MeTTa sources.

The inventory contains repository paths, Git blob identities and byte sizes.
It neither downloads nor executes third-party source, nor grants training rights.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

CATALOG = Path(__file__).with_name("goertzel_source_discovery.json")
OBJECT_ID = re.compile(r"^[0-9a-f]{40}$")
PERMITTED_DISCOVERY_LICENSES = {"MIT", "Apache-2.0"}
FILE_CEILING = 128_000


def audit_registry(payload: dict[str, Any]) -> dict[str, Any]:
    """Reject privilege escalation, missing identities, duplicates and count drift."""
    if (
        payload.get("schema_version") != "metta-goertzel-source-discovery-v1"
        or payload.get("metadata_only") is not True
        or payload.get("training_ready") is not False
        or payload.get("source_content_copied") is not False
        or payload.get("size_quarantine_bytes") != FILE_CEILING
    ):
        raise ValueError("invalid source discovery custody")
    sources = payload.get("sources")
    files = payload.get("source_files")
    if not isinstance(sources, list) or not isinstance(files, list) or not sources:
        raise ValueError("empty or malformed source discovery manifest")
    known = {}
    for entry in sources:
        if not isinstance(entry, dict):
            raise ValueError("malformed source entry")
        identity = entry.get("source_id")
        if not isinstance(identity, str) or not identity or identity in known:
            raise ValueError("duplicate or missing source ID")
        if not isinstance(entry.get("repository"), str) or "/" not in entry["repository"]:
            raise ValueError("missing source repository")
        for field in ("pinned_commit", "tree_sha"):
            if not isinstance(entry.get(field), str) or not OBJECT_ID.fullmatch(entry[field]):
                raise ValueError("invalid pinned Git object identity")
        if entry.get("is_tree_complete") is not True:
            raise ValueError("incomplete upstream tree")
        if not isinstance(entry.get("metta_files"), int) or entry["metta_files"] < 0:
            raise ValueError("bad source file count")
        if not isinstance(entry.get("reported_metta_bytes"), int) or entry["reported_metta_bytes"] < 0:
            raise ValueError("bad source byte count")
        known[identity] = entry
    if [s["source_id"] for s in sources] != sorted(known):
        raise ValueError("source entries are not canonical")
    count: Counter[str] = Counter()
    size: Counter[str] = Counter()
    dispositions: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("malformed file record")
        sid = item.get("source_id")
        name = item.get("path")
        if sid not in known or not isinstance(name, str) or not name.endswith(".metta"):
            raise ValueError("unknown source or non-MeTTa file")
        if name.startswith("/") or ".." in Path(name).parts or "\\" in name:
            raise ValueError("unsafe source path")
        key = (sid, name)
        if key in seen:
            raise ValueError("repeated source file")
        seen.add(key)
        blob = item.get("git_blob_sha")
        if not isinstance(blob, str) or not OBJECT_ID.fullmatch(blob):
            raise ValueError("invalid source blob")
        bytes_ = item.get("size_bytes")
        if bytes_ is not None and (type(bytes_) is not int or bytes_ < 0):
            raise ValueError("invalid source size")
        if item.get("rights_status") != known[sid]["license_spdx"] or item.get("training_ready") is not False:
            raise ValueError("source rights or readiness drift")
        if any(field in item for field in ("content", "source_text", "program", "prompt", "training_label")):
            raise ValueError("source code or training labels embedded in discovery record")
        safe = known[sid]["license_spdx"] in PERMITTED_DISCOVERY_LICENSES and (
            bytes_ is not None and bytes_ <= FILE_CEILING
        )
        expected = "DISCOVERY_ONLY" if safe else "QUARANTINE"
        if item.get("disposition") != expected:
            raise ValueError("ineligible source admission")
        count[sid] += 1
        size[sid] += bytes_ or 0
        dispositions[expected] += 1
    if [(x["source_id"], x["path"]) for x in files] != sorted(seen):
        raise ValueError("file list not canonical")
    for sid, source in known.items():
        if count[sid] != source["metta_files"] or size[sid] != source["reported_metta_bytes"]:
            raise ValueError("source Git-tree totals differ")
    report = payload.get("report")
    expected_report = {
        "sources": len(sources),
        "metta_files": len(files),
        "reported_metta_bytes": sum(size.values()),
        "discovery_only": dispositions["DISCOVERY_ONLY"],
        "quarantined": dispositions["QUARANTINE"],
        "zero_metta_sources": sorted(sid for sid in known if count[sid] == 0),
    }
    if report != expected_report:
        raise ValueError("aggregate source report drift")
    return {"valid": True, **expected_report, "training_ready": False,
            "verified_source_contents": False, "license_review_complete": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=CATALOG)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = audit_registry(json.loads(args.manifest.read_text(encoding="utf-8")))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
