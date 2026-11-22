#!/usr/bin/env python3
"""Restore a missing pinned source snapshot, verify its Git blob SHA before writing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from corpus import LICENSE_ALLOWLIST, ROOT, git_blob_sha1, safe_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for annotation in sorted((args.root / "annotations").rglob("*.json")):
        item = json.loads(annotation.read_text(encoding="utf-8"))
        if item["license_spdx"] not in LICENSE_ALLOWLIST:
            raise ValueError("Unapproved license: " + item["id"])
        destination = safe_file(args.root, item["raw_path"])
        if destination.exists():
            if git_blob_sha1(destination.read_bytes()) != item["upstream_blob_sha1"]:
                raise ValueError(f"Existing source has changed: {item['id']}")
            continue
        parts = item["source_repo"].split("/") + [item["source_commit"]] + item["source_path"].split("/")
        url = "https://raw.githubusercontent.com/" + "/".join(quote(p, safe="") for p in parts)
        if args.dry_run:
            print("Would fetch", item["id"])
            continue
        with urlopen(url, timeout=25) as response:
            data = response.read(1_000_001)
        if len(data) > 1_000_000 or git_blob_sha1(data) != item["upstream_blob_sha1"]:
            raise ValueError("Source size/digest check failed: " + item["id"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        print("Restored", item["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
