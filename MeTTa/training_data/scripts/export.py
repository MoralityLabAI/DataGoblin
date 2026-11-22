#!/usr/bin/env python3
"""Export either the source-only pretraining candidates or their metadata manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corpus import ROOT, safe_file, validate_corpus


def export(root: Path, output: Path, mode: str) -> int:
    rows, _, issues = validate_corpus(root)
    if issues:
        raise ValueError("Corpus validation failed: " + "; ".join(issues))
    if mode not in {"source", "manifest"}:
        raise ValueError(f"Unknown export mode: {mode}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda item: item["id"]):
            if mode == "manifest":
                payload = row
            else:
                # This is source-language-modeling data, never a verified solver trajectory.
                payload = {
                    "id": row["id"],
                    "text": safe_file(root, row["raw_path"]).read_text(encoding="utf-8"),
                    "metadata": {
                        "source_repo": row["source_repo"],
                        "source_commit": row["source_commit"],
                        "source_path": row["source_path"],
                        "upstream_blob_sha1": row["upstream_blob_sha1"],
                        "license_spdx": row["license_spdx"],
                        "source_kind": row["source_kind"],
                        "task_family": row["task_family"],
                        "runtime_hint": row["runtime_hint"],
                        "verification_status": row["verification_status"],
                        "split": row["split"],
                        "training_status": row["training_status"],
                    },
                }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=("source", "manifest"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        rows = export(args.root, args.out, args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Exported {rows} {args.mode} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
