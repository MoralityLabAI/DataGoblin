#!/usr/bin/env python3
"""Measure actual corpus tokens with a named tokenizer; preserve source custody."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from corpus import ROOT, safe_file, validate_corpus


def count_tokens(root: Path, encoding: str, out: Path, report: Path) -> dict:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("Install a pinned tiktoken version before measuring tokens") from exc

    rows, corpus_summary, issues = validate_corpus(root)
    if issues:
        raise ValueError("Invalid source corpus: " + "; ".join(issues))
    encoder = tiktoken.get_encoding(encoding)
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0, "tokens": 0})
    family_tokens: Counter[str] = Counter()
    total = 0
    output = []
    for row in sorted(rows, key=lambda item: item["id"]):
        raw = safe_file(root, row["raw_path"]).read_bytes()
        tokens = len(encoder.encode(raw.decode("utf-8"), disallowed_special=()))
        if tokens < 1:
            raise ValueError("A nonempty source file produced zero tokens: " + row["id"])
        output.append({
            "id": row["id"],
            "source_repo": row["source_repo"],
            "source_commit": row["source_commit"],
            "upstream_blob_sha1": row["upstream_blob_sha1"],
            "license_spdx": row["license_spdx"],
            "task_family": row["task_family"],
            "source_kind": row["source_kind"],
            "verification_status": row["verification_status"],
            "tokenizer_encoding": encoding,
            "token_count": tokens,
            "utf8_bytes": len(raw),
        })
        count = grouped[row["source_repo"]]
        count["files"] += 1
        count["bytes"] += len(raw)
        count["tokens"] += tokens
        family_tokens[row["task_family"]] += tokens
        total += tokens
    summary = {
        "schema_version": "1.0",
        "measurement": "exact tokenizer encode of pinned source text",
        "tokenizer_library": "tiktoken",
        "tokenizer_library_version": version("tiktoken"),
        "encoding": encoding,
        "source_files": len(output),
        "source_bytes": corpus_summary["source_bytes"],
        "source_repositories": corpus_summary["repos"],
        "measured_source_tokens": total,
        "verified_rollout_tokens": 0,
        "upstream_test_assertions_unverified": corpus_summary["embedded_assertions_unverified"],
        "by_repo": dict(sorted(grouped.items())),
        "by_task_family_tokens": dict(sorted(family_tokens.items())),
        "warning": "These counts use the specified tokenizer, not the Bonsai/Qwen native tokenizer.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in output),
        encoding="utf-8",
    )
    report.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--encoding", default="o200k_base")
    parser.add_argument("--out", type=Path, default=ROOT / "token_counts/o200k_base.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/o200k_base.json")
    args = parser.parse_args()
    summary = count_tokens(args.root, args.encoding, args.out, args.report)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
