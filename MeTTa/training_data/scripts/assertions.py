#!/usr/bin/env python3
"""Extract author-written MeTTa test assertions into unverified supervision candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from corpus import ROOT, safe_file, validate_corpus

PATTERN = re.compile(r"!\s*\(\s*(assertEqualToResult|assertEqualMsg|assertEqual|test)\b")


def mask_comments_and_strings(source: str) -> str:
    """Preserve source offsets while masking strings and semicolon comments."""
    masked: list[str] = []
    i = 0
    quoted = False
    commented = False
    while i < len(source):
        ch = source[i]
        if commented:
            masked.append("\n" if ch == "\n" else " ")
            commented = ch != "\n"
        elif quoted:
            masked.append("\n" if ch == "\n" else " ")
            if ch == "\\" and i + 1 < len(source):
                i += 1
                masked.append("\n" if source[i] == "\n" else " ")
            elif ch == '"':
                quoted = False
        elif ch == ";":
            commented = True
            masked.append(" ")
        elif ch == '"':
            quoted = True
            masked.append(" ")
        else:
            masked.append(ch)
        i += 1
    return "".join(masked)


def split_top_level(body: str) -> tuple[list[str], bool]:
    """Split an assertion's top-level arguments, respecting parentheses and strings."""
    tokens: list[str] = []
    current: list[str] = []
    depth = 0
    quoted = False
    commented = False
    escaped = False
    valid = True
    for ch in body:
        if commented:
            if ch == "\n":
                commented = False
                if current:
                    current.append(" ")
            continue
        if quoted:
            current.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
            continue
        if ch == ";":
            commented = True
            continue
        if ch == '"':
            quoted = True
            current.append(ch)
            continue
        if ch == "(":
            depth += 1
            current.append(ch)
            continue
        if ch == ")":
            depth -= 1
            valid = valid and depth >= 0
            current.append(ch)
            continue
        if ch.isspace() and depth == 0:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
                current.clear()
        else:
            current.append(ch)
    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return tokens, valid and depth == 0 and not quoted


def find_assertions(source: str) -> list[dict]:
    mask = mask_comments_and_strings(source)
    output: list[dict] = []
    for match in PATTERN.finditer(mask):
        opening = mask.find("(", match.start(), match.end())
        if opening < 0:
            continue
        depth = 0
        closing = -1
        for pos in range(opening, len(mask)):
            if mask[pos] == "(":
                depth += 1
            elif mask[pos] == ")":
                depth -= 1
                if depth == 0:
                    closing = pos + 1
                    break
        if closing < 0:
            output.append({
                "start_offset": match.start(),
                "end_offset": len(source),
                "assertion_kind": match.group(1),
                "parse_status": "unbalanced",
                "query_expression": None,
                "author_expected_expression": None,
                "assertion_source": source[match.start():],
            })
            continue
        args, valid = split_top_level(source[opening + 1:closing - 1])
        recognized = valid and len(args) >= 3 and args[0] == match.group(1)
        output.append({
            "start_offset": match.start(),
            "end_offset": closing,
            "assertion_kind": match.group(1),
            "parse_status": "parsed" if recognized else "unparsed",
            "query_expression": args[1] if recognized else None,
            "author_expected_expression": args[2] if recognized else None,
            "assertion_source": source[match.start():closing],
        })
    return output


def extract_corpus(root: Path, out: Path, report: Path) -> dict:
    rows, summary, errors = validate_corpus(root)
    if errors:
        raise ValueError("Corpus validation failed: " + "; ".join(errors))
    entries = []
    by_family: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    dedup: Counter[str] = Counter()
    parsed = 0
    for row in sorted(rows, key=lambda item: item["id"]):
        content = safe_file(root, row["raw_path"]).read_text(encoding="utf-8")
        source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        for item in find_assertions(content):
            line_start = content.count("\n", 0, item["start_offset"]) + 1
            line_end = content.count("\n", 0, item["end_offset"]) + 1
            assertion_hash = hashlib.sha256(item["assertion_source"].encode("utf-8")).hexdigest()
            assert_id = hashlib.sha256(
                f"{row['id']}:{item['start_offset']}:{assertion_hash}".encode()
            ).hexdigest()
            by_family[row["task_family"]] += 1
            by_kind[item["assertion_kind"]] += 1
            dedup[assertion_hash] += 1
            parsed += item["parse_status"] == "parsed"
            entries.append({
                "schema_version": "1.0",
                "id": assert_id,
                "source_id": row["id"],
                "source_repo": row["source_repo"],
                "source_commit": row["source_commit"],
                "source_path": row["source_path"],
                "source_blob_sha1": row["upstream_blob_sha1"],
                "source_content_sha256": source_hash,
                "source_license_spdx": row["license_spdx"],
                "task_family": row["task_family"],
                "source_kind": row["source_kind"],
                "runtime_hint": row["runtime_hint"],
                "line_start": line_start,
                "line_end": line_end,
                "assertion_kind": item["assertion_kind"],
                "assertion_content_sha256": assertion_hash,
                "assertion_source": item["assertion_source"],
                "query_expression": item["query_expression"],
                "author_expected_expression": item["author_expected_expression"],
                "parse_status": item["parse_status"],
                "label_provenance": "upstream_author_assertion_unexecuted",
                "verification_status": "not_run",
                "split": "unassigned",
                "rollout_steps": None,
                "training_status": "candidate_for_interpreter_verification",
            })
    details = {
        "schema_version": "1.0",
        "source_files": summary["source_files"],
        "candidate_assertions": len(entries),
        "parsed_query_expected_pairs": parsed,
        "unparsed_or_unbalanced_assertions": len(entries) - parsed,
        "distinct_assertion_texts": len(dedup),
        "exact_duplicate_assertion_occurrences": sum(count - 1 for count in dedup.values()),
        "verification_status": "not_run",
        "verified_rollouts": 0,
        "by_assertion_kind": dict(sorted(by_kind.items())),
        "by_task_family": dict(sorted(by_family.items())),
        "note": "Expected outputs are upstream author claims, not interpreter-verified labels.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in entries),
                   encoding="utf-8")
    report.write_text(json.dumps(details, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "assertion_candidates/author_assertions.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/assertions.json")
    args = parser.parse_args()
    print(json.dumps(extract_corpus(args.root, args.out, args.report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
