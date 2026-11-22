#!/usr/bin/env python3
"""Align recorded Hyperon evaluations with pinned upstream assertions.

Only native interpreter assertion-result observations are certified here.
These receipts do not establish independent semantic oracle correctness or
training-data admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from assertions import find_assertions, mask_comments_and_strings, split_top_level
from corpus import ROOT, safe_file, validate_corpus

ASSERT_OPS = frozenset({
    "assertEqual", "assertEqualMsg", "assertEqualToResult",
    "assertAlphaEqual", "assertAlphaEqualToResult", "test",
})
OP_RE = re.compile(r"^\(\s*([^\s()]+)")
COMPLETE_STDOUT_LIMIT = 16_384


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scan_eval_sites(source: str) -> list[dict[str, Any]]:
    """Locate only top-level ! (expression) calls, ignoring strings and comments."""
    mask = mask_comments_and_strings(source)
    sites: list[dict[str, Any]] = []
    depth = 0
    i = 0
    while i < len(mask):
        c = mask[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced top-level closing parenthesis at offset {i}")
        elif c == "!" and depth == 0:
            start = i
            open_at = i + 1
            while open_at < len(mask) and mask[open_at].isspace():
                open_at += 1
            if open_at >= len(mask) or mask[open_at] != "(":
                raise ValueError(f"unsupported top-level evaluation syntax at offset {start}")
            nest = 0
            end = None
            for k in range(open_at, len(mask)):
                if mask[k] == "(":
                    nest += 1
                elif mask[k] == ")":
                    nest -= 1
                    if nest == 0:
                        end = k + 1
                        break
            if end is None:
                raise ValueError(f"unbalanced evaluation expression at offset {start}")
            form = source[open_at:end]
            op_match = OP_RE.match(form)
            operator = op_match.group(1) if op_match else None
            args, parsed = split_top_level(form[1:-1])
            assertion = operator in ASSERT_OPS
            author_pair = parsed and assertion and len(args) >= 3 and args[0] == operator
            sites.append({
                "start_offset": start,
                "end_offset": end,
                "line_start": source.count("\n", 0, start) + 1,
                "line_end": source.count("\n", 0, end) + 1,
                "eval_source": source[start:end],
                "operator": operator,
                "is_assertion": assertion,
                "query_expression": args[1] if author_pair else None,
                "author_expected_expression": args[2] if author_pair else None,
                "parse_status": "parsed" if author_pair else "not_assertion" if not assertion else "unparsed",
            })
            i = end
            continue
        i += 1
    if depth != 0:
        raise ValueError("unbalanced source expression")
    return sites


def read_complete_output(rec: dict[str, Any], name: str) -> str | None:
    preview = rec.get(name + "_preview")
    byte_count = rec.get(name + "_bytes")
    digest = rec.get(name + "_sha256")
    if not isinstance(preview, str) or not isinstance(byte_count, int) or not isinstance(digest, str):
        return None
    raw = preview.encode("utf-8")
    if len(raw) != byte_count or sha256(raw) != digest or byte_count > COMPLETE_STDOUT_LIMIT:
        return None
    return preview


def verify_receipts(root: Path, receipts_path: Path, out: Path, report: Path) -> dict[str, Any]:
    sources, _, errors = validate_corpus(root)
    if errors:
        raise ValueError("Source validation failed: " + "; ".join(errors))
    source_map = {x["id"]: x for x in sources}
    receipts = [json.loads(s) for s in receipts_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    sites_total = 0
    for receipt in receipts:
        sid = receipt["source_id"]
        if sid in seen or sid not in source_map:
            raise ValueError("Unknown or duplicate execution source: " + sid)
        seen.add(sid)
        src = source_map[sid]
        if (
            receipt["source_git_blob_sha1"] != src["upstream_blob_sha1"]
            or receipt["source_commit"] != src["source_commit"]
            or receipt["runtime"].get("package") != "hyperon"
            or receipt["runtime"].get("version") != "0.2.10"
        ):
            raise ValueError("Source or runtime identity mismatch: " + sid)
        text = safe_file(root, src["raw_path"]).read_text(encoding="utf-8")
        sites = scan_eval_sites(text)
        author = find_assertions(text)
        author_offsets = {x["start_offset"]: x for x in author}
        if len(author_offsets) != src["embedded_test_assertion_count"]:
            raise ValueError("Author-assertion source count mismatch: " + sid)
        native_assertions = [x for x in sites if x["is_assertion"]]
        if len(native_assertions) < src["embedded_test_assertion_count"]:
            raise ValueError("Top-level assertion count below source annotation: " + sid)
        stdout = read_complete_output(receipt, "stdout")
        stderr = read_complete_output(receipt, "stderr")
        supported = (
            receipt["exit_code"] == 0
            and not receipt["timed_out"]
            and receipt["process_observation"] == "exit_zero"
            and stdout is not None
            and stderr is not None
        )
        output_lines = stdout.splitlines() if stdout is not None else []
        if not supported:
            alignment = "incomplete_process_receipt"
        elif len(output_lines) == len(sites):
            alignment = "exact_eval_output_count"
        elif (
            0 < len(output_lines) < len(sites)
            and output_lines[-1].lstrip().startswith("[(Error ")
        ):
            alignment = "terminated_on_error"
        else:
            alignment = "ambiguous_eval_output_count"
        sites_total += len(sites)
        for ix, site in enumerate(sites):
            if not site["is_assertion"]:
                continue
            line = output_lines[ix] if ix < len(output_lines) else None
            if alignment in {"exact_eval_output_count", "terminated_on_error"}:
                if line == "[()]":
                    result = "native_assertion_pass"
                elif line is None:
                    result = "not_observed"
                elif line.lstrip().startswith("[(Error "):
                    result = "native_assertion_error"
                else:
                    result = "other_native_result"
            else:
                result = "alignment_unverified"
            counts[result] += 1
            assertion_source = site["eval_source"]
            key = f"{sid}:{site['start_offset']}:{sha256(assertion_source.encode('utf-8'))}"
            row = {
                "schema_version": "metta-native-assertion-observation-v1",
                "id": sha256(key.encode("utf-8")),
                "source_id": sid,
                "source_repo": src["source_repo"],
                "source_commit": src["source_commit"],
                "source_blob_sha1": src["upstream_blob_sha1"],
                "source_license_spdx": src["license_spdx"],
                "task_family": src["task_family"],
                "runtime": receipt["runtime"],
                "evaluation_ordinal": ix,
                "assertion_ordinal": sum(s["is_assertion"] for s in sites[:ix]),
                "line_start": site["line_start"],
                "line_end": site["line_end"],
                "operator": site["operator"],
                "query_expression": site["query_expression"],
                "author_expected_expression": site["author_expected_expression"],
                "assertion_source": assertion_source,
                "native_output": line if result != "alignment_unverified" else None,
                "native_result": result,
                "stdout_sha256": receipt["stdout_sha256"],
                "stderr_sha256": receipt["stderr_sha256"],
                "process_observation": receipt["process_observation"],
                "alignment_status": alignment,
                "observation_scope": "upstream_native_assertion_execution",
                "independent_semantic_oracle": False,
                "is_generated_rollout": False,
                "split": "unassigned",
                "training_ready": False,
            }
            # The original candidate extractor has a smaller operator vocabulary.
            # Preserve its lexical-offset identity only for exact matches.
            original = author_offsets.get(site["start_offset"])
            if original is not None:
                row["author_assertion_extracted"] = original["assertion_kind"]
            rows.append(row)
    rows.sort(key=lambda x: (x["source_id"], x["evaluation_ordinal"]))
    summary = {
        "schema_version": "metta-native-assertion-observation-v1",
        "source_execution_receipts": len(receipts),
        "source_top_level_evaluations": sites_total,
        "native_assertions": len(rows),
        "outcome_counts": dict(sorted(counts.items())),
        "source_runtime": "hyperon==0.2.10",
        "training_ready": False,
        "verified_training_rollouts": 0,
        "independent_semantic_oracle_labels": 0,
        "evidence": "interpreter stdout aligned with source top-level evaluation order",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--receipts", type=Path, default=ROOT / "runtime_receipts/hyperon_0_2_10.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "native_assertion_verdicts/hyperon_0_2_10.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/hyperon_assertion_verdicts.json")
    args = parser.parse_args()
    summary = verify_receipts(args.root, args.receipts, args.out, args.report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
