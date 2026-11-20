"""Conservative proof statement family and evaluation leakage audit.

Input must be source-only Lean declaration JSONL exported by audit_owned_math.py.
Candidate clusters require human review and never automatically authorize training.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def rows_from_file(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_statements(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    by_lane = Counter()
    by_role = Counter()
    for row in rows:
        if row.get("kind") not in {"lemma", "theorem", "example"}:
            continue
        by_lane[row.get("source_lane", "unknown")] += 1
        by_role[row.get("split_role", "unassigned")] += 1
        digest = row.get("name_free_statement_sha256")
        if digest:
            groups[digest].append(row)
    clusters = []
    overlaps = []
    for digest, members in sorted(groups.items()):
        origins = sorted({(r.get("repository", ""), r.get("path", "")) for r in members})
        roles = sorted({r.get("split_role", "unassigned") for r in members})
        if len(members) > 1:
            cluster = {
                "name_free_statement_sha256": digest,
                "occurrences": len(members),
                "source_files": [{"repository": r, "path": p} for r, p in origins],
                "split_roles": roles,
                "review_required": True,
            }
            clusters.append(cluster)
            if "training_candidate" in roles and any(role in roles for role in ("protected_eval", "development")):
                overlaps.append(cluster)
    return {
        "status": "source_only_unverified",
        "description": "Exact name-free syntax matches; not semantic theorem equivalence or independent proof roots",
        "theorem_like_occurrences": sum(by_lane.values()),
        "name_free_unique_statement_syntaxes": len(groups),
        "repeated_syntax_clusters": len(clusters),
        "repeated_syntax_occurrences": sum(c["occurrences"] for c in clusters),
        "train_vs_eval_overlap_candidates": len(overlaps),
        "source_lanes": dict(sorted(by_lane.items())),
        "split_roles": dict(sorted(by_role.items())),
        "duplicate_clusters": clusters,
        "split_overlap_candidates": overlaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declarations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = group_statements(rows_from_file(args.declarations))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print({k: v for k, v in result.items() if k not in {"duplicate_clusters", "split_overlap_candidates"}})


if __name__ == "__main__":
    main()
