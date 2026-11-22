"""PLN-inspired evidence-lineage shadow ledger for synthetic curation research.

This is independent original Python research code. Official MeTTa PLN uses
EvidenceIDs and two-dimensional truth values; the computation here is only a
bounded demonstration of lineage-aware deduplication. Its strength/confidence
are uncalibrated descriptive scores, not native PLN formulas or proof.

All inputs are untrusted claims. The ledger provides zero authorization to
release a source, update ResearchMembrane, exclude an MCP subtree or alter a
verifier's decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .collect import ROOT, sha256_bytes

SCHEMA = "metta-pln-lineage-shadow-v1"
MAX_CLAIMS = 4096
MAX_LINEAGES = 512
DEFAULT_OUT = ROOT / "runs" / "metta" / "pln_lineage_shadow.json"


@dataclass(frozen=True)
class EvidenceClaim:
    evidence_id: str
    lineage_id: str
    revision: int
    supports: bool

    def __post_init__(self) -> None:
        for value in (self.evidence_id, self.lineage_id):
            if not isinstance(value, str) or not value or len(value) > 160:
                raise ValueError("invalid evidence identity")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be a nonnegative integer")
        if type(self.supports) is not bool:
            raise ValueError("supports must be a boolean claim")


def _stamp(groups: list[tuple[str, int, bool | None]]) -> str:
    """Hash only the selected lineage state, without emitting identifiers."""
    return hashlib.sha256(json.dumps(
        groups, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def summarize_lineage(
    claims: Iterable[EvidenceClaim], *, prior_equivalent_count: float = 2.0,
    max_claims: int = MAX_CLAIMS, max_lineages: int = MAX_LINEAGES,
) -> dict[str, Any]:
    """Summarize by declared independent lineage at its newest observed revision.

    One lineage contributes at most one independent vote. The highest revision
    supersedes older observations *for this shadow computation*, with same-
    revision disagreement excluded pending independent review. Revisions are
    labels from the input, not evidence that remote data is immutable.
    """
    if not 0 < prior_equivalent_count <= 1_000:
        raise ValueError("bad confidence prior equivalent")
    if not 1 <= max_claims <= MAX_CLAIMS or not 1 <= max_lineages <= MAX_LINEAGES:
        raise ValueError("invalid computation resource bounds")
    seen: dict[str, EvidenceClaim] = {}
    repeated_ids = 0
    for observation_index, claim in enumerate(claims):
        if observation_index >= max_claims * 4:
            raise ValueError("evidence observation horizon exceeded")
        if not isinstance(claim, EvidenceClaim):
            raise ValueError("expected typed evidence claims")
        if len(seen) >= max_claims and claim.evidence_id not in seen:
            raise ValueError("evidence claim horizon exceeded")
        prev = seen.get(claim.evidence_id)
        if prev is not None:
            if prev != claim:
                raise ValueError("one evidence ID has conflicting contents")
            repeated_ids += 1
            continue
        seen[claim.evidence_id] = claim
    lineages: dict[str, list[EvidenceClaim]] = {}
    for claim in seen.values():
        lineages.setdefault(claim.lineage_id, []).append(claim)
        if len(lineages) > max_lineages:
            raise ValueError("independent lineage horizon exceeded")
    independent_support = 0
    independent_refute = 0
    conflicted_lineages = 0
    newer_revisions_superseded = 0
    shared_lineage_repeats = 0
    selected_groups = []
    for lineage_id in sorted(lineages):
        group = lineages[lineage_id]
        latest = max(x.revision for x in group)
        live = [x for x in group if x.revision == latest]
        newer_revisions_superseded += len(group) - len(live)
        # Several claims within one same-version lineage count once.
        shared_lineage_repeats += max(0, len(live) - 1)
        values = {x.supports for x in live}
        if len(values) > 1:
            conflicted_lineages += 1
            selected_groups.append((lineage_id, latest, None))
        else:
            selected = values.pop()
            selected_groups.append((lineage_id, latest, selected))
            independent_support += int(selected)
            independent_refute += int(not selected)
    n = independent_support + independent_refute
    strength = round(independent_support / n, 6) if n else None
    nominal_confidence = round(n / (n + prior_equivalent_count), 6) if n else 0.0
    return {
        "schema_version": SCHEMA,
        "method": "independent_lineage_dedup_latest_revision_shadow",
        "evidence_type": "UNTRUSTED_CLAIMS",
        "native_pln_executed": False,
        "pln_truth_formula_implemented": False,
        "confidence_calibrated": False,
        "authorization": "none",
        "training_ready": False,
        "claim_ids_unique": len(seen),
        "replayed_identical_evidence_ids": repeated_ids,
        "declared_lineages": len(lineages),
        "independent_support": independent_support,
        "independent_refute": independent_refute,
        "conflicted_lineages_excluded": conflicted_lineages,
        "old_revisions_superseded": newer_revisions_superseded,
        "shared_lineage_repeats_ignored": shared_lineage_repeats,
        "strength": strength,
        "confidence": nominal_confidence,
        "confidence_prior_equivalent": prior_equivalent_count,
        "stamp_sha256": _stamp(selected_groups),
        "needs_review": bool(conflicted_lineages) or n == 0,
    }


def synthetic_demo() -> dict[str, Any]:
    """Deterministic fictitious evidence about a toy source record."""
    raw = [
        EvidenceClaim("toy:replicaA", "toy:source-A", 1, True),
        EvidenceClaim("toy:replicaB", "toy:source-A", 1, True),
        EvidenceClaim("toy:independent", "toy:source-B", 1, False),
        EvidenceClaim("toy:replicaA", "toy:source-A", 1, True),
    ]
    report = summarize_lineage(raw)
    report["demo_only"] = True
    report["source_sha256"] = sha256_bytes(Path(__file__).read_bytes())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to((ROOT / "MeTTa").resolve()):
        parser.error("shadow receipts must stay outside committed source")
    report = synthetic_demo()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
