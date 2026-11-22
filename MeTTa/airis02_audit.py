"""Independent AIRIS-02 synthetic corpus audit before bounded research fitting.

The auditor refuses source-root mixing, hidden stewardship materialization,
changed labels, inconsistent four-arm allocations and unauthorized release.
Its only positive output is permission for a bounded *synthetic research
comparison*. It is not an EtiquettePolicy VERIFIED_TRAIN decision.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .airis02_fresh import (
    ARMS,
    FROZEN_MOSES_PROGRAM,
    SCHEMA,
    SOURCE_FAMILIES,
    _frozen_motifs,
    build_row,
    canonical,
    partition,
    root_id,
)
from .collect import sha256_bytes


def audit_bundle(path: Path, *, full_replay: bool = True) -> dict[str, Any]:
    manifest_file = path / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "metta-airis02-bundle-v1"
        or manifest.get("training_ready") is not False
        or manifest.get("teacher_labels_in_steward_registry") != 0
        or manifest.get("teacher_labels_in_public_test_registry") != 0
        or manifest.get("new_training_presentations_original_v1") != 0
        or manifest.get("old_sft_seed_overlap") != 0
        or manifest.get("root_exposure_policy") != "one_per_arm_max_four_per_root"
        or manifest.get("arms") != list(ARMS)
        or manifest.get("frozen_moses_key") != FROZEN_MOSES_PROGRAM.key
    ):
        raise ValueError("unsafe or incompatible AIRIS-02 research bundle")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        "train_candidate.jsonl", "development_labels_research.jsonl",
        "public_test_registry.json", "steward_registry.json",
    }:
        raise ValueError("incomplete research source custody")
    for name, expected in files.items():
        if sha256_bytes((path / name).read_bytes()) != expected:
            raise ValueError("research source artifact digest mismatch")
    for name in ("public_test_registry.json", "steward_registry.json"):
        registry = json.loads((path / name).read_text(encoding="utf-8"))
        if (
            registry.get("worlds_opened") != 0
            or registry.get("training_ready") is not False
            or any(k in registry for k in (
                "program", "selected", "oracle", "labels", "answer", "world",
            ))
        ):
            raise ValueError("reserved identity registry contains source contents or labels")
    motifs = _frozen_motifs() if full_replay else None
    if motifs is not None and sha256_bytes(canonical(motifs).encode()) != manifest["frozen_motifs_sha256"]:
        raise ValueError("frozen original-train motif source drift")
    by_split: dict[str, list[dict]] = {}
    by_arm: dict[str, dict[str, int]] = {}
    for filename, split in (
        ("train_candidate.jsonl", "train_candidate"),
        ("development_labels_research.jsonl", "development"),
    ):
        rows = [json.loads(line) for line in (path / filename).read_text(encoding="utf-8").splitlines() if line]
        by_split[split] = rows
        seen: set[tuple[str, str]] = set()
        counts: Counter[str] = Counter()
        by_arm[split] = dict(Counter(row["arm"] for row in rows))
        for row in rows:
            if (
                row.get("schema_version") != SCHEMA
                or row.get("split") != split
                or row.get("training_ready") is not False
                or row.get("native_hyperon_verified") is not False
                or row.get("max_total_presentations_per_root") != len(ARMS)
                or row.get("model_label_authority") != "independent_bounded_synthetic_python_oracle"
                or row.get("source_kind") != "new_synthetic_curation_world"
            ):
                raise ValueError("source-row or teacher custody drift")
            arm = row.get("arm")
            family = row.get("family")
            seed = row.get("source_seed")
            if arm not in ARMS or family not in SOURCE_FAMILIES or type(seed) is not int:
                raise ValueError("unknown intervention arm or forbidden family")
            if row["root"] != root_id(family, seed) or partition(family, seed) != split:
                raise ValueError("source-root split or identity mismatch")
            if row["presentation_number"] != ARMS.index(arm):
                raise ValueError("presentation accounting mismatch")
            key = (row["root"], arm)
            if key in seen:
                raise ValueError("duplicate source-root arm presentation")
            seen.add(key)
            counts[row["root"]] += 1
            if full_replay and row != build_row(family, seed, arm, motifs):
                raise ValueError("independent synthetic-oracle replay mismatch")
        if not rows or any(n != len(ARMS) for n in counts.values()):
            raise ValueError("incomplete root-level four-arm presentation set")
        if any(by_arm[split].get(arm, 0) != len(counts) for arm in ARMS):
            raise ValueError("unequal source-root counts across arms")
    train_roots = {row["root"] for row in by_split["train_candidate"]}
    dev_roots = {row["root"] for row in by_split["development"]}
    if train_roots & dev_roots or not train_roots or not dev_roots:
        raise ValueError("cross-split root contamination")
    if len(train_roots) != manifest.get("train_roots") or len(dev_roots) != manifest.get("development_roots"):
        raise ValueError("root-count manifest drift")
    if (
        len(by_split["train_candidate"]) != manifest.get("train_presentations_total")
        or len(by_split["development"]) != manifest.get("development_diagnostic_presentations")
    ):
        raise ValueError("presentation-count manifest drift")
    return {
        "schema_version": "metta-airis02-research-fit-audit-v1",
        "synthetic_research_fit_eligible": True,
        "general_corpus_training_ready": False,
        "oracle_source_regenerated": full_replay,
        "train_source_roots": len(train_roots),
        "development_source_roots": len(dev_roots),
        "train_rows_per_arm": by_arm["train_candidate"],
        "development_rows_per_arm": by_arm["development"],
        "older_sft_root_overlap": 0,
        "public_test_labels_used": 0,
        "steward_labels_used": 0,
        "maximum_root_presentations": len(ARMS),
        "input_manifest_sha256": sha256_bytes(manifest_file.read_bytes()),
        "verifier_type": "independent_exact_synthetic_python_oracle",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = audit_bundle(args.bundle, full_replay=not args.fast)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
