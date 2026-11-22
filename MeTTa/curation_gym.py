"""Build MeTTa curation decision fixtures using DataGoblin's existing policy.

Programs are advisory MeTTa fact/query proposals. Only EtiquettePolicy
supplies disposition targets, and its simulated labels remain discovery-only.
Evaluation partitions receive unlabeled packets; no sealed labels are exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datagoblin.data_etiquette.policy import EtiquettePolicy
from datagoblin.data_etiquette.records import DataCollectionRecord
from datagoblin.domain.evidence import EvidenceType

from MeTTa.collect import ROOT, proposed_split, sha256_bytes

VARIANTS = (
    "clean_metadata",
    "license_unknown",
    "license_restricted",
    "secret_exposed",
    "hidden_reasoning",
    "model_weight",
    "consent_denied",
    "deletion_requested",
    "unredacted_personal_data",
    "no_provenance",
)
CHANGE_FLAGS = {
    "license_unknown": {"license_status": "UNKNOWN"},
    "license_restricted": {"license_status": "RESTRICTED"},
    "secret_exposed": {"contains_secret": True},
    "hidden_reasoning": {"contains_hidden_reasoning": True},
    "model_weight": {"contains_model_weight": True},
    "consent_denied": {"consent_status": "DENIED"},
    "deletion_requested": {"deletion_requested": True},
    "unredacted_personal_data": {"personal_data_status": "PRESENT", "redacted": False},
    "no_provenance": {"provenance_refs": ()},
}
ALLOWED_FACT_FIELDS = {
    "License": "license_status",
    "Consent": "consent_status",
    "Secret": "contains_secret",
    "HiddenReasoning": "contains_hidden_reasoning",
    "ModelWeight": "contains_model_weight",
    "DeletionRequested": "deletion_requested",
    "PersonalData": "personal_data_status",
    "Redacted": "redacted",
}
DEFAULT_POLICY = ROOT / "configs" / "data_etiquette.yaml"


def atom_safe(raw: object) -> str:
    return "".join(c if c.isascii() and (c.isalnum() or c in "_-") else "_" for c in str(raw))


def policy_record(file_row: dict[str, Any], variant: str, *, source_commit: str) -> DataCollectionRecord:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    root = f'{file_row["source_id"]}:{file_row["path"]}'
    split = proposed_split(file_row)
    is_eval = split != "train_candidate"
    synthetic_payload = json.dumps(
        {"root": root, "variant": variant, "source_commit": source_commit},
        sort_keys=True, separators=(",", ":"),
    )
    content_sha = sha256_bytes(synthetic_payload.encode())
    args: dict[str, Any] = {
        "item_id": sha256_bytes(f"{root}:{variant}".encode())[:32],
        "schema_version": "1.0",
        "content_sha256": content_sha,
        "source_kind": "PUBLIC_CODE_METADATA",
        "purpose_id": "metta_curator_research_v0",
        "lawful_basis": "PUBLIC_PERMISSIVE_SOURCE_METADATA",
        "allowed_uses": ("TRAINING", "EVALUATION"),
        "data_residency_region": "GLOBAL_PUBLIC",
        "sensitivity_class": "PUBLIC_METADATA",
        "license_status": "PERMISSIVE",
        "consent_status": "NOT_REQUIRED",
        "personal_data_status": "NONE",
        "redacted": True,
        "contains_secret": False,
        "contains_hidden_reasoning": False,
        "contains_model_weight": False,
        "evaluation_material": is_eval,
        "sealed_evaluation": False,
        "model_generated": True,
        "generator_id": "MeTTa.curation_gym.v1",
        "evidence_type": EvidenceType.SIMULATED,
        "asserted_by": "MeTTa.curation_gym",
        "verified_by": None,
        "verification_receipt_sha256": None,
        "verification_subject_sha256": None,
        "verification_custody": "NONE",
        "provenance_refs": (f'https://github.com/{file_row["source_id"]}',),
        "dedup_group": root,
        "intended_split": "ordinary_test" if split == "dev_reserved" else ("ood_test" if is_eval else "train"),
        "retention_days": 365,
        "deletion_requested": False,
        "public_metadata": {
            "language": "MeTTa",
            "source_id": file_row["source_id"],
            "git_blob_sha1": file_row["git_blob_sha"],
            "reported_bytes": file_row["bytes"],
            "source_category": file_row["category"],
            "review_flags": file_row["review_flags"],
        },
    }
    args.update(CHANGE_FLAGS.get(variant, {}))
    if variant == "no_provenance":
        args["provenance_refs"] = ()
    if file_row["eligibility"] == "quarantine":
        args["license_status"] = "UNKNOWN"  # source-level review remains unresolved
    return DataCollectionRecord(**args)


def program_for(record: DataCollectionRecord) -> str:
    """Render a simple native MeTTa fact store and a non-authoritative match."""
    key = "item_" + atom_safe(record.item_id)
    facts = [f"({name} {key} {atom_safe(getattr(record, field))})"
             for name, field in ALLOWED_FACT_FIELDS.items()]
    facts.extend([
        f"(HasProvenance {key} {atom_safe(bool(record.provenance_refs))})",
        f"(IntendedSplit {key} {atom_safe(record.intended_split)})",
        f"(Evidence {key} SIMULATED)",
        f"(SourceId {key} {atom_safe(record.public_metadata['source_id'])})",
    ])
    # Multi-atom pattern and query follow the reference interpreter's match
    # grammar. This program still needs native execution before runtime claims.
    query = f"!(match &self (, (License {key} $license) (Evidence {key} $evidence)) ($license $evidence))"
    return "\n".join(facts + [query]) + "\n"


def build_cases(
    catalog_path: Path, policy_path: Path, output_dir: Path,
    *, max_files: int | None = None,
) -> dict[str, Any]:
    if output_dir.resolve().is_relative_to((ROOT / "MeTTa").resolve()):
        raise ValueError("generation output must be outside MeTTa source directory")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != "metta-source-catalog-v1":
        raise ValueError("unsupported source catalog")
    policy = EtiquettePolicy(policy_path)
    sources = {s["source_id"]: s for s in catalog["sources"]}
    rows: list[dict[str, Any]] = []
    for file_row in catalog["files"][:max_files]:
        source_id = file_row["source_id"]
        for variant in VARIANTS:
            record = policy_record(file_row, variant, source_commit=sources[source_id]["commit"])
            verdict = policy.decide(record)
            candidate = {
                "schema_version": "metta-curation-task-v1",
                "root_id": record.dedup_group,
                "case_id": record.item_id,
                "variant": variant,
                "partition": proposed_split(file_row),
                "provenance": {
                    "source_id": source_id, "source_commit": sources[source_id]["commit"],
                    "git_blob_sha1": file_row["git_blob_sha"],
                    "catalog_file": file_row["path"],
                },
                "model_input": {
                    "instruction": "Inspect the MeTTa fact store and propose curation; defer authority to DataGoblin policy.",
                    "program": program_for(record),
                    "public_envelope": record.to_jsonable(),
                },
                "teacher_authority": "DETERMINISTIC_ETIQUETTE_POLICY",
                "teacher_policy_sha256": policy.hash,
                "is_native_metta_verified": False,
                "training_ready": False,
            }
            if candidate["partition"] == "train_candidate":
                candidate["policy_reference"] = {
                    "disposition": verdict.disposition.value,
                    "violations": list(verdict.violations),
                    "required_actions": list(verdict.required_actions),
                    "decision_receipt_hash": verdict.decision_receipt_hash,
                    "candidate_release": "DISCOVERY_ONLY",
                }
            rows.append(candidate)
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "cases.jsonl"
    with dst.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    stats = Counter(x["partition"] for x in rows)
    manifest = {
        "schema_version": "metta-curation-gym-receipt-v1",
        "source_catalog_sha256": sha256_bytes(catalog_path.read_bytes()),
        "policy_sha256": policy.hash,
        "cases_sha256": sha256_bytes(dst.read_bytes()),
        "cases": len(rows),
        "partition_counts": dict(sorted(stats.items())),
        "train_candidate_labels": sum("policy_reference" in x for x in rows),
        "evaluation_labels_written": 0,
        "native_execution_verified": False,
        "training_ready": False,
        "remarks": "Synthetic metadata fixtures. EtiquettePolicy is the label authority; MeTTa code supplies advisory structure.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=ROOT / "MeTTa" / "source_catalog.json")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "metta" / "curation")
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    result = build_cases(args.catalog, args.policy, args.out, max_files=args.max_files)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
