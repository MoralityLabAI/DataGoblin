from __future__ import annotations

from typing import Any

from datagoblin.hashing import sha256_json

from .records import CollectionDisposition, DataCollectionRecord


class DeterministicMockLDTSorter:
    """Metadata-only plumbing control. This is not a trained LDT."""

    sorter_id = "deterministic-mock-ldt-etiquette-v1"

    def sort(self, record: DataCollectionRecord) -> dict[str, Any]:
        flags: list[str] = []
        if record.contains_secret:
            flags.append("secret_or_credential")
        if record.contains_hidden_reasoning:
            flags.append("hidden_reasoning")
        if record.contains_model_weight:
            flags.append("model_weight_or_checkpoint")
        if record.deletion_requested:
            flags.append("deletion_requested")
        if record.consent_status == "DENIED":
            flags.append("consent_denied")
        if record.license_status == "RESTRICTED":
            flags.append("license_restricted")
        if flags:
            disposition = CollectionDisposition.REJECT
            confidence = 0.94
        elif record.sealed_evaluation:
            disposition = CollectionDisposition.SEALED_EVALUATION
            confidence = 0.98
        elif record.evaluation_material or record.intended_split in {"ordinary_test", "ood_test", "evaluation"}:
            disposition = CollectionDisposition.EVALUATION_ONLY
            confidence = 0.92
        elif record.license_status == "UNKNOWN" or record.consent_status == "UNKNOWN":
            disposition = CollectionDisposition.QUARANTINE
            confidence = 0.85
        elif not record.purpose_id or record.lawful_basis == "UNKNOWN" or not record.allowed_uses:
            disposition = CollectionDisposition.QUARANTINE
            flags.append("missing_purpose_or_lawful_basis")
            confidence = 0.86
        elif record.personal_data_status != "NONE" and not record.redacted:
            disposition = CollectionDisposition.QUARANTINE
            flags.append("unredacted_personal_data")
            confidence = 0.88
        elif not record.provenance_refs:
            disposition = CollectionDisposition.QUARANTINE
            flags.append("missing_provenance")
            confidence = 0.82
        elif record.model_generated:
            disposition = CollectionDisposition.DISCOVERY_ONLY
            confidence = 0.78
        else:
            disposition = CollectionDisposition.VERIFIED_TRAIN
            confidence = 0.68
        return {
            "item_id": record.item_id,
            "sorter_id": self.sorter_id,
            "recommended_disposition": disposition.value,
            "flags": sorted(flags),
            "confidence": confidence,
            "abstained": False,
            "authority": "MODEL_RECOMMENDATION_ONLY",
            "record_hash": sha256_json(record),
        }
