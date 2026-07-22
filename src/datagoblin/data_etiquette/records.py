from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from datagoblin.domain.evidence import EvidenceType
from datagoblin.serialization import freeze_jsonable, to_jsonable


class CollectionDisposition(str, Enum):
    REJECT = "REJECT"
    QUARANTINE = "QUARANTINE"
    DISCOVERY_ONLY = "DISCOVERY_ONLY"
    VERIFIED_TRAIN = "VERIFIED_TRAIN"
    EVALUATION_ONLY = "EVALUATION_ONLY"
    SEALED_EVALUATION = "SEALED_EVALUATION"


@dataclass(frozen=True)
class DataCollectionRecord:
    item_id: str
    schema_version: str
    content_sha256: str
    source_kind: str
    purpose_id: str
    lawful_basis: str
    allowed_uses: tuple[str, ...]
    data_residency_region: str
    sensitivity_class: str
    license_status: str
    consent_status: str
    personal_data_status: str
    redacted: bool
    contains_secret: bool
    contains_hidden_reasoning: bool
    contains_model_weight: bool
    evaluation_material: bool
    sealed_evaluation: bool
    model_generated: bool
    generator_id: str | None
    evidence_type: EvidenceType
    asserted_by: str
    verified_by: str | None
    verification_receipt_sha256: str | None
    verification_subject_sha256: str | None
    verification_custody: str
    provenance_refs: tuple[str, ...]
    dedup_group: str
    intended_split: str
    retention_days: int
    deletion_requested: bool
    public_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported collection record schema version")
        if not _is_sha256(self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        for name, digest in (
            ("verification_receipt_sha256", self.verification_receipt_sha256),
            ("verification_subject_sha256", self.verification_subject_sha256),
        ):
            if digest is not None and not _is_sha256(digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not all(
            (
                self.item_id,
                self.purpose_id,
                self.data_residency_region,
                self.sensitivity_class,
                self.asserted_by,
                self.dedup_group,
            )
        ):
            raise ValueError("collection identity, purpose, residency, sensitivity, and assertion fields are required")
        forbidden_public_keys = {
            "content",
            "raw_content",
            "text",
            "filesystem_path",
            "path",
            "secret",
            "credential",
            "password",
            "token",
        }
        public_keys = _nested_keys(self.public_metadata)
        forbidden = sorted(public_keys & forbidden_public_keys)
        if forbidden:
            raise ValueError(f"private fields cannot enter public_metadata: {forbidden}")
        for reference in self.provenance_refs:
            if reference.startswith(("/", "\\")) or (
                len(reference) >= 3 and reference[0].isalpha() and reference[1:3] in {":/", ":\\"}
            ):
                raise ValueError("filesystem paths cannot enter public provenance_refs")
        object.__setattr__(self, "provenance_refs", tuple(self.provenance_refs))
        object.__setattr__(self, "allowed_uses", tuple(self.allowed_uses))
        object.__setattr__(self, "public_metadata", freeze_jsonable(self.public_metadata))

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, value: Mapping[str, Any]) -> DataCollectionRecord:
        return cls(
            item_id=str(value["item_id"]),
            schema_version=str(value["schema_version"]),
            content_sha256=str(value["content_sha256"]),
            source_kind=str(value["source_kind"]),
            purpose_id=str(value["purpose_id"]),
            lawful_basis=str(value["lawful_basis"]),
            allowed_uses=tuple(str(item) for item in value["allowed_uses"]),
            data_residency_region=str(value["data_residency_region"]),
            sensitivity_class=str(value["sensitivity_class"]),
            license_status=str(value["license_status"]),
            consent_status=str(value["consent_status"]),
            personal_data_status=str(value["personal_data_status"]),
            redacted=bool(value["redacted"]),
            contains_secret=bool(value["contains_secret"]),
            contains_hidden_reasoning=bool(value["contains_hidden_reasoning"]),
            contains_model_weight=bool(value["contains_model_weight"]),
            evaluation_material=bool(value["evaluation_material"]),
            sealed_evaluation=bool(value["sealed_evaluation"]),
            model_generated=bool(value["model_generated"]),
            generator_id=str(value["generator_id"]) if value.get("generator_id") else None,
            evidence_type=EvidenceType(value["evidence_type"]),
            asserted_by=str(value["asserted_by"]),
            verified_by=str(value["verified_by"]) if value.get("verified_by") else None,
            verification_receipt_sha256=(
                str(value["verification_receipt_sha256"])
                if value.get("verification_receipt_sha256")
                else None
            ),
            verification_subject_sha256=(
                str(value["verification_subject_sha256"])
                if value.get("verification_subject_sha256")
                else None
            ),
            verification_custody=str(value["verification_custody"]),
            provenance_refs=tuple(str(item) for item in value.get("provenance_refs", ())),
            dedup_group=str(value["dedup_group"]),
            intended_split=str(value["intended_split"]),
            retention_days=int(value["retention_days"]),
            deletion_requested=bool(value["deletion_requested"]),
            public_metadata=dict(value.get("public_metadata", {})),
        )


@dataclass(frozen=True)
class EtiquetteDecision:
    item_id: str
    record_hash: str
    disposition: CollectionDisposition
    violations: tuple[str, ...]
    required_actions: tuple[str, ...]
    evidence_authority: EvidenceType
    policy_version: str
    policy_hash: str
    decision_receipt_hash: str

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        keys = {str(key).lower() for key in value}
        for item in value.values():
            keys.update(_nested_keys(item))
        return keys
    if isinstance(value, (list, tuple)):
        keys: set[str] = set()
        for item in value:
            keys.update(_nested_keys(item))
        return keys
    return set()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
