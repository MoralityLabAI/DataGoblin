from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from datagoblin.domain.evidence import EvidenceType
from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import load_structured

from .evidence_receipts import EvidenceReceiptStore
from .records import CollectionDisposition, DataCollectionRecord, EtiquetteDecision


class EtiquettePolicy:
    def __init__(self, path: Path, *, evidence_store: EvidenceReceiptStore | None = None) -> None:
        self.path = path
        self.config = load_structured(path)
        if self.config.get("schema_version") != "1.0":
            raise ValueError("unsupported etiquette policy schema version")
        self.version = str(self.config["policy_version"])
        self.hash = sha256_file(path)
        self.evidence_store = evidence_store

    def decide(self, record: DataCollectionRecord) -> EtiquetteDecision:
        if record.schema_version != "1.0":
            raise ValueError("unsupported collection record schema version")
        violations: list[str] = []
        actions: list[str] = []
        disposition = self._disposition(record, violations, actions)
        return self._make_decision(record, disposition, violations, actions)

    def decide_batch(
        self,
        records: Sequence[DataCollectionRecord],
        *,
        dataset_partitions: Sequence[str] | None = None,
    ) -> tuple[EtiquetteDecision, ...]:
        if dataset_partitions is not None and len(dataset_partitions) != len(records):
            raise ValueError("dataset partition count must match collection record count")
        decisions = list(self.decide(record) for record in records)
        evaluation_groups = {
            record.dedup_group
            for record in records
            if record.sealed_evaluation
            or record.evaluation_material
            or record.intended_split in {"ordinary_test", "ood_test", "evaluation"}
        }
        if dataset_partitions is not None:
            evaluation_groups.update(
                record.dedup_group
                for record, partition in zip(records, dataset_partitions, strict=True)
                if partition in {"ordinary_test", "ood_test", "sealed_poison"}
            )
        for index, (record, decision) in enumerate(zip(records, decisions, strict=True)):
            if (
                record.dedup_group in evaluation_groups
                and decision.disposition in {CollectionDisposition.VERIFIED_TRAIN, CollectionDisposition.DISCOVERY_ONLY}
            ):
                decisions[index] = self._make_decision(
                    record,
                    CollectionDisposition.QUARANTINE,
                    [*decision.violations, "cross_split_duplicate"],
                    [*decision.required_actions, "rebuild_deduplicated_split_assignment"],
                )
        return tuple(decisions)

    def _make_decision(
        self,
        record: DataCollectionRecord,
        disposition: CollectionDisposition,
        violations: list[str],
        actions: list[str],
    ) -> EtiquetteDecision:
        record_hash = sha256_json(record)
        payload = {
            "item_id": record.item_id,
            "record_hash": record_hash,
            "disposition": disposition,
            "violations": tuple(violations),
            "required_actions": tuple(actions),
            "evidence_authority": record.evidence_type,
            "policy_version": self.version,
            "policy_hash": self.hash,
        }
        return EtiquetteDecision(
            record.item_id,
            record_hash,
            disposition,
            tuple(violations),
            tuple(actions),
            record.evidence_type,
            self.version,
            self.hash,
            sha256_json(payload),
        )

    def _disposition(
        self,
        record: DataCollectionRecord,
        violations: list[str],
        actions: list[str],
    ) -> CollectionDisposition:
        if record.deletion_requested:
            violations.append("deletion_requested")
            actions.append("delete_private_content_and_retain_tombstone")
            return CollectionDisposition.REJECT
        if record.contains_secret:
            violations.append("secret_or_credential")
            actions.append("purge_content_and_rotate_exposed_secret")
            return CollectionDisposition.REJECT
        if record.contains_hidden_reasoning:
            violations.append("hidden_reasoning")
            actions.append("exclude_hidden_reasoning_from_collection")
            return CollectionDisposition.REJECT
        if record.contains_model_weight:
            violations.append("model_weight_or_checkpoint")
            actions.append("remove_binary_weight_material")
            return CollectionDisposition.REJECT
        if record.consent_status == "DENIED":
            violations.append("consent_denied")
            actions.append("delete_or_return_material")
            return CollectionDisposition.REJECT
        if record.license_status == "RESTRICTED":
            violations.append("license_restricted")
            actions.append("exclude_from_training_and_review_permitted_use")
            return CollectionDisposition.REJECT
        if not record.purpose_id or record.lawful_basis == "UNKNOWN" or not record.allowed_uses:
            violations.append("missing_purpose_or_lawful_basis")
            actions.append("record_specific_purpose_lawful_basis_and_allowed_uses")
            return CollectionDisposition.QUARANTINE
        intended_use = "EVALUATION" if record.evaluation_material or record.sealed_evaluation else "TRAINING"
        if intended_use not in record.allowed_uses:
            violations.append("intended_use_not_permitted")
            actions.append("quarantine_and_obtain_a_purpose_bound_use_receipt")
            return CollectionDisposition.QUARANTINE
        if record.sealed_evaluation:
            return CollectionDisposition.SEALED_EVALUATION
        if record.evaluation_material or record.intended_split in {"ordinary_test", "ood_test", "evaluation"}:
            return CollectionDisposition.EVALUATION_ONLY
        if record.license_status == "UNKNOWN":
            violations.append("license_unknown")
            actions.append("obtain_license_or_terms_receipt")
        if record.consent_status == "UNKNOWN":
            violations.append("consent_unknown")
            actions.append("obtain_consent_or_document_not_required")
        if record.personal_data_status != "NONE" and not record.redacted:
            violations.append("unredacted_personal_data")
            actions.append("minimize_and_redact_personal_data")
        if not record.provenance_refs or not record.content_sha256:
            violations.append("missing_provenance")
            actions.append("attach_source_and_content_hash_receipts")
        if record.retention_days <= 0:
            violations.append("invalid_retention")
            actions.append("set_a_finite_retention_period")
        if violations:
            return CollectionDisposition.QUARANTINE
        if record.evidence_type.may_hard_promote:
            if (
                not record.verified_by
                or not record.verification_receipt_sha256
                or record.verification_subject_sha256 != record.content_sha256
                or record.verification_custody not in {"ENVIRONMENT_OWNED", "INDEPENDENT"}
                or self.evidence_store is None
            ):
                violations.append("missing_independent_verification_receipt")
                actions.append("attach_content_bound_environment_or_independent_verification_receipt")
                return CollectionDisposition.QUARANTINE
            verified, reason = self.evidence_store.verify_binding(
                receipt_sha256=record.verification_receipt_sha256,
                subject_sha256=record.content_sha256,
                evidence_type=record.evidence_type,
                verified_by=record.verified_by,
                custody=record.verification_custody,
                asserted_by=record.asserted_by,
                required_claim_types=("CONTENT_INTEGRITY", "POLICY_ELIGIBILITY"),
            )
            if not verified:
                violations.append("invalid_independent_verification_receipt")
                actions.append(f"repair_or_replace_verification_receipt:{reason}")
                return CollectionDisposition.QUARANTINE
        soft_evidence = {
            EvidenceType.HUMAN_REVIEWED,
            EvidenceType.MODEL_INFERRED,
            EvidenceType.EXPERIENCE_INFERRED,
            EvidenceType.SIMULATED,
            EvidenceType.UNKNOWN,
        }
        if record.model_generated or record.evidence_type in soft_evidence:
            if record.model_generated and not record.generator_id:
                violations.append("missing_generator_identity")
                actions.append("attach_generator_and_prompt_lineage")
                return CollectionDisposition.QUARANTINE
            return CollectionDisposition.DISCOVERY_ONLY
        if record.evidence_type.may_hard_promote:
            return CollectionDisposition.VERIFIED_TRAIN
        violations.append("non_authoritative_evidence")
        actions.append("obtain_environment_or_independent_verification")
        return CollectionDisposition.QUARANTINE
