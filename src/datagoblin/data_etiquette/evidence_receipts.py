from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagoblin.domain.evidence import EvidenceType
from datagoblin.hashing import sha256_file, sha256_json
from datagoblin.serialization import load_structured, to_jsonable


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class EvidenceVerificationReceipt:
    schema_version: str
    subject_sha256: str
    evidence_type: EvidenceType
    verifier_id: str
    method_id: str
    custody: str
    asserted_by: str
    verdict: str
    registry_sha256: str
    challenge_sha256: str
    observation_sha256: str
    replay_spec_sha256: str
    claim_types: tuple[str, ...]
    artifact_sha256s: tuple[str, ...]
    receipt_sha256: str

    def payload(self) -> dict[str, Any]:
        value = to_jsonable(self)
        del value["receipt_sha256"]
        return value

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_jsonable(cls, value: Mapping[str, Any]) -> EvidenceVerificationReceipt:
        return cls(
            schema_version=str(value["schema_version"]),
            subject_sha256=str(value["subject_sha256"]),
            evidence_type=EvidenceType(value["evidence_type"]),
            verifier_id=str(value["verifier_id"]),
            method_id=str(value["method_id"]),
            custody=str(value["custody"]),
            asserted_by=str(value["asserted_by"]),
            verdict=str(value["verdict"]),
            registry_sha256=str(value["registry_sha256"]),
            challenge_sha256=str(value["challenge_sha256"]),
            observation_sha256=str(value["observation_sha256"]),
            replay_spec_sha256=str(value["replay_spec_sha256"]),
            claim_types=tuple(str(item) for item in value["claim_types"]),
            artifact_sha256s=tuple(str(item) for item in value["artifact_sha256s"]),
            receipt_sha256=str(value["receipt_sha256"]),
        )


class EvidenceVerifierRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.config = load_structured(path)
        if self.config.get("schema_version") != "1.0":
            raise ValueError("unsupported evidence verifier registry schema")
        self.sha256 = sha256_file(path)
        rows = self.config.get("verifiers")
        if not isinstance(rows, list) or not rows:
            raise ValueError("evidence verifier registry must contain verifiers")
        self.verifiers: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("invalid evidence verifier registry row")
            verifier_id = str(row.get("verifier_id", ""))
            if not verifier_id or verifier_id in self.verifiers:
                raise ValueError("evidence verifier IDs must be non-empty and unique")
            custody = str(row.get("custody", ""))
            if custody not in {"ENVIRONMENT_OWNED", "INDEPENDENT"}:
                raise ValueError(f"invalid verifier custody: {verifier_id}")
            evidence_types = tuple(EvidenceType(value) for value in row.get("evidence_types", ()))
            if not evidence_types or any(not value.may_hard_promote for value in evidence_types):
                raise ValueError(f"verifier may certify only hard evidence: {verifier_id}")
            methods = row.get("methods")
            if not isinstance(methods, list) or not methods:
                raise ValueError(f"verifier must declare replay methods: {verifier_id}")
            method_map: dict[str, dict[str, Any]] = {}
            for method in methods:
                if not isinstance(method, Mapping):
                    raise ValueError(f"invalid verifier replay method: {verifier_id}")
                method_id = str(method.get("method_id", ""))
                replay_spec_sha256 = str(method.get("replay_spec_sha256", ""))
                allowed_claim_types = tuple(str(item) for item in method.get("allowed_claim_types", ()))
                if not method_id or method_id in method_map or not _is_sha256(replay_spec_sha256):
                    raise ValueError(f"invalid verifier replay method commitment: {verifier_id}")
                if not allowed_claim_types or len(set(allowed_claim_types)) != len(allowed_claim_types):
                    raise ValueError(f"verifier replay method must declare unique claim types: {verifier_id}")
                method_map[method_id] = {
                    "replay_spec_sha256": replay_spec_sha256,
                    "allowed_claim_types": allowed_claim_types,
                }
            self.verifiers[verifier_id] = {
                "custody": custody,
                "evidence_types": evidence_types,
                "methods": method_map,
            }

    def validate_receipt(self, receipt: EvidenceVerificationReceipt) -> None:
        if receipt.schema_version != "evidence-verification-receipt.v2":
            raise ValueError("unsupported evidence receipt schema")
        digests = (
            receipt.subject_sha256,
            receipt.registry_sha256,
            receipt.challenge_sha256,
            receipt.observation_sha256,
            receipt.replay_spec_sha256,
            receipt.receipt_sha256,
            *receipt.artifact_sha256s,
        )
        if not all(_is_sha256(value) for value in digests):
            raise ValueError("evidence receipt contains an invalid SHA-256 commitment")
        if not receipt.artifact_sha256s or len(set(receipt.artifact_sha256s)) != len(receipt.artifact_sha256s):
            raise ValueError("evidence receipt requires unique replay artifact commitments")
        if receipt.registry_sha256 != self.sha256:
            raise ValueError("evidence receipt is not bound to the active verifier registry")
        verifier = self.verifiers.get(receipt.verifier_id)
        if verifier is None:
            raise ValueError("evidence receipt names an untrusted verifier")
        if receipt.custody != verifier["custody"]:
            raise ValueError("evidence receipt custody does not match the trusted verifier")
        if receipt.evidence_type not in verifier["evidence_types"]:
            raise ValueError("verifier is not trusted for this evidence type")
        method = verifier["methods"].get(receipt.method_id)
        if method is None or receipt.replay_spec_sha256 != method["replay_spec_sha256"]:
            raise ValueError("evidence receipt replay method is not trusted")
        if (
            not receipt.claim_types
            or len(set(receipt.claim_types)) != len(receipt.claim_types)
            or not set(receipt.claim_types) <= set(method["allowed_claim_types"])
        ):
            raise ValueError("evidence receipt claims exceed the trusted replay method")
        if not receipt.asserted_by or receipt.asserted_by == receipt.verifier_id:
            raise ValueError("proposer cannot independently verify its own evidence")
        if receipt.verdict not in {"PASS", "FAIL", "INCONCLUSIVE"}:
            raise ValueError("invalid evidence receipt verdict")
        if receipt.receipt_sha256 != sha256_json(receipt.payload()):
            raise ValueError("evidence receipt hash mismatch")


class EvidenceReceiptStore:
    def __init__(
        self,
        registry: EvidenceVerifierRegistry,
        receipts: Sequence[EvidenceVerificationReceipt],
    ) -> None:
        self.registry = registry
        self.receipts: dict[str, EvidenceVerificationReceipt] = {}
        for receipt in receipts:
            registry.validate_receipt(receipt)
            if receipt.receipt_sha256 in self.receipts:
                raise ValueError("duplicate evidence receipt hash")
            self.receipts[receipt.receipt_sha256] = receipt

    @classmethod
    def from_jsonl(cls, registry_path: Path, receipts_path: Path) -> EvidenceReceiptStore:
        registry = EvidenceVerifierRegistry(registry_path)
        receipts: list[EvidenceVerificationReceipt] = []
        with receipts_path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"expected evidence receipt object at line {line_number}")
                receipts.append(EvidenceVerificationReceipt.from_jsonable(value))
        return cls(registry, receipts)

    def verify_binding(
        self,
        *,
        receipt_sha256: str | None,
        subject_sha256: str,
        evidence_type: EvidenceType,
        verified_by: str | None,
        custody: str,
        asserted_by: str,
        required_claim_types: Sequence[str] = (),
    ) -> tuple[bool, str]:
        if not receipt_sha256:
            return False, "receipt_hash_missing"
        receipt = self.receipts.get(receipt_sha256)
        if receipt is None:
            return False, "receipt_not_in_trusted_store"
        checks = {
            "subject_mismatch": receipt.subject_sha256 == subject_sha256,
            "evidence_type_mismatch": receipt.evidence_type == evidence_type,
            "verifier_mismatch": receipt.verifier_id == verified_by,
            "custody_mismatch": receipt.custody == custody,
            "proposer_mismatch": receipt.asserted_by == asserted_by,
            "non_passing_verdict": receipt.verdict == "PASS",
            "required_claims_missing": set(required_claim_types) <= set(receipt.claim_types),
        }
        for failure, passed in checks.items():
            if not passed:
                return False, failure
        return True, "verified"

    def passing_receipts_for(
        self,
        *,
        subject_sha256: str,
        asserted_by: str,
        required_claim_types: Sequence[str],
    ) -> tuple[EvidenceVerificationReceipt, ...]:
        return tuple(
            receipt
            for receipt in self.receipts.values()
            if receipt.subject_sha256 == subject_sha256
            and receipt.asserted_by == asserted_by
            and receipt.verdict == "PASS"
            and set(required_claim_types) <= set(receipt.claim_types)
        )


def issue_evidence_receipt(
    registry: EvidenceVerifierRegistry,
    *,
    subject_sha256: str,
    evidence_type: EvidenceType,
    verifier_id: str,
    method_id: str,
    asserted_by: str,
    verdict: str,
    challenge_sha256: str,
    observation_sha256: str,
    claim_types: Sequence[str],
    artifact_sha256s: Sequence[str],
) -> EvidenceVerificationReceipt:
    verifier = registry.verifiers.get(verifier_id)
    if verifier is None:
        raise ValueError("cannot issue receipt for an untrusted verifier")
    method = verifier["methods"].get(method_id)
    if method is None:
        raise ValueError("cannot issue receipt for an untrusted replay method")
    payload = {
        "schema_version": "evidence-verification-receipt.v2",
        "subject_sha256": subject_sha256,
        "evidence_type": evidence_type,
        "verifier_id": verifier_id,
        "method_id": method_id,
        "custody": verifier["custody"],
        "asserted_by": asserted_by,
        "verdict": verdict,
        "registry_sha256": registry.sha256,
        "challenge_sha256": challenge_sha256,
        "observation_sha256": observation_sha256,
        "replay_spec_sha256": method["replay_spec_sha256"],
        "claim_types": tuple(claim_types),
        "artifact_sha256s": tuple(artifact_sha256s),
    }
    receipt = EvidenceVerificationReceipt.from_jsonable(
        {**to_jsonable(payload), "receipt_sha256": sha256_json(payload)}
    )
    registry.validate_receipt(receipt)
    return receipt
