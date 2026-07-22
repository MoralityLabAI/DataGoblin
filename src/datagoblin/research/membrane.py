from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from datagoblin.domain.evidence import EvidenceClaim, EvidenceType
from datagoblin.domain.experiments import LifecycleStatus
from datagoblin.hashing import sha256_json
from datagoblin.serialization import freeze_jsonable, to_jsonable

from .state import ResearchState, TransitionAuthority, transition_status

_DEFAULT_VERIFICATION_AUTHORITIES = {
    "causal_switches_independent_verifier": frozenset({EvidenceType.INDEPENDENT_HARNESS}),
}


class ProposalAction(str, Enum):
    PROPOSE_HYPOTHESIS = "PROPOSE_HYPOTHESIS"
    NARROW_HYPOTHESES = "NARROW_HYPOTHESES"
    REQUEST_EXPERIMENT = "REQUEST_EXPERIMENT"
    RECOMMEND_ABSTENTION = "RECOMMEND_ABSTENTION"
    REQUEST_ABSTRACTION_EXPANSION = "REQUEST_ABSTRACTION_EXPANSION"


@dataclass(frozen=True)
class ResearchProposal:
    action: ProposalAction
    payload: Mapping[str, Any]
    claimed_evidence: EvidenceClaim
    requested_status: LifecycleStatus | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze_jsonable(self.payload))

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class EvidenceVerification:
    state_hash: str
    proposal_hash: str
    evidence_type: EvidenceType
    verifier_id: str
    artifact_refs: tuple[str, ...]
    receipt_hash: str

    def __post_init__(self) -> None:
        if not self.evidence_type.may_hard_promote:
            raise ValueError("verification evidence must be environment-owned or independently verified")
        if not self.verifier_id or not self.artifact_refs:
            raise ValueError("verification requires a verifier identity and artifact references")
        if self.receipt_hash != sha256_json(self._receipt_payload()):
            raise ValueError("verification receipt hash does not match its contents")

    @classmethod
    def issue(
        cls,
        state: ResearchState,
        proposal: ResearchProposal,
        *,
        evidence_type: EvidenceType,
        verifier_id: str,
        artifact_refs: tuple[str, ...],
    ) -> EvidenceVerification:
        state_hash = sha256_json(state)
        proposal_hash = sha256_json(proposal)
        values: dict[str, Any] = {
            "state_hash": state_hash,
            "proposal_hash": proposal_hash,
            "evidence_type": evidence_type,
            "verifier_id": verifier_id,
            "artifact_refs": artifact_refs,
        }
        return cls(
            state_hash,
            proposal_hash,
            evidence_type,
            verifier_id,
            artifact_refs,
            sha256_json(values),
        )

    def matches(self, state: ResearchState, proposal: ResearchProposal) -> bool:
        return self.state_hash == sha256_json(state) and self.proposal_hash == sha256_json(proposal)

    def _receipt_payload(self) -> dict[str, Any]:
        return {
            "state_hash": self.state_hash,
            "proposal_hash": self.proposal_hash,
            "evidence_type": self.evidence_type,
            "verifier_id": self.verifier_id,
            "artifact_refs": self.artifact_refs,
        }

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class MembraneDecision:
    accepted: bool
    before: ResearchState
    after: ResearchState
    reason: str
    claimed_evidence: EvidenceType
    verified_evidence: EvidenceType | None
    disagreement: bool | None
    soft_stored: bool
    verification_receipt_hash: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


class ResearchMembrane:
    def __init__(
        self,
        verification_authorities: Mapping[str, frozenset[EvidenceType]] | None = None,
    ) -> None:
        self.verification_authorities = dict(verification_authorities or _DEFAULT_VERIFICATION_AUTHORITIES)

    def apply(
        self,
        state: ResearchState,
        proposal: ResearchProposal,
        *,
        verification: EvidenceVerification | None = None,
        authority: TransitionAuthority | str = TransitionAuthority.MODEL,
    ) -> MembraneDecision:
        verified_evidence = verification.evidence_type if verification else None
        disagreement = verified_evidence != proposal.claimed_evidence.evidence_type if verified_evidence else None
        if verification is not None and not verification.matches(state, proposal):
            return MembraneDecision(
                False,
                state,
                state,
                "rejected: verification receipt is not bound to this state and proposal",
                proposal.claimed_evidence.evidence_type,
                verified_evidence,
                disagreement,
                True,
                verification.receipt_hash,
            )
        if verification is not None and verification.evidence_type not in self.verification_authorities.get(
            verification.verifier_id, frozenset()
        ):
            return MembraneDecision(
                False,
                state,
                state,
                "rejected: verifier is not trusted for the claimed evidence authority",
                proposal.claimed_evidence.evidence_type,
                verified_evidence,
                disagreement,
                True,
                verification.receipt_hash,
            )
        if verification is not None and verification.verifier_id == proposal.claimed_evidence.asserted_by:
            return MembraneDecision(
                False,
                state,
                state,
                "rejected: proposer cannot independently verify its own evidence",
                proposal.claimed_evidence.evidence_type,
                verified_evidence,
                disagreement,
                True,
                verification.receipt_hash,
            )
        if proposal.requested_status is not None:
            try:
                target = transition_status(state.lifecycle_status, proposal.requested_status, authority=authority)
            except ValueError as exc:
                return MembraneDecision(
                    False,
                    state,
                    state,
                    f"rejected: {exc}",
                    proposal.claimed_evidence.evidence_type,
                    verified_evidence,
                    disagreement,
                    True,
                    verification.receipt_hash if verification else None,
                )
        else:
            target = state.lifecycle_status
        if target in {LifecycleStatus.EVALUATED, LifecycleStatus.PROMOTED} and verification is None:
            return MembraneDecision(
                False,
                state,
                state,
                "audit: evaluation and promotion require a bound independent verification receipt",
                proposal.claimed_evidence.evidence_type,
                None,
                None,
                True,
                None,
            )
        if state.lifecycle_status == LifecycleStatus.EVALUATED and proposal.payload:
            return MembraneDecision(
                False,
                state,
                state,
                "rejected: evaluated records are immutable",
                proposal.claimed_evidence.evidence_type,
                verified_evidence,
                disagreement,
                True,
                verification.receipt_hash if verification else None,
            )
        payload = dict(state.public_payload)
        accepted = tuple(payload.get("accepted_proposals", ()))
        payload["accepted_proposals"] = (*accepted, proposal.to_jsonable())
        frozen_evaluation_hash = state.frozen_evaluation_hash
        if target == LifecycleStatus.VALIDATED:
            candidate = proposal.payload.get("frozen_evaluation_hash")
            if not isinstance(candidate, str) or not candidate:
                return MembraneDecision(
                    False,
                    state,
                    state,
                    "rejected: validation must freeze the evaluation contract hash",
                    proposal.claimed_evidence.evidence_type,
                    verified_evidence,
                    disagreement,
                    True,
                    verification.receipt_hash if verification else None,
                )
            frozen_evaluation_hash = candidate
        candidate_hypothesis_ids = state.candidate_hypothesis_ids
        if target == LifecycleStatus.CRITIQUED and "hypothesis_ids" in proposal.payload:
            candidate_hypothesis_ids = tuple(str(item) for item in proposal.payload["hypothesis_ids"])
        after = replace(
            state,
            lifecycle_status=target,
            candidate_hypothesis_ids=candidate_hypothesis_ids,
            frozen_evaluation_hash=frozen_evaluation_hash,
            public_payload=payload,
        )
        return MembraneDecision(
            True,
            state,
            after,
            "accepted: typed public proposal",
            proposal.claimed_evidence.evidence_type,
            verified_evidence,
            disagreement,
            False,
            verification.receipt_hash if verification else None,
        )
