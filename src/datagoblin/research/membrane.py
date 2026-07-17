from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from datagoblin.domain.evidence import EvidenceClaim, EvidenceType
from datagoblin.domain.experiments import LifecycleStatus
from datagoblin.serialization import to_jsonable

from .state import ResearchState, transition_status


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

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


class ResearchMembrane:
    def apply(
        self,
        state: ResearchState,
        proposal: ResearchProposal,
        *,
        verified_evidence: EvidenceType | None = None,
        authority: str = "model",
    ) -> MembraneDecision:
        effective = verified_evidence or proposal.claimed_evidence.evidence_type
        disagreement = verified_evidence != proposal.claimed_evidence.evidence_type if verified_evidence else None
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
                )
        else:
            target = state.lifecycle_status
        if effective.may_hard_promote and verified_evidence is None and authority == "model":
            return MembraneDecision(
                False,
                state,
                state,
                "audit: authoritative evidence was claimed but not independently verified",
                proposal.claimed_evidence.evidence_type,
                None,
                None,
                True,
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
            )
        payload = dict(state.public_payload)
        payload.setdefault("accepted_proposals", []).append(to_jsonable(proposal.payload))
        after = replace(state, lifecycle_status=target, public_payload=payload)
        return MembraneDecision(
            True,
            state,
            after,
            "accepted: typed public proposal",
            proposal.claimed_evidence.evidence_type,
            verified_evidence,
            disagreement,
            False,
        )
