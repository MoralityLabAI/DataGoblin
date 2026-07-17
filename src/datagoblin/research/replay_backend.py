from __future__ import annotations

from pathlib import Path

from datagoblin.domain.hypotheses import ResearchHypothesis
from datagoblin.domain.recipes import DataRecipe
from datagoblin.serialization import load_structured

from .backend import BackendProposal
from .scout import ScoutFindings
from .state import ResearchState


class ReplayGoblinBackend:
    def __init__(self, proposal_path: Path) -> None:
        self.proposal_path = proposal_path

    def propose(self, state: ResearchState) -> BackendProposal:
        del state
        value = load_structured(self.proposal_path)
        findings = value["scout_findings"]
        return BackendProposal(
            ScoutFindings(
                tuple(findings["highest_priority_gaps"]),
                tuple(findings["surprising_correlations"]),
                tuple(findings["missing_evaluation_coverage"]),
                tuple(findings["data_vs_architecture_explanations"]),
                tuple(findings["competing_hypotheses"]),
                tuple(findings["suggested_discriminating_variables"]),
            ),
            tuple(ResearchHypothesis.from_jsonable(item) for item in value["hypotheses"]),
            DataRecipe.from_jsonable(value["recipe"]),
        )
