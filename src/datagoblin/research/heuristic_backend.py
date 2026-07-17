from __future__ import annotations

from datagoblin.domain.learner import LearnerSnapshot

from .alchemist import Alchemist
from .backend import BackendProposal
from .scout import Scout
from .state import ResearchState


class HeuristicGoblinBackend:
    def __init__(self) -> None:
        self.scout = Scout()
        self.alchemist = Alchemist()

    def propose(self, state: ResearchState) -> BackendProposal:
        snapshot = LearnerSnapshot.from_jsonable(state.public_payload["learner_snapshot"])
        findings, hypotheses = self.scout.analyze(snapshot)
        recipe = self.alchemist.propose(snapshot, findings, hypotheses)
        return BackendProposal(findings, hypotheses, recipe)
