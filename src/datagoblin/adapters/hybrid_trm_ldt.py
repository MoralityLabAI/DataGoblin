from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from datagoblin.domain.evidence import EvidenceType
from datagoblin.domain.research_episode import ResearchEpisode

_SOUNDNESS_MAP = {
    EvidenceType.ENVIRONMENT_VERIFIED: "env_sound_dead",
    EvidenceType.INDEPENDENT_HARNESS: "unknown",
    EvidenceType.HUMAN_REVIEWED: "unknown",
    EvidenceType.MODEL_INFERRED: "model_sound_dead",
    EvidenceType.EXPERIENCE_INFERRED: "experience_sound_dead",
    EvidenceType.SIMULATED: "unknown",
    EvidenceType.UNKNOWN: "unknown",
}


class HybridTRMLDTAdapter:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root.resolve() if root else None

    def evidence_to_soundness(self, evidence: EvidenceType) -> str:
        return _SOUNDNESS_MAP[evidence]

    def episode_frame(self, episode: ResearchEpisode) -> dict[str, Any]:
        evidence = (
            episode.experiment_outcome.evidence_type if episode.experiment_outcome else EvidenceType.MODEL_INFERRED
        )
        return {
            "id": episode.episode_id,
            "family": "datagoblin_research",
            "source": "datagoblin",
            "input_state": episode.learner_snapshot.to_jsonable(),
            "operation": episode.repaired_recipe.to_jsonable(),
            "output_state": episode.experiment_outcome.to_jsonable() if episode.experiment_outcome else {},
            "soundness_type": self.evidence_to_soundness(evidence),
            "label": episode.next_action,
            "metadata": {
                "claimed_evidence": evidence.value,
                "hard_promoted": evidence.may_hard_promote,
            },
        }

    def compatibility(self) -> dict[str, Any]:
        importable = importlib.util.find_spec("research_gym") is not None
        return {
            "root_available": bool(self.root and self.root.exists()),
            "research_gym_importable": importable,
            "narrow_membrane_exercise": "available"
            if importable
            else "skipped: optional upstream absent from sys.path",
        }
