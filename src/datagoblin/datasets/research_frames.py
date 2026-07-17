from __future__ import annotations

from typing import Any

from datagoblin.domain.evidence import EvidenceType
from datagoblin.domain.research_episode import ResearchEpisode


def episode_to_frame(episode: ResearchEpisode) -> dict[str, Any]:
    return {
        "id": episode.episode_id,
        "family": "datagoblin_research_episode",
        "source": "datagoblin",
        "input_state": episode.learner_snapshot.to_jsonable(),
        "operation": episode.repaired_recipe.to_jsonable(),
        "output_state": episode.experiment_outcome.to_jsonable() if episode.experiment_outcome else {},
        "soundness_type": EvidenceType.MODEL_INFERRED.value,
        "label": episode.next_action,
        "metadata": {"constitutional_decision": episode.constitutional_decision},
    }
