from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datagoblin.domain.evidence import EvidenceType
from datagoblin.domain.hypotheses import ResearchHypothesis
from datagoblin.domain.learner import LearnerSnapshot
from datagoblin.serialization import to_jsonable


@dataclass(frozen=True)
class ScoutFindings:
    highest_priority_gaps: tuple[str, ...]
    surprising_correlations: tuple[str, ...]
    missing_evaluation_coverage: tuple[str, ...]
    data_vs_architecture_explanations: tuple[str, ...]
    competing_hypotheses: tuple[str, ...]
    suggested_discriminating_variables: tuple[str, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


class Scout:
    def analyze(self, snapshot: LearnerSnapshot) -> tuple[ScoutFindings, tuple[ResearchHypothesis, ...]]:
        delayed = any(item.capability_id == "delayed_causality" for item in snapshot.capability_observations)
        gap = "delayed_causality" if delayed else snapshot.capability_observations[0].capability_id
        hypotheses = (
            ResearchHypothesis(
                "hyp_color_shortcut",
                "The learner relies on object color rather than functional role and intervention history.",
                "Biased training support makes color a lower-loss shortcut.",
                tuple(item.cluster_id for item in snapshot.failure_clusters),
                (),
                0.65,
                ("hyp_temporal_capacity",),
                ("Role permutation training improves permuted holdout accuracy.",),
                ("Matched role-permutation data does not improve holdout accuracy or reliance diagnostics.",),
                EvidenceType.MODEL_INFERRED,
            ),
            ResearchHypothesis(
                "hyp_temporal_capacity",
                "The learner cannot represent the delayed intervention history with its current features.",
                "The learner representation discards the relevant lagged state.",
                tuple(item.cluster_id for item in snapshot.failure_clusters),
                (),
                0.35,
                ("hyp_color_shortcut",),
                ("Permutation alone does not help, while adding an explicit history feature does.",),
                ("The same architecture learns the effect from a balanced delay curriculum.",),
                EvidenceType.MODEL_INFERRED,
            ),
        )
        findings = ScoutFindings(
            (gap, "role_invariance"),
            ("Training color-role correlation is high while permuted-holdout accuracy is low.",),
            (
                "No counterfactual pair audit is present.",
                "Longer intervention delays are under-sampled.",
            ),
            ("Data shortcut and representational capacity remain competing explanations.",),
            tuple(item.hypothesis_id for item in hypotheses),
            (
                "role permutation",
                "delay width",
                "counterfactual pairing",
                "history-feature availability",
            ),
        )
        return findings, hypotheses
