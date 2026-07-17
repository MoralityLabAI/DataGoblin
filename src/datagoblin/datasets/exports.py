from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datagoblin.constitution.validator import ConstitutionalValidator
from datagoblin.domain.research_episode import ResearchEpisode

from .corruptions import MUTATION_NAMES, apply_mutation


def read_episodes(path: Path) -> list[ResearchEpisode]:
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        values = value if isinstance(value, list) else [value]
    return [ResearchEpisode.from_jsonable(item) for item in values]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def export_sft_jsonl(episodes: Iterable[ResearchEpisode], path: Path) -> None:
    _write_jsonl(
        path,
        (
            {
                "schema_version": "1.0",
                "input": {
                    "learner_snapshot": episode.learner_snapshot.to_jsonable(),
                    "public_research_context": episode.scout_findings,
                },
                "target": {
                    "hypotheses": [item.to_jsonable() for item in episode.hypothesis_set],
                    "recipe": episode.repaired_recipe.to_jsonable(),
                },
            }
            for episode in episodes
        ),
    )


def export_preference_jsonl(episodes: Iterable[ResearchEpisode], path: Path) -> None:
    rows = []
    for episode_index, episode in enumerate(episodes):
        for operator_index, operator in enumerate(MUTATION_NAMES):
            rejected, receipt = apply_mutation(
                episode.repaired_recipe, operator, seed=episode_index * 1000 + operator_index
            )
            rows.append(
                {
                    "schema_version": "1.0",
                    "input": episode.learner_snapshot.to_jsonable(),
                    "chosen": episode.repaired_recipe.to_jsonable(),
                    "rejected": rejected.to_jsonable(),
                    "violation_labels": list(receipt.expected_violation_mask),
                    "critique": receipt.minimal_repair,
                    "mutation_receipt": receipt.to_jsonable(),
                }
            )
    _write_jsonl(path, rows)


def export_verifier_jsonl(episodes: Iterable[ResearchEpisode], path: Path) -> None:
    validator = ConstitutionalValidator()
    rows = []
    for episode in episodes:
        for index, operator in enumerate(MUTATION_NAMES):
            candidate, receipt = apply_mutation(episode.repaired_recipe, operator, index)
            decision = validator.validate(candidate, episode.hypothesis_set)
            rows.append(
                {
                    "candidate": candidate.to_jsonable(),
                    "expected_decision": decision.decision.value,
                    "violation_vector": list(receipt.expected_violation_mask),
                }
            )
    _write_jsonl(path, rows)


def export_outcome_prediction_jsonl(episodes: Iterable[ResearchEpisode], path: Path) -> None:
    rows = []
    for episode in episodes:
        if episode.experiment_outcome is None:
            continue
        rows.append(
            {
                "learner_snapshot": episode.learner_snapshot.to_jsonable(),
                "recipe": episode.repaired_recipe.to_jsonable(),
                "predicted": [item.to_jsonable() for item in episode.repaired_recipe.predicted_outcomes],
                "actual": episode.experiment_outcome.actual_metrics,
                "prediction_error": episode.prediction_error,
            }
        )
    _write_jsonl(path, rows)
