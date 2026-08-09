from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from datagoblin.experiments.ldt_array_hard_surface import ARRAY_STATUSES
from datagoblin.experiments.ldt_edge_set_array_v10 import (
    SetCoupledLoopedStatusModel,
)
from datagoblin.experiments.ldt_listwise_array_v9 import (
    IndependentCandidateEncoder,
    LoopedSetStatusModel,
)


@lru_cache(maxsize=8)
def _composition_space(
    max_candidates: int,
) -> tuple[tuple[tuple[int, ...], ...], dict[tuple[int, ...], int]]:
    if not 1 <= max_candidates <= 12:
        raise ValueError("composition bound is invalid")
    vectors = tuple(
        (selected, dominated, unresolved, unsafe)
        for total in range(max_candidates + 1)
        for selected in range(total + 1)
        for dominated in range(total - selected + 1)
        for unresolved in range(total - selected - dominated + 1)
        for unsafe in (total - selected - dominated - unresolved,)
    )
    return vectors, {vector: index for index, vector in enumerate(vectors)}


@lru_cache(maxsize=16)
def _assignment_table(
    candidate_count: int,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= candidate_count <= max_candidates:
        raise ValueError("assignment count exceeds composition bound")
    assignment_ids = np.arange(
        len(ARRAY_STATUSES) ** candidate_count,
        dtype=np.int64,
    )
    place_values = len(ARRAY_STATUSES) ** np.arange(
        candidate_count,
        dtype=np.int64,
    )
    assignments = (
        (assignment_ids[:, np.newaxis] // place_values) % len(ARRAY_STATUSES)
    ).astype(np.int8)
    _, composition_index = _composition_space(max_candidates)
    composition_ids = np.asarray(
        [
            composition_index[
                tuple(
                    int(np.sum(assignment == status))
                    for status in range(len(ARRAY_STATUSES))
                )
            ]
            for assignment in assignments
        ],
        dtype=np.int32,
    )
    assignments.setflags(write=False)
    composition_ids.setflags(write=False)
    return assignments, composition_ids


@dataclass
class ExactCompositionStatusModel(SetCoupledLoopedStatusModel):
    max_candidates: int = 6
    decoder: str = "factorized"

    def __post_init__(self) -> None:
        self.set_loss_weight = 0.0
        self.set_loss_smoothing = 1e-6
        super().__post_init__()
        if self.max_candidates < 2:
            raise ValueError("composition model needs at least two candidates")
        if self.decoder not in {"factorized", "global_composition_map"}:
            raise ValueError("unknown composition decoder")
        vectors, _ = _composition_space(self.max_candidates)
        self.composition_potential = np.zeros(
            len(vectors),
            dtype=np.float64,
        )

    @property
    def parameter_count(self) -> int:
        return super().parameter_count + self.composition_potential.size

    def _structured_statistics(
        self,
        probabilities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        candidate_count = probabilities.shape[0]
        assignments, composition_ids = _assignment_table(
            candidate_count,
            self.max_candidates,
        )
        log_probabilities = np.log(np.maximum(probabilities, 1e-12))
        local_scores = np.sum(
            log_probabilities[
                np.arange(candidate_count)[np.newaxis, :],
                assignments,
            ],
            axis=1,
        )
        scores = (
            local_scores
            + self.composition_potential[composition_ids]
        )
        maximum = float(np.max(scores))
        unnormalized = np.exp(scores - maximum)
        normalizer = float(np.sum(unnormalized))
        assignment_probabilities = unnormalized / normalizer
        log_partition = maximum + math.log(normalizer)

        status_marginals = np.zeros_like(probabilities)
        for position in range(candidate_count):
            status_marginals[position] = np.bincount(
                assignments[:, position],
                weights=assignment_probabilities,
                minlength=len(ARRAY_STATUSES),
            )
        composition_marginals = np.bincount(
            composition_ids,
            weights=assignment_probabilities,
            minlength=len(self.composition_potential),
        )
        return status_marginals, composition_marginals, log_partition

    def _target_composition_id(self, targets: np.ndarray) -> int:
        composition = tuple(
            int(np.sum(targets == status))
            for status in range(len(ARRAY_STATUSES))
        )
        _, composition_index = _composition_space(self.max_candidates)
        return composition_index[composition]

    def loss(
        self,
        features: np.ndarray,
        targets: Sequence[int],
        *,
        class_weights: Sequence[float] | None = None,
    ) -> float:
        del class_weights
        probabilities = self.predict_proba(features)
        target_array = np.asarray(targets, dtype=np.int64)
        if target_array.shape != (features.shape[0],):
            raise ValueError("invalid composition targets")
        _, _, log_partition = self._structured_statistics(probabilities)
        composition_id = self._target_composition_id(target_array)
        target_score = (
            float(
                np.sum(
                    np.log(
                        np.maximum(
                            probabilities[
                                np.arange(len(target_array)),
                                target_array,
                            ],
                            1e-12,
                        )
                    )
                )
            )
            + float(self.composition_potential[composition_id])
        )
        return log_partition - target_score

    def train_array(
        self,
        features: np.ndarray,
        targets: Sequence[int],
        *,
        learning_rate: float,
        l2: float,
        gradient_clip: float,
        class_weights: Sequence[float],
    ) -> None:
        del class_weights
        if learning_rate <= 0.0 or l2 < 0.0 or gradient_clip <= 0.0:
            raise ValueError("invalid composition optimizer request")
        states, probabilities = self._forward(features)
        target_array = np.asarray(targets, dtype=np.int64)
        if (
            target_array.shape != (features.shape[0],)
            or np.any(target_array < 0)
            or np.any(target_array >= self.status_count)
        ):
            raise ValueError("invalid composition targets")
        status_marginals, composition_marginals, _ = (
            self._structured_statistics(probabilities)
        )
        logits_delta = status_marginals
        logits_delta[
            np.arange(len(target_array)),
            target_array,
        ] -= 1.0
        self._apply_logits_delta(
            features,
            states,
            logits_delta,
            learning_rate=learning_rate,
            l2=l2,
            gradient_clip=gradient_clip,
        )

        composition_id = self._target_composition_id(target_array)
        composition_gradient = composition_marginals.copy()
        composition_gradient[composition_id] -= 1.0
        composition_gradient += l2 * self.composition_potential
        norm = float(np.linalg.norm(composition_gradient))
        scale = min(1.0, gradient_clip / norm) if norm > 0.0 else 1.0
        self.composition_potential -= (
            learning_rate * scale * composition_gradient
        )

    def decode(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.predict_proba(features)
        if self.decoder == "factorized":
            return np.argmax(probabilities, axis=1), probabilities
        assignments, composition_ids = _assignment_table(
            features.shape[0],
            self.max_candidates,
        )
        log_probabilities = np.log(np.maximum(probabilities, 1e-12))
        scores = (
            np.sum(
                log_probabilities[
                    np.arange(features.shape[0])[np.newaxis, :],
                    assignments,
                ],
                axis=1,
            )
            + self.composition_potential[composition_ids]
        )
        return assignments[int(np.argmax(scores))].copy(), probabilities

    def state_dict(self) -> dict[str, Any]:
        vectors, _ = _composition_space(self.max_candidates)
        return {
            **LoopedSetStatusModel.state_dict(self),
            "architecture": "exact_status_composition_array",
            "max_candidates": self.max_candidates,
            "decoder": self.decoder,
            "normalization": "exact_all_status_assignment_enumeration",
            "represented_status_vectors": "all_four_status_assignments",
            "shared_state_gradient": True,
            "class_balancing": "none_proper_likelihood",
            "composition_labels": list(ARRAY_STATUSES),
            "composition_vectors": [list(vector) for vector in vectors],
            "composition_potential": self.composition_potential.tolist(),
        }


def predict_composition_statuses(
    model: ExactCompositionStatusModel,
    encoder: IndependentCandidateEncoder,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    features = encoder.transform_array(
        [candidate["features"] for candidate in row["candidates"]]
    )
    indices, probabilities = model.decode(features)
    return {
        "statuses": [ARRAY_STATUSES[int(index)] for index in indices],
        "status_probabilities": [
            {
                label: float(probabilities[row_index, label_index])
                for label_index, label in enumerate(ARRAY_STATUSES)
            }
            for row_index in range(len(indices))
        ],
    }
