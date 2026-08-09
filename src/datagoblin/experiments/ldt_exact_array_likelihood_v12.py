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
def _selected_subset_masks(candidate_count: int) -> np.ndarray:
    if not 1 <= candidate_count <= 12:
        raise ValueError("candidate count is outside exact subset bounds")
    subset_ids = np.arange(1 << candidate_count, dtype=np.uint16)
    bit_positions = np.arange(candidate_count, dtype=np.uint16)
    masks = ((subset_ids[:, np.newaxis] >> bit_positions) & 1).astype(
        np.float64
    )
    masks.setflags(write=False)
    return masks


@dataclass
class ExactArrayLikelihoodModel(SetCoupledLoopedStatusModel):
    max_candidates: int = 6
    decoder: str = "factorized"

    def __post_init__(self) -> None:
        self.set_loss_weight = 0.0
        self.set_loss_smoothing = 1e-6
        super().__post_init__()
        if self.max_candidates < 2:
            raise ValueError("exact array model needs at least two candidates")
        if self.decoder not in {"factorized", "global_selected_count_map"}:
            raise ValueError("unknown exact-array decoder")
        self.selected_count_potential = np.zeros(
            self.max_candidates + 1,
            dtype=np.float64,
        )

    @property
    def parameter_count(self) -> int:
        return super().parameter_count + self.selected_count_potential.size

    @property
    def selected_index(self) -> int:
        return ARRAY_STATUSES.index("SELECTED")

    def _structured_statistics(
        self,
        probabilities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        candidate_count = probabilities.shape[0]
        if candidate_count > self.max_candidates:
            raise ValueError("array exceeds exact structured bound")
        masks = _selected_subset_masks(candidate_count)
        selected = np.clip(
            probabilities[:, self.selected_index],
            1e-12,
            1.0 - 1e-12,
        )
        not_selected = 1.0 - selected
        subset_counts = np.sum(masks, axis=1).astype(np.int64)
        log_weights = (
            np.sum(np.log(not_selected))
            + masks @ (np.log(selected) - np.log(not_selected))
            + self.selected_count_potential[subset_counts]
        )
        maximum = float(np.max(log_weights))
        unnormalized = np.exp(log_weights - maximum)
        normalizer = float(np.sum(unnormalized))
        subset_probabilities = unnormalized / normalizer
        log_partition = maximum + math.log(normalizer)
        selected_marginals = subset_probabilities @ masks
        count_marginals = np.bincount(
            subset_counts,
            weights=subset_probabilities,
            minlength=self.max_candidates + 1,
        )

        status_marginals = np.zeros_like(probabilities)
        status_marginals[:, self.selected_index] = selected_marginals
        nonselected_indices = [
            index
            for index in range(len(ARRAY_STATUSES))
            if index != self.selected_index
        ]
        conditional_nonselected = (
            probabilities[:, nonselected_indices]
            / not_selected[:, np.newaxis]
        )
        status_marginals[:, nonselected_indices] = (
            (1.0 - selected_marginals)[:, np.newaxis]
            * conditional_nonselected
        )
        return status_marginals, count_marginals, log_partition

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
            raise ValueError("invalid exact-array targets")
        _, _, log_partition = self._structured_statistics(probabilities)
        selected_count = int(np.sum(target_array == self.selected_index))
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
            + float(self.selected_count_potential[selected_count])
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
            raise ValueError("invalid exact-array optimizer request")
        states, probabilities = self._forward(features)
        target_array = np.asarray(targets, dtype=np.int64)
        if (
            target_array.shape != (features.shape[0],)
            or np.any(target_array < 0)
            or np.any(target_array >= self.status_count)
        ):
            raise ValueError("invalid exact-array targets")
        status_marginals, count_marginals, _ = (
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

        selected_count = int(np.sum(target_array == self.selected_index))
        count_gradient = count_marginals.copy()
        count_gradient[selected_count] -= 1.0
        count_gradient += l2 * self.selected_count_potential
        norm = float(np.linalg.norm(count_gradient))
        scale = min(1.0, gradient_clip / norm) if norm > 0.0 else 1.0
        self.selected_count_potential -= (
            learning_rate * scale * count_gradient
        )

    def decode(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.predict_proba(features)
        if self.decoder == "factorized":
            return np.argmax(probabilities, axis=1), probabilities
        selected = np.maximum(
            probabilities[:, self.selected_index],
            1e-12,
        )
        nonselected_indices = np.asarray(
            [
                index
                for index in range(len(ARRAY_STATUSES))
                if index != self.selected_index
            ],
            dtype=np.int64,
        )
        best_offsets = np.argmax(
            probabilities[:, nonselected_indices],
            axis=1,
        )
        best_nonselected = nonselected_indices[best_offsets]
        baseline = np.log(
            np.maximum(
                probabilities[
                    np.arange(len(best_nonselected)),
                    best_nonselected,
                ],
                1e-12,
            )
        )
        selected_delta = np.log(selected) - baseline
        ranked = np.argsort(-selected_delta, kind="stable")
        scores = []
        running_delta = 0.0
        baseline_score = float(np.sum(baseline))
        for count in range(features.shape[0] + 1):
            if count:
                running_delta += float(selected_delta[ranked[count - 1]])
            scores.append(
                float(self.selected_count_potential[count])
                + baseline_score
                + running_delta
            )
        selected_count = int(np.argmax(np.asarray(scores)))
        result = best_nonselected.copy()
        if selected_count:
            result[ranked[:selected_count]] = self.selected_index
        return result, probabilities

    def state_dict(self) -> dict[str, Any]:
        return {
            **LoopedSetStatusModel.state_dict(self),
            "architecture": "exact_globally_normalized_array_status",
            "max_candidates": self.max_candidates,
            "decoder": self.decoder,
            "normalization": "exact_selected_subset_marginalization",
            "represented_status_vectors": "all_four_status_assignments",
            "shared_state_gradient": True,
            "class_balancing": "none_proper_likelihood",
            "selected_count_potential": (
                self.selected_count_potential.tolist()
            ),
        }


def predict_exact_array_statuses(
    model: ExactArrayLikelihoodModel,
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
