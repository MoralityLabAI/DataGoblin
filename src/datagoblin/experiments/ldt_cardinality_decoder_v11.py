from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from datagoblin.experiments.ldt_array_hard_surface import ARRAY_STATUSES
from datagoblin.experiments.ldt_listwise_array_v9 import (
    IndependentCandidateEncoder,
    LoopedSetStatusModel,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.clip(logits - np.max(logits), -40.0, 40.0)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def joint_selected_count_map(
    status_probabilities: np.ndarray,
    count_probabilities: np.ndarray,
) -> np.ndarray:
    """Decode a joint MAP status vector under a learned selected count."""
    if (
        status_probabilities.ndim != 2
        or status_probabilities.shape[1] != len(ARRAY_STATUSES)
        or status_probabilities.shape[0] < 1
        or count_probabilities.ndim != 1
        or len(count_probabilities) <= status_probabilities.shape[0]
        or not np.all(np.isfinite(status_probabilities))
        or not np.all(np.isfinite(count_probabilities))
        or np.any(status_probabilities < 0.0)
        or np.any(count_probabilities < 0.0)
    ):
        raise ValueError("invalid cardinality decoder probabilities")
    selected_index = ARRAY_STATUSES.index("SELECTED")
    nonselected_indices = np.asarray(
        [index for index in range(len(ARRAY_STATUSES)) if index != selected_index],
        dtype=np.int64,
    )
    safe_status = np.maximum(status_probabilities, 1e-12)
    best_nonselected_offsets = np.argmax(
        safe_status[:, nonselected_indices],
        axis=1,
    )
    best_nonselected = nonselected_indices[best_nonselected_offsets]
    baseline = np.log(
        safe_status[np.arange(len(best_nonselected)), best_nonselected]
    )
    selected_delta = (
        np.log(safe_status[:, selected_index]) - baseline
    )
    ranked = np.argsort(-selected_delta, kind="stable")
    count_scores = []
    running_delta = 0.0
    baseline_score = float(np.sum(baseline))
    for count in range(status_probabilities.shape[0] + 1):
        if count > 0:
            running_delta += float(selected_delta[ranked[count - 1]])
        count_scores.append(
            math.log(max(float(count_probabilities[count]), 1e-12))
            + baseline_score
            + running_delta
        )
    selected_count = int(np.argmax(np.asarray(count_scores)))
    result = best_nonselected.copy()
    if selected_count:
        result[ranked[:selected_count]] = selected_index
    return result


@dataclass
class CardinalityLoopedStatusModel(LoopedSetStatusModel):
    max_candidates: int = 6
    cardinality_loss_weight: float = 0.5
    decoder: str = "factorized"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_candidates < 2:
            raise ValueError("cardinality head needs at least two candidates")
        if not 0.0 < self.cardinality_loss_weight <= 2.0:
            raise ValueError("cardinality loss weight is out of bounds")
        if self.decoder not in {"factorized", "joint_selected_count_map"}:
            raise ValueError("unknown cardinality decoder")
        rng = np.random.default_rng(self.seed + 17_171)
        width = self.hidden_width + 1
        scale = 1.0 / math.sqrt(width)
        self.cardinality_weights = rng.uniform(
            -scale,
            scale,
            size=(self.max_candidates + 1, width),
        )
        self.cardinality_bias = np.zeros(
            self.max_candidates + 1,
            dtype=np.float64,
        )

    @property
    def parameter_count(self) -> int:
        return (
            super().parameter_count
            + self.cardinality_weights.size
            + self.cardinality_bias.size
        )

    def _cardinality_features(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        states, _ = self._forward(features)
        pooled = np.mean(states[-1], axis=0)
        return np.concatenate(
            (
                pooled,
                np.asarray(
                    [features.shape[0] / self.max_candidates],
                    dtype=np.float64,
                ),
            )
        )

    def predict_cardinality_proba(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        cardinality_features = self._cardinality_features(features)
        logits = (
            self.cardinality_weights @ cardinality_features
            + self.cardinality_bias
        )
        return _softmax(logits)

    def loss(
        self,
        features: np.ndarray,
        targets: Sequence[int],
        *,
        class_weights: Sequence[float] | None = None,
    ) -> float:
        base = super().loss(
            features,
            targets,
            class_weights=class_weights,
        )
        selected_index = ARRAY_STATUSES.index("SELECTED")
        selected_count = sum(int(target == selected_index) for target in targets)
        probabilities = self.predict_cardinality_proba(features)
        cardinality = -math.log(
            max(float(probabilities[selected_count]), 1e-12)
        )
        return base + self.cardinality_loss_weight * cardinality

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
        super().train_array(
            features,
            targets,
            learning_rate=learning_rate,
            l2=l2,
            gradient_clip=gradient_clip,
            class_weights=class_weights,
        )
        selected_index = ARRAY_STATUSES.index("SELECTED")
        selected_count = sum(int(target == selected_index) for target in targets)
        if selected_count > self.max_candidates:
            raise ValueError("selected count exceeds cardinality head")
        cardinality_features = self._cardinality_features(features)
        probabilities = _softmax(
            self.cardinality_weights @ cardinality_features
            + self.cardinality_bias
        )
        probabilities[selected_count] -= 1.0
        logits_delta = self.cardinality_loss_weight * probabilities
        weight_gradient = (
            np.outer(logits_delta, cardinality_features)
            + l2 * self.cardinality_weights
        )
        bias_gradient = logits_delta
        norm = math.sqrt(
            float(np.sum(weight_gradient * weight_gradient))
            + float(np.sum(bias_gradient * bias_gradient))
        )
        scale = min(1.0, gradient_clip / norm) if norm > 0.0 else 1.0
        self.cardinality_weights -= learning_rate * scale * weight_gradient
        self.cardinality_bias -= learning_rate * scale * bias_gradient

    def decode(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        status_probabilities = self.predict_proba(features)
        if self.decoder == "factorized":
            indices = np.argmax(status_probabilities, axis=1)
        else:
            indices = joint_selected_count_map(
                status_probabilities,
                self.predict_cardinality_proba(features),
            )
        return indices, status_probabilities

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "architecture": (
                "looped_cardinality_permutation_equivariant_status"
            ),
            "max_candidates": self.max_candidates,
            "cardinality_loss_weight": self.cardinality_loss_weight,
            "cardinality_shared_state_gradient": False,
            "decoder": self.decoder,
            "cardinality_weights": self.cardinality_weights.tolist(),
            "cardinality_bias": self.cardinality_bias.tolist(),
        }


def predict_cardinality_statuses(
    model: CardinalityLoopedStatusModel,
    encoder: IndependentCandidateEncoder,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    features = encoder.transform_array(
        [candidate["features"] for candidate in row["candidates"]]
    )
    indices, probabilities = model.decode(features)
    cardinality_probabilities = model.predict_cardinality_proba(features)
    return {
        "statuses": [ARRAY_STATUSES[int(index)] for index in indices],
        "status_probabilities": [
            {
                label: float(probabilities[row_index, label_index])
                for label_index, label in enumerate(ARRAY_STATUSES)
            }
            for row_index in range(len(indices))
        ],
        "selected_count_probabilities": {
            str(count): float(probability)
            for count, probability in enumerate(cardinality_probabilities)
        },
    }
