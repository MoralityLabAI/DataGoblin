from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.experiments.ldt_array_hard_surface import ARRAY_STATUSES
from datagoblin.experiments.ldt_listwise_array_v9 import (
    IndependentCandidateEncoder,
    LoopedSetStatusModel,
)
from datagoblin.experiments.ldt_structured_array_v8 import (
    generate_structured_array_dataset,
)

EDGE_STATISTICS = (
    "mean_signed_delta",
    "mean_absolute_delta",
    "min_signed_delta",
    "max_signed_delta",
)


@dataclass(frozen=True)
class EdgeDifferenceCandidateEncoder:
    """Permutation-equivariant candidate features with one edge-message step."""

    base: IndependentCandidateEncoder = field(
        default_factory=IndependentCandidateEncoder
    )

    def feature_names(self) -> tuple[str, ...]:
        base_names = self.base.feature_names()
        edge_names = tuple(
            f"edge:{statistic}:{name}"
            for statistic in EDGE_STATISTICS
            for name in base_names[1:]
        )
        return (*base_names, *edge_names)

    def transform_array(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        base = self.base.transform_array(candidates)
        if base.shape[0] < 2:
            raise ValueError("edge encoder requires at least two candidates")
        messages = []
        for index in range(base.shape[0]):
            neighbor_mask = np.arange(base.shape[0]) != index
            deltas = base[index][np.newaxis, :] - base[neighbor_mask]
            informative = deltas[:, 1:]
            messages.append(
                np.concatenate(
                    (
                        np.mean(informative, axis=0),
                        np.mean(np.abs(informative), axis=0),
                        np.min(informative, axis=0),
                        np.max(informative, axis=0),
                    )
                )
            )
        result = np.concatenate(
            (base, np.stack(messages)),
            axis=1,
        )
        if result.shape != (
            len(candidates),
            len(self.feature_names()),
        ):
            raise AssertionError("edge feature names and values differ")
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "feature_view": "edge_difference_message",
            "base_encoder": self.base.receipt(),
            "feature_names": list(self.feature_names()),
            "feature_count": len(self.feature_names()),
            "message_steps": 1,
            "edge_statistics": list(EDGE_STATISTICS),
            "pooling": "per_candidate_all_neighbor_statistics",
            "permutation_equivariant": True,
            "fit_split": "none_stateless",
            "executes_verifier_policy": False,
            "uses_targets": False,
        }


@dataclass
class SetCoupledLoopedStatusModel(LoopedSetStatusModel):
    set_loss_weight: float = 0.0
    set_loss_smoothing: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.set_loss_weight <= 2.0:
            raise ValueError("set-coupled loss weight is out of bounds")
        if self.set_loss_smoothing <= 0.0:
            raise ValueError("set-coupled smoothing must be positive")

    @property
    def selected_index(self) -> int:
        return ARRAY_STATUSES.index("SELECTED")

    def _soft_dice_loss_and_probability_gradient(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        selected_probabilities = probabilities[:, self.selected_index]
        selected_targets = (
            targets == self.selected_index
        ).astype(np.float64)
        numerator = (
            2.0
            * float(np.dot(selected_probabilities, selected_targets))
            + self.set_loss_smoothing
        )
        denominator = (
            float(np.sum(selected_probabilities))
            + float(np.sum(selected_targets))
            + self.set_loss_smoothing
        )
        loss = 1.0 - numerator / denominator
        gradient = (
            numerator - 2.0 * selected_targets * denominator
        ) / (denominator * denominator)
        return loss, gradient

    def loss(
        self,
        features: np.ndarray,
        targets: Sequence[int],
        *,
        class_weights: Sequence[float] | None = None,
    ) -> float:
        probabilities = self.predict_proba(features)
        target_array = np.asarray(targets, dtype=np.int64)
        base = super().loss(
            features,
            targets,
            class_weights=class_weights,
        )
        dice, _ = self._soft_dice_loss_and_probability_gradient(
            probabilities,
            target_array,
        )
        return base + self.set_loss_weight * dice

    def _apply_logits_delta(
        self,
        features: np.ndarray,
        states: Sequence[np.ndarray],
        logits_delta: np.ndarray,
        *,
        learning_rate: float,
        l2: float,
        gradient_clip: float,
    ) -> None:
        final = states[-1]
        final_pool = np.mean(final, axis=0)
        status_gradient = logits_delta.T @ final
        status_pool_gradient = np.outer(
            np.sum(logits_delta, axis=0),
            final_pool,
        )
        status_bias_gradient = np.sum(logits_delta, axis=0)
        hidden_delta = logits_delta @ self.status_weights
        pooled_output_delta = (
            np.sum(logits_delta, axis=0) @ self.status_pool_weights
        )
        hidden_delta += pooled_output_delta[np.newaxis, :] / features.shape[0]

        input_gradient = np.zeros_like(self.input_weights)
        self_gradient = (
            None
            if self.self_weights is None
            else np.zeros_like(self.self_weights)
        )
        pool_gradient = (
            None
            if self.pool_weights is None
            else np.zeros_like(self.pool_weights)
        )
        hidden_bias_gradient = np.zeros_like(self.hidden_bias)
        for loop_index in range(self.loop_count - 1, -1, -1):
            current = states[loop_index + 1]
            previous = states[loop_index]
            activation_delta = hidden_delta * (1.0 - current * current)
            input_gradient += activation_delta.T @ features
            hidden_bias_gradient += np.sum(activation_delta, axis=0)
            if (
                self.self_weights is not None
                and self.pool_weights is not None
                and self_gradient is not None
                and pool_gradient is not None
            ):
                previous_pool = np.mean(previous, axis=0)
                self_gradient += activation_delta.T @ previous
                pool_gradient += np.outer(
                    np.sum(activation_delta, axis=0),
                    previous_pool,
                )
                hidden_delta = activation_delta @ self.self_weights
                pooled_delta = (
                    np.sum(activation_delta, axis=0) @ self.pool_weights
                )
                hidden_delta += pooled_delta[np.newaxis, :] / features.shape[0]

        input_gradient += l2 * self.input_weights
        status_gradient += l2 * self.status_weights
        status_pool_gradient += l2 * self.status_pool_weights
        gradients = [
            input_gradient,
            hidden_bias_gradient,
            status_gradient,
            status_pool_gradient,
            status_bias_gradient,
        ]
        if (
            self.self_weights is not None
            and self.pool_weights is not None
            and self_gradient is not None
            and pool_gradient is not None
        ):
            self_gradient += l2 * self.self_weights
            pool_gradient += l2 * self.pool_weights
            gradients.extend((self_gradient, pool_gradient))
        norm = math.sqrt(
            sum(
                float(np.sum(gradient * gradient))
                for gradient in gradients
            )
        )
        scale = min(1.0, gradient_clip / norm) if norm > 0.0 else 1.0
        self.input_weights -= learning_rate * scale * input_gradient
        self.hidden_bias -= learning_rate * scale * hidden_bias_gradient
        self.status_weights -= learning_rate * scale * status_gradient
        self.status_pool_weights -= (
            learning_rate * scale * status_pool_gradient
        )
        self.status_bias -= learning_rate * scale * status_bias_gradient
        if (
            self.self_weights is not None
            and self.pool_weights is not None
            and self_gradient is not None
            and pool_gradient is not None
        ):
            self.self_weights -= learning_rate * scale * self_gradient
            self.pool_weights -= learning_rate * scale * pool_gradient

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
        if learning_rate <= 0.0 or l2 < 0.0 or gradient_clip <= 0.0:
            raise ValueError("invalid set-coupled optimizer request")
        states, probabilities = self._forward(features)
        target_array = np.asarray(targets, dtype=np.int64)
        if (
            target_array.shape != (features.shape[0],)
            or np.any(target_array < 0)
            or np.any(target_array >= self.status_count)
        ):
            raise ValueError("invalid set-coupled targets")
        weights = np.asarray(class_weights, dtype=np.float64)[target_array]
        normalizer = float(np.sum(weights))
        logits_delta = probabilities.copy()
        logits_delta[np.arange(len(target_array)), target_array] -= 1.0
        logits_delta *= weights[:, np.newaxis] / normalizer

        if self.set_loss_weight > 0.0:
            _, probability_gradient = (
                self._soft_dice_loss_and_probability_gradient(
                    probabilities,
                    target_array,
                )
            )
            selected_probabilities = probabilities[:, self.selected_index]
            dice_logits_delta = (
                -probability_gradient[:, np.newaxis]
                * selected_probabilities[:, np.newaxis]
                * probabilities
            )
            dice_logits_delta[:, self.selected_index] += (
                probability_gradient
                * selected_probabilities
            )
            logits_delta += self.set_loss_weight * dice_logits_delta

        self._apply_logits_delta(
            features,
            states,
            logits_delta,
            learning_rate=learning_rate,
            l2=l2,
            gradient_clip=gradient_clip,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "architecture": "edge_difference_permutation_equivariant_status",
            "set_coupled_loss": {
                "family": (
                    "soft_dice_selected_set"
                    if self.set_loss_weight > 0.0
                    else "none"
                ),
                "weight": self.set_loss_weight,
                "smoothing": self.set_loss_smoothing,
            },
        }


def _status_class_weights(
    arrays: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    counts = Counter(
        str(status)
        for row in arrays
        for status in row["target"]["statuses"]
    )
    total = sum(counts.values())
    return tuple(
        total / (len(ARRAY_STATUSES) * counts[label])
        if counts[label]
        else 1.0
        for label in ARRAY_STATUSES
    )


def train_edge_status_model(
    model: SetCoupledLoopedStatusModel,
    encoder: EdgeDifferenceCandidateEncoder,
    arrays: Sequence[Mapping[str, Any]],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    gradient_clip: float,
    seed: int,
    checkpoint_every_epochs: int,
    checkpoint_callback: Any | None = None,
) -> int:
    if (
        not arrays
        or epochs < 1
        or learning_rate <= 0.0
        or l2 < 0.0
        or gradient_clip <= 0.0
        or checkpoint_every_epochs < 1
    ):
        raise ValueError("invalid bounded edge-status training request")
    prepared = [
        (
            encoder.transform_array(
                [candidate["features"] for candidate in row["candidates"]]
            ),
            [
                ARRAY_STATUSES.index(str(status))
                for status in row["target"]["statuses"]
            ],
        )
        for row in arrays
    ]
    class_weights = _status_class_weights(arrays)
    order = list(range(len(prepared)))
    rng = random.Random(seed)
    steps = 0
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        for index in order:
            features, targets = prepared[index]
            model.train_array(
                features,
                targets,
                learning_rate=learning_rate,
                l2=l2,
                gradient_clip=gradient_clip,
                class_weights=class_weights,
            )
            steps += 1
        if checkpoint_callback is not None and (
            epoch % checkpoint_every_epochs == 0 or epoch == epochs
        ):
            checkpoint_callback(epoch, model.state_dict())
    return steps


def generate_edge_set_array_dataset(
    config_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    return generate_structured_array_dataset(config_path, out_dir)
