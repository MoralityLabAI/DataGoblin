from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from datagoblin.experiments.ldt_array_hard_surface import (
    ARRAY_STATUSES,
)
from datagoblin.experiments.ldt_array_sorter import (
    BUCKET_RANK,
    METRIC_FIELDS,
)
from datagoblin.experiments.ldt_structured_array_v8 import (
    generate_structured_array_dataset,
)


@dataclass(frozen=True)
class IndependentCandidateEncoder:
    """Target-independent, permutation-neutral candidate encoder."""

    def feature_names(self) -> tuple[str, ...]:
        names = ["bias"]
        for field in METRIC_FIELDS:
            names.extend(
                (
                    f"{field}:lower",
                    f"{field}:upper",
                    f"{field}:width",
                    f"{field}:lower_missing",
                    f"{field}:upper_missing",
                )
            )
        names.extend(
            (
                "safety:qualified",
                "parameter_bucket:rank",
            )
        )
        return tuple(names)

    def transform(
        self,
        candidate: Mapping[str, Any],
    ) -> np.ndarray:
        values = [1.0]
        for field in METRIC_FIELDS:
            interval = candidate[field]
            lower = (
                0.0
                if interval["lower"] is None
                else float(interval["lower"])
            )
            upper = (
                1.0
                if interval["upper"] is None
                else float(interval["upper"])
            )
            values.extend(
                (
                    lower,
                    upper,
                    upper - lower,
                    float(interval["lower"] is None),
                    float(interval["upper"] is None),
                )
            )
        values.extend(
            (
                float(bool(candidate["safety_qualified"])),
                BUCKET_RANK[
                    str(candidate["trainable_parameter_bucket"])
                ]
                / 3.0,
            )
        )
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (len(self.feature_names()),):
            raise AssertionError(
                "candidate feature names and values differ"
            )
        return result

    def transform_array(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        if not candidates:
            raise ValueError("listwise candidate array is empty")
        return np.stack(
            [self.transform(candidate) for candidate in candidates]
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "feature_view": "independent_candidate",
            "feature_names": list(self.feature_names()),
            "feature_count": len(self.feature_names()),
            "pooling": "permutation_invariant_mean",
            "fit_split": "none_stateless",
            "executes_verifier_policy": False,
        }


@dataclass
class LoopedSetStatusModel:
    input_width: int
    hidden_width: int
    loop_count: int
    seed: int
    status_count: int = len(ARRAY_STATUSES)

    def __post_init__(self) -> None:
        if (
            self.input_width < 1
            or self.hidden_width < 1
            or self.loop_count < 1
            or self.status_count < 2
        ):
            raise ValueError("invalid listwise model dimensions")
        rng = np.random.default_rng(self.seed)
        input_scale = 1.0 / math.sqrt(self.input_width)
        hidden_scale = 1.0 / math.sqrt(self.hidden_width)
        self.input_weights = rng.uniform(
            -input_scale,
            input_scale,
            size=(self.hidden_width, self.input_width),
        )
        self.self_weights = (
            rng.uniform(
                -hidden_scale,
                hidden_scale,
                size=(self.hidden_width, self.hidden_width),
            )
            if self.loop_count > 1
            else None
        )
        self.pool_weights = (
            rng.uniform(
                -hidden_scale,
                hidden_scale,
                size=(self.hidden_width, self.hidden_width),
            )
            if self.loop_count > 1
            else None
        )
        self.hidden_bias = np.zeros(
            self.hidden_width,
            dtype=np.float64,
        )
        self.status_weights = rng.uniform(
            -hidden_scale,
            hidden_scale,
            size=(self.status_count, self.hidden_width),
        )
        self.status_pool_weights = rng.uniform(
            -hidden_scale,
            hidden_scale,
            size=(self.status_count, self.hidden_width),
        )
        self.status_bias = np.zeros(
            self.status_count,
            dtype=np.float64,
        )

    @property
    def parameter_count(self) -> int:
        recurrent = (
            2 * self.hidden_width**2
            if self.loop_count > 1
            else 0
        )
        return (
            self.hidden_width * self.input_width
            + recurrent
            + self.hidden_width
            + 2 * self.status_count * self.hidden_width
            + self.status_count
        )

    def _forward(
        self,
        features: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        if (
            features.ndim != 2
            or features.shape[1] != self.input_width
            or features.shape[0] < 1
        ):
            raise ValueError("invalid listwise feature matrix")
        states = [
            np.zeros(
                (features.shape[0], self.hidden_width),
                dtype=np.float64,
            )
        ]
        input_projection = (
            features @ self.input_weights.T + self.hidden_bias
        )
        for _ in range(self.loop_count):
            pre_activation = input_projection.copy()
            if (
                self.self_weights is not None
                and self.pool_weights is not None
            ):
                previous = states[-1]
                pool = np.mean(previous, axis=0)
                pre_activation += previous @ self.self_weights.T
                pre_activation += pool @ self.pool_weights.T
            states.append(
                np.tanh(
                    np.clip(pre_activation, -20.0, 20.0)
                )
            )
        final = states[-1]
        pool = np.mean(final, axis=0)
        logits = (
            final @ self.status_weights.T
            + pool @ self.status_pool_weights.T
            + self.status_bias
        )
        shifted = np.clip(
            logits - np.max(logits, axis=1, keepdims=True),
            -40.0,
            40.0,
        )
        exponentials = np.exp(shifted)
        return (
            states,
            exponentials
            / np.sum(exponentials, axis=1, keepdims=True),
        )

    def predict_proba(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return self._forward(features)[1]

    def loss(
        self,
        features: np.ndarray,
        targets: Sequence[int],
        *,
        class_weights: Sequence[float] | None = None,
    ) -> float:
        probabilities = self.predict_proba(features)
        target_array = np.asarray(targets, dtype=np.int64)
        weights = (
            np.ones(len(target_array), dtype=np.float64)
            if class_weights is None
            else np.asarray(class_weights, dtype=np.float64)[
                target_array
            ]
        )
        return float(
            np.sum(
                -np.log(
                    np.maximum(
                        probabilities[
                            np.arange(len(target_array)),
                            target_array,
                        ],
                        1e-12,
                    )
                )
                * weights
            )
            / np.sum(weights)
        )

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
        if (
            learning_rate <= 0.0
            or l2 < 0.0
            or gradient_clip <= 0.0
        ):
            raise ValueError("invalid listwise optimizer request")
        states, probabilities = self._forward(features)
        target_array = np.asarray(targets, dtype=np.int64)
        if (
            target_array.shape != (features.shape[0],)
            or np.any(target_array < 0)
            or np.any(target_array >= self.status_count)
        ):
            raise ValueError("invalid listwise targets")
        weights = np.asarray(
            class_weights,
            dtype=np.float64,
        )[target_array]
        normalizer = float(np.sum(weights))
        probabilities[
            np.arange(len(target_array)),
            target_array,
        ] -= 1.0
        logits_delta = (
            probabilities * weights[:, np.newaxis] / normalizer
        )
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
            np.sum(logits_delta, axis=0)
            @ self.status_pool_weights
        )
        hidden_delta += (
            pooled_output_delta[np.newaxis, :]
            / features.shape[0]
        )

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
        for loop_index in range(
            self.loop_count - 1,
            -1,
            -1,
        ):
            current = states[loop_index + 1]
            previous = states[loop_index]
            activation_delta = hidden_delta * (
                1.0 - current * current
            )
            input_gradient += activation_delta.T @ features
            hidden_bias_gradient += np.sum(
                activation_delta,
                axis=0,
            )
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
                hidden_delta = (
                    activation_delta @ self.self_weights
                )
                pooled_delta = (
                    np.sum(activation_delta, axis=0)
                    @ self.pool_weights
                )
                hidden_delta += (
                    pooled_delta[np.newaxis, :]
                    / features.shape[0]
                )

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
        scale = (
            min(1.0, gradient_clip / norm)
            if norm > 0.0
            else 1.0
        )
        self.input_weights -= (
            learning_rate * scale * input_gradient
        )
        self.hidden_bias -= (
            learning_rate * scale * hidden_bias_gradient
        )
        self.status_weights -= (
            learning_rate * scale * status_gradient
        )
        self.status_pool_weights -= (
            learning_rate * scale * status_pool_gradient
        )
        self.status_bias -= (
            learning_rate * scale * status_bias_gradient
        )
        if (
            self.self_weights is not None
            and self.pool_weights is not None
            and self_gradient is not None
            and pool_gradient is not None
        ):
            self.self_weights -= (
                learning_rate * scale * self_gradient
            )
            self.pool_weights -= (
                learning_rate * scale * pool_gradient
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "architecture": "looped_permutation_equivariant_status",
            "input_width": self.input_width,
            "hidden_width": self.hidden_width,
            "loop_count": self.loop_count,
            "status_count": self.status_count,
            "input_weights": self.input_weights.tolist(),
            "self_weights": (
                []
                if self.self_weights is None
                else self.self_weights.tolist()
            ),
            "pool_weights": (
                []
                if self.pool_weights is None
                else self.pool_weights.tolist()
            ),
            "hidden_bias": self.hidden_bias.tolist(),
            "status_weights": self.status_weights.tolist(),
            "status_pool_weights": (
                self.status_pool_weights.tolist()
            ),
            "status_bias": self.status_bias.tolist(),
        }


class StatusCheckpoint(Protocol):
    def __call__(
        self,
        epoch: int,
        state: Mapping[str, Any],
    ) -> None: ...


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


def train_status_model(
    model: LoopedSetStatusModel,
    encoder: IndependentCandidateEncoder,
    arrays: Sequence[Mapping[str, Any]],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    gradient_clip: float,
    seed: int,
    checkpoint_every_epochs: int,
    checkpoint_callback: StatusCheckpoint | None = None,
) -> int:
    if (
        not arrays
        or epochs < 1
        or learning_rate <= 0.0
        or l2 < 0.0
        or gradient_clip <= 0.0
        or checkpoint_every_epochs < 1
    ):
        raise ValueError("invalid bounded listwise training request")
    prepared = [
        (
            encoder.transform_array(
                [
                    candidate["features"]
                    for candidate in row["candidates"]
                ]
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
        if (
            checkpoint_callback is not None
            and (
                epoch % checkpoint_every_epochs == 0
                or epoch == epochs
            )
        ):
            checkpoint_callback(epoch, model.state_dict())
    return steps


def predict_array_statuses(
    model: LoopedSetStatusModel,
    encoder: IndependentCandidateEncoder,
    array_row: Mapping[str, Any],
) -> dict[str, Any]:
    features = encoder.transform_array(
        [
            candidate["features"]
            for candidate in array_row["candidates"]
        ]
    )
    probabilities = model.predict_proba(features)
    indices = np.argmax(probabilities, axis=1)
    return {
        "statuses": [
            ARRAY_STATUSES[int(index)] for index in indices
        ],
        "status_probabilities": [
            {
                label: float(probabilities[row_index, label_index])
                for label_index, label in enumerate(ARRAY_STATUSES)
            }
            for row_index in range(probabilities.shape[0])
        ],
    }


def generate_listwise_array_dataset(
    config_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    return generate_structured_array_dataset(
        config_path,
        out_dir,
    )
