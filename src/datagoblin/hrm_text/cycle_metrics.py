from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

EPSILON = 1e-12


def safe_relative_update(previous: np.ndarray, current: np.ndarray, *, epsilon: float = EPSILON) -> float:
    previous_value = np.asarray(previous, dtype=np.float64)
    current_value = np.asarray(current, dtype=np.float64)
    if previous_value.shape != current_value.shape:
        raise ValueError("recurrent states must have equal shapes")
    denominator = float(np.linalg.norm(previous_value))
    numerator = float(np.linalg.norm(current_value - previous_value))
    if denominator <= epsilon and numerator <= epsilon:
        return 0.0
    return numerator / max(denominator, epsilon)


def pooled_update(previous: np.ndarray, current: np.ndarray, *, epsilon: float = EPSILON) -> float:
    previous_tokens = _token_matrix(previous)
    current_tokens = _token_matrix(current)
    return safe_relative_update(previous_tokens.mean(axis=0), current_tokens.mean(axis=0), epsilon=epsilon)


def token_update_norms(previous: np.ndarray, current: np.ndarray, *, epsilon: float = EPSILON) -> np.ndarray:
    previous_tokens = _token_matrix(previous)
    current_tokens = _token_matrix(current)
    if previous_tokens.shape != current_tokens.shape:
        raise ValueError("token state shapes must match")
    numerator = np.linalg.norm(current_tokens - previous_tokens, axis=-1)
    denominator = np.linalg.norm(previous_tokens, axis=-1)
    both_zero = (numerator <= epsilon) & (denominator <= epsilon)
    values = numerator / np.maximum(denominator, epsilon)
    values[both_zero] = 0.0
    return values


def _token_matrix(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2:
        return array.reshape(1, -1)
    return array.reshape(-1, array.shape[-1])


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values)
    exponential = np.exp(np.clip(values, -700.0, 0.0))
    return exponential / max(float(exponential.sum()), EPSILON)


def jensen_shannon_from_logits(first: np.ndarray, second: np.ndarray) -> float:
    first_probabilities = softmax(_last_logits(first))
    second_probabilities = softmax(_last_logits(second))
    if first_probabilities.shape != second_probabilities.shape:
        raise ValueError("logit projections must have equal shapes")
    midpoint = 0.5 * (first_probabilities + second_probabilities)

    def divergence(values: np.ndarray) -> float:
        selected = values > 0.0
        return float(np.sum(values[selected] * np.log(values[selected] / midpoint[selected])))

    return 0.5 * (divergence(first_probabilities) + divergence(second_probabilities))


def _last_logits(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        return array
    return array.reshape(-1, array.shape[-1])[-1]


def exponential_decay_estimate(updates: Sequence[float], *, epsilon: float = EPSILON) -> float:
    if len(updates) < 2:
        return 0.0
    x = np.arange(len(updates), dtype=np.float64)
    y = np.log(np.maximum(np.asarray(updates, dtype=np.float64), epsilon))
    if np.allclose(y, y[0]):
        return 0.0
    slope = float(np.polyfit(x, y, 1)[0])
    return -slope


def entropy(probabilities: Mapping[str, float] | Sequence[float]) -> float:
    values = np.asarray(
        list(probabilities.values()) if isinstance(probabilities, Mapping) else probabilities,
        dtype=np.float64,
    )
    total = float(values.sum())
    if total <= EPSILON:
        return 0.0
    values = values / total
    selected = values > 0.0
    return float(-np.sum(values[selected] * np.log(values[selected])))


def top_probability_margin(probabilities: Mapping[str, float]) -> tuple[float, float]:
    values = sorted((float(value) for value in probabilities.values()), reverse=True)
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    return values[0], values[0] - values[1]


@dataclass(frozen=True)
class CycleFeatureValues:
    relative_h1_h2_update_norm: float
    mean_final_l_update_norm: float
    maximum_token_level_late_update: float
    exponential_update_decay_estimate: float
    h1_h2_js_logit_divergence: float
    residual_norm_growth: float

    def to_jsonable(self) -> dict[str, float]:
        return {
            "relative_h1_h2_update_norm": self.relative_h1_h2_update_norm,
            "mean_final_l_update_norm": self.mean_final_l_update_norm,
            "maximum_token_level_late_update": self.maximum_token_level_late_update,
            "exponential_update_decay_estimate": self.exponential_update_decay_estimate,
            "h1_h2_js_logit_divergence": self.h1_h2_js_logit_divergence,
            "residual_norm_growth": self.residual_norm_growth,
        }


def compute_cycle_features(
    states: Mapping[str, np.ndarray],
    *,
    step_updates: Sequence[float],
    h1_logits: np.ndarray | None,
    h2_logits: np.ndarray | None,
) -> CycleFeatureValues:
    missing = [name for name in ("H1", "H2", "L4", "L5", "L6") if name not in states]
    if missing:
        raise ValueError(f"missing recurrent states: {', '.join(missing)}")
    h1 = states["H1"]
    h2 = states["H2"]
    final_l_updates = (
        safe_relative_update(states["L4"], states["L5"]),
        safe_relative_update(states["L5"], states["L6"]),
    )
    late_token_updates = token_update_norms(h1, h2)
    h1_norm = float(np.linalg.norm(h1))
    residual_growth = (float(np.linalg.norm(h2)) - h1_norm) / max(h1_norm, EPSILON)
    js_divergence = (
        jensen_shannon_from_logits(h1_logits, h2_logits)
        if h1_logits is not None and h2_logits is not None
        else 0.0
    )
    return CycleFeatureValues(
        relative_h1_h2_update_norm=safe_relative_update(h1, h2),
        mean_final_l_update_norm=float(np.mean(final_l_updates)),
        maximum_token_level_late_update=float(np.max(late_token_updates)) if len(late_token_updates) else 0.0,
        exponential_update_decay_estimate=exponential_decay_estimate(step_updates),
        h1_h2_js_logit_divergence=js_divergence,
        residual_norm_growth=residual_growth if math.isfinite(residual_growth) else 0.0,
    )

