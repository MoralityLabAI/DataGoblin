from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from datagoblin.hashing import sha256_json


@dataclass
class NumpyLogisticLearner:
    learning_rate: float = 0.25
    steps: int = 600
    l2: float = 0.001
    weights: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> NumpyLogisticLearner:
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("invalid feature or label shapes")
        weights = np.zeros(features.shape[1], dtype=np.float64)
        for _ in range(self.steps):
            logits = np.clip(features @ weights, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = features.T @ (probabilities - labels) / len(labels) + self.l2 * weights
            weights -= self.learning_rate * gradient
        self.weights = weights
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("learner is not fitted")
        logits = np.clip(features @ self.weights, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        probabilities = self.predict_proba(features)
        predictions = probabilities >= 0.5
        return {
            "accuracy": float(np.mean(predictions == labels)),
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "log_loss": float(
                -np.mean(
                    labels * np.log(np.clip(probabilities, 1e-9, 1.0))
                    + (1 - labels) * np.log(np.clip(1 - probabilities, 1e-9, 1.0))
                )
            ),
        }

    def receipt(self) -> dict[str, Any]:
        if self.weights is None:
            raise ValueError("learner is not fitted")
        payload = {
            "algorithm": "numpy_logistic_gradient_descent",
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "l2": self.l2,
            "weights": self.weights.tolist(),
        }
        return {**payload, "sha256": sha256_json(payload)}
