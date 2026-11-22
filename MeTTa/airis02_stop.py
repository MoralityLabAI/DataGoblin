"""Train-only calibrated, low-capacity STOP gate for AIRIS-02 repair policies.

This is a binary research classifier trained on the independently verified
synthetic fixed-point labels. It supplies a learned *action* stopping proposal,
not an adaptive recurrent computation-time controller or a verified solution.
The threshold is frozen on TRAIN examples before development evaluation.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .airis02_student import INPUT_DIM


class StopHead:
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.02 / math.sqrt(INPUT_DIM), size=(INPUT_DIM,))
        self.b = np.array(-0.5, dtype=np.float64)
        self.threshold: float | None = None

    @property
    def params(self) -> int:
        return self.w.size + 1

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        logits = np.clip(x @ self.w + self.b, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def fit(
        self, x: np.ndarray, stop_targets: np.ndarray, *,
        steps: int, lr: float, max_false_stop_rate: float = 0.05,
    ) -> dict:
        if not 1 <= steps <= 256 or not 0.001 <= lr <= 0.4:
            raise ValueError("bounded STOP optimization required")
        if not 0.0 < max_false_stop_rate <= 0.2:
            raise ValueError("STOP false-positive target outside safe bound")
        if x.shape[1] != INPUT_DIM or len(x) != len(stop_targets):
            raise ValueError("STOP input/target shape differs")
        if not np.array_equal(stop_targets, stop_targets.astype(bool)):
            raise ValueError("STOP target must be binary")
        positives = int(np.sum(stop_targets))
        if not positives or positives == len(stop_targets):
            raise ValueError("STOP training requires both positive and negative states")
        truth = stop_targets.astype(np.float64)
        for step in range(steps):
            probs = self.probabilities(x)
            delta = (probs - truth) / len(x)
            dw = x.T @ delta + 0.0001 * self.w
            db = np.sum(delta)
            norm = math.sqrt(float(np.sum(dw * dw) + db * db))
            scale = min(1.0, 2.0 / max(norm, 1e-12))
            eta = lr / (1.0 + step / steps)
            self.w -= eta * dw * scale
            self.b -= eta * db * scale
        post = self.probabilities(x)
        negative_scores = post[stop_targets == 0]
        # Strict > quantile allows at most the upper tail of observed training
        # negatives to stop. No development label is read to tune this gate.
        limit = float(np.quantile(negative_scores, 1.0 - max_false_stop_rate))
        self.threshold = float(np.nextafter(max(limit, 0.2), np.inf))
        proposed = post >= self.threshold
        false_stop = int(np.sum(proposed & (stop_targets == 0)))
        true_stop = int(np.sum(proposed & (stop_targets == 1)))
        return {
            "schema_version": "metta-airis02-train-stop-gate-v1",
            "threshold_selected_on": "training_only",
            "threshold": round(self.threshold, 8),
            "train_positive": positives,
            "train_negative": len(stop_targets) - positives,
            "train_true_stop": true_stop,
            "train_false_stop": false_stop,
            "train_true_stop_rate": round(true_stop / positives, 6),
            "train_false_stop_rate": round(false_stop / (len(stop_targets) - positives), 6),
            "nominal_train_false_stop_bound": max_false_stop_rate,
            "no_native_act": True,
            "no_development_label_calibration": True,
            "training_ready": False,
        }

    def should_stop(self, x: np.ndarray) -> bool:
        if self.threshold is None or x.shape != (1, INPUT_DIM):
            raise ValueError("STOP gate must be fitted on the correct public state")
        return bool(self.probabilities(x)[0] >= self.threshold)

    def digest(self) -> str:
        if self.threshold is None:
            raise ValueError("unfitted STOP head has no registered checkpoint")
        data = self.w.tobytes() + self.b.tobytes() + np.asarray([self.threshold]).tobytes()
        return hashlib.sha256(data).hexdigest()
