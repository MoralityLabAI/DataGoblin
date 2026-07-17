from __future__ import annotations

from typing import Protocol

import numpy as np


class Learner(Protocol):
    def fit(self, features: np.ndarray, labels: np.ndarray) -> Learner: ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...
