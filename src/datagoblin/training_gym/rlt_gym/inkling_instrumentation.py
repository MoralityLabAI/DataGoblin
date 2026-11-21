"""Probe contract for instrumented loop runtimes, independent of teacher capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class LoopBoundary:
    task_digest: str
    pass_index: int
    total_passes: int
    layer: int
    token: int
    component: str


@dataclass(frozen=True)
class SparseIntervention:
    kind: Literal["lesion", "donor_patch", "sham", "random"]
    feature_ids: tuple[int, ...]
    basis_digest: str
    dose: float
    seed: int
    donor_task_digest: str | None = None


class LoopProbe(Protocol):
    """Expose the same one-pass map, including carried cache state, for paired probes."""

    def activation(self, point: LoopBoundary) -> Vector: ...

    def advance(self, point: LoopBoundary, state: Vector, *, intervention: SparseIntervention | None = None
                ) -> Vector: ...

    def jvp(self, point: LoopBoundary, state: Vector, tangent: Vector) -> Vector: ...

    def vjp(self, point: LoopBoundary, state: Vector, cotangent: Vector) -> Vector: ...


def stability_sample(
    probe: LoopProbe, point: LoopBoundary, *, seed: int = 7, dose: float = 1e-4, iterations: int = 8,
) -> dict:
    """Return a sampled local spectral-gain estimate and one finite-dose matched probe."""
    if not np.isfinite(dose) or dose <= 0 or type(iterations) is not int or not 1 <= iterations <= 32:
        raise ValueError("invalid stability probe budget")
    state = np.asarray(probe.activation(point), dtype=np.float64)
    if state.ndim != 1 or not state.size or not np.all(np.isfinite(state)):
        raise ValueError("expected one finite flattened loop state")

    def checked(vector: Vector) -> Vector:
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != state.shape or not np.all(np.isfinite(vector)):
            raise ValueError("probe returned an invalid loop state")
        return vector

    direction = np.random.default_rng(seed).normal(size=state.shape)
    direction /= np.linalg.norm(direction)
    for _ in range(iterations):
        gradient = checked(probe.vjp(point, state, checked(probe.jvp(point, state, direction))))
        size = np.linalg.norm(gradient)
        if size == 0:
            break
        direction = gradient / size
    ordinary = checked(probe.advance(point, state))
    perturbed = checked(probe.advance(point, state + dose * direction))
    return {
        "state_norm": float(np.linalg.norm(state)),
        "update_norm": float(np.linalg.norm(ordinary - state)),
        "spectral_gain_estimate": float(np.linalg.norm(checked(probe.jvp(point, state, direction)))),
        "finite_dose_directional_gain": float(np.linalg.norm(perturbed - ordinary) / dose),
        "dose": dose, "seed": seed, "power_iterations_budget": iterations,
        "contraction_certified": False,
    }
