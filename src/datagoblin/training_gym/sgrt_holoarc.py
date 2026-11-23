"""DG-SGRT-01: small, offline geometry for bounded high-dimensional excursions.

This is an engineering prototype, not a trained transformer or a claim of
creative performance. It changes a 12D coordinate while transporting a 16D
fiber through a fixed orthogonal connection. Exact checker custody belongs
to the companion gym.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np

BASE_DIM = 12
FIBER_DIM = 16
DEFAULT_STEPS = 8
ConnectionMode = Literal["flat", "curved"]

# Paired rational gears. Each 2-plane is an exact hypotrochoid over one
# closed sampled circuit. Gear phases and aspect ratios are seed-dependent.
GEARS: tuple[tuple[int, int, float], ...] = (
    (5, 2, 0.80),
    (7, 3, 0.65),
    (8, 3, 0.70),
    (9, 4, 0.55),
    (7, 2, 0.75),
    (11, 4, 0.60),
)


def _anchor(value: np.ndarray) -> np.ndarray:
    anchor = np.asarray(value, dtype=np.float64)
    if anchor.shape != (BASE_DIM,) or not np.isfinite(anchor).all():
        raise ValueError("anchor must be a finite 12D vector")
    return anchor


def arc_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def spiro_path(anchor: np.ndarray, *, seed: int, steps: int = DEFAULT_STEPS) -> np.ndarray:
    """Return a closed 12D sampled path, anchored at a task coordinate."""
    center = _anchor(anchor)
    if steps < 4:
        raise ValueError("steps must be at least four")
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, len(GEARS))
    scales = rng.uniform(0.13, 0.28, len(GEARS))
    path = np.tile(center, (steps + 1, 1))
    for t in range(steps + 1):
        for j, (outer, inner, pen) in enumerate(GEARS):
            theta = 2.0 * np.pi * inner * t / steps + phase[j]
            theta0 = phase[j]
            ratio = (outer - inner) / inner
            x = (outer - inner) * np.cos(theta) + pen * np.cos(ratio * theta)
            y = (outer - inner) * np.sin(theta) - pen * np.sin(ratio * theta)
            x0 = (outer - inner) * np.cos(theta0) + pen * np.cos(ratio * theta0)
            y0 = (outer - inner) * np.sin(theta0) - pen * np.sin(ratio * theta0)
            radius = outer - inner + pen
            path[t, 2 * j] += scales[j] * (x - x0) / radius
            path[t, 2 * j + 1] += scales[j] * (y - y0) / radius
    path[-1] = center  # exact closure despite floating-point trig error
    return path


def random_smooth_path(anchor: np.ndarray, *, seed: int, steps: int = DEFAULT_STEPS) -> np.ndarray:
    """A target-blind, closed Fourier control matched to the spiro path length."""
    center = _anchor(anchor)
    if steps < 4:
        raise ValueError("steps must be at least four")
    rng = np.random.default_rng(seed + 104729)
    t = np.arange(steps + 1, dtype=np.float64) / steps
    path = np.tile(center, (steps + 1, 1))
    for axis in range(BASE_DIM):
        freq = int(rng.integers(1, 4))
        phase = float(rng.uniform(-np.pi, np.pi))
        amp = float(rng.uniform(0.07, 0.22))
        path[:, axis] += amp * (
            np.sin(2.0 * np.pi * freq * t + phase) - np.sin(phase)
        )
    path[-1] = center
    reference = spiro_path(center, seed=seed, steps=steps)
    length = arc_length(path)
    if length <= 0.0:
        raise RuntimeError("degenerate control path")
    path = center + (path - center) * (arc_length(reference) / length)
    path[-1] = center
    return path


def plane_axes(mode: ConnectionMode) -> tuple[tuple[int, int], ...]:
    """Flat axes commute; curved axes overlap and generate nonzero brackets."""
    if mode == "flat":
        return tuple((2 * (i // 2), 2 * (i // 2) + 1) for i in range(BASE_DIM))
    if mode == "curved":
        return tuple(
            (2 * (i // 2), 2 * (i // 2) + 1)
            if i % 2 == 0
            else (2 * (i // 2) + 1, 2 * (i // 2) + 2)
            for i in range(BASE_DIM)
        )
    raise ValueError(f"unknown connection: {mode}")


def _plane_rotation(a: int, b: int, angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rotation = np.eye(FIBER_DIM, dtype=np.float64)
    rotation[a, a] = c
    rotation[b, b] = c
    rotation[a, b] = -s
    rotation[b, a] = s
    return rotation


def step_transport(delta: np.ndarray, *, mode: ConnectionMode, gain: float = 1.6) -> np.ndarray:
    """Ordered product of exact orthogonal rotations, one per base direction."""
    delta = _anchor(delta)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be positive and finite")
    result = np.eye(FIBER_DIM, dtype=np.float64)
    for (a, b), displacement in zip(plane_axes(mode), delta, strict=True):
        result = _plane_rotation(a, b, gain * float(displacement)) @ result
    return result


@dataclass(frozen=True)
class HolonomyTrace:
    path: np.ndarray
    states: np.ndarray
    holonomy: np.ndarray
    mode: ConnectionMode
    dual_pairing_drift: float

    def receipt(self) -> dict[str, object]:
        matrix = self.holonomy
        phases = sorted(float(abs(x)) for x in np.angle(np.linalg.eigvals(matrix)))
        return {
            "schema_version": "dg-sgrt-01-holonomy-v1",
            "mode": self.mode,
            "base_dim": BASE_DIM,
            "fiber_dim": FIBER_DIM,
            "steps": len(self.path) - 1,
            "path_arc_length": arc_length(self.path),
            "path_closure_l2": float(np.linalg.norm(self.path[-1] - self.path[0])),
            "orthogonality_error_fro": float(
                np.linalg.norm(matrix.T @ matrix - np.eye(FIBER_DIM), ord="fro")
            ),
            "determinant": float(np.linalg.det(matrix)),
            "identity_loss": float(1.0 - np.trace(matrix) / FIBER_DIM),
            "canonical_angles_abs_rad": phases[::2],
            "dual_pairing_drift": self.dual_pairing_drift,
        }


def trace_transport(
    path: np.ndarray,
    *,
    mode: ConnectionMode,
    start: np.ndarray | None = None,
    covector: np.ndarray | None = None,
) -> HolonomyTrace:
    """Parallel transport a feature and dual covector over the same closed path."""
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != BASE_DIM or len(points) < 5:
        raise ValueError("path requires at least four 12D edges")
    if not np.isfinite(points).all() or np.linalg.norm(points[-1] - points[0]) > 1e-9:
        raise ValueError("path must be finite and closed")
    initial = np.zeros(FIBER_DIM, dtype=np.float64) if start is None else np.asarray(start, dtype=np.float64)
    if start is None:
        initial[:4] = (0.55, -0.30, 0.45, 0.20)
    dual = initial.copy() if covector is None else np.asarray(covector, dtype=np.float64)
    if initial.shape != (FIBER_DIM,) or dual.shape != (FIBER_DIM,):
        raise ValueError("feature and covector must be 16D")
    if not np.isfinite(initial).all() or not np.isfinite(dual).all():
        raise ValueError("feature and covector must be finite")
    original_pairing = float(dual @ initial)
    state = initial.copy()
    states = [state.copy()]
    product = np.eye(FIBER_DIM, dtype=np.float64)
    maximum_drift = 0.0
    for before, after in pairwise(points):
        q = step_transport(after - before, mode=mode)
        product = q @ product
        state = q @ state
        dual = q @ dual  # Q^{-T}=Q for orthogonal Q
        states.append(state.copy())
        maximum_drift = max(maximum_drift, abs(float(dual @ state) - original_pairing))
    return HolonomyTrace(points, np.stack(states), product, mode, maximum_drift)


def transported_residual(state: np.ndarray, holonomy: np.ndarray, *, gate: float = 0.1) -> np.ndarray:
    """Removable operator-sidecar sketch; no learned weights are implied."""
    value = np.asarray(state, dtype=np.float64)
    if value.ndim != 1 or not 1 <= value.size <= FIBER_DIM:
        raise ValueError("state must be a vector of length 1..16")
    if not 0 <= gate <= 1:
        raise ValueError("gate must be in [0, 1]")
    if holonomy.shape != (FIBER_DIM, FIBER_DIM):
        raise ValueError("holonomy must be 16 by 16")
    lifted = np.zeros(FIBER_DIM, dtype=np.float64)
    lifted[: value.size] = value
    return value + gate * ((holonomy @ lifted - lifted)[: value.size])
