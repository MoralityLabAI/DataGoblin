"""MOLT-05: frozen recurrent-write component restoration and matched shams.

This is an SVD instrument on the MOLT-04 scorer, not native BPD.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datagoblin.training_gym.molt_goblin import digest
from datagoblin.training_gym.molt_retention_train import TiedScorer, features

TOL = 1e-9
RANK = 4
POLICIES = ("intact", "reconstruction", "corrupted", "full_restore", "targeted",
            "equal_rank", "norm_sham", "jacobian_radius_sham")


@dataclass(frozen=True)
class WriteFrame:
    fixed: np.ndarray
    source: np.ndarray
    clean: np.ndarray
    corrupted: np.ndarray
    components: np.ndarray
    donor_indices: np.ndarray


def factorize(model: TiedScorer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.linalg.svd(model.params["h"], full_matrices=False)


def frame(model: TiedScorer, x: np.ndarray, root_id: str,
          factors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> WriteFrame:
    """Corrupt only the pass-2 recurrent preactivation contribution.

    All entries share the public task cue. A fixed nonidentity cyclic permutation
    uses other entries as donor writes, independent of the eventual query.
    """
    p = model.params
    fixed = x @ p["x"] + p["b"]
    h1 = np.tanh(fixed)
    u, s, vt = factors
    components = ((h1 @ u) * s)[:, :, None] * vt[None, :, :]
    source = h1 @ p["h"]
    shift = 1 + int(digest([root_id, "molt05-donor"]), 16) % (len(x) - 1)
    donors = (np.arange(len(x)) + shift) % len(x)
    return WriteFrame(fixed, source, fixed + source, fixed + source[donors], components, donors)


def readout(model: TiedScorer, fixed: np.ndarray, z2: np.ndarray) -> np.ndarray:
    p = model.params
    h3 = np.tanh(fixed + np.tanh(z2) @ p["h"])
    return h3 @ p["o"] + p["c"][0]


def jacobian(model: TiedScorer, fixed: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Exact gradient of each future scalar score at its own pass-2 recipient."""
    p = model.params
    h2 = np.tanh(z2)
    h3 = np.tanh(fixed + h2 @ p["h"])
    return ((1 - h3**2) * p["o"]) @ p["h"].T * (1 - h2**2)


def component_deltas(f: WriteFrame) -> np.ndarray:
    return f.components - f.components[f.donor_indices]


def choose_components(model: TiedScorer, train_worlds,
                      factors: tuple[np.ndarray, np.ndarray, np.ndarray], rank: int = RANK):
    """Rank TRAIN component directions by mean squared signed readout displacement."""
    if not train_worlds or any(w.split != "train" for w in train_worlds):
        raise ValueError("TRAIN-only component selection")
    if not 1 <= rank < model.params["h"].shape[0]:
        raise ValueError("invalid component rank")
    score = np.zeros(model.params["h"].shape[0])
    count = 0
    for w in train_worlds:
        f = frame(model, features(w), w.root_id, factors)
        j = jacobian(model, f.fixed, f.corrupted)
        contribution = np.einsum("nd,nkd->nk", j, component_deltas(f))
        score += (contribution**2).sum(0)
        count += len(j)
    score /= count
    ids = tuple(sorted(range(len(score)), key=lambda i: (-score[i], i))[:rank])
    return ids, score


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _row_space(constraints: np.ndarray) -> np.ndarray:
    # Normalize constraints before determining rank; this keeps small nonzero
    # gradients from disappearing simply because the recipient has large norm.
    norms = np.linalg.norm(constraints, axis=1)
    normalized = constraints[norms > 1e-14] / norms[norms > 1e-14, None]
    if not len(normalized):
        return np.zeros((0, constraints.shape[1]))
    _, s, vt = np.linalg.svd(normalized, full_matrices=False)
    return vt[s > 1e-11]


def matched_vector(delta: np.ndarray, j: np.ndarray, z: np.ndarray,
                   rng: np.random.Generator, *, strong: bool):
    """Distinct sham matching displacement; strong also matches signed Jd and radius.

    Matching z dot d in addition to norm(d) matches norm(z+d). The constraint
    row space contains the full future-output Jacobian for this independent
    per-entry scorer (the full all-entry Jacobian is block diagonal).
    """
    dose = _norm(delta)
    if dose <= 1e-12:
        return delta.copy(), "zero_dose"
    basis = _row_space(np.stack((j, z))) if strong else np.zeros((0, len(z)))
    projection = (basis @ delta) @ basis
    remainder = delta - projection
    radius = _norm(remainder)
    if radius <= 1e-10 * max(1.0, dose):
        return np.zeros_like(delta), "no_distinct_match"
    # Deterministic random direction orthogonal to both the constraints and
    # the target's nullspace component. No outcome-conditioned search.
    direction = rng.standard_normal(len(z))
    direction -= (basis @ direction) @ basis
    direction -= np.dot(direction, remainder) / radius**2 * remainder
    length = _norm(direction)
    if length <= 1e-12:
        return np.zeros_like(delta), "no_distinct_match"
    sham = projection + radius * direction / length
    if abs(_norm(sham) - dose) > TOL or _norm(sham - delta) <= 1e-10:
        return sham, "invalid_match"
    if strong and (abs(float(j @ (sham - delta))) > TOL
                   or abs(_norm(z + sham) - _norm(z + delta)) > TOL):
        return sham, "invalid_match"
    return sham, "matched"


def sham_batch(delta: np.ndarray, j: np.ndarray, z: np.ndarray, rng: np.random.Generator, *, strong: bool):
    vectors, statuses = [], []
    for d, grad, base in zip(delta, j, z, strict=True):
        v, status = matched_vector(d, grad, base, rng, strong=strong)
        vectors.append(v)
        statuses.append(status)
    result = np.stack(vectors)
    active = [i for i, status in enumerate(statuses) if status != "zero_dose"]
    good = all(status in {"matched", "zero_dose"} for status in statuses)
    return result, {
        "all_matched": good, "active": len(active), "matched": statuses.count("matched"),
        "zero_dose": statuses.count("zero_dose"), "unmatched": sum(
            status not in {"matched", "zero_dose"} for status in statuses),
        "max_norm_error": float(np.max(np.abs(np.linalg.norm(result, axis=1) - np.linalg.norm(delta, axis=1)))),
        "max_signed_j_error": float(np.max(np.abs(np.sum(j * (result - delta), axis=1)))),
        "max_radius_error": float(np.max(np.abs(np.linalg.norm(z + result, axis=1)
                                               - np.linalg.norm(z + delta, axis=1)))),
        "mean_direction_distance": float(np.linalg.norm(result - delta, axis=1).mean()),
    }


def interventions(model: TiedScorer, x: np.ndarray, root_id: str,
                  factors: tuple[np.ndarray, np.ndarray, np.ndarray], selected: tuple[int, ...]):
    f = frame(model, x, root_id, factors)
    reconstructed = f.fixed + f.components.sum(1)
    reference = model.forward(x, 3)[0]
    reconstruction_error = float(np.max(np.abs(f.source - f.components.sum(1))))
    native_error = float(np.max(np.abs(reference - readout(model, f.fixed, reconstructed))))
    if max(reconstruction_error, native_error) > TOL:
        raise RuntimeError("reconstruction parity failed")
    components = component_deltas(f)
    target = components[:, list(selected)].sum(1)
    other = [i for i in range(components.shape[1]) if i not in selected]
    rng = np.random.default_rng(int(digest([root_id, "molt05-rank-sham"]), 16))
    ids = rng.choice(other, size=len(selected), replace=False)
    alternate = components[:, ids].sum(1)
    target_norm = np.linalg.norm(target, axis=1)
    alt_norm = np.linalg.norm(alternate, axis=1)
    valid = (target_norm <= 1e-12) | (alt_norm > 1e-12)
    ratio = np.divide(target_norm, alt_norm, out=np.zeros_like(target_norm), where=alt_norm > 1e-12)
    alternate *= ratio[:, None]
    complete = f.source - f.source[f.donor_indices]
    base_scores = {
        "intact": reference,
        "reconstruction": readout(model, f.fixed, reconstructed),
        "corrupted": readout(model, f.fixed, f.corrupted),
        "full_restore": readout(model, f.fixed, f.corrupted + complete),
    }
    conditions = {}
    audits = {}
    for condition, base in (("corrupt", f.corrupted), ("clean", f.clean)):
        j = jacobian(model, f.fixed, base)
        strong_rng = np.random.default_rng(int(digest([root_id, condition, "molt05-strong"]), 16))
        norm_rng = np.random.default_rng(int(digest([root_id, condition, "molt05-norm"]), 16))
        strong, strong_audit = sham_batch(target, j, base, strong_rng, strong=True)
        norm, norm_audit = sham_batch(target, j, base, norm_rng, strong=False)
        # Full-roster conservative execution: an infeasible sham uses the
        # unsteered baseline for that condition, and stays in the denominator.
        scores = dict(base_scores)
        for name, delta, eligible in (("targeted", target, True), ("equal_rank", alternate, bool(valid.all())),
                                      ("norm_sham", norm, norm_audit["all_matched"]),
                                      ("jacobian_radius_sham", strong, strong_audit["all_matched"])):
            scores[name] = readout(model, f.fixed, base + delta if eligible else base)
        conditions[condition] = scores
        audits[condition] = {"strong": strong_audit, "norm": norm_audit,
                             "equal_rank_valid": bool(valid.all()),
                             "max_equal_rank_norm_error": float(np.max(np.abs(target_norm
                                                                            - np.linalg.norm(alternate, axis=1))))}
    return conditions, {"write_reconstruction_max_abs": reconstruction_error,
                        "score_reconstruction_max_abs": native_error,
                        "full_restore_max_abs": float(np.max(np.abs(reference - base_scores["full_restore"]))),
                        "selected": list(selected), "equal_rank_components": ids.tolist(),
                        "mean_target_norm": float(target_norm.mean()), "nonzero_dose_entries": int(
                            np.count_nonzero(target_norm > 1e-12)), "conditions": audits}
