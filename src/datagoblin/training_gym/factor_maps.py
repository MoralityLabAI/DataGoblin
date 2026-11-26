"""Canonical factor-map normalization, selection invariance, and portable manifests.

Implements the exact canonical factor convention for both live SVD and exported
sparse factorizations:

    W_hat = (V U)^T,   n_j = ||U_j||_2
    U'_j = U_j / n_j,  V'[:, j] = V[:, j] * n_j   (for n_j > eps)
    U'_j = 0,          V'[:, j] = 0               (for n_j <= eps, excluded)

Sign-canonicalizes active components and orders columns deterministically by
physical component identity so top-k component selection and injected gradient
projections are invariant to nonzero reciprocal rescalings and column permutations.

Pure NumPy + stdlib so base CI runs without PyTorch.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FACTOR_CONVENTION = "W_hat=(V@U).T; n_j=||U_j||_2; U'_j=U_j/n_j; V'[:,j]=V[:,j]*n_j; sign_max_abs; sorted_id"
REQUIRED_MANIFEST_FIELDS = (
    "path",
    "artifact_sha256",
    "source_checkpoint_sha256",
    "training_step",
    "weight_key",
    "site_identity",
    "factor_convention",
    "fit_settings",
    "source_data_split_digest",
    "code_commit",
)


@dataclass(frozen=True)
class CanonicalFactorMap:
    """Canonicalized (V', U') factor map with stable physical component identities."""

    v: np.ndarray  # [d_in, C] float32
    u: np.ndarray  # [C, d_out] float32 (active rows unit-norm, zero rows all-zero)
    active: np.ndarray  # [C] bool
    component_ids: tuple[str, ...]
    basis_digest: str
    zero_count: int


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _component_id(u_row: np.ndarray, v_col: np.ndarray) -> str:
    """Deterministic physical identity for a sign-canonicalized active component."""
    q_u = np.round(u_row.astype(np.float64), 4).tobytes()
    q_v = np.round(v_col.astype(np.float64), 4).tobytes()
    return hashlib.sha256(q_u + q_v).hexdigest()[:16]


def canonicalize_factors(
    v: np.ndarray,
    u: np.ndarray,
    *,
    eps: float = 1e-12,
) -> CanonicalFactorMap:
    """Canonicalize (V[in, C], U[C, out]) under W_hat = (V @ U)^T.

    Invariant to arbitrary nonzero reciprocal rescalings (V[:,j]*a_j, U[j,:]/a_j)
    and arbitrary component column/row permutations.
    """
    v_arr = np.array(v, dtype=np.float64, copy=True)
    u_arr = np.array(u, dtype=np.float64, copy=True)
    if v_arr.ndim != 2 or u_arr.ndim != 2 or v_arr.shape[1] != u_arr.shape[0]:
        raise ValueError(f"factor shape mismatch: V={v_arr.shape}, U={u_arr.shape}")
    if not (np.isfinite(v_arr).all() and np.isfinite(u_arr).all()):
        raise ValueError("non-finite values in factor matrices")

    n_comp = u_arr.shape[0]
    u_norms = np.linalg.norm(u_arr, axis=1)
    v_norms = np.linalg.norm(v_arr, axis=0)
    active_mask = (u_norms > eps) & (v_norms > eps)

    v_c = np.zeros_like(v_arr, dtype=np.float64)
    u_c = np.zeros_like(u_arr, dtype=np.float64)
    ids: list[str] = []

    for j in range(n_comp):
        if not active_mask[j]:
            ids.append(f"zero_{j:04d}")
            continue
        nj = u_norms[j]
        uj = u_arr[j] / nj
        vj = v_arr[:, j] * nj
        # Sign-canonicalize using the first maximum-magnitude entry of U'_j
        pivot = int(np.argmax(np.abs(uj)))
        if uj[pivot] < 0:
            uj = -uj
            vj = -vj
        u_c[j] = uj
        v_c[:, j] = vj
        ids.append(_component_id(uj, vj))

    # Order active components canonically by physical component_id, zeros last
    order = sorted(range(n_comp), key=lambda j: (0 if active_mask[j] else 1, ids[j], j))
    v_ord = np.ascontiguousarray(v_c[:, order].astype(np.float32))
    u_ord = np.ascontiguousarray(u_c[order, :].astype(np.float32))
    act_ord = np.ascontiguousarray(active_mask[order])
    ids_ord = tuple(ids[j] for j in order)

    digest_h = hashlib.sha256()
    digest_h.update(b"|".join(cid.encode("ascii") for cid in ids_ord))
    return CanonicalFactorMap(
        v=v_ord,
        u=u_ord,
        active=act_ord,
        component_ids=ids_ord,
        basis_digest=digest_h.hexdigest(),
        zero_count=int((~act_ord).sum()),
    )


def canonical_svd_factors(weight: np.ndarray, rank: int, *, eps: float = 1e-12) -> CanonicalFactorMap:
    """Compute canonicalized truncated SVD factors matching W_hat = (V @ U)^T."""
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim != 2 or rank < 1:
        raise ValueError("weight must be 2D and rank >= 1")
    left, s, right = np.linalg.svd(w, full_matrices=False)
    n = min(int(rank), len(s))
    root = np.sqrt(np.maximum(s[:n], 0.0))
    v = right[:n, :].T * root[None, :]
    u = root[:, None] * left[:, :n].T
    return canonicalize_factors(v, u, eps=eps)


def select_and_project_numpy(
    inputs: np.ndarray,
    hist_grad: np.ndarray,
    cur_grad: np.ndarray,
    fmap: CanonicalFactorMap,
    *,
    topk: int,
    mode: str = "bpd",
    ratio: float = 0.2,
    tie_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select top-k physical components and compute injected gradient delta (reference NumPy).

    Returns (selected_indices [B, k], delta [B, ...], admitted [B]).
    """
    if mode not in ("bpd", "dense", "sham"):
        raise ValueError(f"unsupported mode: {mode}")
    h = np.asarray(hist_grad, dtype=np.float32)
    c = np.asarray(cur_grad, dtype=np.float32)
    x = np.asarray(inputs, dtype=np.float32)
    b = h.shape[0]
    h3 = h.reshape(b, -1, fmap.u.shape[1])
    c3 = c.reshape(b, -1, fmap.u.shape[1])
    x3 = x.reshape(b, -1, fmap.v.shape[0])

    reads = np.abs(x3 @ fmap.v).mean(axis=1)  # [B, C]
    coeff = np.abs(h3 @ fmap.u.T).mean(axis=1)  # [B, C]
    scores = reads * coeff
    scores[:, ~fmap.active] = -np.inf

    n_active = int(fmap.active.sum())
    k = min(int(topk), n_active)
    if k == 0:
        return np.zeros((b, 0), dtype=np.int64), np.zeros_like(c), np.zeros(b, dtype=bool)

    # Vectorized deterministic tie-breaking: fmap columns are already canonically sorted
    # by (active_first, component_id), so stable argsort on -round(scores, tie_decimals)
    # breaks all ties identically without any Python loop over batch rows!
    rounded = np.where(np.isfinite(scores), -np.round(scores, tie_decimals), np.inf)
    sel_idx = np.argsort(rounded, axis=1, kind="stable")[:, :k].astype(np.int64)
    u_sel = fmap.u[sel_idx]  # [B, k, d_out]
    # projected = (h3 @ u_sel^T) @ u_sel via vectorized matmul
    coeffs = np.matmul(h3, np.swapaxes(u_sel, 1, 2))  # [B, T, k]
    projected = np.matmul(coeffs, u_sel)  # [B, T, d_out]

    p_flat = projected.reshape(b, -1)
    c_flat = c3.reshape(b, -1)
    p_norm = np.linalg.norm(p_flat, axis=1)
    c_norm = np.linalg.norm(c_flat, axis=1)
    dot = np.sum(p_flat * c_flat, axis=1)
    admitted = (p_norm > 1e-12) & (dot > 0) & (c_norm > 1e-12)

    if mode == "dense":
        chosen = h3
    elif mode == "sham":
        chosen = np.roll(projected, 1, axis=-1)
    else:
        chosen = projected
    z = chosen.reshape(b, -1)
    z_norm = np.linalg.norm(z, axis=1)
    eligible = admitted & (z_norm > 1e-12)
    scale = (ratio * c_norm * eligible.astype(np.float32)) / np.maximum(z_norm, 1e-12)
    delta = (z * scale[:, None]).reshape(c.shape)
    return sel_idx, delta, eligible


def write_factor_manifest(
    manifest_path: Path,
    site_records: dict[str, dict[str, Any]],
) -> Path:
    """Write a portable manifest using paths relative to ``manifest_path.parent``."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, dict[str, Any]] = {}
    for site, rec in site_records.items():
        missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in rec]
        if missing:
            raise ValueError(f"manifest record for {site!r} missing required fields: {missing}")
        art_path = Path(rec["path"])
        if art_path.is_absolute():
            rel = os.path.relpath(art_path, manifest_path.parent).replace("\\", "/")
        else:
            rel = str(art_path).replace("\\", "/")
        out_rec = dict(rec)
        out_rec["path"] = rel
        out_rec["sha256"] = rec["artifact_sha256"]  # backward-compatible alias for LoopedTransformers loader
        normalized[site] = out_rec
    manifest_path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_factor_manifest(manifest_path: Path, *, expected_sha256: str | None = None) -> dict[str, dict[str, Any]]:
    """Load and verify a portable factor manifest, preserving all provenance metadata."""
    manifest_path = Path(manifest_path)
    if expected_sha256 is not None and file_sha256(manifest_path) != expected_sha256:
        raise ValueError("factor manifest SHA-256 mismatch")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("manifest must map site names to metadata dictionaries")
    out: dict[str, dict[str, Any]] = {}
    for site, rec in data.items():
        missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in rec]
        if missing:
            raise ValueError(f"manifest record for {site!r} missing required fields: {missing}")
        rel_path = Path(rec["path"])
        if rel_path.is_absolute():
            raise ValueError(f"manifest path for {site!r} must be manifest-relative, got {rec['path']!r}")
        resolved = (manifest_path.parent / rel_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"factor artifact missing: {resolved}")
        actual_sha = file_sha256(resolved)
        if actual_sha != rec["artifact_sha256"]:
            raise ValueError(f"factor artifact SHA-256 mismatch for {site!r}")
        loaded_rec = dict(rec)
        loaded_rec["resolved_path"] = str(resolved)
        out[site] = loaded_rec
    return out
