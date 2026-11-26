"""Regression tests for canonical factor-map normalization, invariance, and portable manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datagoblin.training_gym.factor_maps import (
    FACTOR_CONVENTION,
    canonical_svd_factors,
    canonicalize_factors,
    file_sha256,
    load_factor_manifest,
    select_and_project_numpy,
    write_factor_manifest,
)


def test_reciprocal_rescaling_and_permutation_invariance() -> None:
    """Equivalent factorizations (rescaled, sign-flipped, permuted) must select identical directions & deltas."""
    rng = np.random.default_rng(2026)
    d_in, d_out, rank = 24, 32, 8
    v0 = rng.standard_normal((d_in, rank)).astype(np.float64)
    u0 = rng.standard_normal((rank, d_out)).astype(np.float64)

    # Arbitrary nonzero rescalings (including negative flips and wide dynamic range 1e-3..1e3)
    scales = np.array([1e-3, -50.0, 0.07, 400.0, -0.004, 12.5, -1.0, 88.0], dtype=np.float64)
    perm = np.array([5, 0, 7, 2, 6, 1, 4, 3])
    v1 = (v0 * scales[None, :])[:, perm]
    u1 = (u0 / scales[:, None])[perm, :]

    # Verify W_hat = (V @ U)^T is identical before canonicalization
    assert np.allclose((v0 @ u0).T, (v1 @ u1).T, atol=1e-9)

    f0 = canonicalize_factors(v0, u0)
    f1 = canonicalize_factors(v1, u1)

    assert f0.basis_digest == f1.basis_digest
    assert f0.component_ids == f1.component_ids
    assert np.allclose(f0.u, f1.u, atol=1e-5)
    assert np.allclose(f0.v, f1.v, atol=1e-4)

    # Verify selected physical directions and injected gradient delta match identically
    x = rng.standard_normal((16, 4, d_in)).astype(np.float32)
    h = rng.standard_normal((16, 4, d_out)).astype(np.float32)
    g = h + 0.3 * rng.standard_normal((16, 4, d_out)).astype(np.float32)

    idx0, d0, adm0 = select_and_project_numpy(x, h, g, f0, topk=3, mode="bpd", ratio=0.2)
    idx1, d1, adm1 = select_and_project_numpy(x, h, g, f1, topk=3, mode="bpd", ratio=0.2)
    assert np.array_equal(idx0, idx1)
    assert np.array_equal(adm0, adm1)
    assert np.allclose(d0, d1, atol=1e-5)


def test_zero_components_excluded_and_ties_deterministic() -> None:
    d_in, d_out = 8, 8
    v = np.eye(d_in, 4, dtype=np.float64)
    u = np.eye(4, d_out, dtype=np.float64)
    u[1, :] = 0.0  # zero component
    v[:, 2] = 0.0  # zero component

    fmap = canonicalize_factors(v, u)
    assert fmap.zero_count == 2
    assert int(fmap.active.sum()) == 2

    # Construct exact score tie between the 2 active components
    x = np.ones((4, 1, d_in), dtype=np.float32)
    h = np.ones((4, 1, d_out), dtype=np.float32)
    g = np.ones((4, 1, d_out), dtype=np.float32)

    idx_a, delta_a, _ = select_and_project_numpy(x, h, g, fmap, topk=1)
    # Permute columns and re-canonicalize
    perm = [3, 2, 1, 0]
    fmap_p = canonicalize_factors(v[:, perm], u[perm, :])
    idx_b, delta_b, _ = select_and_project_numpy(x, h, g, fmap_p, topk=1)
    assert np.array_equal(idx_a, idx_b)
    assert np.allclose(delta_a, delta_b, atol=1e-6)
    assert all(fmap.active[i] for i in idx_a.ravel())


def test_canonical_svd_and_portable_manifest_runs_map_pt(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    w = rng.standard_normal((16, 12)).astype(np.float32)
    fmap = canonical_svd_factors(w, rank=6)
    assert fmap.u.shape == (6, 16) and fmap.v.shape == (12, 6)
    assert np.allclose(np.linalg.norm(fmap.u[fmap.active], axis=1), 1.0, atol=1e-5)

    # Test portable manifest-relative path like runs/map.pt
    art = tmp_path / "runs" / "map.pt"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"dummy-factor-bytes")
    sha = file_sha256(art)
    manifest = tmp_path / "runs" / "map.manifest.json"
    write_factor_manifest(
        manifest,
        {
            "block": {
                "path": str(art),
                "artifact_sha256": sha,
                "source_checkpoint_sha256": "ckpt_sha",
                "training_step": 100,
                "weight_key": "blocks.0.weight",
                "site_identity": "block",
                "factor_convention": FACTOR_CONVENTION,
                "fit_settings": {"components": 6, "steps": 50},
                "source_data_split_digest": "split_sha",
                "code_commit": "0fb4e54",
            }
        },
    )
    loaded = load_factor_manifest(manifest)
    assert loaded["block"]["path"] == "map.pt"  # portable relative path, NOT absolute!
    assert loaded["block"]["training_step"] == 100
    assert loaded["block"]["factor_convention"] == FACTOR_CONVENTION

    # Reject absolute path inside manifest on load
    manifest.write_text(manifest.read_text().replace('"map.pt"', f'"{str(art).replace(chr(92), "/")}"'))
    with pytest.raises(ValueError, match="manifest-relative"):
        load_factor_manifest(manifest)
