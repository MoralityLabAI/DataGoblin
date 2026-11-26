"""Sparse matrix factorization exporter for checkpoint weights (MSE + elementwise L1).

Objective: given a checkpoint weight matrix W[out, in], fit rank-C factors V[in, C], U[C, out]
minimizing:

    L(V, U) = MSE((V @ U)^T, W) + l1_weight * (mean(|V|) + mean(|U|))

This is a sparse weight-matrix factorization of a static checkpoint, NOT a claim of behavioral
BPD component discovery. Exported factors are canonicalized under:

    W_hat = (V @ U)^T,  n_j = ||U_j||_2,  U'_j = U_j / n_j,  V'[:, j] = V[:, j] * n_j

with portable manifest-relative paths and complete provenance/timing metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from datagoblin.training_gym.factor_maps import (
    FACTOR_CONVENTION,
    canonicalize_factors,
    file_sha256,
    write_factor_manifest,
)


def _git_head(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def fit_sparse_factors(
    weight: torch.Tensor,
    *,
    components: int = 16,
    steps: int = 300,
    lr: float = 0.05,
    l1_weight: float = 1e-3,
    seed: int = 101,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fit canonicalized sparse factors (MSE + elementwise L1) with granular CPU/wall timings."""
    torch.manual_seed(seed)
    w = weight.detach().cpu().float()  # [out, in]
    d_out, d_in = w.shape

    t_svd_w, t_svd_c = time.perf_counter(), time.process_time()
    u_svd, s_svd, vt_svd = torch.linalg.svd(w, full_matrices=False)
    r = min(components, len(s_svd))
    v_init = torch.randn(d_in, components) * 0.02
    u_init = torch.randn(components, d_out) * 0.02
    sqrt_s = s_svd[:r].sqrt()
    v_init[:, :r] = vt_svd[:r, :].T * sqrt_s[None, :]
    u_init[:r, :] = (u_svd[:, :r] * sqrt_s[None, :]).T
    svd_wall_s = time.perf_counter() - t_svd_w
    svd_cpu_s = time.process_time() - t_svd_c

    v_param = torch.nn.Parameter(v_init)
    u_param = torch.nn.Parameter(u_init)
    opt = torch.optim.Adam([v_param, u_param], lr=lr)

    t_opt_w, t_opt_c = time.perf_counter(), time.process_time()
    for _ in range(steps):
        recon = (v_param @ u_param).T
        mse = F.mse_loss(recon, w)
        sparsity = v_param.abs().mean() + u_param.abs().mean()
        loss = mse + l1_weight * sparsity
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    opt_wall_s = time.perf_counter() - t_opt_w
    opt_cpu_s = time.process_time() - t_opt_c

    t_diag_w, t_diag_c = time.perf_counter(), time.process_time()
    with torch.no_grad():
        fmap = canonicalize_factors(v_param.detach().cpu().numpy(), u_param.detach().cpu().numpy())
        v_out = torch.from_numpy(fmap.v.copy())
        u_out = torch.from_numpy(fmap.u.copy())
        recon = (v_out @ u_out).T
        cos = float(F.cosine_similarity(w.reshape(1, -1), recon.reshape(1, -1)).item())
        rel_err = float((w - recon).norm().item() / max(w.norm().item(), 1e-8))
        svd_recon = (u_svd[:, :r] * s_svd[:r]) @ vt_svd[:r, :]
        svd_cos = float(F.cosine_similarity(w.reshape(1, -1), svd_recon.reshape(1, -1)).item())
    diag_wall_s = time.perf_counter() - t_diag_w
    diag_cpu_s = time.process_time() - t_diag_c

    stats: dict[str, Any] = {
        "objective": "mse_plus_elementwise_l1_weight_factorization",
        "fit_settings": {
            "components": components,
            "steps": steps,
            "lr": lr,
            "l1_weight": l1_weight,
            "seed": seed,
        },
        "sparse_recon_cosine": round(cos, 6),
        "svd_recon_cosine": round(svd_cos, 6),
        "relative_frobenius_error": round(rel_err, 6),
        "active_components": int(fmap.active.sum()),
        "zero_components": fmap.zero_count,
        "basis_digest": fmap.basis_digest,
        "timing": {
            "initial_svd_wall_s": round(svd_wall_s, 6),
            "initial_svd_cpu_s": round(svd_cpu_s, 6),
            "optimization_wall_s": round(opt_wall_s, 6),
            "optimization_cpu_s": round(opt_cpu_s, 6),
            "diagnostics_wall_s": round(diag_wall_s, 6),
            "diagnostics_cpu_s": round(diag_cpu_s, 6),
        },
    }
    return v_out, u_out, stats


def export_sparse_factor_map(
    *,
    ckpt_path: Path,
    weight_key: str,
    out_pt: Path,
    site_key: str,
    training_step: int = 0,
    source_data_split_digest: str = "unspecified",
    source_model_prep_cpu_s: float | None = None,
    amortization_sessions: int = 1,
    components: int = 16,
    steps: int = 300,
    lr: float = 0.05,
    l1_weight: float = 1e-3,
    seed: int = 101,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Read checkpoint, fit canonical sparse factors, export portable manifest, and record full timing."""
    total_w0, total_c0 = time.perf_counter(), time.process_time()

    t_rd_w, t_rd_c = time.perf_counter(), time.process_time()
    ckpt_sha = file_sha256(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if weight_key not in state:
        raise KeyError(f"{weight_key!r} not in checkpoint keys: {list(state.keys())[:10]}")
    weight = state[weight_key]
    read_wall_s = time.perf_counter() - t_rd_w
    read_cpu_s = time.process_time() - t_rd_c

    v_out, u_out, fit_stats = fit_sparse_factors(
        weight, components=components, steps=steps, lr=lr, l1_weight=l1_weight, seed=seed
    )

    t_exp_w, t_exp_c = time.perf_counter(), time.process_time()
    out_pt = Path(out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    # Save tensor-only weights_only=True-compatible dict alongside metadata sidecar in manifest
    torch.save({"V": v_out, "U": u_out}, out_pt)
    exp_wall_s = time.perf_counter() - t_exp_w
    exp_cpu_s = time.process_time() - t_exp_c

    t_hash_w, t_hash_c = time.perf_counter(), time.process_time()
    art_sha = file_sha256(out_pt)
    hash_wall_s = time.perf_counter() - t_hash_w
    hash_cpu_s = time.process_time() - t_hash_c

    total_wall_s = time.perf_counter() - total_w0
    total_cpu_s = time.process_time() - total_c0

    timing = {
        "checkpoint_read_wall_s": round(read_wall_s, 6),
        "checkpoint_read_cpu_s": round(read_cpu_s, 6),
        **fit_stats["timing"],
        "export_wall_s": round(exp_wall_s, 6),
        "export_cpu_s": round(exp_cpu_s, 6),
        "hashing_wall_s": round(hash_wall_s, 6),
        "hashing_cpu_s": round(hash_cpu_s, 6),
        "total_prep_wall_s": round(total_wall_s, 6),
        "total_prep_cpu_s": round(total_cpu_s, 6),
    }

    commit = code_commit if code_commit is not None else _git_head(Path(__file__).resolve().parents[2])
    manifest_path = out_pt.with_suffix(".manifest.json")
    record = {
        "path": out_pt.name,  # portable manifest-relative path (same directory as manifest)
        "artifact_sha256": art_sha,
        "source_checkpoint_sha256": ckpt_sha,
        "training_step": int(training_step),
        "weight_key": weight_key,
        "site_identity": site_key,
        "factor_convention": FACTOR_CONVENTION,
        "objective": fit_stats["objective"],
        "fit_settings": fit_stats["fit_settings"],
        "source_data_split_digest": source_data_split_digest,
        "code_commit": commit,
        "basis_digest": fit_stats["basis_digest"],
        "sparse_recon_cosine": fit_stats["sparse_recon_cosine"],
        "svd_recon_cosine": fit_stats["svd_recon_cosine"],
        "relative_frobenius_error": fit_stats["relative_frobenius_error"],
        "active_components": fit_stats["active_components"],
        "zero_components": fit_stats["zero_components"],
        "prep_timing": timing,
        "source_model_prep_cpu_s": source_model_prep_cpu_s,
        "amortization_sessions": int(amortization_sessions),
    }
    write_factor_manifest(manifest_path, {site_key: record})
    return {
        "map_path": str(out_pt),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        **record,
    }


# Backward-compatible helper alias
def fit_learned_bpd_factors(
    weight: torch.Tensor,
    *,
    components: int = 16,
    steps: int = 300,
    lr: float = 0.05,
    l1_weight: float = 1e-3,
    seed: int = 101,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    return fit_sparse_factors(
        weight, components=components, steps=steps, lr=lr, l1_weight=l1_weight, seed=seed
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weight-key", required=True)
    p.add_argument("--site-key", default="block")
    p.add_argument("--training-step", type=int, default=0)
    p.add_argument("--split-digest", default="unspecified")
    p.add_argument("--source-prep-cpu-s", type=float, default=None)
    p.add_argument("--amortization-sessions", type=int, default=1)
    p.add_argument("--components", type=int, default=16)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l1-weight", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    info = export_sparse_factor_map(
        ckpt_path=args.ckpt,
        weight_key=args.weight_key,
        out_pt=args.out,
        site_key=args.site_key,
        training_step=args.training_step,
        source_data_split_digest=args.split_digest,
        source_model_prep_cpu_s=args.source_prep_cpu_s,
        amortization_sessions=args.amortization_sessions,
        components=args.components,
        steps=args.steps,
        lr=args.lr,
        l1_weight=args.l1_weight,
        seed=args.seed,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
