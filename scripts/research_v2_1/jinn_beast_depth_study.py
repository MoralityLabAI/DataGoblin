"""Jinn/Beast deep recurrence study (K=1,2,3,6,12,24), compute-matched FF controls, convergence & fresh confirmation.

Extends Workstream B while treating the original 432 archive test roots as SPENT:
  - Builds a fresh independent 600-root (1,200-target) confirmation panel strictly
    disjoint from all 2,304 original archive roots (1,440 train / 432 val / 432 spent test)
    and stores the panel outside Git.
  - Compares tied K-pass recurrence (K = 1, 2, 3, 6, 12, 24; 198 params/frame) with
    exact unrolled BPTT vs straight-through approximation against:
      * fixed_mask_ldt (75 params, 0 corrector MACs)
      * param_matched_ff_1x (198 params, 108 MACs)
      * compute_matched_ff_depth{6,12,24} (untied K-layer residual MLPs, exact 108*K MACs)
      * compute_matched_ff_wide{6,12,24} (wide single-pass MLPs H=12*K, exact 108*K MACs)
  - Verifies exact unrolled BPTT against central finite differences (< 1e-5) up to K=24.
  - Records convergence diagnostics (per-pass delta norms, contraction ratios, accuracy
    trajectories across passes, unrolled gradient norm profiles, final logit norms) and
    reports raw vs guarded critical-policy false accepts separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from jinn_beast_disentangle import (  # noqa: E402
    LANE_TO_IDX,
    AdditiveLDT,
    CorrectorHead,
    _eval_frame,
    _paired_bootstrap_diff,
    _softmax,
)

from datagoblin.research_v2.specialist import (  # noqa: E402
    _load_manifest,
    _policy,
    _read_pinned,
    _score,
    canonical,
    digest,
    root_id,
    semantic_digest,
)

BASE_CELL_MACS = 6 * 12 + 12 * 3  # 108 MACs per pass


class UntiedDeepFeedforwardHead:
    """Compute-matched untied K-layer residual MLP (exact same 108*K MACs, K independent layers)."""

    def __init__(
        self,
        n_features: int,
        own_indices: list[int],
        critical_indices: list[int],
        *,
        depth: int,
        seed: int = 0,
    ) -> None:
        self.own_mask = np.zeros(n_features, dtype=np.float64)
        self.own_mask[own_indices] = 1.0
        self.critical_indices = list(critical_indices)
        self.base = AdditiveLDT(n_features, mask=self.own_mask, seed=seed)
        self.depth = depth
        self.alphas = [1.0 / (depth - j) for j in range(depth)]
        rng = np.random.default_rng(seed + 17)
        self.w1s = [rng.normal(0.0, 0.05, size=(6, 12)).astype(np.float64) for _ in range(depth)]
        self.b1s = [np.zeros(12, dtype=np.float64) for _ in range(depth)]
        self.w2s = [rng.normal(0.0, 0.02, size=(12, 3)).astype(np.float64) for _ in range(depth)]
        self.b2s = [np.zeros(3, dtype=np.float64) for _ in range(depth)]

    @property
    def n_params(self) -> int:
        layer_p = sum(w1.size + b1.size + w2.size + b2.size for w1, b1, w2, b2 in zip(
            self.w1s, self.b1s, self.w2s, self.b2s, strict=True
        ))
        return self.base.n_params + int(layer_p)

    @property
    def corrector_macs(self) -> int:
        return BASE_CELL_MACS * self.depth

    def _static_feats(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        active_cnt = (x @ self.own_mask)[:, None] / max(1.0, float(self.own_mask.sum()))
        crit = x[:, self.critical_indices].any(axis=1, keepdims=True).astype(np.float64)
        return active_cnt, crit

    def forward_corrector(
        self, base_logits: np.ndarray, active_cnt: np.ndarray, crit: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        cur = base_logits.copy()
        cache: list[dict[str, Any]] = []
        rows = np.arange(len(cur))
        for idx, alpha in enumerate(self.alphas):
            probs = _softmax(cur)
            order = np.argsort(probs, axis=1)
            top1, top2 = order[:, -1], order[:, -2]
            margin = (probs[rows, top1] - probs[rows, top2])[:, None]
            feat = np.concatenate([probs, margin, active_cnt, crit], axis=1)
            hid = np.tanh(feat @ self.w1s[idx] + self.b1s[idx])
            delta = hid @ self.w2s[idx] + self.b2s[idx]
            nxt = cur + alpha * delta
            cache.append(
                {
                    "layer": idx,
                    "probs": probs,
                    "top1": top1,
                    "top2": top2,
                    "feat": feat,
                    "hid": hid,
                    "delta": delta,
                    "logits_after": nxt,
                    "alpha": alpha,
                }
            )
            cur = nxt
        return cur, cache

    def backward_corrector(
        self, dlogits: np.ndarray, cache: list[dict[str, Any]]
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]]:
        gw1s = [np.zeros_like(w) for w in self.w1s]
        gb1s = [np.zeros_like(b) for b in self.b1s]
        gw2s = [np.zeros_like(w) for w in self.w2s]
        gb2s = [np.zeros_like(b) for b in self.b2s]
        step_dcur_norms = [0.0] * self.depth
        dcur = dlogits.copy()
        rows = np.arange(len(dcur))
        for step in reversed(cache):
            idx = step["layer"]
            step_dcur_norms[idx] = float(np.linalg.norm(dcur))
            alpha = step["alpha"]
            hid = step["hid"]
            feat = step["feat"]
            probs = step["probs"]
            top1 = step["top1"]
            top2 = step["top2"]
            ddelta = dcur * alpha
            gw2s[idx] = hid.T @ ddelta
            gb2s[idx] = ddelta.sum(axis=0)
            dhid = (ddelta @ self.w2s[idx].T) * (1.0 - hid * hid)
            gw1s[idx] = feat.T @ dhid
            gb1s[idx] = dhid.sum(axis=0)
            dfeat = dhid @ self.w1s[idx].T
            dprobs = dfeat[:, :3].copy()
            dmargin = dfeat[:, 3]
            dprobs[rows, top1] += dmargin
            dprobs[rows, top2] -= dmargin
            dot = np.sum(dprobs * probs, axis=1, keepdims=True)
            dcur = dcur + probs * (dprobs - dot)
        return gw1s, gb1s, gw2s, gb2s, step_dcur_norms

    def fit(self, x_tr: np.ndarray, y_tr: np.ndarray, *, seed: int = 0) -> dict[str, Any]:
        self.base.fit(x_tr, y_tr, epochs=120, lr=0.08, l2=5e-4, seed=seed)
        base_logits_all = self.base.logits(x_tr)
        active_all, crit_all = self._static_feats(x_tr)
        rng = np.random.default_rng(seed + 99)
        n = len(x_tr)
        lr = 0.03
        max_gn = 0.0
        mid_profile: list[float] = []
        for ep in range(80):
            perm = rng.permutation(n)
            for start in range(0, n, 64):
                idx = perm[start : start + 64]
                bl, ac, cr, yb = base_logits_all[idx], active_all[idx], crit_all[idx], y_tr[idx]
                logits, cache = self.forward_corrector(bl, ac, cr)
                probs = _softmax(logits)
                dlog = probs.copy()
                dlog[np.arange(len(yb)), yb] -= 1.0
                dlog /= len(yb)
                gw1s, gb1s, gw2s, gb2s, dnorms = self.backward_corrector(dlog, cache)
                if ep == 40 and start == 0:
                    mid_profile = [round(v, 6) for v in dnorms]
                gn = float(np.sqrt(sum(float(np.sum(g * g)) for g in (*gw1s, *gw2s))))
                max_gn = max(max_gn, gn)
                for j in range(self.depth):
                    self.w2s[j] -= lr * (gw2s[j] + 1e-3 * self.w2s[j])
                    self.b2s[j] -= lr * gb2s[j]
                    self.w1s[j] -= lr * (gw1s[j] + 1e-3 * self.w1s[j])
                    self.b1s[j] -= lr * gb1s[j]
        final_logits, _ = self.forward_corrector(base_logits_all, active_all, crit_all)
        probs = _softmax(final_logits)
        tr_loss = float(-np.log(np.clip(probs[np.arange(n), y_tr], 1e-12, 1.0)).mean())
        return {
            "final_train_loss": round(tr_loss, 6),
            "max_train_grad_norm": round(max_gn, 6),
            "epoch40_dcur_norm_profile": mid_profile,
        }

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        bl = self.base.logits(x)
        ac, cr = self._static_feats(x)
        logits, _ = self.forward_corrector(bl, ac, cr)
        return _softmax(logits)

    def trace_convergence(self, x: np.ndarray, y: np.ndarray, crit: np.ndarray) -> dict[str, Any]:
        bl = self.base.logits(x)
        ac, cr = self._static_feats(x)
        final_logits, cache = self.forward_corrector(bl, ac, cr)
        return _summarize_pass_trace(cache, final_logits, y, crit)


class WideFeedforwardHead:
    """Compute-matched single-pass wide MLP (6 -> 12*mult -> 3, exact same 108*mult MACs)."""

    def __init__(
        self,
        n_features: int,
        own_indices: list[int],
        critical_indices: list[int],
        *,
        compute_mult: int,
        seed: int = 0,
    ) -> None:
        self.own_mask = np.zeros(n_features, dtype=np.float64)
        self.own_mask[own_indices] = 1.0
        self.critical_indices = list(critical_indices)
        self.base = AdditiveLDT(n_features, mask=self.own_mask, seed=seed)
        self.hidden = 12 * compute_mult
        self.compute_mult = compute_mult
        rng = np.random.default_rng(seed + 17)
        scale = 1.0 / np.sqrt(float(compute_mult))
        self.w1 = rng.normal(0.0, 0.05, size=(6, self.hidden)).astype(np.float64)
        self.b1 = np.zeros(self.hidden, dtype=np.float64)
        self.w2 = rng.normal(0.0, 0.02 * scale, size=(self.hidden, 3)).astype(np.float64)
        self.b2 = np.zeros(3, dtype=np.float64)

    @property
    def n_params(self) -> int:
        return self.base.n_params + int(self.w1.size + self.b1.size + self.w2.size + self.b2.size)

    @property
    def corrector_macs(self) -> int:
        return 6 * self.hidden + self.hidden * 3

    def _static_feats(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        active_cnt = (x @ self.own_mask)[:, None] / max(1.0, float(self.own_mask.sum()))
        crit = x[:, self.critical_indices].any(axis=1, keepdims=True).astype(np.float64)
        return active_cnt, crit

    def _forward(
        self, base_logits: np.ndarray, active_cnt: np.ndarray, crit: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        probs = _softmax(base_logits)
        order = np.argsort(probs, axis=1)
        rows = np.arange(len(base_logits))
        margin = (probs[rows, order[:, -1]] - probs[rows, order[:, -2]])[:, None]
        feat = np.concatenate([probs, margin, active_cnt, crit], axis=1)
        hid = np.tanh(feat @ self.w1 + self.b1)
        delta = hid @ self.w2 + self.b2
        return base_logits + delta, feat, hid, delta

    def fit(self, x_tr: np.ndarray, y_tr: np.ndarray, *, seed: int = 0) -> dict[str, Any]:
        self.base.fit(x_tr, y_tr, epochs=120, lr=0.08, l2=5e-4, seed=seed)
        base_logits_all = self.base.logits(x_tr)
        active_all, crit_all = self._static_feats(x_tr)
        rng = np.random.default_rng(seed + 99)
        n = len(x_tr)
        lr = 0.03
        max_gn = 0.0
        for _ in range(80):
            perm = rng.permutation(n)
            for start in range(0, n, 64):
                idx = perm[start : start + 64]
                bl, ac, cr, yb = base_logits_all[idx], active_all[idx], crit_all[idx], y_tr[idx]
                logits, feat, hid, _ = self._forward(bl, ac, cr)
                probs = _softmax(logits)
                dlog = probs.copy()
                dlog[np.arange(len(yb)), yb] -= 1.0
                dlog /= len(yb)
                gw2 = hid.T @ dlog
                gb2 = dlog.sum(axis=0)
                dhid = (dlog @ self.w2.T) * (1.0 - hid * hid)
                gw1 = feat.T @ dhid
                gb1 = dhid.sum(axis=0)
                gn = float(np.sqrt(float(np.sum(gw1 * gw1) + np.sum(gw2 * gw2))))
                max_gn = max(max_gn, gn)
                self.w2 -= lr * (gw2 + 1e-3 * self.w2)
                self.b2 -= lr * gb2
                self.w1 -= lr * (gw1 + 1e-3 * self.w1)
                self.b1 -= lr * gb1
        final_logits, _, _, _ = self._forward(base_logits_all, active_all, crit_all)
        probs = _softmax(final_logits)
        tr_loss = float(-np.log(np.clip(probs[np.arange(n), y_tr], 1e-12, 1.0)).mean())
        return {"final_train_loss": round(tr_loss, 6), "max_train_grad_norm": round(max_gn, 6)}

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        bl = self.base.logits(x)
        ac, cr = self._static_feats(x)
        logits, _, _, _ = self._forward(bl, ac, cr)
        return _softmax(logits)

    def trace_convergence(self, x: np.ndarray, y: np.ndarray, crit: np.ndarray) -> dict[str, Any]:
        bl = self.base.logits(x)
        ac, cr = self._static_feats(x)
        logits, _, _, delta = self._forward(bl, ac, cr)
        raw = logits.argmax(axis=1)
        guarded = raw.copy()
        guarded[crit] = LANE_TO_IDX["reject"]
        return {
            "passes": 1,
            "per_pass_damped_delta_l2": [round(float(np.linalg.norm(delta, axis=1).mean()), 6)],
            "per_pass_raw_delta_l2": [round(float(np.linalg.norm(delta, axis=1).mean()), 6)],
            "tail_to_head_delta_ratio": 1.0,
            "per_pass_guarded_acc": [round(float(np.mean(guarded == y)), 6)],
            "per_pass_raw_crit_fa": [int(np.sum(crit & (raw == LANE_TO_IDX["train"])))],
            "final_logit_l2_mean": round(float(np.linalg.norm(logits, axis=1).mean()), 6),
        }


def _summarize_pass_trace(
    cache: list[dict[str, Any]], final_logits: np.ndarray, y: np.ndarray, crit: np.ndarray
) -> dict[str, Any]:
    damped_norms: list[float] = []
    raw_norms: list[float] = []
    pass_accs: list[float] = []
    pass_raw_fa: list[int] = []
    for step in cache:
        delta = step["delta"]
        alpha = step["alpha"]
        raw_l2 = float(np.linalg.norm(delta, axis=1).mean())
        damped_l2 = float(np.linalg.norm(alpha * delta, axis=1).mean())
        raw_norms.append(round(raw_l2, 6))
        damped_norms.append(round(damped_l2, 6))
        pred_raw = step["logits_after"].argmax(axis=1)
        pred_g = pred_raw.copy()
        pred_g[crit] = LANE_TO_IDX["reject"]
        pass_accs.append(round(float(np.mean(pred_g == y)), 6))
        pass_raw_fa.append(int(np.sum(crit & (pred_raw == LANE_TO_IDX["train"]))))
    ratio = round(damped_norms[-1] / max(1e-9, damped_norms[0]), 6) if damped_norms else 0.0
    return {
        "passes": len(cache),
        "per_pass_damped_delta_l2": damped_norms,
        "per_pass_raw_delta_l2": raw_norms,
        "tail_to_head_delta_ratio": ratio,
        "per_pass_guarded_acc": pass_accs,
        "per_pass_raw_crit_fa": pass_raw_fa,
        "final_logit_l2_mean": round(float(np.linalg.norm(final_logits, axis=1).mean()), 6),
    }


def _fit_and_profile_tied(head: CorrectorHead, x_tr: np.ndarray, y_tr: np.ndarray, *, seed: int) -> dict[str, Any]:
    head.base.fit(x_tr, y_tr, epochs=120, lr=0.08, l2=5e-4, seed=seed)
    base_logits_all = head.base.logits(x_tr)
    active_all, crit_all = head._static_feats(x_tr)
    rng = np.random.default_rng(seed + 99)
    n = len(x_tr)
    lr = 0.03
    max_gn = 0.0
    mid_profile: list[float] = []
    for ep in range(80):
        perm = rng.permutation(n)
        for start in range(0, n, 64):
            idx = perm[start : start + 64]
            bl, ac, cr, yb = base_logits_all[idx], active_all[idx], crit_all[idx], y_tr[idx]
            logits, cache = head.forward_corrector(bl, ac, cr)
            probs = _softmax(logits)
            dlog = probs.copy()
            dlog[np.arange(len(yb)), yb] -= 1.0
            dlog /= len(yb)
            if ep == 40 and start == 0:
                mid_profile = _profile_dcur_norms(head, dlog, cache)
            gw1, gb1, gw2, gb2 = head.backward_corrector(dlog, cache)
            gn = float(np.sqrt(float(np.sum(gw1 * gw1) + np.sum(gw2 * gw2))))
            max_gn = max(max_gn, gn)
            head.w2 -= lr * (gw2 + 1e-3 * head.w2)
            head.b2 -= lr * gb2
            head.w1 -= lr * (gw1 + 1e-3 * head.w1)
            head.b1 -= lr * gb1
    final_logits, _ = head.forward_corrector(base_logits_all, active_all, crit_all)
    probs = _softmax(final_logits)
    tr_loss = float(-np.log(np.clip(probs[np.arange(n), y_tr], 1e-12, 1.0)).mean())
    return {
        "final_train_loss": round(tr_loss, 6),
        "max_train_grad_norm": round(max_gn, 6),
        "epoch40_dcur_norm_profile": mid_profile,
    }


def _profile_dcur_norms(head: CorrectorHead, dlogits: np.ndarray, cache: list[dict[str, Any]]) -> list[float]:
    if head.grad_mode == "straight_through":
        base_norm = round(float(np.linalg.norm(dlogits)), 6)
        return [round(base_norm * step["alpha"], 6) for step in cache]
    dcur = dlogits.copy()
    rows = np.arange(len(dcur))
    norms = [0.0] * len(cache)
    for j in range(len(cache) - 1, -1, -1):
        step = cache[j]
        norms[j] = round(float(np.linalg.norm(dcur)), 6)
        ddelta = dcur * step["alpha"]
        dhid = (ddelta @ head.w2.T) * (1.0 - step["hid"] * step["hid"])
        dfeat = dhid @ head.w1.T
        dprobs = dfeat[:, :3].copy()
        dmargin = dfeat[:, 3]
        dprobs[rows, step["top1"]] += dmargin
        dprobs[rows, step["top2"]] -= dmargin
        dot = np.sum(dprobs * step["probs"], axis=1, keepdims=True)
        dcur = dcur + step["probs"] * (dprobs - dot)
    return norms


def _trace_tied(head: CorrectorHead, x: np.ndarray, y: np.ndarray, crit: np.ndarray) -> dict[str, Any]:
    bl = head.base.logits(x)
    ac, cr = head._static_feats(x)
    cur = bl.copy()
    rows = np.arange(len(cur))
    trace_cache: list[dict[str, Any]] = []
    for alpha in head.alphas:
        probs = _softmax(cur)
        order = np.argsort(probs, axis=1)
        margin = (probs[rows, order[:, -1]] - probs[rows, order[:, -2]])[:, None]
        feat = np.concatenate([probs, margin, ac, cr], axis=1)
        hid = np.tanh(feat @ head.w1 + head.b1)
        delta = hid @ head.w2 + head.b2
        nxt = cur + alpha * delta
        trace_cache.append({"alpha": alpha, "delta": delta, "logits_after": nxt})
        cur = nxt
    return _summarize_pass_trace(trace_cache, cur, y, crit)


def verify_deep_exact_bptt_gradients() -> dict[str, float]:
    """Central finite-difference verification of exact unrolled BPTT at K=1, 3, 6, 12, 24 and untied K=6, 12."""
    rng = np.random.default_rng(20261127)
    n, d = 16, 10
    x = (rng.random((n, d)) < 0.4).astype(np.float64)
    y = rng.integers(0, 3, size=n)
    eps = 1e-6
    out: dict[str, float] = {}

    for k in (1, 3, 6, 12, 24):
        head = CorrectorHead(
            d, own_indices=list(range(6)), critical_indices=[0], passes=k, grad_mode="exact_unrolled", seed=k
        )
        bl = rng.normal(0.0, 1.0, size=(n, 3))
        ac, cr = head._static_feats(x)

        def loss_tied(
            _h: CorrectorHead = head, _bl: np.ndarray = bl, _ac: np.ndarray = ac, _cr: np.ndarray = cr
        ) -> float:
            logits, _ = _h.forward_corrector(_bl, _ac, _cr)
            probs = _softmax(logits)
            return float(-np.log(np.clip(probs[np.arange(n), y], 1e-12, 1.0)).mean())

        logits, cache = head.forward_corrector(bl, ac, cr)
        probs = _softmax(logits)
        dlog = probs.copy()
        dlog[np.arange(n), y] -= 1.0
        dlog /= n
        gw1, gb1, gw2, gb2 = head.backward_corrector(dlog, cache)
        max_rel = 0.0
        for param, grad in ((head.w1, gw1), (head.b1, gb1), (head.w2, gw2), (head.b2, gb2)):
            fp, fg = param.ravel(), grad.ravel()
            for idx in range(min(len(fp), 20)):
                orig = fp[idx]
                fp[idx] = orig + eps
                lp = loss_tied()
                fp[idx] = orig - eps
                lm = loss_tied()
                fp[idx] = orig
                num = (lp - lm) / (2 * eps)
                ana = fg[idx]
                rel = 0.0 if abs(num - ana) <= 1e-7 else abs(num - ana) / max(1e-8, abs(num) + abs(ana))
                max_rel = max(max_rel, float(rel))
        if max_rel >= 1e-5:
            raise RuntimeError(f"tied exact BPTT check failed at K={k}: {max_rel}")
        out[f"tied_K={k}_max_rel_err"] = float(f"{max_rel:.3e}")

    for depth in (6, 12):
        uhead = UntiedDeepFeedforwardHead(d, own_indices=list(range(6)), critical_indices=[0], depth=depth, seed=depth)
        for j in range(depth):
            uhead.w1s[j] *= 4.0
            uhead.w2s[j] *= 4.0
        bl = rng.normal(0.0, 1.0, size=(n, 3))
        ac, cr = uhead._static_feats(x)

        def loss_untied(
            _h: UntiedDeepFeedforwardHead = uhead, _bl: np.ndarray = bl, _ac: np.ndarray = ac, _cr: np.ndarray = cr
        ) -> float:
            logits, _ = _h.forward_corrector(_bl, _ac, _cr)
            probs = _softmax(logits)
            return float(-np.log(np.clip(probs[np.arange(n), y], 1e-12, 1.0)).mean())

        logits, cache = uhead.forward_corrector(bl, ac, cr)
        probs = _softmax(logits)
        dlog = probs.copy()
        dlog[np.arange(n), y] -= 1.0
        dlog /= n
        gw1s, _gb1s, gw2s, _gb2s, _ = uhead.backward_corrector(dlog, cache)
        max_rel = 0.0
        for layer_idx in (0, depth // 2, depth - 1):
            for param, grad in ((uhead.w1s[layer_idx], gw1s[layer_idx]), (uhead.w2s[layer_idx], gw2s[layer_idx])):
                fp, fg = param.ravel(), grad.ravel()
                for idx in range(min(len(fp), 10)):
                    orig = fp[idx]
                    fp[idx] = orig + eps
                    lp = loss_untied()
                    fp[idx] = orig - eps
                    lm = loss_untied()
                    fp[idx] = orig
                    num = (lp - lm) / (2 * eps)
                    ana = fg[idx]
                    rel = 0.0 if abs(num - ana) <= 1e-7 else abs(num - ana) / max(1e-8, abs(num) + abs(ana))
                    max_rel = max(max_rel, float(rel))
        if max_rel >= 1e-5:
            raise RuntimeError(f"untied exact BPTT check failed at depth={depth}: {max_rel}")
        out[f"untied_depth={depth}_max_rel_err"] = float(f"{max_rel:.3e}")
    return out


def build_fresh_confirmation_panel(
    *,
    tag_vocab: list[str],
    policies: dict[str, dict[str, Any]],
    low: Fraction,
    high: Fraction,
    archive_rows: list[dict[str, Any]],
    excluded_roots: set[str],
    excluded_sems: set[str],
    n_roots: int = 600,
    seed: int = 2026112788,
    panel_out: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a fresh independent confirmation panel strictly disjoint from all 2,304 archive roots."""
    if panel_out is not None:
        resolved = panel_out.resolve()
        if any((p / ".git").exists() for p in (resolved, *resolved.parents)):
            raise ValueError("fresh confirmation panel must live strictly outside Git")

    rng = np.random.default_rng(seed)
    card_hist = [len(r["tags"]) for r in archive_rows]
    all_crit = sorted(set(policies["jinn"]["critical_tags"]) | set(policies["beast"]["critical_tags"]))
    non_crit = [t for t in tag_vocab if t not in set(all_crit)]
    crit_rate = float(np.mean([bool(set(r["tags"]) & set(all_crit)) for r in archive_rows]))

    forbidden_roots = set(excluded_roots) | {root_id(r["tags"]) for r in archive_rows}
    forbidden_sems = set(excluded_sems) | {semantic_digest(r["tags"]) for r in archive_rows}

    fresh_rows: list[dict[str, Any]] = []
    seen_roots: set[str] = set()
    attempts = 0
    while len(fresh_rows) < n_roots:
        attempts += 1
        if attempts > n_roots * 50:
            raise RuntimeError("unable to sample enough fresh disjoint roots")
        k = int(rng.choice(card_hist))
        k = max(2, min(len(tag_vocab) - 1, k))
        want_crit = bool(rng.random() < crit_rate)
        if want_crit:
            n_c = 1 if rng.random() < 0.78 else min(2, k)
            chosen_c = rng.choice(all_crit, size=n_c, replace=False).tolist()
            n_nc = max(0, min(len(non_crit), k - n_c))
            chosen_nc = rng.choice(non_crit, size=n_nc, replace=False).tolist() if n_nc > 0 else []
            tags = sorted(set(chosen_c + chosen_nc))
        else:
            n_nc = min(len(non_crit), k)
            tags = sorted(rng.choice(non_crit, size=n_nc, replace=False).tolist())
        rid = root_id(tags)
        sem = semantic_digest(tags)
        if rid in forbidden_roots or sem in forbidden_sems or rid in seen_roots:
            continue
        seen_roots.add(rid)
        joint = {f: _score(tags, policies[f], low, high)["lane"] for f in ("jinn", "beast")}
        fresh_rows.append(
            {
                "row_id": f"fresh-conf-{len(fresh_rows):04d}",
                "root_id": rid,
                "semantic_sha256": sem,
                "split": "fresh_confirmation",
                "tags": tags,
                "joint_label": joint,
            }
        )

    assert len(seen_roots & forbidden_roots) == 0
    fresh_bytes = "".join(canonical(r) + "\n" for r in fresh_rows).encode("utf-8")
    panel_sha = hashlib.sha256(fresh_bytes).hexdigest()
    reg = {
        "schema": "research-v2.1-jinn-fresh-confirmation-roots-v1",
        "held_root_ids": sorted(seen_roots),
        "held_semantic_sha256": sorted(r["semantic_sha256"] for r in fresh_rows),
        "allow_source_ids": ["jinn-beast-audited-tags-v1"],
    }
    reg_bytes = (canonical(reg) + "\n").encode("utf-8")
    reg_sha = hashlib.sha256(reg_bytes).hexdigest()
    if panel_out is not None:
        panel_out.mkdir(parents=True, exist_ok=True)
        (panel_out / "fresh_confirmation.private.jsonl").write_bytes(fresh_bytes)
        (panel_out / "held_registry.json").write_bytes(reg_bytes)

    lane_dist = {
        f: dict(sorted(Counter(r["joint_label"][f] for r in fresh_rows).items())) for f in ("jinn", "beast")
    }
    crit_counts = {
        f: int(sum(1 for r in fresh_rows if set(r["tags"]) & set(policies[f]["critical_tags"])))
        for f in ("jinn", "beast")
    }
    meta = {
        "schema": "research-v2.1-jinn-fresh-confirmation-manifest-1",
        "seed": seed,
        "fresh_roots": len(fresh_rows),
        "fresh_frame_targets": len(fresh_rows) * 2,
        "overlap_with_archive_2304_roots": 0,
        "overlap_with_spent_432_test_roots": 0,
        "overlap_with_external_exclusions": 0,
        "panel_sha256": panel_sha,
        "held_registry_sha256": reg_sha,
        "lane_distribution": lane_dist,
        "critical_examples_by_frame": crit_counts,
        "stored_outside_git": panel_out is not None,
    }
    if panel_out is not None:
        (panel_out / "MANIFEST.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fresh_rows, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclude-registry", type=Path, action="append", default=[])
    parser.add_argument("--fresh-panel-out", type=Path, required=True)
    parser.add_argument("--fresh-roots", type=int, default=600)
    parser.add_argument("--fresh-seed", type=int, default=2026112788)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = time.monotonic()
    grad_check = verify_deep_exact_bptt_gradients()
    print(f"[depth-study] Exact BPTT gradient verification passed: {grad_check}", flush=True)

    manifest = _load_manifest(None)
    payloads = {name: _read_pinned(args.source_root, spec) for name, spec in manifest["files"].items()}
    policies = {frame: _policy(payloads[frame]) for frame in ("jinn", "beast")}
    low, high = Fraction(manifest["reject_maximum"]), Fraction(manifest["train_minimum"])
    rows = [json.loads(line) for line in payloads["dataset"].decode("utf-8").splitlines() if line]

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        for frame, policy in policies.items():
            actual = _score(row["tags"], policy, low, high)
            if row["joint_label"][frame] != actual["lane"]:
                raise RuntimeError(f"label mismatch {row['row_id']}/{frame}")
        splits[row["split"]].append(row)

    excl_roots: set[str] = set()
    excl_sems: set[str] = set()
    for ep in args.exclude_registry:
        if ep.is_file():
            reg = json.loads(ep.read_text(encoding="utf-8"))
            excl_roots.update(reg.get("held_root_ids", []))
            excl_sems.update(reg.get("held_semantic_sha256", []))

    tag_vocab = sorted(set().union(*(set(p["tags"]) for p in policies.values())))
    tag_idx = {t: i for i, t in enumerate(tag_vocab)}
    own_idx = {f: sorted(tag_idx[t] for t in policies[f]["tags"]) for f in ("jinn", "beast")}
    crit_idx = {f: sorted(tag_idx[t] for t in policies[f]["critical_tags"]) for f in ("jinn", "beast")}

    fresh_rows, fresh_meta = build_fresh_confirmation_panel(
        tag_vocab=tag_vocab,
        policies=policies,
        low=low,
        high=high,
        archive_rows=rows,
        excluded_roots=excl_roots,
        excluded_sems=excl_sems,
        n_roots=args.fresh_roots,
        seed=args.fresh_seed,
        panel_out=args.fresh_panel_out,
    )
    print(
        f"[depth-study] Built fresh independent confirmation panel: {fresh_meta['fresh_roots']} roots "
        f"({fresh_meta['fresh_frame_targets']} targets, sha={fresh_meta['panel_sha256'][:12]})",
        flush=True,
    )

    def enc(s_rows: list[dict[str, Any]]):
        xu = np.zeros((len(s_rows), len(tag_vocab)), dtype=np.float64)
        for i, r in enumerate(s_rows):
            for t in r["tags"]:
                xu[i, tag_idx[t]] = 1.0
        xo, ys, cr = {}, {}, {}
        for f in ("jinn", "beast"):
            m = np.zeros(len(tag_vocab), dtype=np.float64)
            m[own_idx[f]] = 1.0
            xo[f] = xu * m[None, :]
            ys[f] = np.array([LANE_TO_IDX[r["joint_label"][f]] for r in s_rows], dtype=np.int64)
            cr[f] = xu[:, crit_idx[f]].any(axis=1)
        return xu, xo, ys, cr

    x_tr, _, y_tr, _ = enc(splits["train"])
    x_sp, x_sp_own, y_sp, crit_sp = enc(splits["test"])
    x_fr, x_fr_own, y_fr, crit_fr = enc(fresh_rows)

    arm_specs: list[tuple[str, dict[str, Any]]] = [
        ("fixed_mask_ldt", {"kind": "fixed_mask_ldt", "macs": 0}),
        ("param_matched_ff_1x", {"kind": "tied", "passes": 1, "grad_mode": "exact_unrolled", "damped": False}),
        ("recurrent_3_pass_exact", {"kind": "tied", "passes": 3, "grad_mode": "exact_unrolled", "damped": True}),
        ("recurrent_6_pass_exact", {"kind": "tied", "passes": 6, "grad_mode": "exact_unrolled", "damped": True}),
        ("recurrent_6_pass_st", {"kind": "tied", "passes": 6, "grad_mode": "straight_through", "damped": True}),
        ("recurrent_12_pass_exact", {"kind": "tied", "passes": 12, "grad_mode": "exact_unrolled", "damped": True}),
        ("recurrent_12_pass_st", {"kind": "tied", "passes": 12, "grad_mode": "straight_through", "damped": True}),
        ("recurrent_24_pass_exact", {"kind": "tied", "passes": 24, "grad_mode": "exact_unrolled", "damped": True}),
        ("recurrent_24_pass_st", {"kind": "tied", "passes": 24, "grad_mode": "straight_through", "damped": True}),
        ("compute_matched_ff_depth6", {"kind": "untied_depth", "depth": 6}),
        ("compute_matched_ff_depth12", {"kind": "untied_depth", "depth": 12}),
        ("compute_matched_ff_depth24", {"kind": "untied_depth", "depth": 24}),
        ("compute_matched_ff_wide6", {"kind": "wide_ff", "mult": 6}),
        ("compute_matched_ff_wide12", {"kind": "wide_ff", "mult": 12}),
        ("compute_matched_ff_wide24", {"kind": "wide_ff", "mult": 24}),
    ]
    seed_list = [20261127 + s for s in range(args.seeds)]

    arm_results: dict[str, Any] = {}
    s0_fresh_g: dict[str, np.ndarray] = {}
    s0_fresh_r: dict[str, np.ndarray] = {}

    for arm_name, spec in arm_specs:
        fresh_runs: dict[str, list[dict[str, Any]]] = {"jinn": [], "beast": []}
        spent_runs: dict[str, list[dict[str, Any]]] = {"jinn": [], "beast": []}
        conv_seed0: dict[str, Any] = {}
        train_diag_seed0: dict[str, Any] = {}
        n_params = 0
        macs = 0
        g_parts, r_parts = [], []

        for s_i, seed in enumerate(seed_list):
            for f_i, f in enumerate(("jinn", "beast")):
                f_seed = seed * 10 + f_i
                kind = spec["kind"]
                if kind == "fixed_mask_ldt":
                    mask = np.zeros(len(tag_vocab), dtype=np.float64)
                    mask[own_idx[f]] = 1.0
                    model: Any = AdditiveLDT(len(tag_vocab), mask=mask, seed=f_seed)
                    model.fit(x_tr, y_tr[f], seed=f_seed)
                    n_params = model.n_params
                    macs = 0
                    tdiag: dict[str, Any] = {}
                elif kind == "tied":
                    model = CorrectorHead(
                        len(tag_vocab),
                        own_indices=own_idx[f],
                        critical_indices=crit_idx[f],
                        passes=spec["passes"],
                        grad_mode=spec["grad_mode"],
                        damped=spec["damped"],
                        seed=f_seed,
                    )
                    tdiag = _fit_and_profile_tied(model, x_tr, y_tr[f], seed=f_seed)
                    n_params = model.n_params
                    macs = BASE_CELL_MACS * spec["passes"]
                elif kind == "untied_depth":
                    model = UntiedDeepFeedforwardHead(
                        len(tag_vocab),
                        own_indices=own_idx[f],
                        critical_indices=crit_idx[f],
                        depth=spec["depth"],
                        seed=f_seed,
                    )
                    tdiag = model.fit(x_tr, y_tr[f], seed=f_seed)
                    n_params = model.n_params
                    macs = model.corrector_macs
                else:
                    model = WideFeedforwardHead(
                        len(tag_vocab),
                        own_indices=own_idx[f],
                        critical_indices=crit_idx[f],
                        compute_mult=spec["mult"],
                        seed=f_seed,
                    )
                    tdiag = model.fit(x_tr, y_tr[f], seed=f_seed)
                    n_params = model.n_params
                    macs = model.corrector_macs

                pu_fr = model.predict_probs(x_fr)
                po_fr = model.predict_probs(x_fr_own[f])
                fresh_runs[f].append(_eval_frame(pu_fr, po_fr, y_fr[f], crit_fr[f]))

                pu_sp = model.predict_probs(x_sp)
                po_sp = model.predict_probs(x_sp_own[f])
                spent_runs[f].append(_eval_frame(pu_sp, po_sp, y_sp[f], crit_sp[f]))

                if s_i == 0:
                    train_diag_seed0[f] = tdiag
                    if kind == "tied":
                        conv_seed0[f] = _trace_tied(model, x_fr, y_fr[f], crit_fr[f])
                    elif kind in ("untied_depth", "wide_ff"):
                        conv_seed0[f] = model.trace_convergence(x_fr, y_fr[f], crit_fr[f])
                    raw = pu_fr.argmax(axis=1)
                    guarded = raw.copy()
                    guarded[crit_fr[f]] = LANE_TO_IDX["reject"]
                    r_parts.append(raw == y_fr[f])
                    g_parts.append(guarded == y_fr[f])

        s0_fresh_r[arm_name] = np.concatenate(r_parts)
        s0_fresh_g[arm_name] = np.concatenate(g_parts)

        def _agg_frames(run_map: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
            sf: dict[str, Any] = {}
            for f in ("jinn", "beast"):
                runs = run_map[f]
                sf[f] = {
                    "guarded_acc_mean": round(float(np.mean([r["guarded_accuracy"] for r in runs])), 6),
                    "guarded_acc_std": round(float(np.std([r["guarded_accuracy"] for r in runs])), 6),
                    "raw_acc_mean": round(float(np.mean([r["raw_accuracy"] for r in runs])), 6),
                    "raw_acc_std": round(float(np.std([r["raw_accuracy"] for r in runs])), 6),
                    "raw_crit_fa_mean": round(float(np.mean([r["raw_critical_false_accepts"] for r in runs])), 4),
                    "raw_crit_fa_max": int(max(r["raw_critical_false_accepts"] for r in runs)),
                    "guarded_crit_fa_mean": round(
                        float(np.mean([r["guarded_critical_false_accepts"] for r in runs])), 4
                    ),
                    "guarded_crit_fa_max": int(max(r["guarded_critical_false_accepts"] for r in runs)),
                    "opp_frame_inv_guarded_mean": round(
                        float(np.mean([r["opposite_frame_invariance_guarded"] for r in runs])), 6
                    ),
                }
            return {
                "combined_guarded_acc_mean": round(
                    (sf["jinn"]["guarded_acc_mean"] + sf["beast"]["guarded_acc_mean"]) / 2, 6
                ),
                "combined_raw_acc_mean": round((sf["jinn"]["raw_acc_mean"] + sf["beast"]["raw_acc_mean"]) / 2, 6),
                "frames": sf,
            }

        fr_agg = _agg_frames(fresh_runs)
        sp_agg = _agg_frames(spent_runs)
        arm_results[arm_name] = {
            "parameters_per_frame": n_params,
            "corrector_macs_per_example": macs,
            "seeds": len(seed_list),
            "fresh_confirmation_1200_targets": fr_agg,
            "spent_test_432_reference_only": sp_agg,
            "train_diagnostics_seed0": train_diag_seed0,
            "convergence_diagnostics_seed0": conv_seed0,
        }
        j_fa = fr_agg["frames"]["jinn"]["raw_crit_fa_mean"]
        b_fa = fr_agg["frames"]["beast"]["raw_crit_fa_mean"]
        print(
            f"[depth-study] {arm_name:28s} params={n_params:4d} macs={macs:4d} "
            f"fresh_g={fr_agg['combined_guarded_acc_mean']:.4f} fresh_r={fr_agg['combined_raw_acc_mean']:.4f} "
            f"raw_fa=({j_fa:.1f}/{b_fa:.1f}) "
            f"spent_g={sp_agg['combined_guarded_acc_mean']:.4f}",
            flush=True,
        )

    contrasts: dict[str, Any] = {}
    pairs = [
        ("fixed_mask_ldt", "param_matched_ff_1x"),
        ("fixed_mask_ldt", "recurrent_6_pass_exact"),
        ("fixed_mask_ldt", "recurrent_12_pass_exact"),
        ("fixed_mask_ldt", "recurrent_24_pass_exact"),
        ("param_matched_ff_1x", "recurrent_6_pass_exact"),
        ("param_matched_ff_1x", "recurrent_12_pass_exact"),
        ("param_matched_ff_1x", "recurrent_24_pass_exact"),
        ("recurrent_6_pass_exact", "recurrent_12_pass_exact"),
        ("recurrent_12_pass_exact", "recurrent_24_pass_exact"),
        ("recurrent_6_pass_st", "recurrent_6_pass_exact"),
        ("recurrent_12_pass_st", "recurrent_12_pass_exact"),
        ("recurrent_24_pass_st", "recurrent_24_pass_exact"),
        ("compute_matched_ff_wide6", "recurrent_6_pass_exact"),
        ("compute_matched_ff_wide12", "recurrent_12_pass_exact"),
        ("compute_matched_ff_wide24", "recurrent_24_pass_exact"),
        ("compute_matched_ff_depth6", "recurrent_6_pass_exact"),
        ("compute_matched_ff_depth12", "recurrent_12_pass_exact"),
        ("compute_matched_ff_depth24", "recurrent_24_pass_exact"),
    ]
    for a, b in pairs:
        contrasts[f"{b}_vs_{a}"] = {
            "guarded": _paired_bootstrap_diff(s0_fresh_g[a], s0_fresh_g[b]),
            "raw": _paired_bootstrap_diff(s0_fresh_r[a], s0_fresh_r[b]),
        }

    report = {
        "schema": "research-v2.1-jinn-beast-deep-recurrence-study-v1",
        "source_id": manifest["source_id"],
        "source_revision": manifest["revision"],
        "spent_test_policy": (
            "Original 432 archive test roots are treated as SPENT after Workstream B; all primary "
            "statistical contrasts are evaluated strictly on the fresh 600-root (1,200-target) panel."
        ),
        "fresh_confirmation_panel": fresh_meta,
        "exact_unrolled_bptt_gradient_check": grad_check,
        "arms": arm_results,
        "paired_contrasts_fresh_1200_targets": contrasts,
        "wall_seconds": round(time.monotonic() - t0, 4),
    }
    report["report_sha256"] = digest(canonical(report).encode("utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[depth-study] DONE report_sha256={report['report_sha256']} ({report['wall_seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
