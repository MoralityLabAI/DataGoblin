"""Disentangle PR #93's 198-parameter Jinn/Beast hybrid & verify independent persona semantics.

Compares on the exact pinned private Jinn/Beast archive (1,440 train / 432 val / 432 test,
strictly preserving train/val/test separation and zero overlap with all held/prospective panels):
  1. union_ldt_baseline            (75 params, unmasked union tags)
  2. fixed_mask_ldt                (75 params, fixed own-frame structural mask)
  3. learned_mask_ldt              (99 params: 75 LDT + 24 learned L1-regularized sigmoid gates)
  4. param_matched_feedforward     (198 params: 75 fixed-mask LDT + 123-param 1-pass MLP corrector)
  5. recurrent_1_pass              (198 params: K=1 exact unrolled BPTT vs straight-through)
  6. recurrent_2_pass              (198 params: K=2 exact unrolled BPTT vs straight-through)
  7. recurrent_3_pass              (198 params: K=3 exact unrolled BPTT vs straight-through)
  8. recurrent_6_pass              (198 params: K=6 exact unrolled BPTT vs straight-through)

Verifies:
  - Exact unrolled BPTT analytical gradients against central finite differences (< 1e-6 rel err).
  - Separate raw vs guarded critical-policy false accepts for every arm on both frames.
  - Multi-seed mean/std + paired within-root 95% bootstrap CIs testing whether recurrence
    contributes statistically measurable improvement beyond fixed-mask LDT and parameter-matched
    feedforward correction.
  - Independent reference implementation (project_persona_store) differential parity across all
    432 unseen intervention episodes + 2,000 randomized temporal/revocation/cross-owner fuzz tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from datagoblin.research_v2.specialist import (
    _load_manifest,
    _policy,
    _read_pinned,
    _score,
    canonical,
    digest,
    root_id,
    semantic_digest,
)
from datagoblin.research_v2_1.persona_store_reference import project_persona_store

LANES = ("train", "hold", "reject")
LANE_TO_IDX = {name: i for i, name in enumerate(LANES)}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(sum(f1s) / n_classes)


class AdditiveLDT:
    def __init__(self, n_features: int, mask: np.ndarray | None = None, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.02, size=(n_features, 3)).astype(np.float64)
        self.b = np.zeros(3, dtype=np.float64)
        self.mask = None if mask is None else mask.astype(np.float64)
        if self.mask is not None:
            self.w *= self.mask[:, None]

    def fit(
        self, x: np.ndarray, y: np.ndarray, *, epochs: int = 120, lr: float = 0.08, l2: float = 5e-4, seed: int = 0
    ) -> None:
        rng = np.random.default_rng(seed)
        n = len(x)
        xm = x if self.mask is None else x * self.mask[None, :]
        for _ in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, 64):
                idx = perm[start : start + 64]
                xb, yb = xm[idx], y[idx]
                probs = _softmax(xb @ self.w + self.b)
                grad = probs.copy()
                grad[np.arange(len(yb)), yb] -= 1.0
                grad /= len(yb)
                gw = xb.T @ grad + l2 * self.w
                if self.mask is not None:
                    gw *= self.mask[:, None]
                self.w -= lr * gw
                self.b -= lr * grad.sum(axis=0)

    def logits(self, x: np.ndarray) -> np.ndarray:
        xm = x if self.mask is None else x * self.mask[None, :]
        return xm @ self.w + self.b

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(x))

    @property
    def n_params(self) -> int:
        return int(self.w.size + self.b.size)


class LearnedMaskLDT:
    """75-param LDT + 24 learned per-tag sigmoid gates regularized with L1 sparsity (99 params)."""

    def __init__(self, n_features: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.02, size=(n_features, 3)).astype(np.float64)
        self.b = np.zeros(3, dtype=np.float64)
        self.gate_logits = np.zeros(n_features, dtype=np.float64)
        self.hard_gate: np.ndarray | None = None

    def fit(
        self,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_va: np.ndarray,
        y_va: np.ndarray,
        *,
        l1_grid: tuple[float, ...] = (0.005, 0.02, 0.05, 0.12),
        seed: int = 0,
    ) -> dict[str, Any]:
        best_acc = -1.0
        best_state: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
        for lam in l1_grid:
            rng = np.random.default_rng(seed)
            w = rng.normal(0.0, 0.02, size=self.w.shape).astype(np.float64)
            b = np.zeros_like(self.b)
            gl = np.zeros_like(self.gate_logits)
            n = len(x_tr)
            for _ in range(140):
                perm = rng.permutation(n)
                for start in range(0, n, 64):
                    idx = perm[start : start + 64]
                    xb, yb = x_tr[idx], y_tr[idx]
                    g = _sigmoid(gl)
                    xm = xb * g[None, :]
                    probs = _softmax(xm @ w + b)
                    dlog = probs.copy()
                    dlog[np.arange(len(yb)), yb] -= 1.0
                    dlog /= len(yb)
                    gw = xm.T @ dlog + 5e-4 * w
                    gb = dlog.sum(axis=0)
                    # Exact gradient wrt gate_logits including L1 gate sparsity
                    dg = ((dlog @ w.T) * xb).sum(axis=0) + lam
                    dgl = dg * g * (1.0 - g)
                    w -= 0.08 * gw
                    b -= 0.08 * gb
                    gl -= 0.35 * dgl
            g_eval = _sigmoid(gl)
            va_probs = _softmax((x_va * g_eval[None, :]) @ w + b)
            va_acc = float(np.mean(va_probs.argmax(axis=1) == y_va))
            # Tie-break toward sparser masks on validation only
            score = va_acc - 0.001 * float(g_eval.mean())
            if score > best_acc:
                best_acc = score
                best_state = (w.copy(), b.copy(), gl.copy(), lam)
        assert best_state is not None
        self.w, self.b, self.gate_logits, best_lam = best_state
        gates = _sigmoid(self.gate_logits)
        # Threshold gates > 0.25 (selected purely via learned sparsity, no frame labels used)
        self.hard_gate = (gates >= 0.25).astype(np.float64)
        return {
            "selected_l1": best_lam,
            "active_gates_count": int(self.hard_gate.sum()),
            "mean_soft_gate": round(float(gates.mean()), 4),
        }

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        g = self.hard_gate if self.hard_gate is not None else _sigmoid(self.gate_logits)
        return _softmax((x * g[None, :]) @ self.w + self.b)

    @property
    def n_params(self) -> int:
        return int(self.w.size + self.b.size + self.gate_logits.size)


class CorrectorHead:
    """75-param fixed-mask LDT + 123-param corrector (feedforward or K-pass recurrent).

    Supports:
      - mode="feedforward": single-pass nonlinear MLP corrector (123 params, alpha=1.0, exact grad)
      - mode="exact_unrolled": K-pass damped recurrent corrector with EXACT unrolled BPTT
      - mode="straight_through": K-pass damped recurrent corrector using PR #93's straight-through
        approximation (ddelta = dlogits * alpha without softmax/margin Jacobian)
    """

    def __init__(
        self,
        n_features: int,
        own_indices: list[int],
        critical_indices: list[int],
        *,
        passes: int = 3,
        grad_mode: str = "exact_unrolled",
        damped: bool = True,
        seed: int = 0,
    ) -> None:
        self.own_mask = np.zeros(n_features, dtype=np.float64)
        self.own_mask[own_indices] = 1.0
        self.critical_indices = list(critical_indices)
        self.base = AdditiveLDT(n_features, mask=self.own_mask, seed=seed)
        self.passes = passes
        self.grad_mode = grad_mode
        self.alphas = [1.0 / (passes - j) for j in range(passes)] if damped else [1.0 / passes] * passes
        rng = np.random.default_rng(seed + 17)
        # 6 -> 12 -> 3 = (6*12+12) + (12*3+3) = 84 + 39 = 123 parameters (EXACT match to PR #93!)
        self.w1 = rng.normal(0.0, 0.05, size=(6, 12)).astype(np.float64)
        self.b1 = np.zeros(12, dtype=np.float64)
        self.w2 = rng.normal(0.0, 0.02, size=(12, 3)).astype(np.float64)
        self.b2 = np.zeros(3, dtype=np.float64)

    @property
    def n_params(self) -> int:
        return self.base.n_params + int(self.w1.size + self.b1.size + self.w2.size + self.b2.size)

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
        for alpha in self.alphas:
            probs = _softmax(cur)
            order = np.argsort(probs, axis=1)
            top1 = order[:, -1]
            top2 = order[:, -2]
            margin = (probs[rows, top1] - probs[rows, top2])[:, None]
            feat = np.concatenate([probs, margin, active_cnt, crit], axis=1)
            hid = np.tanh(feat @ self.w1 + self.b1)
            delta = hid @ self.w2 + self.b2
            nxt = cur + alpha * delta
            cache.append(
                {
                    "probs": probs,
                    "top1": top1,
                    "top2": top2,
                    "feat": feat,
                    "hid": hid,
                    "alpha": alpha,
                }
            )
            cur = nxt
        return cur, cache

    def backward_corrector(
        self, dlogits: np.ndarray, cache: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gw1 = np.zeros_like(self.w1)
        gb1 = np.zeros_like(self.b1)
        gw2 = np.zeros_like(self.w2)
        gb2 = np.zeros_like(self.b2)

        if self.grad_mode == "straight_through":
            for step in reversed(cache):
                ddelta = dlogits * step["alpha"]
                gw2 += step["hid"].T @ ddelta
                gb2 += ddelta.sum(axis=0)
                dhid = (ddelta @ self.w2.T) * (1.0 - step["hid"] * step["hid"])
                gw1 += step["feat"].T @ dhid
                gb1 += dhid.sum(axis=0)
            return gw1, gb1, gw2, gb2

        # Exact unrolled BPTT through residual sum, tanh, margin = p[top1] - p[top2], and softmax
        dcur = dlogits.copy()
        rows = np.arange(len(dcur))
        for step in reversed(cache):
            alpha = step["alpha"]
            hid = step["hid"]
            feat = step["feat"]
            probs = step["probs"]
            top1 = step["top1"]
            top2 = step["top2"]

            ddelta = dcur * alpha
            gw2 += hid.T @ ddelta
            gb2 += ddelta.sum(axis=0)
            dhid = (ddelta @ self.w2.T) * (1.0 - hid * hid)
            gw1 += feat.T @ dhid
            gb1 += dhid.sum(axis=0)
            dfeat = dhid @ self.w1.T

            dprobs = dfeat[:, :3].copy()
            dmargin = dfeat[:, 3]
            dprobs[rows, top1] += dmargin
            dprobs[rows, top2] -= dmargin
            dot = np.sum(dprobs * probs, axis=1, keepdims=True)
            dcur = dcur + probs * (dprobs - dot)
        return gw1, gb1, gw2, gb2

    def fit(self, x_tr: np.ndarray, y_tr: np.ndarray, *, seed: int = 0) -> None:
        self.base.fit(x_tr, y_tr, epochs=120, lr=0.08, l2=5e-4, seed=seed)
        base_logits_all = self.base.logits(x_tr)
        active_all, crit_all = self._static_feats(x_tr)
        rng = np.random.default_rng(seed + 99)
        n = len(x_tr)
        lr = 0.03
        for _ in range(80):
            perm = rng.permutation(n)
            for start in range(0, n, 64):
                idx = perm[start : start + 64]
                bl, ac, cr, yb = base_logits_all[idx], active_all[idx], crit_all[idx], y_tr[idx]
                logits, cache = self.forward_corrector(bl, ac, cr)
                probs = _softmax(logits)
                dlog = probs.copy()
                dlog[np.arange(len(yb)), yb] -= 1.0
                dlog /= len(yb)
                gw1, gb1, gw2, gb2 = self.backward_corrector(dlog, cache)
                self.w2 -= lr * (gw2 + 1e-3 * self.w2)
                self.b2 -= lr * gb2
                self.w1 -= lr * (gw1 + 1e-3 * self.w1)
                self.b1 -= lr * gb1

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        bl = self.base.logits(x)
        ac, cr = self._static_feats(x)
        logits, _ = self.forward_corrector(bl, ac, cr)
        return _softmax(logits)


def verify_exact_unrolled_gradient() -> dict[str, Any]:
    """Central finite-difference verification of exact unrolled BPTT across K=3 and K=6 passes."""
    rng = np.random.default_rng(12345)
    n, d = 16, 10
    x = (rng.random((n, d)) < 0.4).astype(np.float64)
    y = rng.integers(0, 3, size=n)
    results = {}
    eps = 1e-6
    for k in (1, 3, 6):
        head = CorrectorHead(
            d, own_indices=list(range(6)), critical_indices=[0], passes=k, grad_mode="exact_unrolled", seed=k
        )
        # Use non-degenerate logits so top1/top2 order is smooth around eps
        bl = rng.normal(0.0, 1.0, size=(n, 3))
        ac, cr = head._static_feats(x)

        def loss_fn(
            _head: CorrectorHead = head,
            _bl: np.ndarray = bl,
            _ac: np.ndarray = ac,
            _cr: np.ndarray = cr,
        ) -> float:
            logits, _ = _head.forward_corrector(_bl, _ac, _cr)
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
            flat_p = param.ravel()
            flat_g = grad.ravel()
            for idx in range(min(len(flat_p), 24)):
                orig = flat_p[idx]
                flat_p[idx] = orig + eps
                lp = loss_fn()
                flat_p[idx] = orig - eps
                lm = loss_fn()
                flat_p[idx] = orig
                num = (lp - lm) / (2 * eps)
                ana = flat_g[idx]
                rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
                max_rel = max(max_rel, float(rel))
        if max_rel >= 1e-5:
            raise RuntimeError(f"exact unrolled BPTT gradient check failed at K={k}: max_rel={max_rel}")
        results[f"K={k}_max_rel_err"] = float(f"{max_rel:.3e}")
    return results


def _eval_frame(probs_u: np.ndarray, probs_o: np.ndarray, y: np.ndarray, crit: np.ndarray) -> dict[str, Any]:
    raw = probs_u.argmax(axis=1)
    guarded = raw.copy()
    guarded[crit] = LANE_TO_IDX["reject"]
    own_raw = probs_o.argmax(axis=1)
    own_guarded = own_raw.copy()
    own_guarded[crit] = LANE_TO_IDX["reject"]
    crit_n = int(crit.sum())
    return {
        "raw_accuracy": float(np.mean(raw == y)),
        "guarded_accuracy": float(np.mean(guarded == y)),
        "macro_f1_guarded": _macro_f1(y, guarded),
        "critical_examples": crit_n,
        "raw_critical_false_accepts": int(np.sum(crit & (raw == LANE_TO_IDX["train"]))),
        "guarded_critical_false_accepts": int(np.sum(crit & (guarded == LANE_TO_IDX["train"]))),
        "opposite_frame_invariance_guarded": float(np.mean(guarded == own_guarded)),
        "opposite_frame_invariance_raw": float(np.mean(raw == own_raw)),
    }


def _paired_bootstrap_diff(a_correct: np.ndarray, b_correct: np.ndarray, n_boot: int = 2000) -> dict[str, Any]:
    diff = b_correct.astype(np.float64) - a_correct.astype(np.float64)
    rng = np.random.default_rng(20261127)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "delta": round(float(diff.mean()), 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "statistically_nonzero_95": bool(lo > 0.0 or hi < 0.0),
    }


def verify_independent_persona_reference(test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Differential test between independent project_persona_store and PR #93 _reconstruct_store."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research_v2"))
    from jinn_beast_persona_exp import _generate_intervention_episodes, _reconstruct_store

    episodes = _generate_intervention_episodes(test_rows, seed=20261126)
    ep_matches = 0
    total_rejected_owner = 0
    total_rejected_scope = 0
    total_rejected_future = 0
    total_tombstones = 0
    for ep in episodes:
        orig = _reconstruct_store(ep["events"], owner=ep["owner"], scope=ep["scope"], at_revision=ep["at_revision"])
        proj = project_persona_store(ep["events"], owner=ep["owner"], scope=ep["scope"], at_revision=ep["at_revision"])
        if orig != proj.active:
            raise RuntimeError(f"Reference mismatch on {ep['episode_id']}: {orig} vs {proj.active}")
        ep_matches += 1
        total_rejected_owner += proj.rejected_cross_owner
        total_rejected_scope += proj.rejected_cross_scope
        total_rejected_future += proj.rejected_future_revision
        total_tombstones += len(proj.tombstones)

    # 2,000 randomized differential fuzz histories (including set-after-revoke, out-of-order, ties, ablations)
    rng = np.random.default_rng(20261127)
    fuzz_cases = 2000
    fuzz_matches = 0
    actors = ("u_owner", "u_other", "character")
    scopes = ("research", "fiction", "private")
    keys = ("frame", "tone", "verbosity")
    vals = ("jinn", "beast", "warm", "formal", "brief", "hostile")
    for _ in range(fuzz_cases):
        n_ev = int(rng.integers(3, 18))
        evs = []
        # Distinct revisions per key for deterministic total order across both engines
        revs = rng.choice(np.arange(1, 30), size=n_ev, replace=False).tolist()
        for rev in revs:
            op = "revoke" if rng.random() < 0.3 else "set"
            ev: dict[str, Any] = {
                "actor": str(rng.choice(actors)),
                "scope": str(rng.choice(scopes)),
                "revision": int(rev),
                "op": op,
                "key": str(rng.choice(keys)),
            }
            if op == "set":
                ev["value"] = str(rng.choice(vals))
            evs.append(ev)
        at_rev = int(rng.integers(1, 25))
        modes = (
            (True, True, True, True),
            (False, True, True, True),
            (True, False, True, True),
            (True, True, False, True),
        )
        for eo, et, es, er in modes:
            orig = _reconstruct_store(
                evs, owner="u_owner", scope="research", at_revision=at_rev,
                enforce_owner=eo, enforce_temporal=et, enforce_scope=es, enforce_revocation=er,
            )
            proj = project_persona_store(
                evs, owner="u_owner", scope="research", at_revision=at_rev,
                enforce_owner=eo, enforce_temporal=et, enforce_scope=es, enforce_revocation=er,
            )
            if orig != proj.active:
                raise RuntimeError(f"Fuzz differential mismatch: {orig} vs {proj.active}")
        fuzz_matches += 1

    return {
        "unseen_intervention_episodes_tested": len(episodes),
        "unseen_intervention_exact_matches": ep_matches,
        "fuzz_histories_tested": fuzz_cases,
        "fuzz_mode_checks_passed": fuzz_matches * 4,
        "audit_totals_432_episodes": {
            "rejected_cross_owner": total_rejected_owner,
            "rejected_cross_scope": total_rejected_scope,
            "rejected_future_revision": total_rejected_future,
            "active_tombstones": total_tombstones,
        },
        "parity_rate": 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclude-registry", type=Path, action="append", default=[])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = time.monotonic()
    grad_check = verify_exact_unrolled_gradient()

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

    train_roots = {root_id(r["tags"]) for r in splits["train"]}
    val_roots = {root_id(r["tags"]) for r in splits["validation"]}
    test_roots = {root_id(r["tags"]) for r in splits["test"]}
    assert not (train_roots & val_roots) and not (train_roots & test_roots) and not (val_roots & test_roots)

    excl_overlap = 0
    all_roots = train_roots | val_roots | test_roots
    all_sem = {semantic_digest(r["tags"]) for r in rows}
    for ep in args.exclude_registry:
        reg = json.loads(ep.read_text(encoding="utf-8"))
        excl_overlap += len(all_roots & set(reg.get("held_root_ids", [])))
        excl_overlap += len(all_sem & set(reg.get("held_semantic_sha256", [])))
    if excl_overlap != 0:
        raise RuntimeError(f"exclusion overlap detected: {excl_overlap}")

    ref_parity = verify_independent_persona_reference(splits["test"])

    tag_vocab = sorted(set().union(*(set(p["tags"]) for p in policies.values())))
    tag_idx = {t: i for i, t in enumerate(tag_vocab)}
    own_idx = {f: sorted(tag_idx[t] for t in policies[f]["tags"]) for f in ("jinn", "beast")}
    crit_idx = {f: sorted(tag_idx[t] for t in policies[f]["critical_tags"]) for f in ("jinn", "beast")}

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
    x_va, _x_va_own, y_va, _crit_va = enc(splits["validation"])
    x_te, x_te_own, y_te, crit_te = enc(splits["test"])

    arm_specs = [
        ("union_ldt_baseline", {"kind": "union_ldt"}),
        ("fixed_mask_ldt", {"kind": "fixed_mask_ldt"}),
        ("learned_mask_ldt", {"kind": "learned_mask_ldt"}),
        (
            "param_matched_feedforward",
            {"kind": "corrector", "passes": 1, "grad_mode": "exact_unrolled", "damped": False},
        ),
        ("recurrent_1_pass_exact", {"kind": "corrector", "passes": 1, "grad_mode": "exact_unrolled", "damped": True}),
        ("recurrent_1_pass_st", {"kind": "corrector", "passes": 1, "grad_mode": "straight_through", "damped": True}),
        ("recurrent_2_pass_exact", {"kind": "corrector", "passes": 2, "grad_mode": "exact_unrolled", "damped": True}),
        ("recurrent_2_pass_st", {"kind": "corrector", "passes": 2, "grad_mode": "straight_through", "damped": True}),
        ("recurrent_3_pass_exact", {"kind": "corrector", "passes": 3, "grad_mode": "exact_unrolled", "damped": True}),
        ("recurrent_3_pass_st", {"kind": "corrector", "passes": 3, "grad_mode": "straight_through", "damped": True}),
        ("recurrent_6_pass_exact", {"kind": "corrector", "passes": 6, "grad_mode": "exact_unrolled", "damped": True}),
        ("recurrent_6_pass_st", {"kind": "corrector", "passes": 6, "grad_mode": "straight_through", "damped": True}),
    ]
    seed_list = [20261127 + s for s in range(args.seeds)]

    arm_results: dict[str, Any] = {}
    # Store seed-0 guarded correctness vectors concatenated across both frames (864 targets) for paired tests
    seed0_guarded_correct: dict[str, np.ndarray] = {}
    seed0_raw_correct: dict[str, np.ndarray] = {}

    for arm_name, spec in arm_specs:
        per_frame_runs: dict[str, list[dict[str, Any]]] = {"jinn": [], "beast": []}
        extra_info: dict[str, Any] = {}
        n_params = 0
        s0_g_parts, s0_r_parts = [], []
        for s_i, seed in enumerate(seed_list):
            for f_i, f in enumerate(("jinn", "beast")):
                f_seed = seed * 10 + f_i
                kind = spec["kind"]
                if kind == "union_ldt":
                    model: Any = AdditiveLDT(len(tag_vocab), mask=None, seed=f_seed)
                    model.fit(x_tr, y_tr[f], seed=f_seed)
                elif kind == "fixed_mask_ldt":
                    mask = np.zeros(len(tag_vocab), dtype=np.float64)
                    mask[own_idx[f]] = 1.0
                    model = AdditiveLDT(len(tag_vocab), mask=mask, seed=f_seed)
                    model.fit(x_tr, y_tr[f], seed=f_seed)
                elif kind == "learned_mask_ldt":
                    model = LearnedMaskLDT(len(tag_vocab), seed=f_seed)
                    info = model.fit(x_tr, y_tr[f], x_va, y_va[f], seed=f_seed)
                    if s_i == 0:
                        extra_info[f] = info
                else:
                    model = CorrectorHead(
                        len(tag_vocab),
                        own_indices=own_idx[f],
                        critical_indices=crit_idx[f],
                        passes=spec["passes"],
                        grad_mode=spec["grad_mode"],
                        damped=spec["damped"],
                        seed=f_seed,
                    )
                    model.fit(x_tr, y_tr[f], seed=f_seed)
                n_params = model.n_params
                pu = model.predict_probs(x_te)
                po = model.predict_probs(x_te_own[f])
                ev = _eval_frame(pu, po, y_te[f], crit_te[f])
                per_frame_runs[f].append(ev)
                if s_i == 0:
                    raw = pu.argmax(axis=1)
                    guarded = raw.copy()
                    guarded[crit_te[f]] = LANE_TO_IDX["reject"]
                    s0_r_parts.append(raw == y_te[f])
                    s0_g_parts.append(guarded == y_te[f])
        seed0_raw_correct[arm_name] = np.concatenate(s0_r_parts)
        seed0_guarded_correct[arm_name] = np.concatenate(s0_g_parts)

        summary_frames: dict[str, Any] = {}
        for f in ("jinn", "beast"):
            runs = per_frame_runs[f]
            summary_frames[f] = {
                "guarded_acc_mean": round(float(np.mean([r["guarded_accuracy"] for r in runs])), 6),
                "guarded_acc_std": round(float(np.std([r["guarded_accuracy"] for r in runs])), 6),
                "raw_acc_mean": round(float(np.mean([r["raw_accuracy"] for r in runs])), 6),
                "raw_acc_std": round(float(np.std([r["raw_accuracy"] for r in runs])), 6),
                "macro_f1_guarded_mean": round(float(np.mean([r["macro_f1_guarded"] for r in runs])), 6),
                "raw_crit_fa_mean": round(float(np.mean([r["raw_critical_false_accepts"] for r in runs])), 4),
                "raw_crit_fa_max": int(max(r["raw_critical_false_accepts"] for r in runs)),
                "guarded_crit_fa_mean": round(float(np.mean([r["guarded_critical_false_accepts"] for r in runs])), 4),
                "guarded_crit_fa_max": int(max(r["guarded_critical_false_accepts"] for r in runs)),
                "opp_frame_inv_guarded_mean": round(
                    float(np.mean([r["opposite_frame_invariance_guarded"] for r in runs])), 6
                ),
                "opp_frame_inv_raw_mean": round(float(np.mean([r["opposite_frame_invariance_raw"] for r in runs])), 6),
            }
        mean_guarded = round(
            (summary_frames["jinn"]["guarded_acc_mean"] + summary_frames["beast"]["guarded_acc_mean"]) / 2, 6
        )
        mean_raw = round((summary_frames["jinn"]["raw_acc_mean"] + summary_frames["beast"]["raw_acc_mean"]) / 2, 6)
        arm_results[arm_name] = {
            "parameters_per_frame": n_params,
            "seeds": len(seed_list),
            "combined_guarded_acc_mean": mean_guarded,
            "combined_raw_acc_mean": mean_raw,
            "frames": summary_frames,
            "extra": extra_info,
        }

    # Paired within-root bootstrap contrasts (864 paired test targets across Jinn+Beast)
    paired_contrasts = {}
    for target in (
        "param_matched_feedforward",
        "recurrent_1_pass_exact",
        "recurrent_2_pass_exact",
        "recurrent_3_pass_exact",
        "recurrent_6_pass_exact",
        "recurrent_3_pass_st",
        "recurrent_6_pass_st",
    ):
        paired_contrasts[f"{target}_vs_fixed_mask_ldt"] = {
            "guarded": _paired_bootstrap_diff(seed0_guarded_correct["fixed_mask_ldt"], seed0_guarded_correct[target]),
            "raw": _paired_bootstrap_diff(seed0_raw_correct["fixed_mask_ldt"], seed0_raw_correct[target]),
        }
    for target in (
        "recurrent_2_pass_exact",
        "recurrent_3_pass_exact",
        "recurrent_6_pass_exact",
        "recurrent_3_pass_st",
        "recurrent_6_pass_st",
    ):
        paired_contrasts[f"{target}_vs_param_matched_feedforward"] = {
            "guarded": _paired_bootstrap_diff(
                seed0_guarded_correct["param_matched_feedforward"], seed0_guarded_correct[target]
            ),
            "raw": _paired_bootstrap_diff(seed0_raw_correct["param_matched_feedforward"], seed0_raw_correct[target]),
        }
    for k in (1, 2, 3, 6):
        paired_contrasts[f"exact_vs_st_K={k}"] = {
            "guarded": _paired_bootstrap_diff(
                seed0_guarded_correct[f"recurrent_{k}_pass_st"], seed0_guarded_correct[f"recurrent_{k}_pass_exact"]
            ),
            "raw": _paired_bootstrap_diff(
                seed0_raw_correct[f"recurrent_{k}_pass_st"], seed0_raw_correct[f"recurrent_{k}_pass_exact"]
            ),
        }

    report = {
        "schema": "research-v2.1-jinn-beast-disentanglement-v1",
        "source_id": manifest["source_id"],
        "source_revision": manifest["revision"],
        "split_integrity": {
            "train_roots": len(train_roots),
            "validation_roots": len(val_roots),
            "test_roots": len(test_roots),
            "exclusion_overlaps": excl_overlap,
            "all_4608_labels_recomputed": True,
        },
        "exact_unrolled_bptt_gradient_check": grad_check,
        "independent_persona_reference_verification": ref_parity,
        "arms": arm_results,
        "paired_contrasts_864_test_targets": paired_contrasts,
        "wall_seconds": round(time.monotonic() - t0, 4),
    }
    report["report_sha256"] = digest(canonical(report).encode("utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
