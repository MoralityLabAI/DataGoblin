"""MMS-07 Fable review: executable mathematical checks.

Pure NumPy + stdlib. These are deterministic mathematical and simulation checks
that accompany research/dg_gym/MMS_07_FABLE_REVIEW.md. They are NOT training
experiments on any gym and produce no learning-benefit evidence.

Checks that touch repository code import the pinned DataGoblin modules
(factor_maps, fast_gradient_bank, mms_utility) and exercise them unmodified.

    python research/dg_gym/mms07_math_checks.py --dg-src src --out receipt.json

Each check returns a dict of numbers and raises AssertionError when a stated
mathematical claim fails. Timing fields are informational and excluded from
the pass/fail status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CHECKS: dict[str, Any] = {}


def check(fn):
    CHECKS[fn.__name__] = fn
    return fn


# ----------------------------------------------------------------------------
# A small tied recurrent network with hand-written reverse and forward mode.
#   h0 = tanh(x E^T + e);  z_k = h_k W^T + b;  h_{k+1} = tanh(h_k + z_k)
#   y = h_K H^T + c;  loss = 0.5 * sum((y - t)^2) / B
# This is the LBF-20D `recur3` topology (tied Linear block, residual tanh).
# ----------------------------------------------------------------------------
class TiedNet:
    def __init__(self, rng, din=12, width=16, dout=4, visits=3, scale=0.6):
        self.K = visits
        self.p = {
            "E": rng.standard_normal((width, din)) * scale / math.sqrt(din),
            "e": rng.standard_normal(width) * 0.1,
            "W": rng.standard_normal((width, width)) * scale / math.sqrt(width),
            "b": rng.standard_normal(width) * 0.1,
            "H": rng.standard_normal((dout, width)) * scale / math.sqrt(width),
            "c": rng.standard_normal(dout) * 0.1,
        }

    def forward(self, x, p=None):
        p = self.p if p is None else p
        hs = [np.tanh(x @ p["E"].T + p["e"])]
        zs = []
        for _ in range(self.K):
            z = hs[-1] @ p["W"].T + p["b"]
            zs.append(z)
            hs.append(np.tanh(hs[-1] + z))
        y = hs[-1] @ p["H"].T + p["c"]
        return y, hs, zs

    def loss(self, x, t, p=None):
        y, _, _ = self.forward(x, p)
        return 0.5 * float(((y - t) ** 2).sum()) / len(x)

    def backward(self, x, t, inject=None):
        """Reverse mode. inject=(m, delta) adds delta to dL/dz_m exactly as a
        PyTorch tensor hook on the module output would (upstream flow is modified)."""
        p = self.p
        y, hs, _ = self.forward(x)
        dy = (y - t) / len(x)
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["H"] = dy.T @ hs[-1]
        g["c"] = dy.sum(0)
        dh = dy @ p["H"]
        cots = [None] * self.K
        for k in range(self.K - 1, -1, -1):
            da = dh * (1.0 - hs[k + 1] ** 2)
            cz = da.copy()
            cots[k] = da.copy()
            if inject is not None and inject[0] == k:
                cz = cz + inject[1]
            g["W"] += cz.T @ hs[k]
            g["b"] += cz.sum(0)
            dh = da + cz @ p["W"]
        dpre = dh * (1.0 - hs[0] ** 2)
        g["E"] = dpre.T @ x
        g["e"] = dpre.sum(0)
        return g, cots, hs

    def pull_delta(self, x, m, delta, truncate=None):
        """d(tau) = parameter-space image of an injection delta at visit m, with the
        upstream propagation cut after `truncate` earlier visits (None = full)."""
        p = self.p
        _, hs, _ = self.forward(x)
        g = {k: np.zeros_like(v) for k, v in p.items()}
        cz = delta
        g["W"] += cz.T @ hs[m]
        g["b"] += cz.sum(0)
        dh = cz @ p["W"]
        steps = 0
        for k in range(m - 1, -1, -1):
            if truncate is not None and steps >= truncate:
                return g
            da = dh * (1.0 - hs[k + 1] ** 2)
            g["W"] += da.T @ hs[k]
            g["b"] += da.sum(0)
            dh = da + da @ p["W"]
            steps += 1
        if truncate is None or steps < truncate:
            dpre = dh * (1.0 - hs[0] ** 2)
            g["E"] = dpre.T @ x
            g["e"] = dpre.sum(0)
        return g

    def jvp_site(self, x, m, v):
        """Forward mode: q = D_theta z_m [v] (total derivative, all upstream paths)."""
        p = self.p
        _, hs, _ = self.forward(x)
        dh = (1.0 - hs[0] ** 2) * (x @ v["E"].T + v["e"])
        for k in range(m + 1):
            dz = dh @ p["W"].T + hs[k] @ v["W"].T + v["b"]
            if k == m:
                return dz
            dh = (1.0 - hs[k + 1] ** 2) * (dh + dz)
        raise AssertionError


def pdot(a, b):
    return float(sum((a[k] * b[k]).sum() for k in a))


def padd(a, b, s=1.0):
    return {k: a[k] + s * b[k] for k in a}


# ----------------------------------------------------------------------------
# Problem 1
# ----------------------------------------------------------------------------
@check
def p1_scalar_counterexample():
    """The registered counterexample: positive local alignment, reversed parameter gradient."""
    def L(th):
        return ((th + 1) ** 2 + (th - 1.1) ** 2) / 2

    c1, c2 = 1.0, -1.1
    g = c1 + c2
    delta = 0.2
    d = delta  # A_1 = dz_1/dtheta = 1
    eta = 0.1
    out = {
        "g": g, "g_steered": g + d, "local_dot": delta * c1, "param_dot": g * d,
        "L0": L(0.0), "L_ordinary": L(-eta * g), "L_steered": L(-eta * (g + d)),
        # decomposition q = K c + rho with K = A A^T = 1 and rho = A r = c2
        "K_c": 1.0 * c1, "rho": c2, "q": 1.0 * c1 + c2, "certificate_q_dot_delta": (c1 + c2) * delta,
    }
    assert abs(out["L0"] - 1.105) < 1e-12 and abs(out["L_ordinary"] - 1.1041) < 1e-12
    assert abs(out["L_steered"] - 1.1061) < 1e-12
    assert out["local_dot"] > 0 > out["param_dot"] and out["certificate_q_dot_delta"] < 0
    return out


@check
def p1_anisotropy_counterexample():
    """Single untied cut site (no off-site paths): local alignment still fails when the
    site kernel K = A A^T is anisotropic. z = A theta, A = diag(1, 10)."""
    A = np.diag([1.0, 10.0])
    t = np.array([-1.0, -0.1])
    def L(th):
        return 0.5 * float(((A @ th - t) ** 2).sum())

    th = np.zeros(2)
    c = A @ th - t  # cotangent at the site
    g = A.T @ c
    delta = 0.2 * np.array([1.0, -0.5])
    d = A.T @ delta
    eta = 0.005
    out = {
        "c": c.tolist(), "delta": delta.tolist(),
        "cos_local": float(delta @ c / np.linalg.norm(delta) / np.linalg.norm(c)),
        "param_dot": float(g @ d),
        "L0": L(th), "L_ordinary": L(th - eta * g), "L_steered": L(th - eta * (g + d)),
        "kappa_K": 100.0, "cone_threshold": (100 - 1) / (100 + 1),
    }
    assert out["cos_local"] > 0.8 and out["param_dot"] < 0
    assert out["L_steered"] > out["L_ordinary"]
    assert out["cos_local"] < out["cone_threshold"]  # consistent with Theorem 1.2
    return out


@check
def p1_adjoint_identity_tied_net():
    """<v, d> = <q, delta> with q = D_theta z_m [v]; local variant; truncation family."""
    rng = np.random.default_rng(7)
    worst = 0.0
    worst_fd = 0.0
    worst_ratio = 0.0
    for _ in range(20):
        net = TiedNet(rng)
        B = 8
        x = rng.standard_normal((B, 12))
        t = rng.standard_normal((B, 4))
        m = 1
        delta = rng.standard_normal((B, 16)) * 0.1
        g, _, hs = net.backward(x, t)
        g_inj, _, _ = net.backward(x, t, inject=(m, delta))
        d_hook = padd(g_inj, g, -1.0)
        d_full = net.pull_delta(x, m, delta)
        assert max(np.abs(d_hook[k] - d_full[k]).max() for k in g) < 1e-12
        v = {k: rng.standard_normal(val.shape) for k, val in g.items()}
        q = net.jvp_site(x, m, v)
        lhs, rhs = pdot(v, d_full), float((q * delta).sum())
        worst = max(worst, abs(lhs - rhs) / (abs(lhs) + 1e-12))
        # forward-mode q against central finite differences
        eps = 1e-6
        zp = net.forward(x, padd(net.p, v, eps))[2][m]
        zm = net.forward(x, padd(net.p, v, -eps))[2][m]
        worst_fd = max(worst_fd, float(np.abs((zp - zm) / (2 * eps) - q).max()))
        # local (truncate=0) certificate: q_loc = x_m V_W^T + v_b
        d_loc = net.pull_delta(x, m, delta, truncate=0)
        q_loc = hs[m] @ v["W"].T + v["b"]
        assert abs(pdot(v, d_loc) - float((q_loc * delta).sum())) < 1e-10
        # Theorem 1.5 with m = 2: |d_W - d_W^loc|_F <= |delta|_F |W| sum_j |I+W|^(m-1-j) |X_j|
        m2 = 2
        dfull2, dloc2 = net.pull_delta(x, m2, delta), net.pull_delta(x, m2, delta, truncate=0)
        nW = np.linalg.norm(net.p["W"], 2)
        nIW = np.linalg.norm(np.eye(16) + net.p["W"], 2)
        bound = np.linalg.norm(delta) * nW * sum(nIW ** (m2 - 1 - j) * np.linalg.norm(hs[j], 2) for j in range(m2))
        gap = np.linalg.norm(dfull2["W"] - dloc2["W"])
        assert gap <= bound + 1e-12
        worst_ratio = max(worst_ratio, gap / bound)
    assert worst < 1e-10 and worst_fd < 1e-6
    return {"max_rel_error_adjoint": worst, "max_abs_error_jvp_vs_fd": worst_fd, "trials": 20,
            "max_truncation_gap_over_bound": worst_ratio}


@check
def p1_local_alignment_failure_rate_synthetic():
    """SYNTHETIC ILLUSTRATION on random-init 3-visit tied tanh nets (not a gym measurement).
    delta_b = rho*|c_b|*(cos(phi) c_hat + sin(phi) n_hat), n_hat random orthogonal.
    Report how often positive local alignment gives negative parameter-space alignment,
    and how aligned the cotangent c is with the pushforward q = A g."""
    rng = np.random.default_rng(11)
    rho = 0.2
    grid = [0.1, 0.3, 0.6, 0.9, 1.0]
    neg_full = {c: 0 for c in grid}
    neg_loc = {c: 0 for c in grid}
    per_sample = {c: [] for c in grid}
    cos_cq, cos_cq_loc = [], []
    n_trials = 400
    for _ in range(n_trials):
        net = TiedNet(rng, scale=1.2)
        B = 32
        x = rng.standard_normal((B, 12))
        t = rng.standard_normal((B, 4))
        m = 1
        g, cots, hs = net.backward(x, t)
        c = cots[m]
        cn = np.linalg.norm(c, axis=1, keepdims=True)
        chat = c / cn
        q = net.jvp_site(x, m, g)
        q_loc = hs[m] @ g["W"].T + g["b"]
        cos_cq.append(float((c * q).sum() / np.linalg.norm(c) / np.linalg.norm(q)))
        cos_cq_loc.append(float((c * q_loc).sum() / np.linalg.norm(c) / np.linalg.norm(q_loc)))
        n = rng.standard_normal(c.shape)
        n -= (n * chat).sum(1, keepdims=True) * chat
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        for cosphi in grid:
            delta = rho * cn * (cosphi * chat + math.sqrt(1 - cosphi**2) * n)
            assert (delta * c).sum() > 0
            if (q * delta).sum() < 0:
                neg_full[cosphi] += 1
            if (q_loc * delta).sum() < 0:
                neg_loc[cosphi] += 1
            per_sample[cosphi].append(float(((q * delta).sum(1) < 0).mean()))
    return {
        "trials": n_trials,
        "frac_negative_param_alignment_full_propagation": {str(k): v / n_trials for k, v in neg_full.items()},
        "frac_negative_param_alignment_local_injection": {str(k): v / n_trials for k, v in neg_loc.items()},
        "per_sample_frac_with_negative_<q_b,delta_b>": {str(k): float(np.mean(v)) for k, v in per_sample.items()},
        "batch_cos(c, q_full) quartiles": np.quantile(cos_cq, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist(),
        "batch_cos(c, q_local) quartiles": np.quantile(cos_cq_loc, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist(),
    }


@check
def p1_first_order_prediction_and_dominance():
    """Loss change relative to the ordinary SGD step equals -eta <q, delta> + O(eta^2);
    with per-sample norm budgets the first-order optimum is delta_b parallel to q_b."""
    rng = np.random.default_rng(3)
    net = TiedNet(rng, scale=1.0)
    B, m, rho, eta = 16, 1, 0.2, 1e-4
    x = rng.standard_normal((B, 12))
    t = rng.standard_normal((B, 4))
    g, cots, _ = net.backward(x, t)
    c = cots[m]
    cn = np.linalg.norm(c, axis=1, keepdims=True)
    q = net.jvp_site(x, m, g)
    L_ord = net.loss(x, t, padd(net.p, g, -eta))
    res = {}
    cands = {
        "collinear_c": rho * c,
        "q_aligned": rho * cn * q / np.linalg.norm(q, axis=1, keepdims=True),
        "anti_q": -rho * cn * q / np.linalg.norm(q, axis=1, keepdims=True),
    }
    n = rng.standard_normal(c.shape)
    n -= (n * q).sum(1, keepdims=True) * q / (q * q).sum(1, keepdims=True)
    cands["orthogonal_to_q"] = rho * cn * n / np.linalg.norm(n, axis=1, keepdims=True)
    for name, delta in cands.items():
        d = net.pull_delta(x, m, delta)
        actual = net.loss(x, t, padd(net.p, padd(g, d), -eta)) - L_ord
        pred = -eta * float((q * delta).sum())
        res[name] = {"predicted_first_order": pred, "actual": actual}
        assert abs(actual - pred) <= 5e-3 * abs(pred) + 5e-9, (name, actual, pred)
    assert res["q_aligned"]["actual"] <= res["collinear_c"]["actual"]
    assert abs(res["orthogonal_to_q"]["actual"]) < 1e-3 * abs(res["q_aligned"]["actual"])
    res["efficiency_of_collinear_c_vs_q_optimum"] = (
        res["collinear_c"]["predicted_first_order"] / res["q_aligned"]["predicted_first_order"]
    )
    return res


@check
def p1_cone_bound_tightness():
    """delta^T K c >= |delta||c| ((M+mu)/2 cos(phi) - (M-mu)/2), with equality attained."""
    rng = np.random.default_rng(5)
    min_slack = np.inf
    for _ in range(20000):
        n = int(rng.integers(2, 7))
        Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
        lam = np.exp(rng.uniform(-3, 3, n))
        K = (Q * lam) @ Q.T
        c, dl = rng.standard_normal(n), rng.standard_normal(n)
        S = np.linalg.qr(np.stack([c, dl], 1))[0]
        ev = np.linalg.eigvalsh(S.T @ K @ S)
        mu, M = ev[0], ev[-1]
        cosphi = c @ dl / np.linalg.norm(c) / np.linalg.norm(dl)
        bound = np.linalg.norm(c) * np.linalg.norm(dl) * ((M + mu) / 2 * cosphi - (M - mu) / 2)
        min_slack = min(min_slack, (dl @ K @ c - bound) / (np.linalg.norm(c) * np.linalg.norm(dl) * M))
    assert min_slack > -1e-9
    # tight instance: K = diag(M, mu), c at angle th from axis 1, delta = reflection across axis 2
    M, mu, phi = 9.0, 1.0, math.acos(0.8)
    th = (math.pi - phi) / 2
    c = np.array([math.cos(th), math.sin(th)])
    dl = np.array([-math.cos(th), math.sin(th)])
    K = np.diag([M, mu])
    eq = dl @ K @ c - ((M + mu) / 2 * math.cos(phi) - (M - mu) / 2)
    assert abs(c @ dl - math.cos(phi)) < 1e-12 and abs(eq) < 1e-12
    return {"min_normalized_slack_random": float(min_slack), "tight_instance_residual": float(eq),
            "tight_instance_cos": 0.8, "tight_instance_threshold_(kappa-1)/(kappa+1)": 0.8,
            "tight_instance_value": float(dl @ K @ c)}


@check
def p1_tied_recurrence_local_vs_full():
    """h1=(1+w)x, z1=w*h1. Local dz1/dw = h1; total dz1/dw = h1 + w*x.
    For w in (-1,-1/2) the two have opposite signs although |dz1/dz0| = |w| < 1."""
    x = 1.0
    rows = []
    for w in (-0.6, -0.5, -0.4, 0.3):
        h1 = (1 + w) * x
        local, total = h1, h1 + w * x
        lam = abs(w)
        rows.append({"w": w, "local": local, "total": total, "lambda": lam,
                     "bound_|total-local|<=lambda*|x0|": abs(total - local) <= lam * abs(x) + 1e-15})
        assert rows[-1]["bound_|total-local|<=lambda*|x0|"]
    assert rows[0]["local"] > 0 > rows[0]["total"] and abs(rows[1]["total"]) < 1e-15
    return {"rows": rows}


def adam_update(g, m_prev, v_prev, t, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m_prev + (1 - b1) * g
    v = b2 * v_prev + (1 - b2) * g * g
    mh = m / (1 - b1**t)
    vh = v / (1 - b2**t)
    return mh / (np.sqrt(vh) + eps), m, v


def adam_sensitivity(g, m_prev, v_prev, t, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m_prev + (1 - b1) * g
    v = b2 * v_prev + (1 - b2) * g * g
    mh, vh = m / (1 - b1**t), v / (1 - b2**t)
    a_t, b_t = (1 - b1) / (1 - b1**t), (1 - b2) / (1 - b2**t)
    s = np.sqrt(vh)
    return a_t / (s + eps) - b_t * mh * g / (s * (s + eps) ** 2)


@check
def p1_adam_sensitivity():
    """du/dg for Adam: closed form vs finite differences; zero response at t=1;
    negative response for a coordinate whose gradient exceeds its history."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(2000):
        t = int(rng.integers(1, 400))
        g = rng.standard_normal() * 2
        mp = rng.standard_normal() * (1 - 0.9 ** (t - 1))
        vp = abs(rng.standard_normal()) * (1 - 0.999 ** (t - 1)) + (1e-4 if t > 1 else 0.0)
        if t == 1:
            mp = vp = 0.0
        h = 1e-6 * max(1.0, abs(g))
        fd = (adam_update(g + h, mp, vp, t)[0] - adam_update(g - h, mp, vp, t)[0]) / (2 * h)
        D = adam_sensitivity(g, mp, vp, t)
        worst = max(worst, abs(fd - D) / (abs(D) + 1e-3))
    assert worst < 1e-4, worst
    # t = 1: u = g/(|g|+eps): response ~ eps/(|g|+eps)^2 ~ 0
    d_t1 = float(adam_sensitivity(0.5, 0.0, 0.0, 1))
    # nine steps of gradient 1.0, then a gradient of 3.0 at t = 10
    mp = vp = 0.0
    for t in range(1, 10):
        _, mp, vp = adam_update(1.0, mp, vp, t)
    D10 = float(adam_sensitivity(3.0, mp, vp, 10))
    u_plain = float(adam_update(3.0, mp, vp, 10)[0])
    u_steer = float(adam_update(3.6, mp, vp, 10)[0])
    assert D10 < 0 and u_steer < u_plain
    ratio = {
        str(t): ((1 - 0.999) / (1 - 0.999**t)) / ((1 - 0.9) / (1 - 0.9**t)) for t in (1, 10, 50, 100, 400, 1000, 10000)
    }
    return {"max_rel_error_closed_form_vs_fd": worst, "sensitivity_t1": d_t1,
            "sensitivity_t10_g3_after_nine_g1": D10, "u_t10(g=3.0)": u_plain, "u_t10(g=3.6)": u_steer,
            "b_t/a_t": ratio,
            "note": "g*d > 0 for d = +0.6 yet the Adam step along the coordinate shrinks"}


@check
def p1_truncated_gradient_angle_budget():
    """If angle(d, g_tr) <= alpha and angle(g_tr, G) <= beta with alpha+beta < pi/2 then <d, G> > 0;
    at alpha + beta = pi/2 the planar configuration gives <d, G> = 0 (tight)."""
    rng = np.random.default_rng(2)
    viol = 0

    def ang(a, b):
        return math.acos(max(-1.0, min(1.0, a @ b / np.linalg.norm(a) / np.linalg.norm(b))))

    for _ in range(50000):
        n = int(rng.integers(2, 8))
        G, gtr, d = (rng.standard_normal(n) for _ in range(3))
        if ang(d, gtr) + ang(gtr, G) < math.pi / 2 - 1e-9 and d @ G <= 0:
            viol += 1
    beta = math.radians(35)
    alpha = math.pi / 2 - beta
    G = np.array([1.0, 0.0])
    gtr = np.array([math.cos(beta), math.sin(beta)])
    d = np.array([math.cos(beta + alpha), math.sin(beta + alpha)])
    assert viol == 0 and abs(d @ G) < 1e-12
    return {"violations": viol, "tight_planar_inner_product": float(d @ G)}


@check
def p1_smoothness_sufficient_condition():
    """On quadratics with Hessian H (L = lambda_max): if <g,d> > eta*L*|d|*(|g| + |d|/2) the steered
    SGD step beats the ordinary one. The MMS-06 review condition <g,d> > (L eta/2)(|g+d|^2+|g|^2)
    is strictly stronger (it implies ours)."""
    rng = np.random.default_rng(21)
    ours = theirs = ours_and_better = theirs_only = 0
    for _ in range(40000):
        n = int(rng.integers(2, 6))
        Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
        lam = np.exp(rng.uniform(-2, 1, n))
        H = (Q * lam) @ Q.T
        Ls = lam.max()
        th = rng.standard_normal(n)
        g = H @ th
        dirn = g / np.linalg.norm(g) + 0.7 * rng.standard_normal(n)
        d = 0.3 * np.linalg.norm(g) * dirn / np.linalg.norm(dirn)
        eta = float(np.exp(rng.uniform(-6, 0))) / Ls
        t_s, t_o = th - eta * (g + d), th - eta * g
        better = 0.5 * t_s @ H @ t_s < 0.5 * t_o @ H @ t_o
        c_ours = g @ d > eta * Ls * np.linalg.norm(d) * (np.linalg.norm(g) + np.linalg.norm(d) / 2)
        c_theirs = g @ d > (Ls * eta / 2) * ((g + d) @ (g + d) + g @ g)
        ours += c_ours
        theirs += c_theirs
        if c_ours:
            assert better
            ours_and_better += 1
        if c_theirs and not c_ours:
            theirs_only += 1
    assert theirs_only == 0 and ours > theirs
    return {"cases": 40000, "ours_holds": int(ours), "review_condition_holds": int(theirs),
            "review_holds_but_ours_fails": int(theirs_only), "violations_of_ours": int(ours - ours_and_better)}


@check
def p1_adam_scale_null():
    """Adam is homogeneous of degree zero in the gradient history (eps -> 0): scaling every
    gradient by (1 + rho) leaves the trajectory unchanged. A perfectly aligned replay at a cut
    site that covers every sample is therefore a no-op under AdamW with decoupled decay."""
    rng = np.random.default_rng(22)
    H = np.diag(np.exp(rng.uniform(-2, 2, 10)))

    def run(scale, hit_frac=1.0, steps=300, lr=2e-3, wd=1e-4):
        r = np.random.default_rng(5)
        th = np.ones(10)
        m = v = np.zeros(10)
        for t in range(1, steps + 1):
            noise = r.standard_normal((8, 10))
            per_sample = th @ H + noise  # eight per-sample gradients
            w = np.where(r.random(8) < hit_frac, scale, 1.0)[:, None]
            g = (w * per_sample).mean(0)
            u, m, v = adam_update(g, m, v, t, eps=1e-12)
            th = th * (1 - lr * wd) - lr * u
        return th

    base = run(1.0)
    full = run(1.2)
    partial = run(1.2, hit_frac=0.6)
    out = {"max_abs_diff_all_samples_scaled": float(np.abs(full - base).max()),
           "max_abs_diff_60pct_of_samples_scaled": float(np.abs(partial - base).max())}
    assert out["max_abs_diff_all_samples_scaled"] < 1e-9 < out["max_abs_diff_60pct_of_samples_scaled"]
    return out


@check
def p1_control_variate_table():
    """SAGA-form estimator with an arbitrary (stale, compressed, partially missing) table is
    unbiased; variance ratio at the optimal coefficient equals 1 - r^2."""
    rng = np.random.default_rng(9)
    N, p, B = 512, 24, 16
    G = rng.standard_normal((N, p)) + rng.standard_normal(p)  # per-example gradients
    T = 0.7 * G + 0.5 * rng.standard_normal((N, p))  # stale / perturbed table entries
    T[rng.random(N) < 0.3] = 0.0  # evicted or never written slots count as zero entries
    gbar, tbar = G.mean(0), T.mean(0)
    Gc, Tc = G - gbar, T - tbar
    cov = float((Gc * Tc).sum() / N)
    var_g, var_t = float((Gc**2).sum() / N), float((Tc**2).sum() / N)
    alpha = cov / var_t
    r2 = cov**2 / (var_g * var_t)
    trials = 40000
    idx = rng.integers(0, N, size=(trials, B))
    plain = G[idx].mean(1)
    cv = (G[idx] - alpha * T[idx]).mean(1) + alpha * tbar
    bias = float(np.abs(cv.mean(0) - gbar).max())
    var_plain = float(((plain - gbar) ** 2).sum(1).mean())
    var_cv = float(((cv - gbar) ** 2).sum(1).mean())
    se = math.sqrt(var_g / B / trials)
    assert bias < 6 * se, (bias, se)
    assert abs(var_cv / var_plain - (1 - r2)) < 0.03
    assert abs(var_plain - var_g / B) / (var_g / B) < 0.03
    return {"max_abs_bias": bias, "mc_standard_error_per_coordinate": se, "alpha_star": alpha, "r_squared": r2,
            "variance_ratio_mc": var_cv / var_plain, "variance_ratio_theory_1-r^2": 1 - r2}


@check
def p3_delay_independent_stability():
    """theta_{t+1} = (1-a) theta_t - rho*a*theta_{t-tau}: all roots inside the unit disk for
    every delay whenever |1-a| + rho*a < 1; beyond that, stability is delay dependent."""
    def radius(a, rho, tau):
        coeffs = np.zeros(tau + 2)
        coeffs[0], coeffs[1], coeffs[-1] = 1.0, -(1 - a), rho * a
        return float(np.abs(np.roots(coeffs)).max())
    worst = 0.0
    for a in np.linspace(0.02, 1.98, 50):
        for rho in (0.1, 0.2, 0.5, 0.9):
            if abs(1 - a) + rho * a < 1 - 1e-9:
                for tau in (1, 2, 5, 17, 60):
                    worst = max(worst, radius(a, rho, tau))
    assert worst < 1.0
    a, rho = 1.75, 0.2  # a(1+rho) = 2.1 > 2
    radii = {str(tau): radius(a, rho, tau) for tau in (1, 2, 3, 4, 5, 6)}
    assert max(radii.values()) > 1.0
    return {"max_radius_inside_sufficient_region": worst, "radii_a=1.75_rho=0.2_by_delay": radii,
            "plain_gd_radius_a=1.75": abs(1 - 1.75)}


# ----------------------------------------------------------------------------
# Problem 2 (exercises the pinned repository code)
# ----------------------------------------------------------------------------
def scaled_norm(x):
    s = float(np.max(np.abs(x))) if x.size else 0.0
    if s == 0.0 or not math.isfinite(s):
        return s
    return s * float(np.sqrt(np.sum((x / s) ** 2)))


def canonicalize_invariant(v, u, *, rel_prune=1e-7, merge_tol=1e-10):
    """Reference canonical form using only rank-one invariants.

    Returns unit factors (u_hat [C', d_out], v_hat [d_in, C']), log-magnitudes
    log m_j = log|u_j| + log|v_j|, and a status per input component. Activity depends
    on m_j relative to |W_hat|_F only, so it is invariant under reciprocal scaling,
    sign flips and permutations, and equivariant under W -> s W.
    """
    v = np.asarray(v, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    C = u.shape[0]
    comps = []
    status = []
    for j in range(C):
        nu, nv = scaled_norm(u[j]), scaled_norm(v[:, j])
        if nu == 0.0 or nv == 0.0:
            status.append("exact_zero")
            continue
        status.append("kept")
        comps.append([u[j] / nu, v[:, j] / nv, math.log(nu) + math.log(nv)])
    if not comps:
        return {"u_hat": np.zeros((0, u.shape[1])), "v_hat": np.zeros((v.shape[0], 0)), "logm": np.zeros(0),
                "status_in": status, "status_out": [], "fro": 0.0}

    def merge(comps, axis):
        out = []
        for uh, vh, lm in comps:
            hit = None
            for item in out:
                cs = float((item[axis] * (uh if axis == 0 else vh)).sum())
                if abs(cs) >= 1 - merge_tol:
                    hit = (item, 1.0 if cs > 0 else -1.0)
                    break
            if hit is None:
                out.append([uh.copy(), vh.copy(), lm])
                continue
            item, sgn = hit
            base = max(item[2], lm)
            other = 1 - axis  # the factor that is summed
            summed = math.exp(item[2] - base) * item[other] + sgn * math.exp(lm - base) * (vh if axis == 0 else uh)
            ns = scaled_norm(summed)
            if ns == 0.0:
                item[2] = -math.inf
            else:
                item[other] = summed / ns
                item[2] = base + math.log(ns)
        return out

    comps = merge(merge(comps, 0), 1)
    lm = np.array([c[2] for c in comps])
    fin = np.isfinite(lm)
    base = lm[fin].max() if fin.any() else 0.0
    w = np.where(fin, np.exp(lm - base), 0.0)
    Uh = np.stack([c[0] for c in comps])
    Vh = np.stack([c[1] for c in comps], axis=1)
    gram = (Uh @ Uh.T) * (Vh.T @ Vh)
    fro_rel = math.sqrt(max(float(w @ gram @ w), 0.0))  # |W_hat|_F / exp(base)
    out_status = []
    for j in range(len(comps)):
        if not fin[j]:
            out_status.append("cancelled")
        elif fro_rel == 0.0 or w[j] < np.finfo(np.float32).eps * fro_rel:
            out_status.append("unrepresentable")
        elif w[j] < rel_prune * fro_rel:
            out_status.append("pruned")
        else:
            out_status.append("active")
    order = np.argsort(-np.where(fin, lm, -np.inf), kind="stable")
    return {"u_hat": Uh[order], "v_hat": Vh[:, order], "logm": lm[order], "status_in": status,
            "status_out": [out_status[i] for i in order], "fro": fro_rel * math.exp(base) if fin.any() else 0.0}


def match_components(a, b):
    """Greedy matching on the rank-one cosine (u_a.u_b)(v_a.v_b); sign and scale free."""
    cosr = (a["u_hat"] @ b["u_hat"].T) * (a["v_hat"].T @ b["v_hat"])
    work = cosr.copy()
    pairs = []
    for _ in range(min(work.shape)):
        i, j = np.unravel_index(np.argmax(work), work.shape)
        pairs.append((int(i), int(j), float(cosr[i, j])))
        work[i, :] = -np.inf
        work[:, j] = -np.inf
    matched = min(p[2] for p in pairs)
    off = cosr.copy()
    for i, j, _ in pairs:
        off[i, j] = -np.inf
    competitor = float(off.max()) if np.isfinite(off).any() else -1.0
    return {"pairs": pairs, "min_matched_cos": matched, "max_competitor_cos": competitor}


@check
def p2_activity_gauge_defect(fm):
    """(1,1) and (1e13,1e-13) are the same rank-one matrix; current activity differs."""
    a = fm.canonicalize_factors(np.array([[1.0]]), np.array([[1.0]]))
    b = fm.canonicalize_factors(np.array([[1e-13]]), np.array([[1e13]]))
    c = fm.canonicalize_factors(np.array([[1e13]]), np.array([[1e-13]]))
    out = {"active(1,1)": bool(a.active[0]), "active(v=1e-13,u=1e13)": bool(b.active[0]),
           "active(v=1e13,u=1e-13)": bool(c.active[0])}
    assert out["active(1,1)"] and not out["active(v=1e-13,u=1e13)"] and not out["active(v=1e13,u=1e-13)"]
    ra = canonicalize_invariant(np.array([[1.0]]), np.array([[1.0]]))
    rb = canonicalize_invariant(np.array([[1e-13]]), np.array([[1e13]]))
    rc = canonicalize_invariant(np.array([[1e200]]), np.array([[1e-200]]))
    assert ra["status_out"] == rb["status_out"] == rc["status_out"] == ["active"]
    assert abs(ra["logm"][0] - rb["logm"][0]) < 1e-9 and abs(rc["logm"][0]) < 1e-9
    out["invariant_reference_status"] = [ra["status_out"], rb["status_out"], rc["status_out"]]
    return out


@check
def p2_extreme_scale_imbalance(fm):
    """Same defect at a larger imbalance: factors (1e200, 1e-200) of the all-ones matrix.
    The plain norm overflows to inf and the component is silently dropped."""
    u = np.full((1, 2), 1e200)
    v = np.full((2, 1), 1e-200)
    assert np.allclose((v @ u).T, np.ones((2, 2)))  # the physical matrix is exactly representable
    with np.errstate(all="ignore"):
        f = fm.canonicalize_factors(v, u)
        plain = float(np.linalg.norm(u[0]))
    recon = (f.v.astype(np.float64) @ f.u.astype(np.float64)).T
    ref = canonicalize_invariant(v, u)
    out = {"current_active": bool(f.active[0]), "current_reconstruction_max": float(np.abs(recon).max()),
           "plain_norm_of_u": plain, "scaled_norm_of_u": scaled_norm(u[0]),
           "reference_status": ref["status_out"], "reference_logm": ref["logm"].tolist()}
    assert math.isinf(plain) and not out["current_active"] and out["current_reconstruction_max"] == 0.0
    assert ref["status_out"] == ["active"] and abs(ref["logm"][0] - math.log(2.0)) < 1e-9
    return out


@check
def p2_component_id_instability(fm):
    """Rounded-hash identity: (i) -0.0 vs +0.0, (ii) sign pivot near ties,
    (iii) aliasing below the 1e-4 quantum, (iv) quantization boundary."""
    v = np.array([[1.0], [0.5]])
    f1 = fm.canonicalize_factors(v, np.array([[1.0, 1e-6, 0.0]]))
    f2 = fm.canonicalize_factors(v, np.array([[1.0, -1e-6, 0.0]]))
    f3 = fm.canonicalize_factors(v, np.array([[0.6, -0.6000001, 0.1]]))
    f4 = fm.canonicalize_factors(v, np.array([[0.6000001, -0.6, 0.1]]))
    g1 = fm.canonicalize_factors(np.array([[1.0], [0.00002]]), np.array([[1.0, 0.0, 0.0]]))
    g2 = fm.canonicalize_factors(np.array([[1.0], [-0.00002 + 0.00006]]), np.array([[1.0, 0.0, 0.0]]))
    b1 = fm.canonicalize_factors(np.array([[0.12344999], [1.0]]), np.array([[1.0, 0.0, 0.0]]))
    b2 = fm.canonicalize_factors(np.array([[0.12345001], [1.0]]), np.array([[1.0, 0.0, 0.0]]))
    out = {
        "ids_differ_for_pm_1e-6_entry": f1.component_ids[0] != f2.component_ids[0],
        "max_abs_diff_u_pm_1e-6": float(np.abs(f1.u - f2.u).max()),
        "ids_differ_at_sign_pivot_tie": f3.component_ids[0] != f4.component_ids[0],
        "u_sign_flipped_at_pivot_tie": bool(np.sign(f3.u[0, 0]) != np.sign(f4.u[0, 0])),
        "alias_distinct_v_same_id": g1.component_ids[0] == g2.component_ids[0],
        "alias_v_abs_diff": 0.00002,
        "ids_differ_across_quantum_boundary_2e-8": b1.component_ids[0] != b2.component_ids[0],
    }
    assert out["ids_differ_for_pm_1e-6_entry"] and out["ids_differ_at_sign_pivot_tie"]
    assert out["alias_distinct_v_same_id"] and out["ids_differ_across_quantum_boundary_2e-8"]
    # tolerance matching is stable in all four situations
    r = [canonicalize_invariant(v, np.array([[1.0, s * 1e-6, 0.0]])) for s in (1, -1)]
    m12 = match_components(r[0], r[1])
    r34 = [canonicalize_invariant(v, np.array([x])) for x in ([0.6, -0.6000001, 0.1], [0.6000001, -0.6, 0.1])]
    m34 = match_components(r34[0], r34[1])
    assert m12["min_matched_cos"] > 1 - 1e-10 and m34["min_matched_cos"] > 1 - 1e-10
    out["rank_one_cosine_match_pm_1e-6"] = m12["min_matched_cos"]
    out["rank_one_cosine_match_pivot_tie"] = m34["min_matched_cos"]
    return out


@check
def p2_selection_scale_and_cancelling_pair(fm):
    """(i) Absolute tie rounding breaks invariance to positive rescaling of the inputs.
    (ii) A cancelling pair (u, v), (u, -v) contributes zero to W but wins top-k selection."""
    rng = np.random.default_rng(4)
    d_in = d_out = 8
    V = rng.standard_normal((d_in, 6))
    U = rng.standard_normal((6, d_out))
    f = fm.canonicalize_factors(V, U)
    x = rng.standard_normal((5, d_in))
    h = rng.standard_normal((5, d_out))
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    c = h + 0.1 * rng.standard_normal((5, d_out))
    s1, _, _ = fm.select_and_project_numpy(x, h, c, f, topk=2)
    s2, _, _ = fm.select_and_project_numpy(x * 1e-8, h, c, f, topk=2)
    out = {"selection_changes_under_x_scale_1e-8": bool((np.sort(s1, 1) != np.sort(s2, 1)).any())}
    assert out["selection_changes_under_x_scale_1e-8"]

    # true map: two orthogonal unit components; null pair along a third orthogonal write direction
    e = np.eye(d_out)
    Vt = np.stack([np.eye(d_in)[0], np.eye(d_in)[1]], 1)
    Ut = np.stack([e[0], e[1]])
    vn = np.eye(d_in)[2] * 1e3
    Vp = np.concatenate([Vt, vn[:, None], -vn[:, None]], 1)
    Up = np.concatenate([Ut, e[2][None], e[2][None]], 0)
    assert np.allclose((Vp @ Up).T, (Vt @ Ut).T)
    fp = fm.canonicalize_factors(Vp, Up)
    xx = np.ones((1, d_in))
    hh = (e[0] + e[1] + e[2])[None] / math.sqrt(3)
    cc = hh.copy()
    sel, delta, _ = fm.select_and_project_numpy(xx, hh, cc, fp, topk=2)
    chosen_u = fp.u[sel[0]]
    out["active_count_with_null_pair"] = int(fp.active.sum())
    out["selected_write_directions"] = chosen_u.round(3).tolist()
    out["delta_direction"] = (delta[0] / np.linalg.norm(delta[0])).round(3).tolist()
    W = (Vp @ Up).T
    out["norm_of_W_restricted_to_selected_write_direction"] = float(np.linalg.norm(e[2] @ W))
    assert np.allclose(np.abs(chosen_u), np.stack([e[2], e[2]]))
    assert out["norm_of_W_restricted_to_selected_write_direction"] == 0.0
    ref = canonicalize_invariant(Vp, Up)
    out["reference_status"] = ref["status_out"]
    assert ref["status_out"].count("active") == 2 and "cancelled" in ref["status_out"]
    return out


@check
def p2_gl_mixing(fm):
    """General GL(C) mixing preserves W and the full-span projector, not component sets,
    not sub-selections, and not the outer-product sum P."""
    rng = np.random.default_rng(6)
    d_in, d_out, C = 10, 9, 4
    V = rng.standard_normal((d_in, C))
    U = rng.standard_normal((C, d_out))
    R = rng.standard_normal((C, C)) + 2 * np.eye(C)
    V2, U2 = V @ R, np.linalg.solve(R, U)
    assert np.allclose(V @ U, V2 @ U2)
    f1, f2 = fm.canonicalize_factors(V, U), fm.canonicalize_factors(V2, U2)
    def proj(a):
        return a.T @ np.linalg.pinv(a @ a.T) @ a

    u1, u2 = f1.u.astype(np.float64), f2.u.astype(np.float64)
    out = {
        "full_span_projector_diff": float(np.abs(proj(u1) - proj(u2)).max()),
        "outer_product_sum_diff": float(np.abs(u1.T @ u1 - u2.T @ u2).max()),
        "top2_subspace_projector_diff": float(np.abs(proj(u1[:2]) - proj(u2[:2])).max()),
        "shared_component_ids": len(set(f1.component_ids) & set(f2.component_ids)),
    }
    assert out["full_span_projector_diff"] < 1e-5 and out["outer_product_sum_diff"] > 1e-2
    assert out["top2_subspace_projector_diff"] > 1e-2 and out["shared_component_ids"] == 0
    # invariant-mask criterion: a diagonal mask D is gauge invariant iff D R = R D
    D = np.diag([1.0, 1.0, 0.0, 0.0])
    Rb = np.block([[R[:2, :2], np.zeros((2, 2))], [np.zeros((2, 2)), R[2:, 2:]]])
    lhs = V @ D @ U
    assert np.allclose(lhs, (V @ Rb) @ D @ np.linalg.solve(Rb, U))
    assert not np.allclose(lhs, V2 @ D @ U2)
    out["block_mask_invariant_under_block_gauge"] = True
    return out


@check
def p2_outer_sum_vs_projector(fm):
    """P = U_S^T U_S versus Pi = U_S^T (U_S U_S^T)^+ U_S."""
    e = np.eye(3)
    US = np.stack([e[0], e[0], e[1]])
    P = US.T @ US
    Pi = US.T @ np.linalg.pinv(US @ US.T) @ US
    h = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
    ang = math.degrees(math.acos((P @ h) @ (Pi @ h) / np.linalg.norm(P @ h) / np.linalg.norm(Pi @ h)))
    out = {"duplicate_opnorm_P": float(np.linalg.norm(P, 2)), "duplicate_angle_deg": ang}
    assert abs(out["duplicate_opnorm_P"] - 2.0) < 1e-12 and abs(ang - 18.434948822922) < 1e-6

    # two unit components at 60 degrees: admission sign disagreement in both directions
    U2 = np.array([[1.0, 0.0], [0.5, math.sqrt(3) / 2]])
    G = U2 @ U2.T
    lam, Wv = np.linalg.eigh(G)
    E = (U2.T @ Wv) / np.sqrt(lam)  # orthonormal principal directions e_i (columns)
    hh = E @ np.array([1.0, 1.0])
    c_rej = E @ np.array([1.0, -0.5])   # eigen order ascending: lam = (0.5, 1.5)
    c_acc = E @ np.array([-1.0, 0.5])
    P2 = U2.T @ U2
    out["gram_eigs_60deg"] = lam.tolist()
    out["false_reject"] = {"h.c (Pi = I)": float(hh @ c_rej), "h^T P c": float(hh @ P2 @ c_rej)}
    out["false_accept"] = {"h.c (Pi = I)": float(hh @ c_acc), "h^T P c": float(hh @ P2 @ c_acc)}
    assert out["false_reject"]["h.c (Pi = I)"] > 0 > out["false_reject"]["h^T P c"]
    assert out["false_accept"]["h.c (Pi = I)"] < 0 < out["false_accept"]["h^T P c"]

    # worst-case angle between P h and Pi h is the first antieigenvalue of the Gram matrix
    rng = np.random.default_rng(8)
    worst = 1.0
    for _ in range(200000):
        a = rng.standard_normal(2)
        cosv = (lam * a * a).sum() / math.sqrt((lam**2 * a * a).sum() * (a * a).sum())
        worst = min(worst, cosv)
    bound = 2 * math.sqrt(lam[0] * lam[-1]) / (lam[0] + lam[-1])
    assert worst >= bound - 1e-12 and worst - bound < 1e-3
    out["min_cos(Ph, Pi h)_sampled"] = worst
    out["antieigenvalue_bound"] = bound

    # the canonical SVD map is orthonormal: P = Pi exactly, so the two replay arms coincide
    Wm = rng.standard_normal((64, 64))
    f = fm.canonical_svd_factors(Wm, rank=16)
    Gs = f.u.astype(np.float64) @ f.u.astype(np.float64).T
    out["svd_map_gram_minus_identity_max"] = float(np.abs(Gs - np.eye(16)).max())
    assert out["svd_map_gram_minus_identity_max"] < 1e-5

    # dual basis: oblique projector O_j = u_j utilde_j^T is component specific
    dual = np.linalg.solve(G, U2)
    O1 = np.outer(U2[0], dual[0])
    out["O_1 u_1 = u_1"] = bool(np.allclose(O1 @ U2[0], U2[0]))
    out["O_1 u_2 = 0"] = bool(np.allclose(O1 @ U2[1], 0))
    out["matched_filter_crosstalk_u1.u2"] = float(U2[0] @ U2[1])
    out["variance_inflation_(G^-1)_11"] = float(np.linalg.inv(G)[0, 0])
    assert out["O_1 u_1 = u_1"] and out["O_1 u_2 = 0"]

    # admission disagreement rate for a correlated (non-orthogonal) dictionary, k = 4 of d = 64
    dis = []
    for rho_c in (0.0, 0.5, 0.8, 0.95):
        base = rng.standard_normal(64)
        base /= np.linalg.norm(base)
        cnt = tot = 0
        for _ in range(300):
            Uc = math.sqrt(rho_c) * base + math.sqrt(1 - rho_c) * rng.standard_normal((4, 64)) / 8.0
            Uc /= np.linalg.norm(Uc, axis=1, keepdims=True)
            Pc = Uc.T @ Uc
            Pic = Uc.T @ np.linalg.pinv(Uc @ Uc.T) @ Uc
            hh = rng.standard_normal((50, 64))
            cc = hh + 2.0 * rng.standard_normal((50, 64))
            a = np.einsum("bi,ij,bj->b", hh, Pc, cc) > 0
            b = np.einsum("bi,ij,bj->b", hh, Pic, cc) > 0
            cnt += int((a != b).sum())
            tot += 50
        dis.append({"mean_pairwise_cos": rho_c, "admission_disagreement_rate": cnt / tot})
    out["synthetic_admission_disagreement"] = dis
    assert dis[0]["admission_disagreement_rate"] < dis[-1]["admission_disagreement_rate"]
    return out


# ----------------------------------------------------------------------------
# Problem 3 / 4 / 5
# ----------------------------------------------------------------------------
@check
def p3_staleness_depth_scaling():
    """SYNTHETIC ILLUSTRATION (random-init tied tanh nets, K = 8 visits): relative change of the
    cached cotangent dL/dz_k per unit parameter perturbation, by visit. Theorem 3.1 predicts
    larger relative staleness for sites with more remaining recurrent depth K - k."""
    rng = np.random.default_rng(41)
    out = {}
    for scale in (0.5, 0.9, 1.3):
        acc = np.zeros(8)
        trials = 200
        for _ in range(trials):
            net = TiedNet(rng, visits=8, scale=scale)
            x = rng.standard_normal((16, 12))
            t = rng.standard_normal((16, 4))
            _, c0, _ = net.backward(x, t)
            v = {k: rng.standard_normal(val.shape) for k, val in net.p.items()}
            nv = math.sqrt(sum(float((v[k] ** 2).sum()) for k in v))
            eps = 1e-3
            old = net.p
            net.p = padd(old, v, eps / nv)
            _, c1, _ = net.backward(x, t)
            net.p = old
            acc += np.array([np.linalg.norm(c1[k] - c0[k]) / np.linalg.norm(c0[k]) for k in range(8)]) / eps
        prof = acc / trials
        assert all(prof[k] > prof[k + 1] for k in range(7))
        out[f"init_scale_{scale}"] = prof.round(3).tolist()
    out["columns"] = "visit k = 0..7 (remaining depth K-k = 8..1)"
    return out


@check
def p3_refresh_square_root_law():
    """Renewal-reward optimum for refresh period R with linear quality decay b(a)=b0(1-a/tau),
    baseline progress gbar per step, step cost c, refresh cost Cr:
        R* = (-Cr + sqrt(Cr^2 + 2 c Cr tau (1 + gbar/b0))) / c."""
    def rate(R, tau, c, Cr, gbar, b0):
        return (R * gbar + b0 * (R - R * R / (2 * tau))) / (R * c + Cr)
    rows = []
    for tau, Cr, gbar in [(100, 0.45, 0.0), (500, 0.45, 0.0), (2000, 0.45, 0.0), (500, 50.0, 0.0),
                          (2000, 50.0, 0.0), (500, 0.45, 3.0), (500, 50.0, 3.0)]:
        c, b0 = 1.0, 1.0
        closed = (-Cr + math.sqrt(Cr * Cr + 2 * c * Cr * tau * (1 + gbar / b0))) / c
        grid = np.linspace(1, tau, 200001)
        num = float(grid[np.argmax(rate(grid, tau, c, Cr, gbar, b0))])
        rows.append({"tau_steps": tau, "refresh_cost_in_steps": Cr, "gbar/b0": gbar, "R_closed": closed,
                     "R_numeric": num, "sqrt_approx": math.sqrt(2 * tau * Cr * (1 + gbar / b0)),
                     "interior": closed < tau})
        if closed < tau:
            assert abs(num - closed) / closed < 2e-3, rows[-1]
    return {"rows": rows}


@check
def p4_scheduler_null_simulation(mu):
    """Null world: every action leaves the regret curve unchanged. The pinned scheduler is
    driven exactly as in mms06_lbf20d_campaign.py (40-step windows, raw window regret change
    as feedback). A calibrated controller should pick the cheapest action (ordinary)."""
    rng = np.random.default_rng(12)
    runs, windows = 300, 10
    counts = {"__ordinary__": 0, "block@0.5": 0, "block@4": 0}
    first_ordinary = []
    ordinary_while_improving = 0
    improving_windows = 0
    def regret(w):
        return 1.0 + 0.5 * math.exp(-w / 3.0)  # monotone improving, identical for all actions

    for run in range(runs):
        sch = mu.UtilityDirectedScheduler(sites=("block",), ram_tiers_mib=(0.5, 4.0), lambda_ram=0.005,
                                          exploration_bonus=0.02, history_budget_bytes=128 * 1024, cov_dim=8,
                                          research_mode=True, exploration_budget=3)
        last = regret(0) + 0.01 * rng.standard_normal()
        site, tier = mu.ORDINARY_SITE, 0.0
        got_ord = None
        for w in range(windows):
            if w > 0:
                cur = regret(w) + 0.01 * rng.standard_normal()
                ctx = mu.FeedbackContext("lbf20d", "mean_regret", -1, 1.0, "split", 40, "ck", "opt", site,
                                         "ordinary_none" if site == mu.ORDINARY_SITE else "basis")
                sch.observe_feedback(mu.FeedbackSignal(step=40 * w, trajectory_id=f"r{run}_w{w}", site=site,
                                                       ram_mib=tier, partition="dev_train",
                                                       signal_kind="lbf_bellman_regret", delta_v=cur - last,
                                                       delta_cpu_seconds=0.33 if site == mu.ORDINARY_SITE else 0.5,
                                                       delta_ram_mib=tier, context=ctx))
                last = cur
            sel = mu.FeedbackContext("lbf20d", "mean_regret", -1, 1.0, "split", 40, "ck", "opt", "block", "block_svd")
            ch = sch.select(context=sel, policy="utility", remaining_cpu_seconds=None,
                            live_ram_headroom_bytes=8 << 20)
            site, tier = ch.site, ch.ram_mib
            sch.record_action(f"r{run}_w{w + 1}", 40 * w, site, tier)
            key = site if site == mu.ORDINARY_SITE else f"block@{tier:g}"
            counts[key] += 1
            improving_windows += 1
            if site == mu.ORDINARY_SITE:
                ordinary_while_improving += 1
                if got_ord is None:
                    got_ord = w
        first_ordinary.append(-1 if got_ord is None else got_ord)
    total = sum(counts.values())
    share = {k: v / total for k, v in counts.items()}
    out = {"runs": runs, "windows_per_run": windows, "selection_share": share,
           "runs_never_selecting_ordinary": int(sum(1 for f in first_ordinary if f < 0)),
           "ordinary_share": share["__ordinary__"]}
    assert share["__ordinary__"] < 0.10, share
    return out


@check
def p4_ratio_utility_negative_numerator(mu):
    """With equal negative gains, gain/cost ranks the costlier action higher."""
    sch = mu.UtilityDirectedScheduler(sites=("block",), ram_tiers_mib=(0.5, 4.0), lambda_ram=0.0,
                                      exploration_bonus=0.0, research_mode=True, exploration_budget=5)
    for tier, cpu in ((0.5, 1.0), (4.0, 10.0)):
        ctx = mu.FeedbackContext("t", "m", 1, 1.0, "s", 1, "c", "o", "block", "b")
        sch.observe_feedback(mu.FeedbackSignal(step=1, trajectory_id=f"x{tier}", site="block", ram_mib=tier,
                                               partition="dev_train", signal_kind="lbf_bellman_regret",
                                               delta_v=-0.1, delta_cpu_seconds=cpu, delta_ram_mib=tier, context=ctx))
    ctx = mu.FeedbackContext("t", "m", 1, 1.0, "s", 1, "c", "o", "block", "b")
    ch = sch.select(context=ctx, policy="utility")
    out = {"chosen_tier": ch.ram_mib, "reason": ch.reason, "utility": ch.utility}
    assert ch.reason == "budgeted_exploration" and ch.ram_mib == 4.0
    return out


@check
def p4_estimator_comparison():
    """Noisy quadratic, SGD, 40-step windows. Treatment = learning rate x1.2 (a stand-in for a
    dose-like intervention); NULL treatment = identical to ordinary. Compare three estimators of
    the incremental window value: (a) sequential raw progress as fed to the pinned scheduler,
    (b) randomized windows, (c) paired clones with common random numbers."""
    rng = np.random.default_rng(31)
    lam = np.exp(np.linspace(-1.5, 1.0, 20))
    eta, sigma, W, n_win, runs = 0.05, 0.5, 40, 8, 400
    def loss(th):
        return 0.5 * float((lam * th * th).sum())

    def window(th, mult, noise):
        for t in range(W):
            th = th - eta * mult * (lam * th + sigma * noise[t])
        return th

    res = {}
    for label, mult in (("null", 1.0), ("lr_x1.2", 1.2)):
        seq, clone = [], []
        rnd = {w: ([], []) for w in range(n_win)}  # per window index: (treated, control) progress
        for _ in range(runs):
            th = np.full(20, 2.0)
            # (a) sequential: treatment in windows 0..3, ordinary in 4..7 (scheduler ordering)
            prog = []
            t2 = th.copy()
            for w in range(n_win):
                nz = rng.standard_normal((W, 20))
                new = window(t2, mult if w < 4 else 1.0, nz)
                prog.append(loss(t2) - loss(new))
                t2 = new
            seq.append(np.mean(prog[:4]) - np.mean(prog[4:]))
            # (b) randomized assignment per window, (c) paired clones from the same state
            t3 = th.copy()
            for w in range(n_win):
                nz = rng.standard_normal((W, 20))
                treat = rng.random() < 0.5
                a_new, o_new = window(t3, mult, nz), window(t3, 1.0, nz)
                clone.append(loss(o_new) - loss(a_new))
                rnd[w][0 if treat else 1].append(loss(t3) - loss(a_new if treat else o_new))
                t3 = a_new if treat else o_new
        pooled_t = [y for w in rnd for y in rnd[w][0]]
        pooled_c = [y for w in rnd for y in rnd[w][1]]
        strat = [np.mean(rnd[w][0]) - np.mean(rnd[w][1]) for w in rnd]
        strat_var = [np.var(rnd[w][0]) / len(rnd[w][0]) + np.var(rnd[w][1]) / len(rnd[w][1]) for w in rnd]
        res[label] = {
            "sequential_raw_progress_estimate": float(np.mean(seq)),
            "randomized_unstratified_estimate": float(np.mean(pooled_t) - np.mean(pooled_c)),
            "randomized_unstratified_se": float(
                math.sqrt(np.var(pooled_t) / len(pooled_t) + np.var(pooled_c) / len(pooled_c))
            ),
            "randomized_stratified_by_window_estimate": float(np.mean(strat)),
            "randomized_stratified_by_window_se": float(math.sqrt(sum(strat_var)) / n_win),
            "paired_clone_estimate": float(np.mean(clone)),
            "paired_clone_se": float(np.std(clone) / math.sqrt(len(clone))),
            "per_window_sd_unstratified": float(np.std(pooled_t + pooled_c)),
            "per_window_sd_clone": float(np.std(clone)),
            "windows": len(clone),
        }
    null = res["null"]
    assert null["sequential_raw_progress_estimate"] > 1.0  # spurious "gain" for a treatment that does nothing
    assert abs(null["paired_clone_estimate"]) < 1e-12  # exactly zero under common random numbers
    assert abs(null["randomized_stratified_by_window_estimate"]) < 4 * null["randomized_stratified_by_window_se"]
    trt = res["lr_x1.2"]
    assert trt["paired_clone_estimate"] > 5 * trt["paired_clone_se"]
    seq_gap = abs(trt["sequential_raw_progress_estimate"] - null["sequential_raw_progress_estimate"])
    assert seq_gap < 0.05 * null["sequential_raw_progress_estimate"]
    return res


@check
def p7_receipt_deductions(receipts):
    """Read-only deductions from the committed MMS-06 receipts (no new measurements)."""
    if receipts is None:
        return {"skipped": "receipts directory not found"}
    v = json.loads((receipts / "mms06_lbf20d_vec1p5_summary.json").read_text())["summaries"]
    s4 = json.loads((receipts / "mms06_lbf20d_4s_summary.json").read_text())["summaries"]
    pts = [(v["fixed_cpu"]["ordinary"]["mean_updates"], v["fixed_cpu"]["ordinary"]["mean_regret"]),
           (v["fixed_exposure"]["ordinary"]["mean_updates"], v["fixed_exposure"]["ordinary"]["mean_regret"]),
           (s4["fixed_cpu"]["ordinary"]["mean_updates"], s4["fixed_cpu"]["ordinary"]["mean_regret"])]
    mem_updates = {k: a["mean_updates"] for k, a in v["fixed_cpu"].items() if k != "ordinary"}
    lookup = {k: a["mean_ledger_cpu"]["lookup"] for k, a in v["fixed_exposure"].items()}
    out = {"ordinary_(updates, final_dev_regret)": pts,
           "ordinary_regret_non_decreasing_in_updates": all(pts[i][1] <= pts[i + 1][1] for i in range(2)),
           "max_memory_arm_updates_at_1.5s": max(mem_updates.values()),
           "ordinary_updates_at_1.5s": pts[0][0],
           "scheduler_lookup_cpu_over_always_on_svd_refreshed": lookup["utility_scheduler"] / lookup["svd_refreshed"],
           "per_seed_delta_V_sd_dense_1.5s": float(np.std(v["fixed_cpu"]["dense"]["per_seed_delta_V"], ddof=1)),
           "per_seed_delta_V_sd_svd_frozen_1.5s": float(
               np.std(v["fixed_cpu"]["svd_frozen"]["per_seed_delta_V"], ddof=1))}
    assert out["ordinary_regret_non_decreasing_in_updates"]
    assert out["max_memory_arm_updates_at_1.5s"] < out["ordinary_updates_at_1.5s"]
    return out


@check
def p4_confidence_sequence_coverage():
    """Two-sided normal-mixture boundary |S_n| >= sqrt((n+r) log((n+r)/(r alpha^2))) for unit-variance
    Gaussian increments: crossing probability over n <= 2000 stays below alpha."""
    rng = np.random.default_rng(13)
    alpha, r, n_max, paths = 0.05, 20.0, 2000, 20000
    n = np.arange(1, n_max + 1)
    bound = np.sqrt((n + r) * np.log((n + r) / (r * alpha * alpha)))
    crossed = 0
    for _ in range(paths // 2000):
        S = np.cumsum(rng.standard_normal((2000, n_max)), axis=1)
        crossed += int((np.abs(S) >= bound).any(axis=1).sum())
    rate = crossed / paths
    assert rate <= alpha
    return {"alpha": alpha, "rho": r, "horizon": n_max, "paths": paths, "crossing_rate": rate,
            "radius_of_mean_at_n=8": float(bound[7] / 8), "radius_of_mean_at_n=64": float(bound[63] / 64)}


@check
def p5_hit_rate_laws(fb):
    """(i) Collision-free transient law reproduces the hit counts in the MMS receipts.
    (ii) Fixed random hash, uniform access: steady-state hit rate (1 - exp(-lam))/lam, lam = N/Cap.
    (iii) The repository hash on arithmetic key sets departs strongly from the random model,
          including the documented RMP default shape at power-of-two MiB tiers."""
    N, B, steps = 12288, 128, 240
    expected = sum(B * (1 - (1 - 1 / N) ** (B * s)) for s in range(steps))
    out = {"expected_hits_collision_free": expected, "receipt_hits_LT_MMS01_05": 19376,
           "receipt_hits_bank_profile": 19402, "queries": B * steps}
    rng = np.random.default_rng(14)
    sims = []
    for _ in range(200):
        seen = np.zeros(N, dtype=bool)
        hits = 0
        for _step in range(steps):
            k = rng.integers(0, N, B)
            hits += int(seen[k].sum())
            seen[k] = True
        sims.append(hits)
    out["mc_mean"], out["mc_sd"] = float(np.mean(sims)), float(np.std(sims))
    assert abs(19376 - expected) < 3 * out["mc_sd"] and abs(19402 - expected) < 3 * out["mc_sd"]

    def splitmix(k):
        z = k.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))

    rows = []
    for ratio in (0.25, 0.5, 1.0, 2.0, 4.0):
        cap = int(N * ratio)
        tags = np.full(cap, -1, dtype=np.int64)
        hits = q = 0
        for s in range(6000):
            k = rng.integers(0, N, B)
            with np.errstate(over="ignore"):
                slot = (splitmix(k) % np.uint64(cap)).astype(np.int64)
            if s >= 3000:
                hits += int((tags[slot] == k).sum())
                q += B
            tags[slot] = k
        lam = 1 / ratio
        law = (1 - math.exp(-lam)) / lam
        rows.append({"cap/N": ratio, "steady_hit_rate_mixing_hash": hits / q, "law_(1-exp(-lam))/lam": law,
                     "upper_bound_min(1,cap/N)": min(1.0, ratio)})
        assert abs(hits / q - law) < 0.02
    out["steady_state"] = rows
    out["cap_over_N_needed_for_hit_rate"] = {}
    for target in (0.5, 0.8, 0.9, 0.95):
        lo, hi = 1e-6, 50.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if (1 - math.exp(-mid)) / mid > target:
                lo = mid
            else:
                hi = mid
        out["cap_over_N_needed_for_hit_rate"][str(target)] = 1 / lo

    def distinct_slots(keys, cap):
        bank = object.__new__(fb.FastGradientBank)  # pinned hash code without allocating the payload
        bank.hash_scheme, bank.capacity = fb.SIGNED_HASH, int(cap)
        return len(np.unique(bank._slots(keys)[1]))

    def rnd(cap, n):
        return cap * (1 - (1 - 1 / cap) ** n)

    # LBF-20D campaign keys {3 i + 1}, row (64,) = 144 bytes
    keys = np.arange(N, dtype=np.int64) * 3 + 1
    lbf = []
    for mib in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0):
        cap = int(mib * (1 << 20)) // 144
        lbf.append({"mib": mib, "capacity": cap, "distinct_slots": distinct_slots(keys, cap),
                    "random_model": round(rnd(cap, N), 1), "keys": N})
    out["lbf20d_keys_3i+1"] = lbf
    assert lbf[5]["distinct_slots"] == N  # 4 MiB happens to be injective (zero collisions in receipts)
    assert lbf[2]["distinct_slots"] < 0.35 * lbf[2]["random_model"]  # 1.5 MiB is pathological
    sweep = [distinct_slots(keys, cap) / rnd(cap, N) for cap in range(1000, 60001, 7)]
    out["lbf20d_capacity_sweep_1000_60000"] = {"frac_below_half_of_random": float(np.mean(np.array(sweep) < 0.5)),
                                               "min_ratio_to_random": float(min(sweep))}
    out["lbf20d_capacity_53237_distinct_slots"] = distinct_slots(keys, 53237)
    # RMP default: hidden 128, sequence 64, visits 8, middle visit 4 -> keys {8 i + 4}, row 16,400 bytes
    rmp = []
    for n_keys in (2048, 20000):
        k8 = np.arange(n_keys, dtype=np.int64) * 8 + 4
        for mib in (128, 512, 2048, 4096):
            cap = (mib << 20) // 16400
            rmp.append({"example_ids": n_keys, "mib": mib, "capacity": cap, "distinct_slots": distinct_slots(k8, cap),
                        "random_model": round(rnd(cap, n_keys), 1),
                        "distinct_slots_capacity_minus_1": distinct_slots(k8, cap - 1)})
    out["rmp_default_keys_8i+4"] = rmp
    assert all(r["distinct_slots"] <= 2892 for r in rmp)
    assert rmp[-1]["distinct_slots"] < 0.2 * rmp[-1]["random_model"]
    # the same keys through the mixing hash behave like the random model
    k8 = np.arange(20000, dtype=np.int64) * 8 + 4
    with np.errstate(over="ignore"):
        mixed = len(np.unique(splitmix(k8) % np.uint64((4096 << 20) // 16400)))
    out["rmp_keys_mixing_hash_distinct_at_4096MiB"] = int(mixed)
    assert mixed > 0.95 * 20000
    return out


@check
def p5_capacity_numbers():
    def row(T, d):
        return 2 * T * d + 16

    out = {
        "lbf20d_row_bytes": row(1, 64), "lbf20d_rows_4MiB": (4 << 20) // row(1, 64),
        "rmp_row_bytes_T64_d128": row(64, 128), "rmp_rows_4MiB": (4 << 20) // row(64, 128),
        "rmp_rows_512MiB": (512 << 20) // row(64, 128), "rmp_rows_4GiB": (4 << 30) // row(64, 128),
        "trm_xs_row_bytes_T384_d176": row(384, 176), "trm_xs_rows_4GiB": (4 << 30) // row(384, 176),
        "hypothetical_T128_d128_rows_4MiB": (4 << 20) // row(128, 128),
        "trm_xs_full_corpus_GiB_one_site_visit": 193000 * 4 * row(384, 176) / 2**30,
        "trm_xs_rank16_row_bytes": 2 * 384 * 16 * 2 + 16,
    }
    assert out["lbf20d_rows_4MiB"] == 29127 and out["trm_xs_rows_4GiB"] == 31771
    assert out["hypothetical_T128_d128_rows_4MiB"] == 127
    return out


@check
def p5_sandbox_microbenchmark(fb, fm):
    """INFORMATIONAL timing on the sandbox that produced this receipt (not the MMS-06 host).
    Separates a per-call fixed cost from a per-element cost for bank read+write and projection."""
    rng = np.random.default_rng(15)
    rows = []
    for name, B, T, d in (("lbf20d", 128, 1, 64), ("mid", 64, 16, 128), ("rmp", 64, 64, 128), ("trm_xs", 16, 384, 176)):
        shape = (d,) if T == 1 else (T, d)
        bank = fb.FastGradientBank(64.0, shape)
        f = fm.canonical_svd_factors(rng.standard_normal((d, d)), rank=16)
        g = rng.standard_normal((B, *shape)).astype(np.float32)
        x = rng.standard_normal((B, *shape)).astype(np.float32)
        keys = np.arange(B, dtype=np.int64)
        bank.write(keys, g, 0)
        reps = 30
        t0 = time.perf_counter()
        for s in range(1, reps + 1):
            old, _ = bank.read(keys, s, 1 << 20)
            bank.write(keys, g, s)
        t_bank = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            fm.select_and_project_numpy(x, old, g, f, topk=4)
        t_proj = (time.perf_counter() - t0) / reps
        rows.append({"shape": name, "B": B, "T": T, "d": d, "elements_BTd": B * T * d,
                     "bank_read_write_ms": 1e3 * t_bank, "select_project_ms": 1e3 * t_proj})
    e = np.array([r["elements_BTd"] for r in rows], dtype=float)
    A = np.stack([np.ones_like(e), e], 1)
    fit = {}
    for key in ("bank_read_write_ms", "select_project_ms"):
        y = np.array([r[key] for r in rows])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        fit[key] = {"fixed_ms": float(coef[0]), "ns_per_element": float(coef[1] * 1e6)}
    return {"rows": rows, "affine_fit": fit}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve()
    ap.add_argument("--dg-src", type=Path, default=here.parents[2] / "src")
    ap.add_argument("--receipts", type=Path, default=here.parent / "receipts")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(args.dg_src))
    from datagoblin.training_gym import factor_maps as fm
    from datagoblin.training_gym import fast_gradient_bank as fb
    from datagoblin.training_gym import mms_utility as mu

    mods = {"fm": fm, "fb": fb, "mu": mu, "receipts": args.receipts if args.receipts.is_dir() else None}
    results: dict[str, Any] = {}
    status = "PASS"
    for name, fn in CHECKS.items():
        params = fn.__code__.co_varnames[: fn.__code__.co_argcount]
        t0 = time.perf_counter()
        try:
            results[name] = {"status": "PASS", "result": fn(*[mods[p] for p in params])}
        except AssertionError as exc:  # a stated claim failed
            results[name] = {"status": "FAIL", "error": repr(exc)}
            status = "FAIL"
        results[name]["seconds"] = round(time.perf_counter() - t0, 3)
        print(f"{results[name]['status']:4s} {name} ({results[name]['seconds']} s)", flush=True)

    digests = {}
    for mod in (fm, fb, mu):
        pth = Path(mod.__file__)
        digests[pth.name] = hashlib.sha256(pth.read_bytes()).hexdigest()
    receipt = {
        "status": status,
        "scope": "MATHEMATICAL_AND_SIMULATION_CHECKS_ONLY; no gym training; no learning-benefit evidence",
        "script_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "module_sha256": digests,
        "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
        "checks": results,
    }
    text = json.dumps(receipt, indent=2, default=float) + "\n"
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    print(f"STATUS {status}: {sum(r['status'] == 'PASS' for r in results.values())}/{len(results)} checks")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
