"""MMS-07 CV engineering reference. Synthetic checks only; no gym training."""
from __future__ import annotations

import json

import numpy as np


class CoefficientBank:
    """Fixed codec and logical key universe; exact tags; FP16 payload, FP64 sum.

    Modulo indexing is deliberately used to test collisions. This is a reference,
    not the proposed production hash. Missing logical keys have zero coefficients.
    Call estimate before writing any entries from the current minibatch.
    """

    def __init__(self, n, capacity, dictionary):
        self.dictionary = np.array(dictionary, dtype=np.float64, copy=True)
        if self.dictionary.ndim != 2 or not np.isfinite(self.dictionary).all():
            raise ValueError("dictionary must be a finite matrix")
        if not 0 < capacity <= n or self.dictionary.shape[1] == 0:
            raise ValueError("invalid capacity or rank")
        self.dictionary.flags.writeable = False
        self.n = n
        self.tags = np.full(capacity, -1, dtype=np.int64)
        self.rows = np.zeros((capacity, self.dictionary.shape[1]), dtype=np.float16)
        self.total = np.zeros(self.rows.shape[1], dtype=np.float64)

    def _keys(self, keys):
        keys = np.asarray(keys)
        if keys.ndim != 1 or keys.dtype.kind not in "iu" or np.any(keys >= self.n) or np.any(keys < 0):
            raise ValueError("invalid logical keys")
        return keys.astype(np.int64)

    def read(self, keys):
        keys = self._keys(keys)
        slots = keys % len(self.tags)
        hit = self.tags[slots] == keys
        return np.where(hit[:, None], self.rows[slots], 0).astype(np.float64), hit

    def write(self, keys, coefficients):
        keys = self._keys(keys)
        with np.errstate(over="ignore", invalid="ignore"):
            rows = np.asarray(coefficients, dtype=np.float16)
        if rows.shape != (len(keys), self.rows.shape[1]) or not np.isfinite(rows).all():
            raise ValueError("invalid or unrepresentable coefficients")
        for key, row in zip(keys, rows):
            slot = key % len(self.tags)
            self.total -= self.rows[slot].astype(np.float64)
            self.tags[slot] = key
            self.rows[slot] = row
            self.total += row.astype(np.float64)

    def evict(self, key):
        key = int(self._keys([key])[0])
        slot = key % len(self.tags)
        if self.tags[slot] == key:
            self.total -= self.rows[slot].astype(np.float64)
            self.rows[slot] = 0
            self.tags[slot] = -1

    def encode(self, gradients):
        """Least-squares synthesis coordinates, including nonorthogonal atoms."""
        return np.linalg.lstsq(self.dictionary, np.asarray(gradients).T, rcond=None)[0].T

    def estimate(self, keys, gradients, alpha, probabilities=None):
        """Uniform target; draws IID with replacement from declared full-support p.

        alpha, dictionary, tags, payload and total must be fixed before the draw.
        Returned gradient is unbiased conditionally on that state (exact arithmetic).
        """
        keys = self._keys(keys)
        g = np.asarray(gradients, dtype=np.float64)
        if len(keys) == 0 or g.shape != (len(keys), self.dictionary.shape[0]) or not np.isfinite(g).all():
            raise ValueError("invalid gradient batch")
        if not np.isfinite(alpha):
            raise ValueError("invalid alpha")
        if probabilities is None:
            weights = np.ones(len(keys))
        else:
            p = np.asarray(probabilities, dtype=np.float64)
            if p.shape != (self.n,) or not np.isfinite(p).all() or np.any(p <= 0) or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0):
                raise ValueError("probabilities must be normalized with full support")
            weights = 1.0 / (self.n * p[keys])
        a, _ = self.read(keys)
        return (weights[:, None] * (g - alpha * (a @ self.dictionary.T))).mean(0) + alpha * (self.dictionary @ self.total) / self.n


def population_stats(g, t, p):
    """Exact single-draw trace moments for a uniform target and sampling law p."""
    w = 1 / (len(p) * p[:, None])
    x, y = w * g - g.mean(0), w * t - t.mean(0)
    vg, vt, cov = [float(np.sum(p[:, None] * v)) for v in (x*x, y*y, x*y)]
    alpha = cov / vt if vt > 0 else 0.0
    return vg, vt, cov, alpha


def run_checks():
    rng = np.random.default_rng(7127)
    n, dim, rank = 64, 32, 4
    D = np.linalg.qr(rng.normal(size=(dim, rank)))[0]
    g = rng.normal(size=(n, rank)) @ D.T + 0.25 * rng.normal(size=(n, dim))
    stale = g + 0.2 * rng.normal(size=g.shape)
    checks = {}
    for capacity in (16, 64):
        bank = CoefficientBank(n, capacity, D)
        bank.write(np.arange(n), bank.encode(stale))
        bank.write([1, 1, 17], bank.encode(stale[[2, 3, 4]]))
        bank.evict(17)
        keys = np.arange(n)
        rows, hit = bank.read(keys)
        np.testing.assert_array_equal(bank.total, rows.sum(0))
        t = rows @ D.T
        for label, p in (("uniform", np.full(n, 1/n)), ("nonuniform", np.arange(1, n+1)/2080)):
            vg, vt, cov, alpha = population_stats(g, t, p)
            estimates = np.array([bank.estimate([i], g[i:i+1], alpha, p) for i in keys])
            mean = (p[:, None] * estimates).sum(0)
            variance = float((p[:, None] * (estimates - mean)**2).sum())
            np.testing.assert_allclose(mean, g.mean(0), atol=1e-12, rtol=0)
            np.testing.assert_allclose(variance, vg - 2*alpha*cov + alpha**2*vt, atol=1e-12)
            assert variance <= vg + 1e-12
            checks[f"{capacity}_{label}"] = {"hit_count": int(hit.sum()), "alpha_oracle": alpha, "variance_ratio": variance/vg, "max_bias": float(np.max(np.abs(mean-g.mean(0))))}
        before = bank.total.copy()
        try:
            bank.write([2], [[1e100]*rank])
        except ValueError:
            pass
        else:
            raise AssertionError("overflow accepted")
        np.testing.assert_array_equal(bank.total, before)
    # Nonorthogonal codec: inner products are not synthesis coefficients.
    D2 = D.copy()
    D2[:, 1] = 0.8*D[:, 0] + 0.6*D[:, 1]
    bank = CoefficientBank(n, n, D2)
    encoded = bank.encode(g)
    np.testing.assert_allclose(encoded @ D2.T, (g @ D) @ D.T, atol=1e-12)
    # A structured synthetic signal tests matched rank, not a learned BPD advantage.
    p = np.full(n, 1/n)
    for label, codec in (("signal_dictionary", D), ("random_dictionary", np.linalg.qr(rng.normal(size=(dim, rank)))[0])):
        bank = CoefficientBank(n, n, codec)
        bank.write(np.arange(n), bank.encode(stale))
        t = bank.read(np.arange(n))[0] @ codec.T
        vg, vt, cov, alpha = population_stats(g, t, p)
        checks[label] = {"rank": rank, "alpha_oracle": alpha, "variance_ratio": (vg-2*alpha*cov+alpha**2*vt)/vg}
    return {"status": "PASS", "scope": "SYNTHETIC_FIXED_STATE_ONLY; oracle alpha is diagnostic, not an online policy", "checks": checks}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, allow_nan=False))
