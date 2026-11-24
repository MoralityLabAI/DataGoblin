"""Distill a coordinated pair of one-bit parity codes from the frozen training games."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from receipt_gym import CODES, LINEAR, TRAIN_PRIORS, Game, controls, parity
from torch.nn import functional as F

TOL = 1e-12


def _pack(bits):
    return sum(int(bit) << i for i, bit in enumerate(bits))


@dataclass(frozen=True)
class CodeTeacher:
    low_mask: int
    high_mask: int
    endpoint_risks: np.ndarray
    train_error: float
    bracket: tuple[int, int]

    def affine_risks(self, prior):
        """All 128 fixed-code Bayes risks from the two training endpoints."""
        if not 0 <= prior <= 1:
            raise ValueError("prior must be between zero and one")
        return (1 - prior) * self.endpoint_risks[0] + prior * self.endpoint_risks[1]

    def margin(self, prior):
        risks = self.affine_risks(prior)
        return float(risks[self.low_mask // 2] - risks[self.high_mask // 2])

    def hard_code(self, prior):
        """Resolve ties as a whole table, with a fixed public naming of the two bits."""
        mask = self.high_mask if self.margin(prior) > TOL else self.low_mask
        return CODES[mask // 2].copy()

    def target_logits(self, prior):
        """Signed risk margin in prior units on disagreements and fixed unit signs elsewhere."""
        low, high = CODES[[self.low_mask // 2, self.high_mask // 2]]
        slope = self.margin(1) - self.margin(0)
        return np.where(low == high, 2 * low - 1, (2 * high - 1) * self.margin(prior) / slope)

    def external_control(self, prior):
        risks = self.affine_risks(prior)
        best = float(risks.min())
        chosen = _pack(self.hard_code(prior))
        return {
            "mask": chosen,
            "risk": float(risks[chosen // 2]),
            "oracle": best,
            "optimal_masks": (2 * np.flatnonzero(risks <= best + TOL)).tolist(),
        }


@lru_cache(maxsize=1)
def build_teacher():
    """Enumerate only TRAIN_PRIORS and pick a common interior winner in each regime."""
    train = np.asarray(TRAIN_PRIORS, dtype=float)
    if train[0] != 0 or train[-1] != 1 or not np.all(np.diff(train) > 0):
        raise ValueError("the training prior grid must span [0, 1] in increasing order")
    risks = np.stack([controls(parity(prior))[0] for prior in train])
    endpoints = risks[[0, -1]].copy()
    interpolated = (1 - train[:, None]) * endpoints[0] + train[:, None] * endpoints[1]
    error = float(np.max(np.abs(risks - interpolated)))
    if error > TOL:
        raise AssertionError("endpoint interpolation disagrees with enumerated training risk")

    winners = risks <= risks.min(axis=1, keepdims=True) + TOL
    nonlinear = ~(CODES[:, None] == LINEAR[None]).all(axis=2).any(axis=1)
    nonlinear[0] = False
    interior = (train > 0) & (train < 1) & winners[:, nonlinear].any(axis=1)
    if not interior.any():
        raise ValueError("no interior training prior has a nonlinear optimum")
    low_options = 2 * np.flatnonzero(winners[interior].all(axis=0) & nonlinear)

    # Keep one high-endpoint winner that also wins every training row for which
    # an endpoint winner is optimal. The same tie rule applies to every arm.
    high_rows = winners[:, winners[-1]].any(axis=1)
    high_options = 2 * np.flatnonzero(winners[high_rows].all(axis=0) & winners[-1])
    if not len(low_options) or not len(high_options):
        raise ValueError("training regimes have no common optimal codes")
    high = int(min(high_options))
    low = min(map(int, low_options), key=lambda mask: ((mask ^ high).bit_count(), mask))
    diff = risks[:, low // 2] - risks[:, high // 2]
    if diff[0] >= -TOL or diff[-1] <= TOL or not np.all(np.diff(diff) > 0):
        raise ValueError("training codes must show a single interior preference change")
    if np.any(np.minimum(risks[:, low // 2], risks[:, high // 2]) > risks.min(axis=1) + TOL):
        raise ValueError("selected pair misses a training optimum")
    bracket = (int(np.flatnonzero(diff < -TOL)[-1]), int(np.flatnonzero(diff > TOL)[0]))
    endpoints.setflags(write=False)
    return CodeTeacher(low, high, endpoints, error, bracket)


def teacher_receiver_brier(model, game, code):
    """Proper realized Brier with the ordinary receiver inputs and a fixed hard upload."""
    if code.shape != (8,) or not ((code == 0) | (code == 1)).all():
        raise ValueError("teacher code must be an eight-world binary truth table")
    return game.risks(code, game.guesses(model))[0]


def fit_coordinated(model, seed, steps=1200):
    """Fit the provided CPU model in place. Its caller controls its initial weights."""
    if steps <= 0 or any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("use positive steps and a CPU model")
    torch.set_num_threads(1)
    model.train()
    teacher = build_teacher()
    games = [Game(parity(prior)) for prior in TRAIN_PRIORS]
    codes = [torch.tensor(teacher.hard_code(p), dtype=torch.float32) for p in TRAIN_PRIORS]
    features = torch.cat([g.p for g in games])
    targets = torch.tensor(np.stack([teacher.target_logits(p) for p in TRAIN_PRIORS]), dtype=torch.float32)
    low, high = teacher.bracket
    gap = float(TRAIN_PRIORS[high] - TRAIN_PRIORS[low])
    # Normalize point errors by the train-only margin size, floored by the
    # observed preference-change bracket. Keep unit weight on agreeing worlds.
    scales = targets.abs().clamp_min(gap)
    disagree = codes[low] != codes[high]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    rng = np.random.default_rng(seed)
    trace = []
    for step in range(steps):
        if step % len(games) == 0:
            order = rng.permutation(len(games))
        index = int(order[step % len(games)])
        logits = model.encode(features).reshape_as(targets)
        error = logits - targets
        point = F.smooth_l1_loss(error / scales, torch.zeros_like(error))
        # Match the local slope at the two observed bracketing priors. No extra
        # prior examples, analytic root targets or receiver metadata are used.
        secant = (error[high, disagree] - error[low, disagree]) / gap
        slope = F.smooth_l1_loss(secant, torch.zeros_like(secant))
        brier = teacher_receiver_brier(model, games[index], codes[index])
        loss = point + 0.25 * slope + brier
        # Anneal the second half to reduce stochastic receiver updates moving
        # a nearly indifferent producer across its learned decision boundary.
        fraction = step / max(steps - 1, 1)
        optimizer.param_groups[0]["lr"] = 0.003 * (1 - 0.95 * max(0.0, 2 * fraction - 1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            trace.append(
                {
                    "step": step + 1,
                    "prior": float(TRAIN_PRIORS[index]),
                    "point": float(point.detach()),
                    "secant": float(slope.detach()),
                    "teacher_brier": float(brier.detach()),
                }
            )

    diagnostics = []
    model.eval()
    with torch.no_grad():
        for prior, game, target, expected in zip(TRAIN_PRIORS, games, targets, codes, strict=True):
            logits = model.encode(game.p)
            learned = (logits >= 0).float()
            guesses = game.guesses(model)
            risk, ideal = game.risks(learned, guesses)
            diagnostics.append(
                {
                    "prior": float(prior),
                    "teacher_mask": _pack(expected),
                    "learned_raw_mask": _pack(learned),
                    "teacher_brier": float(game.risks(expected, guesses)[0]),
                    "learned_brier": float(risk),
                    "learned_code_bayes": float(ideal),
                    "max_logit_error": float((logits - target).abs().max()),
                }
            )
    return {
        "seed": int(seed),
        "steps": int(steps),
        "train_priors": list(TRAIN_PRIORS),
        "low_mask": teacher.low_mask,
        "high_mask": teacher.high_mask,
        "bracket_train_priors": [float(TRAIN_PRIORS[low]), float(TRAIN_PRIORS[high])],
        "max_train_affine_control_error": teacher.train_error,
        "trace": trace,
        "train_diagnostics": diagnostics,
    }
