"""Train-only checks for the coordinated parity teacher and receiver contract."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "aleph_rlt_placement"))

import receipt_coordination as coordination  # noqa: E402
from delayed_parity import risk  # noqa: E402
from receipt_gym import CODES, TRAIN_PRIORS, Game, controls, parity  # noqa: E402
from receipt_models import ReceiptModel  # noqa: E402


def is_nonlinear(code):
    return any(code[x ^ y] != (code[x] ^ code[y]) for x in range(8) for y in range(8))


def test_tie_break_uses_common_interior_optima_and_canonical_names():
    teacher = coordination.build_teacher()
    low, high = teacher.bracket
    common = set(range(0, 256, 2))
    for prior in TRAIN_PRIORS[1 : low + 1]:
        risks, _ = controls(parity(prior))
        common.intersection_update(2 * np.flatnonzero(risks <= risks.min() + 1e-12))
    candidates = sorted(int(m) for m in common if is_nonlinear(CODES[m // 2]))
    assert len(candidates) > 1
    final, _ = controls(parity(TRAIN_PRIORS[-1]))
    high_choice = int(2 * np.flatnonzero(final <= final.min() + 1e-12)[0])
    assert teacher.high_mask == high_choice
    assert teacher.low_mask == min(reversed(candidates), key=lambda m: ((m ^ high_choice).bit_count(), m))
    assert all(teacher.hard_code(p)[0] == 0 for p in TRAIN_PRIORS)
    assert TRAIN_PRIORS[low] < TRAIN_PRIORS[high]


def test_affine_control_and_signed_world_margins_match_independent_train_risks():
    teacher = coordination.build_teacher()
    low, high = CODES[[teacher.low_mask // 2, teacher.high_mask // 2]]
    disagree = low != high
    endpoint_logits = np.stack([teacher.target_logits(p) for p in (0, 1)])
    dir_risks = [[np.mean([float(risk(code, u, t)) for u in range(1, 8)]) for t in range(1, 8)] for code in (low, high)]
    for p in TRAIN_PRIORS:
        external = teacher.external_control(p)
        exhaustive = controls(parity(p))[0]
        np.testing.assert_allclose(teacher.affine_risks(p), exhaustive, atol=1e-14, rtol=0)
        assert external["mask"] in external["optimal_masks"]
        assert external["oracle"] == pytest.approx(exhaustive.min())
        low_risk, high_risk = [(1 - p) * np.mean(v) + p * v[0] for v in dir_risks]
        assert teacher.margin(p) == pytest.approx(low_risk - high_risk)
        target = teacher.target_logits(p)
        np.testing.assert_allclose(target, (1 - p) * endpoint_logits[0] + p * endpoint_logits[1], atol=1e-14)
        np.testing.assert_array_equal(target > 0, teacher.hard_code(p))
        np.testing.assert_array_equal(target[~disagree], 2 * low[~disagree] - 1)
        slope = teacher.margin(1) - teacher.margin(0)
        np.testing.assert_allclose(target[disagree] * slope, (2 * high[disagree] - 1) * (low_risk - high_risk))


def test_teacher_forcing_preserves_public_receiver_information_for_both_codes():
    teacher = coordination.build_teacher()
    game = Game(parity(TRAIN_PRIORS[1]))

    class Receiver:
        def __init__(self):
            self.calls = []

        def decode(self, features, side, message):
            self.calls.append((features.clone(), side.clone(), message.clone()))
            return features[:, 0] * 0.25 + side * 0.75 - message

    receiver = Receiver()
    for mask in (teacher.low_mask, teacher.high_mask):
        code = torch.tensor(CODES[mask // 2], dtype=torch.float32)
        score = coordination.teacher_receiver_brier(receiver, game, code)
        features, side, message = receiver.calls[-1]
        torch.testing.assert_close(features, game.r)
        torch.testing.assert_close(side, game.side)
        torch.testing.assert_close(message, game.msg)
        q = len(game.w)
        logits = game.r[:q, 0] * 0.25 + game.s * 0.75 - code[:, None]
        expected = ((logits.sigmoid() - game.y).square() * game.w).sum() / 8
        torch.testing.assert_close(score, expected)
    with pytest.raises(ValueError, match="binary"):
        coordination.teacher_receiver_brier(receiver, game, torch.full((8,), 0.25))


def test_short_fit_stays_on_training_priors_and_emits_valid_receipt(monkeypatch):
    requested = []

    def train_only_parity(prior):
        assert prior in TRAIN_PRIORS
        requested.append(prior)
        return parity(prior)

    monkeypatch.setattr(coordination, "parity", train_only_parity)
    torch.manual_seed(7)
    model = ReceiptModel("D")
    result = coordination.fit_coordinated(model, 7, steps=2)
    assert set(requested) == set(TRAIN_PRIORS)
    assert result["steps"] == 2
    assert [row["prior"] for row in result["train_diagnostics"]] == list(TRAIN_PRIORS)
    assert all(np.isfinite(row["learned_brier"]) for row in result["train_diagnostics"])
    json.dumps(result, allow_nan=False)
