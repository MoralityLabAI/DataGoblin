"""Posterior fitting and frozen producer checks on train-only fixtures."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "aleph_rlt_placement"))

from receipt_gym import Game, parity  # noqa: E402
from receipt_models import ReceiptModel  # noqa: E402
from receipt_readout import balanced, fit, new_receiver, prepare, weight_hash  # noqa: E402


def test_train_frequency_loss_is_realized_brier_less_ideal():
    sender = ReceiptModel("C").eval().requires_grad_(False)
    game = Game(parity(0.05))
    frame, post, weight = prepare(sender, [game])[0]
    with torch.no_grad():
        code = (sender.encode(game.p) >= 0).float()
    guess = torch.linspace(0.01, 0.99, post.numel()).reshape_as(post)
    actual, ideal = frame.risks(code, guess)
    torch.testing.assert_close((weight * (guess - post).square()).sum(), actual - ideal)
    assert weight.sum() == pytest.approx(1.0)
    assert weight.shape == (2, 2, 49)


def test_separate_receiver_changes_with_identical_frozen_parity_code():
    torch.set_num_threads(1)
    sender = ReceiptModel("D").eval().requires_grad_(False)
    game = Game(parity(0.05))
    receiver = new_receiver("C", 137)
    old_sender, old_receiver = weight_hash(sender), weight_hash(receiver)
    tracked = fit(receiver, sender, prepare(sender, [game]), game.p, 137, steps=2)
    assert weight_hash(sender) == old_sender == tracked["sender_weight_sha256"]
    assert weight_hash(receiver) != old_receiver
    assert tracked["retention"][0]["sha256"] == tracked["retention"][-1]["sha256"]
    assert [p["max_abs_drift"] for p in tracked["retention"]] == [0, 0]
    with pytest.raises(ValueError, match="overlap"):
        fit(sender, sender, prepare(sender, [game]), game.p, 137, steps=1)


def test_family_balancing_does_not_overweight_larger_families():
    rows = {
        "one/0": {"brier": 0.1, "raw_mask": 2},
        "two/0": {"brier": 0.2, "raw_mask": 4},
        "two/1": {"brier": 0.4, "raw_mask": 6},
    }
    assert balanced(rows)["overall"] == {"brier": pytest.approx(0.2)}
