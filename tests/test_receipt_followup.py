"""Controls for the second one-bit receipt tournament."""

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "aleph_rlt_placement"))

from delayed_parity import risk  # noqa: E402
from receipt_exposure import fit_matched_outcome  # noqa: E402
from receipt_gym import CODES, LINEAR, TRAIN_PRIORS, Game, parity  # noqa: E402
from receipt_models import ReceiptModel  # noqa: E402


def test_exposure_control_batches_all_priors_and_integrates_same_bayes_risk():
    torch.manual_seed(42)
    model = ReceiptModel("MLP")
    model.eval()
    with torch.no_grad():
        expected = []
        for prior in TRAIN_PRIORS:
            game = Game(parity(prior))
            expected.append(float(game.risks(model.encode(game.p).sigmoid(), game.guesses(model))[1]))
    trace = fit_matched_outcome(model, 101, steps=1)
    assert trace[0][2] == pytest.approx(np.mean(expected), abs=1e-6)


def test_coordinate_query_and_side_admit_more_than_a_linear_upload():
    # Neutral reference game, independent of all seeded native fixtures.
    def brier(code):
        return sum(risk(code, 1 << side, 1 << query) for side in range(3) for query in range(3)) / 9

    selector_code = [((x >> 2) & 1) if x & 2 else (x & 1) for x in range(8)]
    assert min(brier(code.tolist()) for code in CODES) == brier(selector_code) == Fraction(11, 108)
    assert min(brier(code.tolist()) for code in LINEAR) == Fraction(1, 9)


def test_half_completed_coordinated_change_pays_positive_regret_at_the_tie():
    q = Fraction(4, 53)

    def loss(mask):
        code = CODES[mask // 2].tolist()
        weighted = [
            ((1 - q) / 7 + q * (target == 1)) * risk(code, side, target) / 7
            for target in range(1, 8)
            for side in range(1, 8)
        ]
        return sum(weighted)

    assert loss(46) == loss(170)
    for halfway in (46 ^ 4, 46 ^ 128):
        assert loss(halfway) - loss(46) == Fraction(1, 1484)
