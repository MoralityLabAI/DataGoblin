"""Independent finite-oracle and hard-boundary checks for the optional CPU pilot."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "aleph_rlt_placement"))

from delayed_parity import gains  # noqa: E402
from receipt_gym import CODES, Game, controls, parity  # noqa: E402
from receipt_ldt import make_games  # noqa: E402
from receipt_models import ReceiptModel  # noqa: E402


def test_exact_threshold_and_side_only():
    majority = sum(int(x.bit_count() >= 2) << x for x in range(8))
    store_first = 170
    for prior in (0, 0.07, 4 / 53, 0.08, 1):
        risk, linear = controls(parity(prior))
        assert risk[0] == pytest.approx(3 / 14)
        assert min(risk) == pytest.approx(3 / 14 - max(1 / 14 + prior / 56, 3 / 49 + 15 * prior / 98))
        assert abs(linear - risk[store_first // 2]) < 1e-12
        assert risk[(majority if prior <= 4 / 53 else store_first) // 2] == pytest.approx(min(risk))
    at_zero = controls(parity(0))[0]
    for code in (18, 100, 224):
        g = gains(CODES[code // 2].tolist(), range(1, 8))
        assert at_zero[0] - at_zero[code // 2] == pytest.approx(float(sum(g) / 7))


def test_soft_channel_is_a_mixture_of_hard_events():
    game = Game(parity(0.25))
    p = torch.linspace(0.1, 0.8, 8)
    guess = torch.linspace(0.02, 0.98, 196).reshape(2, 2, 49)
    actual = game.risks(p, guess)[0]
    expected = torch.tensor(0.0)
    for x in range(8):
        for m, prob in enumerate((1 - p[x], p[x])):
            value = guess[m, game.s[x].long(), torch.arange(49)]
            expected += (game.w * prob / 8 * (value - game.y[x]).square()).sum()
    torch.testing.assert_close(actual, expected)


def test_matched_init_stateless_receiver_and_binary_contract():
    torch.manual_seed(84)
    c = ReceiptModel("C")
    torch.manual_seed(84)
    d = ReceiptModel("D")
    for k, v in c.state_dict().items():
        torch.testing.assert_close(v, d.state_dict()[k])
    game = Game(parity(0.25))
    for m in (c, d):
        before = game.guesses(m)
        m.encode(torch.randn(8, 12))
        torch.testing.assert_close(game.guesses(m), before, rtol=0, atol=0)
        with pytest.raises(ValueError):
            m.decode(game.r, game.side, game.msg + 0.25)


def test_fresh_families_and_seed_replay():
    train = make_games("train", 101)
    fresh = make_games("test", 202)
    assert {k.rsplit("/", 1)[0] for k in train}.isdisjoint(k.rsplit("/", 1)[0] for k in fresh)
    replay = make_games("test", 202)
    other = make_games("test", 303)
    assert any(
        not np.array_equal(v[0], other[k][0])
        or not np.array_equal(v[1], other[k][1])
        or not np.array_equal(v[3], other[k][3])
        for k, v in fresh.items()
    )
    for k, raw in fresh.items():
        for a, b in zip(raw, replay[k], strict=True):
            np.testing.assert_array_equal(a, b)
        Game(raw)
        risk, fixed = controls(raw)
        assert min(risk) <= fixed + 1e-12 <= risk[0] + 1e-10


def test_certified_single_flip_barrier():
    for prior in (0.0, 0.07, 1.0):
        risk, _ = controls(parity(prior))
        base = risk[170 // 2]
        for x in range(8):
            candidate = 170 ^ (1 << x)
            if candidate & 1:
                candidate ^= 255
            assert risk[candidate // 2] - base == pytest.approx((15 * prior - 1) / 196)


def test_native_transfer_relations():
    for key, (_, queries, labels, side, _) in make_games("test", 117).items():
        high = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(bool)
        majority = high.sum(axis=1) >= 2
        for i, query in enumerate(queries):
            axis = np.flatnonzero(query[6:9]).item()
            np.testing.assert_array_equal(side[:, i], high[:, axis])
            if key.startswith("array_two_opponents/"):
                shared = np.flatnonzero(query[:3] * query[3:6]).item()
                wanted = majority | high[:, shared]
            else:
                isolated = np.flatnonzero(query[3:6]).item()
                wanted = majority & high[:, isolated]
            np.testing.assert_array_equal(labels[:, i], wanted)
