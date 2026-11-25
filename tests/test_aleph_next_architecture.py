"""Independent verifier, sharing, causality and matched-perturbation tests for ALEPH-01/02."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "aleph_next"))

from arch_scoring import prepare  # noqa: E402
from arch_spectral import norms  # noqa: E402
from architecture_lab import (  # noqa: E402
    ARMS,
    HELD,
    TRAIN,
    MiniAleph,
    contract,
    op,
    sample,
    tensorize,
    training_batch,
    verdict,
)


def boolean_op(a, b, idx):
    return (a != b, a and b, a or b, (not a) or b)[idx]


def test_scalar_truth_verifier_and_target_is_absent_from_inputs():
    rng = np.random.default_rng(849)
    assert set(HELD).isdisjoint(TRAIN)
    for a in range(16):
        for b in range(16):
            for idx in range(4):
                scalar = sum(int(boolean_op(bool(a & (1 << i)), bool(b & (1 << i)), idx)) << i for i in range(4))
                assert op(a, b, idx) == scalar
    for task in range(4):
        for pair in TRAIN:
            example = sample(rng, task, pair)
            a, b, c, h, g, first, second, _, wanted = example
            result = []
            for i in range(4):
                a0, b0, c0, h0, g0 = (bool(x & (1 << i)) for x in (a, b, c, h, g))
                truth = boolean_op(boolean_op(a0, b0, first), c0, second)
                correct = truth if task < 2 else (boolean_op(h0, c0, second) if task == 2 else truth) != g0
                result.append(int(correct))
            assert wanted == sum(v << i for i, v in enumerate(result))
            if task >= 2:
                assert any(result)
            x, labels, task_tensor = tensorize([example])
            other, _, _ = tensorize([(*example[:-1], wanted ^ 15)])
            torch.testing.assert_close(x, other)
            assert verdict(2 * labels - 1, labels, task_tensor).item() == 1
    assert op(5, 3, 0) == 6


def test_stream_sharing_causality_and_norm_matched_corruption():
    torch.set_num_threads(1)
    x, _, _, kinds, _, noise = training_batch(137, 0, 4, 16)
    level = torch.full((4,), 1.25)
    for arm in ARMS:
        torch.manual_seed(91)
        m = MiniAleph(16, arm).eval()
        if arm.startswith("gate") or arm in ("magnitude3", "visit3"):
            with torch.no_grad():
                m.gate[1] = 0.65
        visits = []
        hook = m.u.register_forward_hook(lambda *args, visited=visits: visited.append(1))
        with torch.no_grad():
            logits, stats = m(x, kinds, level, noise, trace=True)
            hook.remove()
            assert len(visits) == ARMS[arm]
            torch.testing.assert_close(stats["injected_relative"][0], level)
            x2, n2 = x.clone(), noise.clone()
            x2[:, 2:, :5] *= -1
            n2[:, 2:] *= -1
            result = m(x2, kinds, level, n2)[0][-1]
            torch.testing.assert_close(result[:, :2], logits[-1][:, :2], rtol=0, atol=0)
        count = sum(p.numel() for p in m.parameters())
        assert count in (60 * 16**2 + 28 * 16 + 1, 60 * 16**2 + 28 * 16 + 3)


def test_initial_learned_clocks_reproduce_fixed_controls():
    x, _, _, k, lvl, n = training_batch(137, 1, 3, 16)
    torch.set_num_threads(1)
    for arm, reference in (
        ("magnitude3", "damp3"),
        ("visit3", "damp3"),
        ("gate_source2", "source2"),
        ("gate_local2", "local2"),
    ):
        torch.manual_seed(25)
        m = MiniAleph(16, arm).eval()
        with torch.no_grad():
            actual = m(x, k, lvl, n)[0][-1]
            expected = m(x, k, lvl, n, reference)[0][-1]
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_basis_sweep_recovers_known_linear_spectrum():
    class Diagonal:
        def core(self, x, source, arm, visit):
            return x * torch.tensor([0.5, 2.0]), None

    x = torch.ones(1, 2, 2)
    measured = norms(Diagonal(), x, x, "direct3", 0, 0.01)
    assert measured["operator_norm"] == pytest.approx(2.0, abs=1e-4)
    assert measured["normalized_frobenius"] == pytest.approx((2.125) ** 0.5, abs=1e-4)
    assert measured["operator_norm_j_minus_i"] == pytest.approx(1.0, abs=1e-4)


def test_architecture_scorer_requires_all_fits_before_private_draw(tmp_path):
    cfg = {"seeds": [137], "arms": ["direct2"], "width": 16, "steps": 1, "batch": 2}
    (tmp_path / "training.json").write_text(json.dumps({"contract": contract(cfg), "fits": {}}))
    with pytest.raises(ValueError, match="complete all"):
        prepare(tmp_path)
    assert not (tmp_path / "private.seed").exists()
