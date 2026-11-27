"""Small residual-value regression; no model endpoint or private data."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


def test_residual_two_step_decomposition_and_replay():
    script_dir = Path(__file__).resolve().parents[2] / "scripts" / "research_v2_2"
    with patch.object(sys, "path", [str(script_dir), *sys.path]):
        import residual_value as rv

    for seed in range(4):
        for world in (rv.make_world(np.random.default_rng(seed), False), rv.parity4(np.random.default_rng(seed))):
            _, y, bases = rv.ResidualWorld(world).examples2()
            assert np.all(np.asarray(bases)[:, 0] <= np.asarray(bases)[:, 1] + 1e-12)
            assert np.all(np.asarray(bases)[:, 1] <= np.asarray(y) + 1e-12)
            rv.clear()
            exact = rv.evaluate([world], "oracle")
            assert np.allclose(exact[:, 1], 0)
            assert rv.evaluate([world], "two_step")[0, 0] <= exact[0, 0] + 1e-12
