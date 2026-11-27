"""Reachable-state parity and exact shortlist bounds for RV22-K."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


def test_cached_depth2_reachable_state_parity():
    script_dir = Path(__file__).resolve().parents[2] / "scripts" / "research_v2_2"
    with patch.object(sys, "path", [str(script_dir), *sys.path]):
        import adaptive_width as k

    for kind in ("iid", "parity3", "iid16", "parity4"):
        rng = np.random.default_rng(9000 + len(kind))
        for _ in range(32):
            d = k.generate(rng, kind)
            w, legacy = k.FastWorld(d), k.World(d)
            seen = set()
            stack = [(w.full, d["budget"], 0)]
            while stack:
                mask, budget, used = stack.pop()
                state = mask, budget, used
                if state in seen:
                    continue
                seen.add(state)
                assert list(w.legal(mask, budget, used)) == list(legacy.legal(mask, budget, used))
                for i, z, o in w.legal(mask, budget, used):
                    q2 = w.q2(mask, budget, i, z, o)
                    assert q2 == legacy.q(mask, budget, used, i, z, o, 2)
                    assert q2 <= w.upper2(mask, budget, i, z, o)
                    stack.extend((m, budget - w.costs[i], used | 1 << i) for m in (z, o))
            tree, _ = legacy.policy("two_step")
            assert tree == w.tree("exact") == w.tree("bounded")
            k.clear()
