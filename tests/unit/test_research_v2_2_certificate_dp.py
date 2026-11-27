"""Exact continuation reuse, mutations and comparison against legacy full DP."""
import copy
import random
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_certificate_transport_and_rejection():
    directory = Path(__file__).resolve().parents[2] / "scripts" / "research_v2_2"
    with patch.object(sys, "path", [str(directory), *sys.path]):
        import certificate_dp as l
        from decision_value import World

    rng = random.Random(2026112815)
    for j in range(32):
        d = l.world(rng, 12 if j % 2 == 0 else 16, 6 if j % 2 == 0 else 8)
        m = l.CertifiedDP()
        variants = l.variants(d, rng)
        for field in ("budget", "correct_reward", "wrong_reward"):
            v = copy.deepcopy(d)
            v[field] += 1
            variants.append((field, v))
        v = copy.deepcopy(d)
        v["hypotheses"][0]["class"] = "B" if v["hypotheses"][0]["class"] == "A" else "A"
        variants.append(("label", v))
        for name, v in variants:
            a, b = l.MaskDP(v).run()
            c, e, stats = m.run(v)
            w = World(v)
            k = w.value(w.full, v["budget"], 0, len(w.tests))
            mass = w.mass(w.full)
            assert a * mass == k * b and c * mass == k * e
            if name in ("perm_scale", "split_duplicate", "conditioned"):
                assert stats["root_hit"] == 1
            if name in ("budget", "correct_reward", "wrong_reward", "label", "cost_mutation"):
                assert stats["root_hit"] == 0
            World.mass.cache_clear()
            World.a_mass.cache_clear()
            World.value.cache_clear()
    d = l.world(rng, 12, 6)
    m = l.CertifiedDP(capacity=1)
    m.run(d)
    assert not m.verified and not m.policies
    for name in ("A", "B", "defer"):
        v = copy.deepcopy(d)
        v["tests"][0]["id"] = name
        with pytest.raises(ValueError, match="unsupported test"):
            l.root(v)
