"""RV22-N: root identity, split isolation, verification and arm activation."""

import copy
import sys
from pathlib import Path
from random import Random
from unittest.mock import patch

import pytest


def test_economic_memory_contract():
    scripts = Path(__file__).resolve().parents[2] / "scripts" / "research_v2_2"
    with patch.object(sys, "path", [str(scripts), *sys.path]):
        import certificate_dp as old
        import economic_memory as m

    train = {
        m.key(d)[0]
        for i, kind in enumerate(m.KINDS[:3])
        for j in range(12)
        for d in m.stream(2026120100 + 100*i + j, kind, 60)
    }
    seen = set(train)
    for i, kind in enumerate(m.KINDS):
        keys = {
            m.key(d)[0]
            for j in range(16)
            for d in m.stream(2026128000 + 100*i + j, kind, 48)
        }
        assert not seen.intersection(keys)
        seen.update(keys)

    rng = Random(661)
    for i in range(8):
        d = m.world(rng, 12 if i % 2 == 0 else 16)
        variant = m.variant(d, rng)
        mutated = copy.deepcopy(d)
        mutated["tests"][0]["cost"] += 1
        assert m.key(d)[0] == m.key(variant)[0] != m.key(mutated)[0]
        for x in (d, variant, mutated):
            solver = m.DP(x)
            v, tree = solver.run()
            assert v == old.MaskDP(x).run()[0] == m.replay(x, tree)
            assert m.check(solver) == solver.edges
            root = solver.full, x["budget"]
            value, action = solver.memo[root]
            solver.memo[root] = value + 1, action
            with pytest.raises(ValueError, match="invalid Bellman"):
                m.check(solver)

    model = {"weights": [100.] + [0.] * 10, "shuffled": [100.] + [0.] * 10,
             "hit_ms": 0., "check_ratio": 1.}
    d = m.world(rng, 12)
    ds = [d, m.variant(d, rng), d, m.variant(d, rng)]
    ref = m.evaluate(ds, "cold", model)[1]
    for arm in m.ARMS:
        stats, values = m.evaluate(ds, arm, model)
        assert values == ref
        assert stats["peak_payload"] <= m.CAP
        if arm in ("trusted_lru", "checked_lru", "trusted_learned", "checked_learned"):
            assert stats["hits"] == 3 and stats["admissions"] == 1
        if arm == "checked_second":
            assert stats["hits"] == 2 and stats["checks"] == 1
    stats, values = m.evaluate(ds, "checked_lru", model, cap=1)
    assert values == ref and stats["hits"] == 0 and stats["peak_payload"] == 0
