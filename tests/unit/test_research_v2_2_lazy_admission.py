"""Regression checks for exact memory admission."""
import sys
from pathlib import Path
from random import Random
from unittest.mock import patch

import numpy as np
import pytest


def test_lazy_certificate_admission_and_capacity():
    script_dir = Path(__file__).resolve().parents[2] / "scripts" / "research_v2_2"
    with patch.object(sys, "path", [str(script_dir), *sys.path]):
        import lazy_admission as m

    d = m.world(Random(93), 12, 6)
    v = dict(m.variants(d, Random(94)))
    ds = [d, v["perm_scale"], v["split_duplicate"], d]
    ref = m.evaluate(ds, "cold")[1]
    for arm in m.ARMS:
        stats, values = m.evaluate(ds, arm, np.ones(7))
        assert ref == values
        assert stats["peak_payload_bytes"] <= m.CAP
        if arm in ("canonical_selected", "checked_root"):
            assert stats["hits"] == 3
        if arm == "checked_root_lazy":
            assert stats["hits"] == 2 and stats["admissions"] == 1
    s = m.Store(5)
    s.put(b"abc", b"de")
    s.put(b"x", b"y")
    assert s.bytes == 2 and s.evictions == 1
    assert s.get(b"abc") is None
    s.put(b"large", b"record")
    assert s.bytes == 2
    solver, _, tree = m.solve(d)
    records, _ = m.checked_records(solver, tree)
    assert records
    key = ((1 << len(solver.h)) - 1, d["budget"])
    a, b = solver.memo[key]
    solver.memo[key] = (a + 1, b)
    with pytest.raises(ValueError, match="invalid Bellman"):
        m.check(solver)
    solver.memo.pop(key)
    with pytest.raises(ValueError, match="missing checked root"):
        m.check(solver)
