"""Independent finite checks for the nonlinear uplink pilot."""

import sys
from fractions import Fraction
from itertools import combinations_with_replacement, product
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "aleph_next"))
import nonlinear_lab as lab


@pytest.fixture(scope="module")
def programs():
    return lab.build_programs()


def test_exact_integer_brier_and_actual_linear_witness(programs):
    g = programs[0]
    p = g["profiles"][0]
    w = p["endpoints"][0] // 64
    brute = []
    for code in lab.CODES:
        total = Fraction()
        for q, weight in enumerate(w):
            for m, s in product((0, 1), repeat=2):
                worlds = [i for i in range(8) if code[i] == m and g["side"][i, q] == s]
                if worlds:
                    yes = sum(int(g["y"][i, q]) for i in worlds)
                    total += int(weight) * Fraction(yes * (len(worlds) - yes), 8 * len(worlds))
        brute.append(total * lab.UNIT)
    assert all(int(x) == y for x, y in zip(g["risks"] @ w, brute, strict=True))
    assert min(brute) == 1834000
    assert min(brute[k] for k in g["linear"]) == 2205840
    assert 2 * brute.index(min(brute)) == 116


def test_gates_reject_every_simple_winner_and_transfer_interpolates(programs):
    for g in programs:
        _, shortcuts = lab.shortcut_codes(g["y"])
        assert len({p["code"] for p in g["profiles"]}) == 2
        for p in g["profiles"]:
            for t in (0, 0.25, 0.75, 1):
                r = g["risks"] @ lab.mix(p, t) / lab.UNIT
                assert p["code"] == int(r.argmin())
                assert sum(abs(r - r.min()) < 1e-14) == 1
                assert r[sorted(shortcuts)].min() - r.min() >= 0.001 - 1e-13
        assert g["side"].shape == g["y"].shape == (8, 18)
        for j, q in enumerate(g["features"]):
            np.testing.assert_array_equal(g["side"][:, j], lab.WORLDS[:, np.flatnonzero(q[6:9]).item()])


def test_generated_predicates_and_slot_timing(programs):
    for g in programs:
        for j, q in enumerate(g["features"]):
            if g["family"] == "conditional_obligation":
                gate = np.flatnonzero(q[:3]).item()
                left = np.flatnonzero(abs(q[3:6]) == 1).item()
                right = np.flatnonzero(abs(q[3:6]) == 2).item()
                y = np.where(
                    lab.WORLDS[:, gate] ^ (q[gate] < 0),
                    lab.WORLDS[:, right] ^ (q[3 + right] < 0),
                    lab.WORLDS[:, left] ^ (q[3 + left] < 0),
                )
            else:
                a, b = q[:3] != 0, q[3:6] != 0
                negative = (q[:3] < 0) | (q[3:6] < 0)
                x = lab.WORLDS ^ negative
                y = x[:, a].any(1) & x[:, b].any(1)
            np.testing.assert_array_equal(y, g["y"][:, j])
        p, q = lab.public_inputs(g, 0)
        np.testing.assert_array_equal(p[:, 3:], np.tile(p[0, 3:], (8, 1)))
        assert p.shape == (8, 12) and q.shape == (18, 12)


def test_real_threshold_class_is_complete():
    scored = []
    for weights in product(range(-3, 4), repeat=3):
        values = lab.WORLDS @ weights
        scored.extend(lab.canonical(values > cutoff) for cutoff in np.unique(values))
    threshold = set(scored)
    assert len(threshold) == 52

    def pair_sums(worlds):
        return {tuple(lab.WORLDS[i] + lab.WORLDS[j]) for i, j in combinations_with_replacement(worlds, 2)}

    for code, truth in enumerate(lab.CODES):
        # Equal sums of two positive and two negative points preclude any real affine separator.
        obstruction = pair_sums(np.flatnonzero(truth)) & pair_sums(np.flatnonzero(1 - truth))
        assert bool(obstruction) == (code not in threshold)


def test_partitions_and_budgeted_independent_selection(programs):
    parts = lab.central_partitions()
    assert parts.shape == (2795, 8) and len(set(map(tuple, parts))) == 2795
    refs = lab.design(programs, 0, np.full(8, 0.5))
    for b, v in refs.items():
        assert len(set(v["authorized_sites"])) == b
        assert v["side_only"] >= v["linear_optimum"] > v["nonlinear_optimum"]
        assert v["native_ldt"] + 1e-12 >= v["nonlinear_optimum"]
    assert refs[8]["native_ldt"] == pytest.approx(refs[8]["round_robin"])
    # A concrete two-site joint enumeration checks factorization with one chosen local code per site.
    a, b = programs[:2]
    q0 = 2
    ca = lab.CODES[a["profiles"][0]["code"]]
    cb = lab.CODES[b["profiles"][1]["code"]]
    joint = Fraction()
    for m, n, s in product((0, 1), repeat=3):
        worlds = [(i, j) for i, j in product(range(8), repeat=2) if ca[i] == m and cb[j] == n and a["side"][i, q0] == s]
        if worlds:
            yes = sum(int(a["y"][i, q0]) for i, j in worlds)
            joint += Fraction(yes * (len(worlds) - yes), 64 * len(worlds))
    assert joint == Fraction(int(a["risks"][a["profiles"][0]["code"], q0]), lab.UNIT)
