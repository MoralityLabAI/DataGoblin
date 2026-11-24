"""Public fixture checks for two-rival LDT transfer, independent of model scoring."""

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "aleph_rlt_placement"))

import receipt_ldt
from receipt_relations import make_relations, verified_side_sha256


@pytest.fixture(scope="module")
def panels():
    return make_relations("train", 84031), make_relations("test", 84033)


def _status(key):
    return "SELECTED" if key.startswith("point_selected_") else "UNRESOLVED"


def _family(key):
    return key.rsplit("/", 1)[0]


def test_relation_holdout_and_matched_slot_support(panels):
    train, test = panels
    assert (len(train), len(test)) == (24, 16)
    assert {_family(k) for k in train}.isdisjoint(_family(k) for k in test)
    marginal_support = []
    joint_support = []
    for split, games in (("train", train), ("test", test)):
        contexts = defaultdict(set)
        mixtures = defaultdict(set)
        marginal, joint = set(), set()
        orientations = defaultdict(set)
        for key, (_, queries, _, _, weights) in games.items():
            assert len({tuple(row) for row in queries}) == len(queries) == 6
            contexts[_family(key)].add(frozenset(tuple(row) for row in queries))
            mixtures[_family(key)].add(tuple(weights))
            for q in queries:
                a = frozenset(np.flatnonzero(q[:3]))
                b = frozenset(np.flatnonzero(q[3:6]))
                axis = np.flatnonzero(q[6:9]).item()
                assert a and b
                shape = len(a), len(b)
                orientations[_status(key)].add(shape)
                if split == "train":
                    assert a.isdisjoint(b) or (shape == (2, 2) and len(a & b) == 1)
                else:
                    assert a < b or b < a
                joint.add((_status(key), a, b))
                for slot, mask in enumerate((a, b)):
                    marginal.add((_status(key), shape, slot, mask, axis))
        assert all(len(group) == 4 for group in contexts.values())
        assert all(len(group) == 4 for group in mixtures.values())
        wanted = {(1, 2), (2, 1), (2, 2)} if split == "train" else {(1, 2), (2, 1)}
        assert orientations == {"SELECTED": wanted, "UNRESOLVED": wanted}
        marginal_support.append(marginal)
        joint_support.append(joint)
    assert marginal_support[1] <= marginal_support[0]
    assert joint_support[0].isdisjoint(joint_support[1])


def test_native_targets_side_and_information_boundary(panels):
    high = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(bool)
    for games in panels:
        for key, raw in games.items():
            p, q, y, s, w = raw
            assert [a.shape for a in raw] == [(8, 12), (6, 12), (8, 6), (8, 6), (6,)]
            np.testing.assert_array_equal(p[:, :3] > 0.5, high)
            np.testing.assert_array_equal(p[:, 3:], np.repeat(p[:1, 3:], 8, axis=0))
            np.testing.assert_array_equal(q[:, 6:9].sum(0), np.full(3, 2))
            assert (w > 0).all() and w.sum() == pytest.approx(1)
            np.testing.assert_allclose(p[0, 6:9], w @ q[:, 6:9])
            np.testing.assert_allclose(p[0, 3:6].sum(), 1)
            flags = (0, 1, 0) if _status(key) == "SELECTED" else (1, 1, 1)
            np.testing.assert_array_equal(q[:, 9:], np.tile(flags, (6, 1)))
            for column, row in enumerate(q):
                a, b = row[:3].astype(bool), row[3:6].astype(bool)
                # A point rival dominates iff all focused measurements are low.
                # An interval rival abstains iff any focused measurement is high.
                expected = high[:, a].any(1) & high[:, b].any(1)
                np.testing.assert_array_equal(y[:, column], expected)
                np.testing.assert_array_equal(s[:, column], high[:, np.flatnonzero(row[6:9]).item()])
                if "contained_" in key:
                    np.testing.assert_array_equal(y[:, column], high[:, np.flatnonzero(a & b).item()])


def test_both_slots_change_training_outcomes_with_fixed_other_inputs(panels):
    train, _ = panels
    changed_slots = set()
    comparisons = defaultdict(dict)
    for key, (_, queries, labels, _, _) in train.items():
        for i, q in enumerate(queries):
            for slot, (changed, fixed) in enumerate(((q[:3], q[3:6]), (q[3:6], q[:3]))):
                context = (_status(key), slot, tuple(fixed), tuple(q[6:]))
                outcomes = comparisons[context]
                current = tuple(labels[:, i])
                if any(previous != current for mask, previous in outcomes.items() if mask != tuple(changed)):
                    changed_slots.add((_status(key), slot))
                outcomes[tuple(changed)] = current
    assert changed_slots == {("SELECTED", 0), ("SELECTED", 1), ("UNRESOLVED", 0), ("UNRESOLVED", 1)}


def test_replay_and_checked_side_digest(panels):
    train, _ = panels
    replay = make_relations("train", 84031)
    other = make_relations("train", 91019)
    for key, raw in train.items():
        for left, right in zip(raw, replay[key], strict=True):
            np.testing.assert_array_equal(left, right)
        assert not np.array_equal(raw[0][:, :3], other[key][0][:, :3])
        assert not np.array_equal(raw[-1], other[key][-1])
    raw = train["point_selected_disjoint_1_2/0"]
    pinned = "0610a526a48c76e04348e52b5aa3fa9d9e17d8314169f67af58aa1497754d3e5"
    assert verified_side_sha256(raw, pinned) == pinned
    with pytest.raises(ValueError, match="digest mismatch"):
        verified_side_sha256(raw, "0" * 64)
    corrupted = raw[3].copy()
    corrupted[1, 2] ^= 1
    with pytest.raises(ValueError, match="calibration predicate"):
        verified_side_sha256((*raw[:3], corrupted, raw[4]))


def test_v1_training_bytes_preserved():
    digest = hashlib.sha256()
    for key, values in receipt_ldt.make_games("train", 5521).items():
        digest.update(key.encode() + bytes(1))
        for array in values:
            digest.update(np.ascontiguousarray(array).tobytes())
    assert digest.hexdigest() == "eea126c4434d52d9d332c021fe90b9b715e3b1844aeb3b8e28654761da8272ee"


def test_native_policy_runs_three_candidates_per_target(monkeypatch):
    native = receipt_ldt.derive_array_target
    counts = defaultdict(int)

    def traced(candidates, *, tolerance):
        assert len(candidates) == 3
        intervals = [candidates[2][field] for field in receipt_ldt.METRIC_FIELDS]
        uncertain = any(metric["upper"] > metric["lower"] for metric in intervals)
        counts[uncertain] += 1
        return native(candidates, tolerance=tolerance)

    monkeypatch.setattr(receipt_ldt, "derive_array_target", traced)
    make_relations("train", 83317)
    assert counts == {False: 12 * 6 * 8, True: 12 * 6 * 8}
