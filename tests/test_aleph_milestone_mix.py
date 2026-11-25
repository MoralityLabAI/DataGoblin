"""Regression checks for the frozen 10M stream builder and replay."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/aleph_next/candidate01"))
from milestone_mix import ReferenceLane, audit_pack, require_train


def test_pack_hash_binds_lanes_and_query_alignment():
    p = {
        "input_ids": [2, 3, 5, 7] + [0] * 2044,
        "segment_ids": [0, 0, 1, 1] + [-1] * 2044,
        "position_ids": [0, 1, 0, 1] + [0] * 2044,
        "labels_at_query": [3, -100, 7, -100] + [-100] * 2044,
        "nonpadding": 4,
    }
    a, b = hashlib.sha256(), hashlib.sha256()
    assert audit_pack(p, "verified", a) == (4, 2)
    audit_pack(p, "procedural", b)
    assert a.digest() != b.digest()
    p["labels_at_query"][1] = 5
    with pytest.raises(ValueError, match="crosses document"):
        audit_pack(p, "verified", a)


def test_one_admitted_view_and_family_custody():
    row = {
        "lane": "composition",
        "family": "composition/train-tree",
        "root": "public-test",
        "presentation": 1,
        "split": "train",
        "objective": "checked_response",
        "target_mask": [False, False, True],
        "response_start": 2,
        "receipt_sha256": "a" * 64,
    }
    mapping = {"composition": "verified"}
    require_train(row, "verified", mapping, set())
    for change, forbidden in [
        ({"split": "final"}, set()),
        ({"presentation": 2}, set()),
        ({"family": "composition/sealed"}, {"sealed"}),
        ({"receipt_sha256": None}, set()),
        ({"target_mask": [False, True, True]}, set()),
        ({}, {"public-test"}),
    ]:
        with pytest.raises(ValueError):
            require_train({**row, **change}, "verified", mapping, forbidden)


def test_independent_reader_refuses_prompt_label(tmp_path):
    from contextlib import ExitStack

    row = {"family": "demo", "token_ids": [7, 8, 9, 10], "target_mask": [False, False, True, True]}
    (tmp_path / "data").write_text(json.dumps(row) + "\n", encoding="utf8")
    (tmp_path / "manifest.json").write_text('{"shards":[{"path":"data"}]}', encoding="utf8")
    pack = {"input_ids": row["token_ids"], "position_ids": [0, 1, 2, 3], "nonpadding": 4}
    with ExitStack() as stack:
        ref = ReferenceLane(tmp_path, "verified", stack)
        # This first label is the correct next ID, but its query still belongs to the masked prompt.
        with pytest.raises(ValueError, match="supervision differ"):
            ref.verify({**pack, "labels_at_query": [8, 9, 10, -100]})
    with ExitStack() as stack:
        ref = ReferenceLane(tmp_path, "verified", stack)
        ref.verify({**pack, "labels_at_query": [-100, 9, 10, -100]})
        assert ref.family_labels == {"demo": 2}
