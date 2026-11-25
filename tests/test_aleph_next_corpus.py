"""Cross-check the four-bit envelope, replay custody and task-specific routing."""

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training" / p) for p in ("aleph_next", "aleph_rlt_placement")]

from continual_lab import Student, four_oracle, game4  # noqa: E402
from joint_query import table  # noqa: E402
from receipt_models import ReceiptModel  # noqa: E402
from research_stream import pack  # noqa: E402


def test_four_bit_oracle_matches_world_enumeration():
    low, high, lower_mask, first = four_oracle()
    for prior in (0, 16 / 667, 1):
        _, _, y, side, weights = game4(prior)
        for mask in (0, first, lower_mask, 0x399A, 0xF0A0):
            code = np.array([mask >> i & 1 for i in range(16)])
            actual = 0
            for s in (0, 1):
                for m in (0, 1):
                    selected = (side == s) & (code[:, None] == m)
                    total = selected.sum(0)
                    yes = (y * selected).sum(0)
                    actual += ((yes * (total - yes) / np.maximum(total, 1)) * weights).sum() / 16
            assert actual == pytest.approx((1 - prior) * low[mask // 2] + prior * high[mask // 2])


def test_route_keeps_prior_bit_and_both_old_readouts():
    torch.manual_seed(19)
    parent = ReceiptModel("C")
    model = Student("route", parent.state_dict())
    x = torch.randn(6, 12)
    side = torch.tensor([0, 1, 0, 1, 0, 1])
    for task in ("p3", "ldt", "p4"):
        model.select(task)
        assert torch.equal(parent.encode(x), model.encode(x))
    model.select("ldt")
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model.encode(x).sum().backward()
    opt.step()
    assert not torch.equal(parent.encode(x), model.encode(x))
    for task in ("p3", "p4"):
        model.select(task)
        assert torch.equal(parent.encode(x), model.encode(x))
        assert torch.equal(parent.decode(x, side, side.flip(0)), model.decode(x, side, side.flip(0)))


def test_corpus_admission_hashes_and_excludes_held(tmp_path):
    row = dict(family="ldt-train-a", split="train", prompt="Audit 1/2. ", answer="1/3", receiver_absolute_error=1 / 6)
    out = tmp_path / "pack"
    report = pack([row, row], out)
    assert report["rows"] == 1 and report["packed_targets"] == 4
    line = json.loads((out / "train.jsonl").read_bytes())
    assert line["receipt_sha256"] == hashlib.sha256((out / "audit-receipts.jsonl").read_bytes().rstrip()).hexdigest()
    with pytest.raises(ValueError, match="provenance"):
        pack([{**row, "split": "test"}], tmp_path / "excluded")


def test_joint_query_regret_and_deployment_switch():
    below = table(Fraction(1, 4))
    assert below["optimum_two"] == "AB" and below["greedy_regret"] == "1/8"
    assert table(Fraction(1, 2))["greedy_regret"] == "0"
    above = table(Fraction(3, 4))
    assert "C" in above["optimum_two"] and above["greedy_regret"] == "0"
