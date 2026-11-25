"""Matched continuation streams, optimizer recovery and A-only horizon selection."""

import sys
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "aleph_next"))
import anchor14_train as anchor  # noqa: E402
from corpus import REPAIR, SEP, Row  # noqa: E402
from study import choose  # noqa: E402


@pytest.fixture
def inputs():
    train = {}
    for lane, choices in (("text", 256), ("lean", 2), ("rule", 32), ("counter", 33)):
        train[lane] = [
            Row(lane, f"{lane}{i}", (48 + i, 16), (33, 17) if lane == "text" else (i % choices,), choices)
            for i in range(4)
        ]
    train["lean_pos"] = [r for r in train["lean"] if r.good == (1,)]
    train["lean_neg"] = [r for r in train["lean"] if r.good == (0,)]
    buckets = {}
    for prev in range(4):
        for op in (16, 17):
            for arg in range(4):
                key = (prev, op, arg)
                target = arg if op == 16 else (prev + arg) % 4
                row = Row("repair", str(key), (REPAIR, 1, 2, SEP, *key, SEP), (target,), 4)
                buckets[key] = [row]
    train["repair"] = [rows[0] for rows in buckets.values()]
    return train, buckets


def test_native_packet_preserved_and_balanced_incoming_train_curriculum(inputs):
    train, buckets = inputs
    seq = [anchor.logical_rows(train, buckets, 137, step) for step in range(8)]
    seen = []
    for step, rows in enumerate(seq):
        for tag, r in rows:
            if tag == "suffix":
                seen.append(r.prompt[2:5])
        assert Counter(r.good for k, r in rows if k == "compact") == {(x,): 2 for x in range(4)}
        assert anchor.input_digest(rows) == anchor.input_digest(anchor.logical_rows(train, buckets, 137, step))
        assert all(len(g) <= 8 and max(len(r.prompt) for _, r in g) * len(g) <= 192 for g in anchor.microbatches(rows))
    assert set(seen) == set(buckets) and len(seen) == 32
    for block in range(2):
        rows = [r for packet in seq[4 * block : 4 * block + 4] for _, r in packet]
        assert not (Counter(choose(train, 137, 32 + block, 1)) - Counter(rows))


def test_atomic_optimizer_continuation(inputs, tmp_path):
    class Mini(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(512, 8)
            self.head = torch.nn.Linear(8, 512)

        def forward(self, x, arm, passes, reader):
            return self.head(self.embed(x))

    torch.set_num_threads(1)
    torch.manual_seed(19)
    net = Mini()
    opt = torch.optim.AdamW(net.parameters(), lr=0.00035, weight_decay=0.01, foreach=False)
    train, buckets = inputs
    first, second = [anchor.logical_rows(train, buckets, 137, j) for j in (0, 1)]
    anchor.earn_step(net, opt, first, "direct", 0)
    ck = tmp_path / "latest.pt"
    anchor.save_tensor(ck, anchor.latest_payload({}, {"step": 1}, net, opt, {}, []))
    ref = anchor.earn_step(net, opt, second, "direct", 1)
    payload = torch.load(ck, weights_only=True)
    other = Mini()
    other.load_state_dict(payload["model"])
    reset = torch.optim.AdamW(other.parameters(), lr=0.1, foreach=False)
    reset.load_state_dict(payload["optimizer"])
    assert anchor.earn_step(other, reset, second, "direct", 1) == ref
    assert all(torch.equal(x, y) for x, y in zip(net.parameters(), other.parameters(), strict=True))


def test_selection_requires_three_paired_sources_and_rejects_b_development(tmp_path, monkeypatch):
    monkeypatch.setattr(anchor, "recipe", lambda: "pin")
    for h in (64, 128):
        for arm in anchor.ARMS:
            for seed in anchor.SEEDS:
                folder = tmp_path / "cells" / f"{arm}-{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                r = dict(
                    step=h,
                    stream=str(seed),
                    tokens=100,
                    labels=27,
                    padded=111,
                    prefix_stream="x",
                    prefix_tokens=1,
                    prefix_padded=2,
                    params=anchor.PARAMS,
                    visits=24,
                    roots_sha256="r",
                    recipe="pin",
                )
                if arm == "direct":
                    r["development"] = {"competent": h == 128 and seed != 163}
                anchor.atomic_json(folder / f"h{h:04d}.json", r)
    assert anchor.select(tmp_path)["horizon"] == 128
    b = tmp_path / "cells/global_full_damp-137/h0064.json"
    data = anchor.read(b)
    data["development"] = {"competent": True}
    anchor.atomic_json(b, data)
    with pytest.raises(ValueError, match="leaked"):
        anchor.select(tmp_path)
    data.pop("development")
    data["labels"] += 1
    anchor.atomic_json(b, data)
    with pytest.raises(ValueError, match="unmatched"):
        anchor.select(tmp_path)
