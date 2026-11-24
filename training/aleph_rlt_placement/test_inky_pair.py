"""Read-position, sharing, optimizer and segment parity checks for the proposed pair."""

import torch
from gym import Aleph, schedule
from inky_pair import InkyPair
from packing import reference_logits
from torch.nn import functional as F


def test_read_and_train():
    torch.manual_seed(137)
    torch.set_num_threads(1)
    pair = InkyPair(16, 32, heads=2, kv=1, core_ff=70)
    original = Aleph(16, 32, heads=2, kv=1, core_ff=70)
    original.load_state_dict(pair.state_dict())
    ids = torch.tensor([[2, 5, 4, 3, 7, 1]])
    assert torch.equal(pair(ids), original(ids))
    visits = []
    captured = {}

    def watch(index):
        def hook(module, inputs, output):
            visits.append(index)
            if index == 0:
                captured["prelude"] = output
            if index == 5:
                captured["front"] = output
            if index == 6 and visits.count(6) == 2:
                expected = torch.lerp(captured["front"], captured["prelude"], 0.25)
                torch.testing.assert_close(inputs[0], expected)

        return hook

    hooks = [block.register_forward_hook(watch(i)) for i, block in enumerate(pair.blocks)]
    try:
        out = pair(ids, "prelude_reread")
    finally:
        for hook in hooks:
            hook.remove()
    assert visits == schedule(2, 1)
    assert not torch.equal(out, pair(ids))
    opt = torch.optim.AdamW(pair.parameters(), lr=0.001)
    F.cross_entropy(out[0, :-1], ids[0, 1:]).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pair.parameters())
    before = pair.blocks[1].q.weight.detach().clone()
    opt.step()
    assert not torch.equal(before, pair.blocks[1].q.weight)


def test_shape_causality_and_packs():
    with torch.device("meta"):
        full = InkyPair(2048, 49152, heads=16, kv=4, core_ff=8960)
        assert sum(p.numel() for p in full.parameters()) == 999352320
        for arm in ("direct", "prelude_reread"):
            assert full(torch.empty((1, 4), dtype=torch.long), arm).shape == (1, 4, 49152)
    torch.manual_seed(163)
    torch.set_num_threads(1)
    pair = InkyPair(16, 32, heads=2, kv=1, core_ff=70)
    pack = {"input_ids": [2, 5, 4, 3, 7, 1], "nonpadding": 6, "segment_ids": [0, 0, 0, 1, 1, 1]}
    for arm in ("direct", "prelude_reread"):
        earlier = reference_logits(pair, pack, arm)
        edited = reference_logits(pair, dict(pack, input_ids=[2, 5, 0, 3, 7, 1]), arm)
        torch.testing.assert_close(earlier[:, :2], edited[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(earlier[:, 3:], edited[:, 3:], rtol=0, atol=0)
