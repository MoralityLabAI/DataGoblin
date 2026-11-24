"""Offline implementation audit. Run next to gym.py after the original smoke."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import torch
from gym import ARMS, Aleph, batch, checkpoint, coefficients, digest, schedule
from torch.nn import functional as F


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def step(model, opt, index, arm):
    x, y = batch(f"audit:{index}", 3, "cpu")
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x, arm, 3, 1)[:, -1], y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    return x, y, loss.detach()


def audit():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = Aleph(16)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    x, _ = batch(12, 2, "cpu")
    word = schedule(3, 2)
    called = []
    handles = [b.register_forward_hook(lambda module, _in, _out: called.append(id(module))) for b in model.blocks]
    with torch.no_grad():
        model(x, "pass_rlt", 3, 2)
    for h in handles:
        h.remove()
    assert called == [id(model.blocks[i]) for i in word] and len(set(called)) == 14
    addresses = [p.data_ptr() for p in model.parameters()]
    assert len(set(addresses)) == len(addresses)
    count = sum(p.numel() for p in model.parameters())
    assert count == 64 * 16 + 14 * (15 * 16**2 + 2 * 16) + 16
    with torch.no_grad():
        for arm in ARMS:
            logits = model(x, arm, 3, 2)
            for cut in (1, 7, 19):
                changed = x.clone()
                changed[:, cut:] = (changed[:, cut:] + 17) % 64
                assert torch.equal(logits[:, :cut], model(changed, arm, 3, 2)[:, :cut]), arm
            assert addresses == [p.data_ptr() for p in model.parameters()]
        for r in (1, 2, 3, 4, 6):
            assert torch.equal(model(x, "direct", r), model(x, "terminal", r))
        for arm, damp in (("core_edges", "core_edges_damp"), ("pass_rlt", "pass_damp")):
            assert torch.equal(model(x, arm, 2), model(x, damp, 2))
            assert not torch.equal(model(x, arm, 3), model(x, damp, 3))
        assert not torch.equal(model(x, "pass_rlt", 3), model(x, "core_edges", 3))
        trace = []
        model(x, "pass_rlt", 3, trace=trace)
        for row in trace:
            assert abs(row["anchor_delta_rms"] - row["alpha"] * row["proposed_delta_rms"]) < 1e-5
    twin = Aleph(16)
    twin.load_state_dict(model.state_dict())
    twin_opt = torch.optim.AdamW(twin.parameters(), lr=0.001)
    assert equal(step(model, opt, 0, "direct"), step(twin, twin_opt, 0, "terminal"))
    assert equal(model.state_dict(), twin.state_dict()) and equal(opt.state_dict(), twin_opt.state_dict())
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(model.parameters(), twin.parameters(), strict=True))
    path = Path.cwd() / f".placement-audit-{uuid4().hex}.pt"
    try:
        cfg = {"arm": "pass_rlt", "r": 3}
        step(model, opt, 1, cfg["arm"])
        checkpoint(path, model, opt, 2, cfg)
        expected_rng = torch.rand(4)
        reference = step(model, opt, 2, cfg["arm"])
        resumed = Aleph(16)
        new_opt = torch.optim.AdamW(resumed.parameters())
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["cfg"] == cfg
        resumed.load_state_dict(state["model"])
        new_opt.load_state_dict(state["opt"])
        torch.random.set_rng_state(state["rng"])
        assert torch.equal(expected_rng, torch.rand(4))
        assert equal(reference, step(resumed, new_opt, state["step"], cfg["arm"]))
        assert equal(model.state_dict(), resumed.state_dict()) and equal(opt.state_dict(), new_opt.state_dict())
    finally:
        path.unlink(missing_ok=True)
        path.with_suffix(".tmp").unlink(missing_ok=True)
    return {
        "scope": "CPU float32 exact",
        "torch": str(torch.__version__),
        "unique_blocks": 14,
        "visits_checked": len(word),
        "parameters_at_width16": count,
        "causal_suffix_tests": len(ARMS) * 3,
        "paired_terminal_adam_update": "bitwise_equal",
        "resume_model_moments_batch_next_loss_and_rng": "bitwise_equal",
        "restart_batch_digest": digest(reference[0].tolist()),
        "restart_loss": float(reference[-1]),
        "physical_and_pass_clocks": {a: coefficients(a, 3, 1) for a in ARMS},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    receipt = json.dumps(audit(), indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(receipt + "\n", encoding="utf-8")
    print(receipt)
