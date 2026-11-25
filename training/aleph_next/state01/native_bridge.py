"""Native Aleph A state-label adapter and bounded smoke; no checkpoint writes."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from state02 import expand, words

LEGACY = Path(__file__).resolve().parents[2] / 'aleph_rlt_placement'
# This script imports the existing dense recipe when run in the repository.
sys.path.insert(0, str(LEGACY))


def actor_input(ops):
    """Pilot-only token namespace; gold states never enter actor input."""
    starts = torch.arange(4).repeat(len(ops)//4).unsqueeze(1)
    prefix = torch.full((len(ops), 1), 260, dtype=torch.long)
    sep = torch.full_like(prefix, 262)
    return torch.cat([prefix, sep, starts, ops + 32], 1)


def state_logits(model, ops):
    return model(actor_input(ops), 'direct', 2, 1)[:, 3:, :4]


def objective(z, y, arm, shuffle=None):
    end = F.cross_entropy(z[:, -1], y[:, -1])
    if arm == 'terminal':
        return end
    if arm not in ('state', 'shuffled'):
        raise ValueError(arm)
    aux = y[:, :-1]
    if arm == 'shuffled':
        if shuffle is None:
            raise ValueError('explicit independent shuffle RNG required')
        aux = torch.stack([v[torch.randperm(len(v), generator=shuffle)]
                           for v in aux.unbind(1)], 1)
    return end + F.cross_entropy(z[:, :-1].reshape(-1, 4), aux.reshape(-1))


def smoke(model):
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(2026112503)
    ops, y = expand(words(g, 2, 8, 'partial'))
    changed = ops.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 3
    with torch.no_grad():
        z1, z2 = state_logits(model, ops), state_logits(model, changed)
        delta = (z1[:, :4] - z2[:, :4]).abs().max().item()
        torch.testing.assert_close(z1[:, :4], z2[:, :4], atol=1e-5, rtol=1e-5)
    z = state_logits(model, ops)
    loss = objective(z, y, 'state')
    opt = torch.optim.AdamW(model.parameters(), lr=.00035, weight_decay=.01, foreach=False)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True).item()
    assert grad > 0
    opt.step()
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                positions=actor_input(ops).numel(), final_labels=len(y),
                auxiliary_labels=y[:, :-1].numel(), loss=loss.item(),
                preclip_grad_norm=grad, prefix_max_delta=delta, optimizer_updates=1)


if __name__ == '__main__':
    from study import dense
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=256)
    args = parser.parse_args()
    torch.manual_seed(137)
    model = dense(args.width)
    if args.width == 256:
        assert sum(p.numel() for p in model.parameters()) == 14_179_584
    print(json.dumps(smoke(model), indent=2))
