"""Outcome-only parity control with matched sender exposure and annealing."""

import numpy as np
import torch
from receipt_gym import TRAIN_PRIORS, Game, parity


def fit_matched_outcome(model, seed, steps=1200):
    games = [Game(parity(p)) for p in TRAIN_PRIORS]
    features = torch.cat([g.p for g in games])
    weights = torch.stack([g.w for g in games])
    # The eight priors share worlds, queries, labels and verifier outcomes.
    common = games[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    rng = np.random.default_rng(seed)
    trace = []
    for step in range(steps):
        if step % len(games) == 0:
            order = rng.permutation(len(games))
        i = int(order[step % len(games)])
        prob = model.encode(features).reshape(len(games), 8).sigmoid()
        event = torch.stack((1 - prob, prob), dim=1)[:, :, None, :, None] * common.side_mask / 8
        mass = event.sum(3)
        yes = (event * common.y).sum(3)
        ideal = (((mass - yes) * yes / mass.clamp_min(1e-15)) * weights[:, None, None]).sum((1, 2, 3)).mean()
        realized = games[i].risks(prob[i], games[i].guesses(model))[0]
        loss = (realized + ideal) / 2
        fraction = step / max(steps - 1, 1)
        optimizer.param_groups[0]["lr"] = 0.003 * (1 - 0.95 * max(0.0, 2 * fraction - 1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            trace.append([step + 1, float(realized.detach()), float(ideal.detach())])
    return trace
