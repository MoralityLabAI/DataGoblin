"""Stateless one-bit receipt actors for a matched two-slot architecture comparison."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ReceiptModel(nn.Module):
    """Score a one-bit upload or a binary downstream target from actor-local inputs.

    C gives both candidate options the observations and compares their final scores.
    D assigns the 12 features to evidence and the supplied bits to a proposal.
    Evidence evolves independently and the proposal reads it on both shared visits.
    The ordinary MLP concatenates the same available observations and actor type.
    """

    def __init__(self, arm: str):
        super().__init__()
        if arm not in {"C", "D", "MLP"}:
            raise ValueError("arm must be C, D, or MLP")
        self.arm = arm
        width = 32
        if arm == "MLP":
            self.network = nn.Sequential(
                nn.Linear(15, width),
                nn.SiLU(),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, 1),
            )
            return

        self.embed = nn.Linear(16, width)
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(2 * width, 2 * width)
        self.down = nn.Linear(2 * width, width)
        self.score = nn.Linear(width, 1, bias=False)

        # Fixed wiring stays with the arm when matched parameters are copied.
        visible = torch.ones(2, 15)
        if arm == "D":
            visible[0, 12:14] = 0
            visible[1, :12] = 0
        self.register_buffer("visible", visible, persistent=False)
        self.register_buffer("role", torch.tensor([-1.0, 1.0]), persistent=False)
        peers = [1, 0] if arm == "C" else [0, 0]
        self.register_buffer("peers", torch.tensor(peers), persistent=False)
        readout = [-1.0, 1.0] if arm == "C" else [0.0, 1.0]
        self.register_buffer("readout", torch.tensor(readout), persistent=False)

    def _logit(self, features: Tensor, side: Tensor, message: Tensor, actor: float) -> Tensor:
        actor_tag = side.new_full(side.shape, actor)
        metadata = torch.stack((side, message, actor_tag), dim=1)
        observation = torch.cat((features, metadata), dim=1)
        if self.arm == "MLP":
            return self.network(observation).squeeze(-1)

        slots = observation.unsqueeze(1) * self.visible
        roles = self.role.expand(features.shape[0], -1).unsqueeze(-1)
        hidden = self.embed(torch.cat((slots, roles), dim=-1))
        for _ in range(2):
            local = self.norm(hidden)
            other = local.index_select(1, self.peers)
            delta = self.down(F.silu(self.up(torch.cat((local, other), dim=-1))))
            hidden = hidden + 0.5 * delta
        scores = self.score(hidden).squeeze(-1)
        return (scores * self.readout).sum(dim=1)

    def _features(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != 12:
            raise ValueError("features must have shape [N, 12]")
        weight = self.network[0].weight if self.arm == "MLP" else self.embed.weight
        return features.to(dtype=weight.dtype)

    def encode(self, features: Tensor) -> Tensor:
        """Return upload logits from producer features, before any receiver query."""
        features = self._features(features)
        missing = features.new_zeros(features.shape[0])
        return self._logit(features, missing, missing, -1.0)

    def decode(self, features: Tensor, side: Tensor, message: Tensor) -> Tensor:
        """Return target logits from receiver features and two actual binary values."""
        features = self._features(features)
        if side.shape != (features.shape[0],) or message.shape != side.shape:
            raise ValueError("side and message must each have shape [N]")
        if not (((side == 0) | (side == 1)).all() & ((message == 0) | (message == 1)).all()):
            raise ValueError("side and message must be hard bits in {0, 1}")
        side = side.to(features) * 2 - 1
        message = message.to(features) * 2 - 1
        return self._logit(features, side, message, 1.0)
