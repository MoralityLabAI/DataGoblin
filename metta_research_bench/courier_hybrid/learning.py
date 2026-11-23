"""CPU LoopedPointer fitting with matched frames and finite budget-aware losses."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from ..looped.model import BUCKETS, FIELDS, LEARNING_RATE, LOOPS, LoopedPointer, weight_hash
from .controller import Observation, Proposal
from .native import ATTEST_FIRST, ATTEST_SECOND, COLLECT, DELIVER, ISSUE, PLACES, STOP
from .oracle import OBJECTIVES, continuations, objective_weights, state_graph

POSITION_ROWS = (*range(10), 7, 8, 10)


def tokenize(obs: Observation) -> tuple[Tensor, Tensor]:
    world, state = obs.world, obs.state
    rows = torch.zeros((STOP + 1, FIELDS), dtype=torch.float32)
    for site in range(PLACES):
        rows[site, :9] = torch.tensor(
            (
                1,
                site == state.place,
                site in world.offices,
                site in world.depots,
                site == world.desk,
                site in world.sources[0],
                site in world.sources[1],
                site / 6,
                state.permit,
            )
        )
    for source, destination, gate in world.arcs:
        rows[source, 9 + destination + PLACES * gate] = 1
    for a in range(PLACES, STOP + 1):
        rows[a, :11] = torch.tensor(
            (
                2,
                state.place / 6,
                state.permit,
                state.cargo / 2,
                bool(state.evidence & 1),
                bool(state.evidence & 2),
                obs.moves_left / 12,
                obs.remaining_calls / 13,
                (a - PLACES) / 5,
                state.place in world.sources[0],
                state.place in world.sources[1],
            )
        )
    codes = ((rows + 2) * 8).round().long() + torch.arange(FIELDS) * BUCKETS
    allowed = torch.zeros(STOP + 1, dtype=torch.bool)
    allowed[list(obs.legal)] = True
    return codes, allowed


class CourierPointer(LoopedPointer):
    """Reuse the 44,417 weights and share two action-position embeddings."""

    def __init__(self, obligation_aware: bool = False):
        super().__init__()
        self.obligation_aware = obligation_aware

    def encoded(self, tokens: Tensor) -> Tensor:
        return self.core.input_norm(self.core.embedding(tokens).mean(dim=2)) + self.positions[list(POSITION_ROWS)]

    def visit(self, hidden: Tensor, encoded: Tensor, index: int) -> Tensor:
        if self.obligation_aware and index > 0:
            slots = (ISSUE, COLLECT, DELIVER, ATTEST_FIRST, ATTEST_SECOND) if index == 1 else (STOP,)
            injection = torch.zeros_like(encoded)
            injection[:, slots] = 0.25 * encoded[:, slots]
            hidden = hidden + injection
        return self.core.block_at(index)(hidden)

    def readout(self, hidden: Tensor) -> Tensor:
        return self.action(self.core.final_norm(hidden)).squeeze(-1)

    def unmasked(self, tokens: Tensor) -> Tensor:
        encoded = hidden = self.encoded(tokens)
        for index in range(LOOPS):
            hidden = self.visit(hidden, encoded, index)
        return self.readout(hidden)

    def forward(self, tokens: Tensor, allowed: Tensor) -> Tensor:
        return self.unmasked(tokens).masked_fill(~allowed, -torch.inf)

    @torch.inference_mode()
    def propose(self, obs: Observation) -> Proposal:
        code, mask = tokenize(obs)
        raw = self.unmasked(code[None])[0]
        masked = raw.masked_fill(~mask, -torch.inf)
        return Proposal(int(masked.argmax()), float(masked.softmax(dim=-1)[STOP]), int(raw.argmax()), LOOPS)


def weighted_set_nll(logits: Tensor, allowed: Tensor, weights: Tensor) -> Tensor:
    """Negative log mass of feasible actions, with optional unnormalized [0, 1] weights."""
    masked = logits.masked_fill(~allowed, -torch.inf)
    targets = masked + weights.clamp_min(torch.finfo(weights.dtype).tiny).log()
    targets = targets.masked_fill(weights == 0, -torch.inf)
    return (masked.logsumexp(dim=-1) - targets.logsumexp(dim=-1)).mean()


@dataclass
class Training:
    tokens: Tensor
    masks: Tensor
    targets: dict[str, Tensor]
    frames: list[dict]
    receipts: list[dict]
    completion: Tensor


def training_data(native, sources, horizon: int, ceiling: int) -> Training:
    tokens, masks, frames, receipts = [], [], [], []
    targets = {objective: [] for objective in OBJECTIVES}
    for root, world in sources:
        graph, distances = state_graph(world, native.checked_successors)
        seen = set()
        for reverse in (False, True):
            route = []
            state = world.initial
            while True:
                moves, calls = horizon - len(route), ceiling - len(route)
                outcomes = continuations(state, graph, distances, moves, calls)
                weights = objective_weights(world, state, outcomes)
                if state not in seen:
                    code, mask = tokenize(Observation(world, state, (*graph[state], STOP), moves, calls))
                    tokens.append(code)
                    masks.append(mask)
                    for name, weight in weights.items():
                        targets[name].append([weight.get(a, 0.0) for a in range(STOP + 1)])
                    frames.append(
                        {
                            "root": root.world_sha256,
                            "state": asdict(state),
                            "moves_left": moves,
                            "calls_left": calls,
                            "legal": sorted([*graph[state], STOP]),
                            "weights": weights,
                            "complete": state.complete,
                        }
                    )
                    seen.add(state)
                action = sorted(weights["shortest"], reverse=reverse)[0]
                if action == STOP:
                    break
                route.append(action)
                state = graph[state][action]
            session = native.session(world, 2, horizon)
            if not session.certify(tuple(route), macro=True).accepted or not session.stop().accepted:
                raise RuntimeError("Native training trajectory failed certification")
            receipts.append({"root": root.world_sha256, "receipts": [asdict(r) for r in session.receipts]})
    return Training(
        torch.stack(tokens),
        torch.stack(masks),
        {k: torch.tensor(v) for k, v in targets.items()},
        frames,
        receipts,
        torch.tensor([row["complete"] for row in frames], dtype=torch.float32),
    )


def fit(
    initial: CourierPointer, data: Training, objective: str, seed: int, updates: int, batch_size: int
) -> tuple[CourierPointer, dict]:
    model = copy.deepcopy(initial)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    weights = data.targets[objective]
    if (
        (weights < 0).any()
        or (weights > 1).any()
        or (weights[~data.masks] != 0).any()
        or (weights.sum(dim=-1) == 0).any()
        or not torch.isfinite(weights).all()
    ):
        raise ValueError("Expected nonempty, bounded weights over legal actions")
    losses = []
    for _ in range(updates):
        batch = torch.randint(len(data.tokens), (batch_size,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.tokens[batch], data.masks[batch])
        loss = weighted_set_nll(logits, data.masks[batch], weights[batch])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 44417:
        raise RuntimeError("Courier model parameter matching failed")
    return model.eval(), {
        "seed": seed,
        "objective": objective,
        "parameters": parameters,
        "initial_sha256": weight_hash(initial),
        "checkpoint_weights_sha256": weight_hash(model),
        "updates": updates,
        "examples_per_update": batch_size,
        "draws": updates * batch_size,
        "first_loss": losses[0],
        "last_ten_loss": sum(losses[-10:]) / len(losses[-10:]),
    }


def fit_stop_gate(model: CourierPointer, data: Training) -> tuple[float, float]:
    """Fit a one-feature calibration of STOP confidence using native-labelled train frames."""
    with torch.no_grad():
        confidence = torch.cat(
            [
                model(code, mask).softmax(dim=-1)[:, STOP]
                for code, mask in zip(data.tokens.split(128), data.masks.split(128), strict=True)
            ]
        )
    features = torch.stack((torch.ones_like(confidence), confidence), dim=-1)
    coefficients = torch.zeros(2, requires_grad=True)
    optimizer = torch.optim.SGD([coefficients], lr=0.5)
    class_weights = torch.where(data.completion > 0, 0.5 / data.completion.mean(), 0.5 / (1 - data.completion.mean()))
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(
            features @ coefficients, data.completion, weight=class_weights
        )
        (loss + 0.01 * coefficients[1].square()).backward()
        optimizer.step()
    intercept, slope = coefficients.detach().tolist()
    return intercept, slope
