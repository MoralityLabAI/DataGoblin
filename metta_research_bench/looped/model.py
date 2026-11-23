"""CPU pointer policy built on the upstream weight-tied LoopedDecoderLM."""
from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch
from research_gym.neural.looped_decoder import LoopedDecoderLM
from torch import Tensor, nn

from MeTTa.airis02_student import MAX_ITEMS
from MeTTa.ram_trajectory import STOP, RepairVerifier, typed_proposal
from MeTTa.ram_trajectory_policy import features, slot
from MeTTa.sft_oracle import edit, first_state, make_task

from ..taskset import MAX_DECISIONS

UPSTREAM_COMMIT = "950d4d594d87a1f74c6139a928a9b6cb4ab70617"
HIDDEN = 32
HEADS = 4
LOOPS = 3
FIELDS = 24
BUCKETS = 41
BATCH_SIZE = 48
MAX_PROPOSALS = 4
LEARNING_RATE = 0.0015
TOKENIZER_ID = "public-slot-feature-buckets-v1"


def upstream_revision() -> str:
    import inspect

    repo = Path(inspect.getfile(LoopedDecoderLM)).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    changes = subprocess.check_output(["git", "diff", "HEAD", "--", "research_gym", "lsa"], cwd=repo)
    if commit != UPSTREAM_COMMIT or changes:
        raise RuntimeError("Use the pinned upstream LoopedTransformers checkout without edits")
    return commit


class LoopedPointer(nn.Module):
    """One upstream decoder block visited three times, with a shared slot readout."""

    def __init__(self) -> None:
        super().__init__()
        self.core = LoopedDecoderLM(
            vocab_size=FIELDS * BUCKETS, hidden_size=HIDDEN, num_heads=HEADS, physical_modules=1,
            expanded_visits=LOOPS, tying="fully_tied", parameter_visits=frozenset(range(LOOPS)),
            retained_state_edges=frozenset(range(LOOPS - 1)), supervision_points=(LOOPS - 1,),
            alpha=1.0, beta=1.0, normalization="pre", carry_policy="reset",
        )
        self.positions = nn.Parameter(torch.randn(MAX_ITEMS + 1, HIDDEN) * 0.02)
        self.action = nn.Linear(HIDDEN, 1)

    def forward(self, tokens: Tensor, allowed: Tensor) -> Tensor:
        state = self.core.input_norm(self.core.embedding(tokens).mean(dim=2)) + self.positions
        hidden = self.core.states(state)[-1]
        return self.action(self.core.final_norm(hidden)).squeeze(-1).masked_fill(~allowed, -torch.inf)


def tokenize(world: dict, selected: list[str]) -> tuple[Tensor, Tensor]:
    items, stop = features(world, selected)
    stop_row = np.pad(stop, (0, FIELDS - len(stop)))
    slots = np.vstack((items[:, :FIELDS], stop_row))
    buckets = np.rint((np.clip(slots, -2, 3) + 2) * 8).astype(np.int64)
    codes = buckets + np.arange(FIELDS) * BUCKETS
    allowed = np.r_[np.arange(MAX_ITEMS) < len(world["items"]), True]
    return torch.from_numpy(codes), torch.from_numpy(allowed)


def batch_examples(trajectories: list[tuple]) -> tuple[Tensor, Tensor, Tensor]:
    tokens, masks, labels = [], [], []
    for world, selected, actions in trajectories:
        for action in actions:
            code, mask = tokenize(world, selected)
            tokens.append(code)
            masks.append(mask)
            labels.append(MAX_ITEMS if action == STOP else slot(action))
            selected = edit(selected, action)
    return torch.stack(tokens), torch.stack(masks), torch.as_tensor(labels)


def weight_hash(model: LoopedPointer) -> str:
    result = hashlib.sha256()
    for key, value in model.state_dict().items():
        result.update(key.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def fit(initial: LoopedPointer, batch: tuple, seed: int, updates: int) -> tuple[LoopedPointer, dict]:
    model = copy.deepcopy(initial)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    tokens, masks, targets = batch
    distributions = targets.ndim == 2
    if distributions and (targets.shape != masks.shape or not torch.isfinite(targets).all()
                          or (targets < 0).any() or (targets[~masks] != 0).any()
                          or not torch.allclose(targets.sum(dim=-1), torch.ones_like(targets[:, 0]))):
        raise ValueError("Expected one probability distribution over legal actions per training row")
    losses = []
    for _ in range(updates):
        index = torch.randint(len(tokens), size=(BATCH_SIZE,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(tokens[index], masks[index])
        if distributions:
            # Keep illegal logits out of the normalizer without multiplying zero mass by negative infinity.
            log_probs = prediction.log_softmax(dim=-1).masked_fill(~masks[index], 0)
            loss = -(targets[index] * log_probs).sum(dim=-1).mean()
        else:
            loss = nn.functional.cross_entropy(prediction, targets[index])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return model.eval(), {
        "seed": seed, "initial_sha256": weight_hash(initial), "checkpoint_sha256": weight_hash(model),
        "supervised_examples": len(tokens), "supervised_action_tokens": updates * BATCH_SIZE,
        "optimizer_updates": updates, "parameters": sum(p.numel() for p in model.parameters()),
        "first_loss": losses[0], "final_loss_average": sum(losses[-10:]) / len(losses[-10:]),
    }


@torch.inference_mode()
def solve(model: LoopedPointer, data, *, stop_offset: float = 0.0, stop_guard=None,
          audit: list | None = None) -> tuple[list[dict], dict]:
    world = make_task(data.family, data.source_seed)
    selected = first_state(world, data.source_seed)
    verifier = RepairVerifier(world)
    visited = {tuple(selected)}
    actions = []
    iterations = vetoes = 0
    extra_calls = interventions = rank_probes = native_queries = 0

    def totals():
        metrics = {"policy_verifier_calls": verifier.checks, "passes": iterations, "vetoes": vetoes}
        if stop_guard is not None:
            metrics.update(witness_verifier_calls=extra_calls, witness_interventions=interventions,
                           witness_rank_probes=rank_probes, witness_native_queries=native_queries)
        return metrics

    for step in range(MAX_DECISIONS):
        code, allowed = tokenize(world, selected)
        logits = model(code[None], allowed[None])[0]
        if stop_offset:
            logits[MAX_ITEMS] += stop_offset
        iterations += LOOPS
        accepted = False
        attempts = 0
        attempted = set()
        for index in logits.argsort(descending=True):
            j = int(index)
            if not allowed[j]:
                continue
            if j == MAX_ITEMS:
                if stop_guard is not None and step < MAX_DECISIONS - 1:
                    result = stop_guard(world, selected, visited, attempted, MAX_PROPOSALS - attempts,
                                        extra_calls, verifier.refine)
                    extra_calls += len(result["checks"])
                    vetoes += sum(not row["receipt"]["accepted"] for row in result["checks"])
                    rank_probes += result["rank_probes"]
                    native_queries += int(result["native"] is not None)
                    if audit is not None:
                        audit.append({"step": step, "before": selected, "witness": result})
                    if result["chosen"] is not None:
                        choice = result["checks"][result["chosen"]]
                        actions.append(choice["action"])
                        selected = choice["receipt"]["selected_after"]
                        visited.add(tuple(selected))
                        interventions += 1
                        accepted = True
                        break
                actions.append(STOP)
                return actions, totals()
            key = world["items"][j]["id"]
            action = {"op": "remove" if key in selected else "add", "id": key}
            if tuple(edit(selected, action)) in visited:
                continue
            if attempts == MAX_PROPOSALS:
                break
            attempts += 1
            attempted.add(key)
            result = verifier.refine(typed_proposal(action), selected)
            if audit is not None:
                audit.append({"step": step, "before": selected, "action": action, "receipt": result})
            if result["accepted"]:
                selected = result["selected_after"]
                visited.add(tuple(selected))
                actions.append(action)
                accepted = True
                break
            vetoes += 1
        if not accepted:
            break
    return actions, totals()
