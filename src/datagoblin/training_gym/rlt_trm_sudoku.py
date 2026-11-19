"""Attention Sudoku TRM inference with matched carry-state schedules."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .rlt_hrm_sudoku import report, slow_coefficients
from .rlt_hrm_sudoku_weights import read_layout

H_CYCLES, L_CYCLES, SEGMENTS = 3, 6, 16
HIDDEN, HEADS, PUZZLE_PREFIX = 512, 8, 16
ARMS = ("direct", "answer_rlt", "answer_damping", "reasoning_rlt", "reasoning_damping")


def feedback(arm: str):
    if arm not in ARMS:
        raise ValueError("unknown TRM feedback")
    answer = "direct"
    reasoning = "direct"
    if arm.startswith("answer"):
        answer = "global_rlt" if arm.endswith("_rlt") else "global_damping"
    if arm.startswith("reasoning"):
        reasoning = "global_rlt" if arm.endswith("_rlt") else "global_damping"
    return slow_coefficients(answer, 1, H_CYCLES), slow_coefficients(reasoning, 1, L_CYCLES)


def checkpoint_weights(checkpoint, torch):
    weights = {}
    for name, (layout, offset) in read_layout(checkpoint).items():
        is_float32 = layout.dtype == "FloatStorage"
        raw = np.memmap(checkpoint, mode="c", dtype="<f4" if is_float32 else "<u2",
                        offset=offset, shape=layout.shape)
        tensor = torch.from_numpy(raw)
        weights[name.removeprefix("_orig_mod.model.inner.")] = (
            tensor.to(torch.bfloat16) if is_float32 else tensor.view(torch.bfloat16)
        )
    return weights


def mix(state, endpoint, coefficient):
    if coefficient == 1:
        return endpoint
    state_fp32 = state.float()
    return (state_fp32 + coefficient * (endpoint.float() - state_fp32)).to(endpoint.dtype)


class SudokuTRM:
    def __init__(self, weights, torch):
        self.weights = weights
        self.torch = torch
        self.fn = torch.nn.functional
        self.dim = HIDDEN // HEADS
        self.sequence = 81 + PUZZLE_PREFIX
        indices = torch.arange(0, self.dim, 2, dtype=torch.float32)
        frequencies = 1 / (10000 ** (indices / self.dim))
        angles = torch.outer(torch.arange(self.sequence, dtype=torch.float32), frequencies).repeat(1, 2)
        self.cos, self.sin = angles.cos()[:, None], angles.sin()[:, None]
        self.layers = [
            tuple(weights[f"L_level.layers.{i}.{part}.weight"] for part in (
                "self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj",
            ))
            for i in range(2)
        ]

    def rotate(self, x):
        fp32 = x.float()
        lo, hi = fp32.chunk(2, dim=-1)
        return (fp32 * self.cos + self.torch.cat((-hi, lo), dim=-1) * self.sin).to(x.dtype)

    def normalize(self, x):
        fp32 = x.float()
        scale = self.torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + 1e-5)
        return (fp32 * scale).to(x.dtype)

    def attention(self, x, qkv, output):
        packed = self.fn.linear(x, qkv).reshape(self.sequence, 3 * HEADS, self.dim)
        q, k, v = packed.chunk(3, dim=1)
        attention = self.fn.scaled_dot_product_attention(
            self.rotate(q).transpose(0, 1), self.rotate(k).transpose(0, 1), v.transpose(0, 1),
        )
        return self.fn.linear(attention.transpose(0, 1).reshape(self.sequence, HIDDEN), output)

    def stack(self, x, injection):
        x = x + injection
        for qkv, projection, gate_up, down in self.layers:
            x = self.normalize(x + self.attention(x, qkv, projection))
            gate, up = self.fn.linear(x, gate_up).chunk(2, dim=-1)
            x = self.normalize(x + self.fn.linear(self.fn.silu(gate) * up, down))
        return x

    def rollout(self, givens, arm, guard):
        w = self.weights
        if len(givens) != 81:
            raise ValueError("expected 81 given cells")
        torch = self.torch
        sequence = torch.tensor([int(c) + 1 for c in givens], dtype=torch.int32)
        prefix = torch.zeros((PUZZLE_PREFIX, HIDDEN), dtype=torch.bfloat16)
        prefix[0] = w["puzzle_emb.weights"][0]
        inputs = math.sqrt(HIDDEN) * torch.cat((prefix, w["embed_tokens.embedding_weight"][sequence]))
        answer, latent = w["H_init"], w["L_init"]
        eta_answers, eta_reasoning = feedback(arm)
        for segment in range(1, SEGMENTS + 1):
            for eta_answer in eta_answers:
                guard()
                injection = answer + inputs
                for eta_latent in eta_reasoning:
                    latent = mix(latent, self.stack(latent, injection), eta_latent)
                answer = mix(answer, self.stack(answer, latent), eta_answer)
            digits = self.fn.linear(answer[PUZZLE_PREFIX:], w["lm_head.weight"]).argmax(-1).tolist()
            yield {
                "segment": segment, "prediction": "".join(str(max(0, digit - 1)) for digit in digits),
                "slow_rms": answer.float().square().mean().sqrt().item(),
                "reasoning_rms": latent.float().square().mean().sqrt().item(),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream-ldt", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.out, args.upstream_ldt, arms=ARMS, model="trm")


if __name__ == "__main__":
    main()
