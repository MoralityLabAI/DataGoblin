"""NumPy CPU inference for the released Sudoku HRM tensor layout."""

from __future__ import annotations

import math

import numpy as np

from .rlt_hrm_sudoku_weights import read_layout


def bf16(x):
    """Round fp32 to bfloat16 (nearest even), retaining fp32 storage for NumPy."""
    bits = x.view(np.uint32)
    rounded = (bits + (0x7FFF + ((bits >> 16) & 1))) & 0xFFFF0000
    return rounded.view(np.float32)


def checkpoint_arrays(path):
    result = {}
    prefix = "_orig_mod.model.inner."
    for name, (layout, offset) in read_layout(path).items():
        if not name.startswith(prefix):
            raise ValueError("unexpected release checkpoint prefix")
        name = name.removeprefix(prefix)
        dtype = "<f4" if layout.dtype == "FloatStorage" else "<u2"
        raw = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=layout.shape)
        result[name] = bf16(raw) if dtype == "<f4" else (raw.astype(np.uint32) << 16).view(np.float32)
    return result


def linear(x, weight):
    return bf16(x @ weight.T)


def rms_norm(x):
    return bf16(x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + 1e-5))


class SudokuHRM:
    def __init__(self, weights, config):
        self.hidden = config["hidden_size"]
        self.heads = config["num_heads"]
        self.dhead = self.hidden // self.heads
        self.h_cycles, self.l_cycles = config["H_cycles"], config["L_cycles"]
        if (self.hidden, self.heads, config["puzzle_emb_ndim"], config["pos_encodings"]) != (512, 8, 512, "rope"):
            raise ValueError("unsupported Sudoku checkpoint architecture")
        if (config["H_layers"], config["L_layers"], config["expansion"]) != (4, 4, 4):
            raise ValueError("unsupported Sudoku stack shape")
        frequencies = 1.0 / (10000.0 ** (np.arange(0, self.dhead, 2, dtype=np.float32) / self.dhead))
        angles = np.outer(np.arange(82, dtype=np.float32), frequencies)
        angles = np.concatenate((angles, angles), axis=-1)
        self.cos, self.sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
        self.weights = weights
        self.high = self.layers("H", config["H_layers"])
        self.low = self.layers("L", config["L_layers"])

    def layers(self, kind, n):
        return [
            tuple(self.weights[f"{kind}_level.layers.{i}.{part}.weight"] for part in (
                "self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj",
            ))
            for i in range(n)
        ]

    def rotate(self, vectors):
        a, b = np.split(vectors, 2, axis=-1)
        return bf16(vectors * self.cos + np.concatenate((-b, a), axis=-1) * self.sin)

    def attention(self, x, qkv, output):
        sequence = x.shape[0]
        mixed = linear(x, qkv).reshape(sequence, 3 * self.heads, self.dhead)
        q, k, v = np.split(mixed, 3, axis=1)
        q = self.rotate(q).transpose(1, 0, 2)
        k = self.rotate(k).transpose(1, 2, 0)
        density = (q @ k) / math.sqrt(self.dhead)
        e = np.exp(density - density.max(axis=-1, keepdims=True))
        aggregate = (e / e.sum(axis=-1, keepdims=True)) @ v.transpose(1, 0, 2)
        return linear(bf16(aggregate.transpose(1, 0, 2).reshape(sequence, self.hidden)), output)

    def stack(self, x, injection, layers):
        x = bf16(x + injection)
        for qkv, projection, gate_up, down in layers:
            x = rms_norm(bf16(x + self.attention(x, qkv, projection)))
            gate, up = np.split(linear(x, gate_up), 2, axis=-1)
            activation = bf16(gate / (1 + np.exp(-np.clip(gate, -80, 80))))
            x = rms_norm(bf16(x + linear(bf16(activation * up), down)))
        return x

    def rollout(self, givens, coefficients, guard):
        w = self.weights
        sequence = np.asarray([int(c) + 1 for c in givens])
        puzzle_embedding = w["puzzle_emb.weights"][:1]
        inputs = bf16(np.concatenate((puzzle_embedding, w["embed_tokens.embedding_weight"][sequence]))
                      * math.sqrt(self.hidden))
        high, low = w["H_init"], w["L_init"]
        for visit, eta in enumerate(coefficients, 1):
            guard()
            for _ in range(self.l_cycles):
                low = self.stack(low, bf16(high + inputs), self.low)
            candidate = self.stack(high, low, self.high)
            high = candidate if eta == 1 else bf16(high + eta * (candidate - high))
            if visit % self.h_cycles == 0:
                logits = linear(high[1:], w["lm_head.weight"])
                raw = logits.argmax(axis=-1)
                yield {
                    "segment": visit // self.h_cycles,
                    "prediction": "".join(str(max(0, token - 1)) for token in raw),
                    "slow_rms": float(np.sqrt(np.mean(high**2))),
                }
