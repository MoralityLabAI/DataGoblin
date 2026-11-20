"""CPU inference for the frozen Lean Loop-XS tactic scorer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured, load_structured

MODES = ("direct", "rlt", "matched_damping")
TACTICS = ("simp", "omega", "grind", "exact?", "simp_all", "assumption", "rfl", "decide")
PAD, BOS, EOS = 256, 257, 258


def candidate_tokens(tactics=TACTICS):
    encoded = [tactic.encode("utf-8") for tactic in tactics]
    width = max(map(len, encoded)) + 1
    if width > 96:
        raise ValueError("tactic and EOS must fit in the checkpoint's decoder window")
    inputs = np.full((len(tactics), width), PAD, dtype=np.int64)
    targets = inputs.copy()
    for idx, tactic in enumerate(encoded):
        data = list(tactic)
        inputs[idx, : len(data) + 1] = [BOS, *data]
        targets[idx, : len(data) + 1] = [*data, EOS]
    return inputs, targets


def scores(logits, targets):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    picked = np.take_along_axis(shifted, targets[..., None], axis=-1)[..., 0]
    total = picked - np.log(np.exp(shifted).sum(axis=-1))
    mask = targets != PAD
    return (total * mask).sum(axis=-1) / mask.sum(axis=-1)


def gelu(values):
    # Evaluate the standard-normal CDF with a rational erf approximation.
    scaled = values * np.float32(1 / math.sqrt(2))
    t = 1 / (1 + np.float32(0.3275911) * np.abs(scaled))
    poly = (((np.float32(1.061405429) * t - 1.453152027) * t + 1.421413741)
            * t - 0.284496736) * t + 0.254829592
    erf = np.sign(scaled) * (1 - poly * t * np.exp(-scaled * scaled))
    return values * (0.5 + 0.5 * erf)


class LoopTacticRanker:
    def __init__(self, weights, feedback):
        self.weights = weights
        self.feedback = feedback
        self.inputs, self.targets = candidate_tokens()

    def linear(self, x, name):
        w = self.weights
        return x @ w[f"{name}.weight"].T + w[f"{name}.bias"]

    def norm(self, x, name):
        center = x - x.mean(axis=-1, keepdims=True)
        normalized = center / np.sqrt(np.mean(center * center, axis=-1, keepdims=True) + 1e-5)
        return normalized * self.weights[f"{name}.weight"] + self.weights[f"{name}.bias"]

    def attention(self, query, memory, name, heads, causal=False):
        dim = query.shape[-1] // heads
        q = self.linear(query, name + ".q").reshape(*query.shape[:-1], heads, dim).swapaxes(-3, -2)
        k, v = np.split(self.linear(memory, name + ".kv"), 2, axis=-1)
        k, v = (x.reshape(*x.shape[:-1], heads, dim).swapaxes(-3, -2) for x in (k, v))
        similarity = (q @ k.swapaxes(-2, -1)) * np.float32(1 / math.sqrt(dim))
        if causal:
            allowed = np.tri(query.shape[-2], dtype=bool)
            similarity = np.where(allowed, similarity, -np.inf)
        probability = np.exp(similarity - similarity.max(axis=-1, keepdims=True))
        probability /= probability.sum(axis=-1, keepdims=True)
        mixed = (probability @ v).swapaxes(-3, -2).reshape(query.shape)
        return self.linear(mixed, name + ".o")

    def block(self, state, name, heads, memory=None):
        base = self.norm(state, name + ".n1")
        value = state + self.attention(base, base, name + ".att", heads, causal=memory is not None)
        if memory is not None:
            query = self.norm(value, name + ".n2")
            value = value + self.attention(query, memory, name + ".cross", heads)
        hidden = gelu(self.linear(self.norm(value, name + ".n3"), name + ".ff.0"))
        return value + self.linear(hidden, name + ".ff.2")

    def predict(self, raw, mode, depth, tactics=None):
        tokens = np.frombuffer(raw.encode("utf-8")[-384:], dtype=np.uint8)
        w = self.weights
        input_embedding = w["emb.weight"][tokens] + w["pos"][: len(tokens)]
        state = np.zeros_like(input_embedding)
        inputs, targets = (self.inputs, self.targets) if tactics is None else candidate_tokens(tactics)
        decoder_input = w["demb.weight"][inputs] + w["dpos"][: inputs.shape[1]]
        batch_size = len(inputs) if tactics is None else 4
        for index in range(depth):
            endpoint = self.block(state + input_embedding, "fast", 8)
            state = self.feedback(state, endpoint, mode, index, depth)
            context = self.linear(self.norm(state, "enc_norm"), "mem_proj")
            result = []
            for offset in range(0, len(inputs), batch_size):
                batch = slice(offset, offset + batch_size)
                decoded = decoder_input[batch]
                for layer in (0, 1):
                    decoded = self.block(decoded, f"dec.{layer}", 4, context)
                logits = self.linear(self.norm(decoded, "head.0"), "head.1")
                result.append(scores(logits, targets[batch]))
            yield np.concatenate(result)


def export(root, checkpoints, torch, guard):
    from lcgrt.feedback import advance
    from lcgrt.tactic_lm import TacticLM, TLMConfig

    examples = (
        ("n : Nat\n⊢ 0 + n = n", (4, 8), None),
        ("a b : Nat\nh : a + 7 ≤ b\n⊢ a + 5 < b", (4, 8), None),
        ("xs : List Nat\n⊢ (xs.take 2).reverse = xs.reverse.drop (xs.length - 2)", (6,), (
            "simp [List.reverse_take, List.reverse_append]", "rw [List.reverse_take]",
            "exact List.reverse_take", "apply List.reverse_drop", "simp [List.take_append, List.reverse_take]",
        )),
    )
    models = {}
    for mode in MODES:
        guard()
        source = checkpoints / f"loop_{mode}_s131"
        path = source / "best.pt"
        receipt = load_structured(source / "best.json")
        metadata = load_structured(source / "config.json")
        if receipt["sha256"] != sha256_file(path) or receipt["step"] != 3400:
            raise ValueError("unexpected Lean checkpoint hash or training milestone")
        ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        cfg = TLMConfig(**ckpt["config"])
        signature = (cfg.arch, cfg.width, cfg.heads, cfg.dec_width, cfg.dec_heads, cfg.dec_layers,
                     cfg.fast_steps, cfg.feedback)
        if signature != ("loop", 240, 8, 128, 4, 2, 8, mode):
            raise ValueError("expected the matched Loop-XS architecture")
        with torch.inference_mode():
            model = TacticLM(cfg)
        model.load_state_dict(ckpt["model"], assign=True)
        model.eval()
        tensors = {k: tensor.numpy() for k, tensor in model.state_dict().items()}
        if any(tensor.dtype != np.float32 for tensor in tensors.values()):
            raise ValueError("expected float32 inference weights")
        ranker = LoopTacticRanker(tensors, advance)
        max_error = 0.0
        checks = {False: 0, True: 0}
        with torch.inference_mode():
            for text, depths, tactics in examples:
                decoder, targets = candidate_tokens(TACTICS if tactics is None else tactics)
                raw = list(text.encode("utf-8"))
                enc = torch.tensor([raw])
                mask = torch.ones_like(enc, dtype=torch.bool)
                for depth in depths:
                    trace = []
                    model.encode(enc, mask, fast_steps=depth, trace=trace)
                    for entry, predicted in zip(trace, ranker.predict(text, mode, depth, tactics), strict=True):
                        mem = model.mem_proj(model.enc_norm(entry["state"])).expand(len(decoder), -1, -1)
                        reference = model.decode(torch.from_numpy(decoder), mem, mask.expand(len(decoder), -1))
                        expected = scores(reference.numpy(), targets)
                        max_error = max(max_error, float(np.max(np.abs(expected - predicted))))
                        orders = (np.argsort(-values, kind="stable") for values in (expected, predicted))
                        if not np.array_equal(*orders):
                            raise ValueError("NumPy tactic ordering differs from Torch reference")
                        checks[tactics is not None] += 1
        if max_error > 0.0002:
            raise ValueError("NumPy tactic scores differ from Torch reference")
        archive = root / f"{mode}.npz"
        np.savez(archive, **tensors)
        models[mode] = {
            "checkpoint_sha256": receipt["sha256"], "array_sha256": sha256_file(archive),
            "training_step": receipt["step"], "identity_hash": metadata["identity_hash"],
            "rank_checks": checks[False], "grounded_rank_checks": checks[True], "maximum_logprob_error": max_error,
        }
        del ranker, tensors, model, ckpt
    dump_structured(root / "model-parity.json", {
        "models": models, "torch": torch.__version__, "numpy": np.__version__,
        "ranker_source_sha256": sha256_file(Path(__file__)),
    })
