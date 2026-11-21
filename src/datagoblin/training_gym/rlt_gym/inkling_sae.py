"""Fit and check a bounded CPU top-k SAE on sealed Inkling activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured

from .inkling import _new_dir
from .inkling_trace import WIDTH
from .runner import read_jsonl


def _load_split(path: Path, meta: dict, key: str, split: str) -> tuple[np.ndarray, list[dict]]:
    archive, index = path / f"{split}.npz", path / f"{split}.jsonl"
    if sha256_file(archive) != meta["activation_sha256"] or sha256_file(index) != meta["rows_sha256"]:
        raise ValueError("SAE input differs from its activation receipt")
    with np.load(archive) as arrays:
        matrix = arrays[key]
    rows = read_jsonl(index)
    if (matrix.shape != (meta["rows"], WIDTH) or matrix.dtype != np.float32
            or not np.isfinite(matrix).all() or len(rows) != meta["rows"]
            or any(row["split"] != split for row in rows)):
        raise ValueError("SAE activation and lineage rows disagree")
    return matrix, rows


def _encode(x: np.ndarray, encoder: np.ndarray, bias: np.ndarray, top_k: int) -> np.ndarray:
    positive = np.maximum(x @ encoder.T + bias, 0)
    indices = np.argpartition(positive, -top_k, axis=1)[:, -top_k:]
    codes = np.zeros_like(positive)
    np.put_along_axis(codes, indices, np.take_along_axis(positive, indices, axis=1), axis=1)
    return codes


def _evaluate(matrix, center, encoder, bias, decoder, top_k):
    error = baseline = active = 0.0
    for offset in range(0, len(matrix), 64):
        x = matrix[offset:offset + 64] - center
        codes = _encode(x, encoder, bias, top_k)
        residual = codes @ decoder - x
        error += float(np.sum(residual * residual, dtype=np.float64))
        baseline += float(np.sum(x * x, dtype=np.float64))
        active += np.count_nonzero(codes)
    return {
        "rows": len(matrix), "mse": error / matrix.size,
        "bias_only_mse": baseline / matrix.size,
        "explained_vs_train_bias": 1 - error / baseline if baseline else None,
        "mean_active_features": active / len(matrix),
    }


def fit_sae(
    inputs: Path, out: Path, *, features: int = 256, top_k: int = 16,
    steps: int = 96, batch_size: int = 24, seed: int = 17, learning_rate: float = 0.02,
) -> dict:
    if (not 1 <= top_k <= features <= 4096 or not 1 <= steps <= 4096
            or not 1 <= batch_size <= 128 or not 0 < learning_rate <= 0.1):
        raise ValueError("SAE fit exceeds its supported budget")
    source = inputs / "manifest.json"
    meta = json.loads(source.read_text(encoding="utf-8"))
    if meta.get("schema_version") != "rlt_gym.inkling_sae_inputs.v1":
        raise ValueError("SAE fit requires sealed lineage-separated matrices")
    train, train_rows = _load_split(inputs, meta["streams"]["train"], meta["key"], "train")
    calibration, cal_rows = _load_split(inputs, meta["streams"]["calibration"], meta["key"], "calibration")
    train_roots = {row["lineage_id"] for row in train_rows}
    cal_roots = {row["lineage_id"] for row in cal_rows}
    if len(train) < 16 or not cal_roots or train_roots & cal_roots:
        raise ValueError("SAE requires 16 training rows and disjoint calibration roots")

    rng = np.random.default_rng(seed)
    center = train.mean(axis=0)
    encoder = rng.standard_normal((features, WIDTH), dtype=np.float32)
    encoder /= np.sqrt(WIDTH)
    bias = np.zeros(features, dtype=np.float32)
    decoder = encoder.copy()
    decoder /= np.maximum(np.linalg.norm(decoder, axis=1, keepdims=True), 1e-8)
    before = _evaluate(train, center, encoder, bias, decoder, top_k)
    for _ in range(steps):
        x = train[rng.integers(len(train), size=batch_size)] - center
        codes = _encode(x, encoder, bias, top_k)
        feedback = (codes @ decoder - x) * (2 / batch_size)
        latent_gradient = (feedback @ decoder.T) * (codes > 0)
        gradients = (latent_gradient.T @ x, latent_gradient.sum(axis=0), codes.T @ feedback)
        for parameter, gradient in zip((encoder, bias, decoder), gradients, strict=True):
            parameter -= gradient * (learning_rate / max(1.0, np.linalg.norm(gradient) / 10))
        decoder /= np.maximum(np.linalg.norm(decoder, axis=1, keepdims=True), 1e-8)
    if any(not np.isfinite(array).all() for array in (encoder, bias, decoder)):
        raise ValueError("SAE training produced nonfinite parameters")

    _new_dir(out)
    dictionary = out / "sae_dictionary.npz"
    np.savez(dictionary, activation_key=np.asarray(meta["key"]), encoder_weight=encoder,
             encoder_bias=bias, pre_bias=center, decoder=decoder, top_k=np.asarray(top_k, dtype=np.int64))
    passes = np.asarray([row["pass_index"] for row in cal_rows])
    report = {
        "schema_version": "rlt_gym.inkling_topk_sae.v1", "source_inputs_sha256": sha256_file(source),
        "weights_sha256": meta["weights_sha256"], "dictionary_sha256": sha256_file(dictionary),
        "features": features, "top_k": top_k, "steps": steps, "batch_size": batch_size,
        "seed": seed, "learning_rate": learning_rate,
        "train_roots": len(train_roots), "calibration_roots": len(cal_roots),
        "training_before": before, "training_after": _evaluate(train, center, encoder, bias, decoder, top_k),
        "calibration": _evaluate(calibration, center, encoder, bias, decoder, top_k),
        "calibration_by_pass": {
            str(p): _evaluate(calibration[passes == p], center, encoder, bias, decoder, top_k)
            for p in np.unique(passes)
        },
        "intervention_applied": False, "vpd_selection_completed": False,
    }
    dump_structured(out / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    print(json.dumps(fit_sae(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
