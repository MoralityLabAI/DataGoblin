"""Fit matched feedback predictors to real Inkling pass-boundary trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured
from datagoblin.training_gym.rlt_feedback import MODES, feedback_weight

from .inkling import _new_dir
from .inkling_sae import _encode
from .inkling_trace import WIDTH


def _matrices(prompts: Path, trace: Path, sealed: dict, max_tokens: int) -> dict[str, np.ndarray]:
    plan = json.loads((prompts / "manifest.json").read_text(encoding="utf-8"))
    splits = {row["sample_key"]: row["split"] for row in plan["samples"]}
    passes = sealed["passes"]
    positions: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
    with (trace / "rows.tsv").open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            positions[row["sample_key"], int(row["token_index"])][int(row["pass_index"])] = index
    groups = defaultdict(list)
    for (key, _), by_pass in sorted(positions.items()):
        groups[splits[key]].append([by_pass[p] for p in range(passes + 1)])
    data = np.memmap(trace / "activations.f32", mode="r", dtype="<f4", shape=(sealed["rows"], WIDTH))
    result = {}
    for split in ("train", "calibration"):
        indices = np.asarray(groups[split], dtype=np.int64)
        if len(indices) < 16:
            raise ValueError("each split needs at least 16 complete token trajectories")
        take = np.linspace(0, len(indices) - 1, min(max_tokens, len(indices)), dtype=np.int64)
        result[split] = data[indices[take].T]
    return result


def _design(state: np.ndarray, visit: int, depth: int) -> np.ndarray:
    clock = np.broadcast_to(np.eye(depth, dtype=state.dtype)[visit], (len(state), depth))
    return np.concatenate((state, clock), axis=1)


def fit_feedback(
    prompts: Path, trace: Path, inputs: Path, sae: Path, out: Path, *,
    max_tokens: int = 1024, ridge: float = 0.1,
) -> dict:
    if not 16 <= max_tokens <= 2048 or not 0 < ridge <= 100:
        raise ValueError("invalid feedback fit budget")
    prompt_path = prompts / "manifest.json"
    trace_path, input_path, sae_path = (path / "manifest.json" for path in (trace, inputs, sae))
    sealed, exported, fitted = (
        json.loads(path.read_text(encoding="utf-8")) for path in (trace_path, input_path, sae_path)
    )
    dictionary = sae / "sae_dictionary.npz"
    if (sealed["passes"] < 3 or sealed["source_prompt_manifest_sha256"] != sha256_file(prompt_path)
            or exported["trace_manifest_sha256"] != sha256_file(trace_path)
            or fitted["source_inputs_sha256"] != sha256_file(input_path)
            or fitted["dictionary_sha256"] != sha256_file(dictionary)
            or any(sha256_file(trace / name) != sealed["files"][name]
                   for name in ("activations.f32", "rows.tsv", "native-complete.txt"))):
        raise ValueError("feedback fit requires a bound SAE and at least three observed core passes")
    with np.load(dictionary) as weights:
        encoder, bias, center, decoder = (weights[name] for name in (
            "encoder_weight", "encoder_bias", "pre_bias", "decoder",
        ))
        top_k = int(weights["top_k"])
    arrays = _matrices(prompts, trace, sealed, max_tokens)
    latent = {split: np.stack([_encode(x - center, encoder, bias, top_k) for x in matrix])
              for split, matrix in arrays.items()}
    train, cal = latent["train"], latent["calibration"]
    depth = sealed["passes"]
    heads = {}
    scores = {}
    for mode in MODES:
        eta = [feedback_weight(mode, visit, depth) for visit in range(depth)]
        designs = np.concatenate([_design(train[i], i, depth) * w for i, w in enumerate(eta)])
        targets = np.concatenate([train[i + 1] - (1 - w) * train[i] for i, w in enumerate(eta)])
        a, y = designs.astype(np.float64), targets.astype(np.float64)
        prior = np.eye(a.shape[1]) * ridge
        head = np.linalg.solve(a.T @ a + prior, a.T @ y).astype(np.float32)
        heads[mode] = head
        state = cal[0].copy()
        metrics = []
        for visit, weight in enumerate(eta):
            endpoint = _design(state, visit, depth) @ head
            state = (1 - weight) * state + weight * endpoint
            latent_error = state - cal[visit + 1]
            raw_error = (state @ decoder + center) - arrays["calibration"][visit + 1]
            metrics.append({
                "pass": visit + 1, "eta": weight,
                "latent_mse": float(np.mean(latent_error * latent_error, dtype=np.float64)),
                "activation_mse": float(np.mean(raw_error * raw_error, dtype=np.float64)),
            })
        scores[mode] = metrics
    _new_dir(out)
    heads_path = out / "heads.npz"
    np.savez(heads_path, **heads)
    report = {
        "schema_version": "rlt_gym.inkling_trace_feedback_fit.v1",
        "train_tokens": train.shape[1], "calibration_tokens": cal.shape[1], "ridge": ridge,
        "train_roots": fitted["train_roots"], "calibration_roots": fitted["calibration_roots"],
        "weights_sha256": sealed["weights_sha256"], "trace_manifest_sha256": sha256_file(trace_path),
        "dictionary_sha256": fitted["dictionary_sha256"], "heads_sha256": sha256_file(heads_path),
        "calibration": scores, "scope": "heldout_prefill_trajectory_prediction",
    }
    dump_structured(out / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prompts", "trace", "inputs", "sae", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--ridge", type=float, default=0.1)
    print(json.dumps(fit_feedback(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
