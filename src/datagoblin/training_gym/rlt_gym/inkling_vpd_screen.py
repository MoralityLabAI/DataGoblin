"""Prepare a lineage-paired SAE feature shortlist for intervention work."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from datagoblin.hashing import sha256_file
from datagoblin.serialization import dump_structured

from .inkling import _manifest, _new_dir
from .inkling_sae import _encode, _load_split
from .inkling_trace import NATIVE_ANSWER, WIDTH, boundary_sites


def _root_effects(matrix, rows, encoder, bias, center, top_k, last_pass):
    pairs = defaultdict(dict)
    roots = {row["lineage_id"] for row in rows}
    for index, row in enumerate(rows):
        depth = row["pass_index"]
        if depth in (1, last_pass):
            pair = pairs[row["lineage_id"], row["token_index"]]
            if depth in pair:
                raise ValueError("duplicate feature pairing at one root and token")
            pair[depth] = index

    grouped = defaultdict(list)
    for (root, _), indices in pairs.items():
        if 1 in indices and last_pass in indices:
            grouped[root].append((indices[1], indices[last_pass]))
    if grouped.keys() != roots:
        raise ValueError("every SAE root needs a matched first/final token pair")

    effects, counts = [], {}
    for root, indices in sorted(grouped.items()):
        summed = np.zeros(len(encoder), dtype=np.float64)
        for start in range(0, len(indices), 64):
            early, late = np.asarray(indices[start:start + 64]).T
            a = _encode(matrix[early] - center, encoder, bias, top_k)
            b = _encode(matrix[late] - center, encoder, bias, top_k)
            summed += (b - a).sum(axis=0, dtype=np.float64)
        counts[root] = len(indices)
        effects.append(summed / len(indices))
    return np.stack(effects), counts


def _max_stat(oriented: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, int, bool]:
    count = len(oriented)
    scale = np.maximum(np.sqrt(np.mean(oriented**2, axis=0)), 1e-12)
    standardized = oriented / scale
    observed = standardized.mean(axis=0)
    exact = count <= 12
    if exact:
        signs = 2 * ((np.arange(1 << count)[:, None] >> np.arange(count)) & 1) - 1
    else:
        signs = np.random.default_rng(seed).choice((-1, 1), size=(4096, count))
    null_max = (signs @ standardized / count).max(axis=1)
    hits = (null_max[:, None] >= observed[None, :] - 1e-12).sum(axis=0)
    adjusted = hits / len(signs) if exact else (hits + 1) / (len(signs) + 1)
    return observed, adjusted, len(signs), exact


def _last_token_pairs(prompts: Path, trace: Path, meta: dict, target: int) -> tuple[dict, list[dict]]:
    prompt_path, trace_path = prompts / "manifest.json", trace / "manifest.json"
    sealed = _manifest(trace_path)
    samples = _manifest(prompt_path)["samples"]
    index, activations = trace / "rows.tsv", trace / "activations.f32"
    if (sealed["input_format"] != NATIVE_ANSWER or sealed["intervention_applied"]
            or sealed["passes"] < target or sealed["source_prompt_manifest_sha256"] != sha256_file(prompt_path)
            or meta["trace_manifest_sha256"] != sha256_file(trace_path)
            or sha256_file(index) != sealed["files"]["rows.tsv"]
            or sha256_file(activations) != sealed["files"]["activations.f32"]):
        raise ValueError("last-token screening requires the unmodified native answer trace")
    positions = {
        (row["sample_key"], str(visit)): str(sealed["tokens_by_sample"][row["sample_key"]] - 1)
        for row in samples for visit in (1, target)
    }
    states = {}
    sites = boundary_sites(sealed["passes"])
    with index.open(encoding="utf-8", newline="") as rows, activations.open("rb") as vectors:
        for offset, row in enumerate(csv.DictReader(rows, delimiter="\t")):
            position = (row["sample_key"], row["pass_index"])
            if position not in positions or positions[position] != row["token_index"]:
                continue
            if position in states or row["tensor"] != sites[int(row["pass_index"])]:
                raise ValueError("last-token trace has duplicate or mismatched sites")
            vectors.seek(offset * WIDTH * 4)
            state = np.frombuffer(vectors.read(WIDTH * 4), dtype="<f4")
            if state.shape != (WIDTH,) or not np.isfinite(state).all():
                raise ValueError("last-token trace contains an invalid activation")
            states[position] = state
    if states.keys() != positions.keys():
        raise ValueError("last-token screen requires both visits at every root")
    return states, samples


def screen_features(
    inputs: Path, sae: Path, out: Path, *, shortlist: int = 16, seed: int = 7,
    prompts: Path | None = None, trace: Path | None = None, target_pass: int | None = None,
) -> dict:
    """Fix the feature IDs and signs on train, then screen them on calibration roots."""
    if not 1 <= shortlist <= 64:
        raise ValueError("invalid feature shortlist budget")
    source, fitted_path = inputs / "manifest.json", sae / "manifest.json"
    meta, fitted = _manifest(source), _manifest(fitted_path)
    dictionary = sae / "sae_dictionary.npz"
    if (meta["schema_version"] != "rlt_gym.inkling_sae_inputs.v1"
            or fitted["source_inputs_sha256"] != sha256_file(source)
            or fitted["dictionary_sha256"] != sha256_file(dictionary)):
        raise ValueError("feature screen needs a dictionary bound to the trace inputs")
    with np.load(dictionary) as weights:
        encoder, bias, center = (weights[label] for label in ("encoder_weight", "encoder_bias", "pre_bias"))
        top_k = int(weights["top_k"])
    if prompts is None and trace is None and target_pass is None:
        train, train_rows = _load_split(inputs, meta["streams"]["train"], meta["key"], "train")
        cal, cal_rows = _load_split(inputs, meta["streams"]["calibration"], meta["key"], "calibration")
        last_pass = max(row["pass_index"] for row in train_rows)
        if last_pass <= 1 or max(row["pass_index"] for row in cal_rows) != last_pass:
            raise ValueError("feature screen requires matched first and final core passes")
        fit, train_pairs = _root_effects(train, train_rows, encoder, bias, center, top_k, last_pass)
        heldout, cal_pairs = _root_effects(cal, cal_rows, encoder, bias, center, top_k, last_pass)
        position_receipt = {"token_selection": "all_paired_prefill_tokens"}
    else:
        if prompts is None or trace is None or target_pass not in (2, 3):
            raise ValueError("last-token screening requires prompts, trace and a target pass of two or three")
        last_pass = target_pass
        states, samples = _last_token_pairs(prompts, trace, meta, last_pass)
        splits = {}
        for split in ("train", "calibration"):
            rows = [row for row in samples if row["split"] == split]
            counts = dict.fromkeys((row["lineage_id"] for row in rows), 1)
            if len(counts) != len(rows) or len(rows) < 4:
                raise ValueError("last-token screen requires four distinct roots per split")
            keys = [row["sample_key"] for row in rows]
            early = np.stack([states[key, "1"] for key in keys])
            late = np.stack([states[key, str(last_pass)] for key in keys])
            effects = _encode(late - center, encoder, bias, top_k) - _encode(early - center, encoder, bias, top_k)
            splits[split] = (effects.astype(np.float64), counts)
        fit, train_pairs = splits["train"]
        heldout, cal_pairs = splits["calibration"]
        position_receipt = {
            "token_selection": "last_prefill_token", "position_trace_sha256": meta["trace_manifest_sha256"],
            "position_prompts_sha256": sha256_file(prompts / "manifest.json"),
        }
    if min(len(train_pairs), len(cal_pairs)) < 4 or train_pairs.keys() & cal_pairs.keys():
        raise ValueError("the screen requires four disjoint roots per split")
    means = fit.mean(axis=0)
    identifiers = np.flatnonzero(np.abs(means) > 1e-12)
    if not len(identifiers):
        raise ValueError("training roots provide no directional contrast")
    ranking = np.lexsort((identifiers, -np.abs(means[identifiers])))
    identifiers = identifiers[ranking[:shortlist]]
    signs = np.sign(means[identifiers])
    statistic, adjusted, permutations, exact = _max_stat(heldout[:, identifiers] * signs, seed)
    features = [{
        "feature": int(feature), "train_direction": int(direction), "training_mean_difference": float(means[feature]),
        "calibration_mean_difference": float(heldout[:, feature].mean()), "oriented_statistic": float(value),
        "max_stat_adjusted_p": float(p), "admitted_for_intervention": bool(value > 0 and p <= 0.05),
    } for feature, direction, value, p in zip(identifiers, signs, statistic, adjusted, strict=True)]
    _new_dir(out)
    result = {
        "schema_version": "rlt_gym.inkling_vpd_feature_screen.v1", "contrast": [1, last_pass],
        "source_inputs_sha256": sha256_file(source), "dictionary_sha256": sha256_file(dictionary),
        "train_pairs_by_root": train_pairs, "calibration_pairs_by_root": cal_pairs,
        "permutation_unit": "root", "permutations": permutations, "exact_permutation": exact, "seed": seed,
        "features": features, "admitted_features": [row["feature"] for row in features
                                                   if row["admitted_for_intervention"]],
        **position_receipt,
        "intervention_applied": False, "bpd_mechanism_verified": False,
    }
    dump_structured(out / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("inputs", "sae", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--shortlist", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--target-pass", type=int, choices=(2, 3))
    result = screen_features(**vars(parser.parse_args()))
    print(json.dumps({"admitted_features": result["admitted_features"], "permutations": result["permutations"]}))


if __name__ == "__main__":
    main()
