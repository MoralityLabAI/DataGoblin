"""Fit a depth selector on verified native rollouts and check held-out roots."""

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

from .inkling import _manifest, _new_dir
from .inkling_rlt_trace import _design
from .inkling_sae import _encode
from .inkling_trace import NATIVE_ANSWER, WIDTH
from .runner import read_jsonl


def _verified_pairs(
    folder: Path, split: str, sealed: dict, samples: list[dict],
) -> list[tuple[dict, dict]]:
    report = _manifest(folder / "manifest.json")
    controls = report["capture_controls"]
    matrix = folder / "depth_matrix.jsonl"
    if (report["split"] != split or controls["native_input_format"] != NATIVE_ANSWER
            or controls["native_prompt_manifest_sha256"] != sealed["source_prompt_manifest_sha256"]
            or controls["native_weights_sha256"] != sealed["weights_sha256"]
            or controls["native_runtime_sha256"] != sealed["runtime_sha256"]
            or sha256_file(matrix) != report["files"]["depth_matrix"]):
        raise ValueError("routing requires the same frozen prompts and verified native answer matrix")
    found = defaultdict(dict)
    for row in read_jsonl(matrix):
        if row["split"] != split or row["fixture"] or row["passes"] not in (1, 3):
            raise ValueError("invalid verified routing row")
        identity = row["task_digest"]
        if row["passes"] in found[identity]:
            raise ValueError("duplicate routing outcome")
        found[identity][row["passes"]] = row
    pairs = []
    for sample in samples:
        arms = found[sample["task_digest"]]
        if set(arms) != {1, 3} or any(not arm["request_attempted"] or arm["layer_visit_token_proxy"] is None
                                     for arm in arms.values()):
            raise ValueError("routing comparison requires both completed depth arms at every sampled root")
        pairs.append((arms[1], arms[3]))
    return pairs


def _sum_choices(choices: np.ndarray, pairs: list[tuple[dict, dict]]) -> dict:
    selected = [pair[int(prefers_deep)] for pair, prefers_deep in zip(pairs, choices, strict=True)]
    return {
        "roots": len(selected), "verified_correct": sum(row["passed"] for row in selected),
        "deep_choices": int(choices.sum()), "loop_passes": sum(row["passes"] for row in selected),
        "reported_verifier_calls": sum(row["verifier_calls"] for row in selected),
        "layer_visit_token_proxy": sum(row["layer_visit_token_proxy"] for row in selected),
    }


def _threshold(strength: np.ndarray, pairs: list[tuple[dict, dict]]) -> float:
    candidates = [*np.unique(strength), np.nextafter(strength.max(), np.inf)]

    def rank(value):
        outcome = _sum_choices(strength >= value, pairs)
        return (-outcome["verified_correct"], outcome["layer_visit_token_proxy"], outcome["deep_choices"])

    return float(min(candidates, key=rank))


def fit_depth_selector(
    prompts: Path, trace: Path, sae: Path, heads: Path, train: Path, calibration: Path, out: Path,
) -> dict:
    prompt_path, trace_path = prompts / "manifest.json", trace / "manifest.json"
    meta, sealed = _manifest(prompt_path), _manifest(trace_path)
    head_receipt = _manifest(heads / "manifest.json")
    dictionary, head_file = sae / "sae_dictionary.npz", heads / "heads.npz"
    prompt_sha = sha256_file(prompt_path)
    if (sealed["passes"] != 3 or sealed["input_format"] != NATIVE_ANSWER
            or sealed["source_prompt_manifest_sha256"] != prompt_sha
            or head_receipt["trace_manifest_sha256"] != sha256_file(trace_path)
            or head_receipt["dictionary_sha256"] != sha256_file(dictionary)
            or head_receipt["heads_sha256"] != sha256_file(head_file)
            or any(sha256_file(trace / file) != sealed["files"][file] for file in ("rows.tsv", "activations.f32"))):
        raise ValueError("routing needs the matching three-pass native trace and fitted heads")
    pairs, samples = {}, {}
    for split, scored in (("train", train), ("calibration", calibration)):
        samples[split] = [row for row in meta["samples"] if row["split"] == split]
        pairs[split] = _verified_pairs(scored, split, sealed, samples[split])
    if any(len(rows) < 4 for rows in samples.values()):
        raise ValueError("routing needs at least four roots in each split")

    last_rows = {}
    with (trace / "rows.tsv").open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            key = row["sample_key"]
            if row["pass_index"] == "0" and int(row["token_index"]) == sealed["tokens_by_sample"][key] - 1:
                last_rows[key] = index
    data = np.memmap(trace / "activations.f32", mode="r", dtype="<f4", shape=(sealed["rows"], WIDTH))
    with np.load(dictionary) as w:
        prelude = {
            split: _encode(data[[last_rows[row["sample_key"]] for row in rows]] - w["pre_bias"],
                           w["encoder_weight"], w["encoder_bias"], int(w["top_k"]))
            for split, rows in samples.items()
        }
    fit = {}
    with np.load(head_file) as weights:
        for mode in MODES:
            strengths = {}
            for split, initial in prelude.items():
                state = initial
                for visit in range(3):
                    eta = feedback_weight(mode, visit, 3)
                    state = (1 - eta) * state + eta * (_design(state, visit, 3) @ weights[mode])
                    if visit == 0:
                        early = state
                strengths[split] = np.linalg.norm(state - early, axis=1).astype(np.float64)
            cut = _threshold(strengths["train"], pairs["train"])
            fit[mode] = {"threshold": cut,
                         **{split: _sum_choices(strength >= cut, pairs[split])
                            for split, strength in strengths.items()}}
    baseline = {f"always_{depth}": _sum_choices(np.full(len(pairs["calibration"]), depth == 3), pairs["calibration"])
                for depth in (1, 3)}
    _new_dir(out)
    report = {
        "schema_version": "rlt_gym.inkling_depth_selector.v1", "fitted": fit, "calibration_baselines": baseline,
        "train_score_sha256": sha256_file(train / "manifest.json"),
        "calibration_score_sha256": sha256_file(calibration / "manifest.json"),
        "trace_feedback_sha256": sha256_file(heads / "manifest.json"),
        "scope": "counterfactual_routing_on_paired_verified_native_rollouts",
    }
    dump_structured(out / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("prompts", "trace", "sae", "heads", "train", "calibration", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    print(json.dumps(fit_depth_selector(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
