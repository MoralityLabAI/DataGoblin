"""Publish answer-free aggregate architecture records from a finished local assay."""

import argparse
import json
from pathlib import Path
from statistics import mean

from arch_join import digest


def export(directory):
    meta = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    spectra_path = directory / "spectra.json"
    spectra = {}
    if spectra_path.exists():
        spec = json.loads(spectra_path.read_text(encoding="utf-8"))
        if spec["scored_manifest_sha256"] != digest(directory / "training.json") or spec["held_seed_sha256"] != digest(
            directory / "private.seed"
        ):
            raise ValueError("spectral audit belongs to a different fit or panel")
        spectra = spec["cells"]
    rows = []
    for line in (directory / "integration.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        fit = item["model"]
        key = f"{fit['arm']}-{fit['seed']}"
        filename = key + ".pt"
        actual = digest(directory / filename)
        if actual != fit["checkpoint_sha256"] or meta["fits"][key] != fit:
            raise ValueError("checkpoint or aggregate identity mismatch")
        gate = summary["gates"].get(fit["arm"])
        failed = [] if gate is None else [name for name, passed in gate["checks"].items() if not passed]
        width, batch, steps = (item["recipe"][k] for k in ("width", "batch", "steps"))
        passes = fit["core_visits"]
        logical = fit["parameters"] + 2 * (passes - 1) * (15 * width**2 + 2 * width)
        if fit["arm"] in ("magnitude3", "visit3"):
            logical += 2 * (passes - 2)
        # QK^T and AV use 4*width FLOPs per query/key pair. Backpropagating
        # their two matrix products adds two forward equivalents. The foil
        # gets one forward. Fused kernel padding and softmax are unobserved.
        attention_factor = steps * 4 * batch * width * (3 * fit["native_blocks_per_example"] + 1)
        mechanism = dict(item["mechanistic_high"])
        if key in spectra:
            severe = [row for row in spectra[key] if row["strength"] == 1.25]
            for metric in ("operator_norm", "operator_norm_j_minus_i", "normalized_frobenius"):
                mechanism["basis_sweep_high_" + metric] = [
                    mean(row[metric] for row in severe if row["visit"] == visit) for visit in range(1, passes + 1)
                ]
        rows.append(
            {
                "key": item["id"],
                "checkpoint": filename,
                "sha256": actual,
                "architecture": f"Dense causal Aleph GQA/SwiGLU with two physically shared middle blocks, {fit['arm']}",
                "params": fit["parameters"],
                "passes": fit["core_visits"],
                "seed": fit["seed"],
                "steps": item["recipe"]["steps"],
                "metrics": {
                    **item["benchmark"],
                    "first_training_batch_bce": fit["training_bce"][0][1],
                    "last_training_batch_bce": fit["training_bce"][-1][1],
                    "max_sampled_gradient_norm_before_clip": max(fit["sampled_gradient_norm"]),
                    "processed_positions": fit["positions"],
                    "traced_flops": fit["counted_flops_all_updates"],
                    "estimated_attention_flops_causal_pairs": attention_factor * 10,
                    "estimated_attention_flops_square_tiles": attention_factor * 16,
                    "native_logical_parameter_applications": logical,
                    "wall_seconds": fit["wall_seconds"],
                    "sampled_peak_rss_mib": fit["sampled_peak_rss_mib"],
                },
                "mechanism": mechanism,
                "failures": ("Failed registered checks: " + ", ".join(failed) + ". " if failed else "")
                + "Four held operator compositions in a synthetic four-position verifier.",
            }
        )
    if len(rows) != len(meta["fits"]):
        raise ValueError("unregistered fit missing")
    (directory / "atlas-input.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    export(parser.parse_args().directory)
