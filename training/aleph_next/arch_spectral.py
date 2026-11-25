"""Local singular-value estimates for the frozen three-pass recurrent core."""

import argparse
import json
import math
from pathlib import Path

import torch
from architecture_lab import ARMS, HELD, MiniAleph, evaluation_rows, rms, token_rms
from gym import atomic_json, file_sha


@torch.inference_mode()
def norms(model, x, source, arm, visit, factor):
    dim = x.numel()
    step = factor * float(rms(x).item())
    perturb = step * torch.eye(dim).reshape(dim, *x.shape[1:])
    inputs = torch.cat((x + perturb, x - perturb))
    values = model.core(inputs, source.expand_as(inputs), arm, visit)[0].reshape(2, dim, dim)
    jac = ((values[0] - values[1]) / (2 * step)).double().T
    singular = torch.linalg.svdvals(jac)
    return {
        "operator_norm": float(singular[0]),
        "normalized_frobenius": float(jac.norm() / math.sqrt(dim)),
        "operator_norm_j_minus_i": float(torch.linalg.matrix_norm(jac - torch.eye(dim, dtype=jac.dtype), ord=2)),
    }


@torch.inference_mode()
def run(directory):
    manifest = json.loads((directory / "training.json").read_text())
    scored = directory / "scores.json"
    if scored.exists():
        summary = json.loads(scored.read_text())
    else:
        from arch_scoring import verified

        _, summary = verified(directory)
    if (
        file_sha(directory / "training.json") != summary["training_sha256"]
        or file_sha(directory / "private.seed") != summary["seed_sha256"]
    ):
        raise ValueError("scoring and provenance mismatch")
    width = manifest["contract"]["recipe"]["width"]
    if width > 64:
        raise ValueError("basis sweep is registered only for miniatures of width at most 64")
    p = int((directory / "private.seed").read_text())
    features = evaluation_rows(p, width, 16)[0]
    result = {
        "source_sha256": file_sha(Path(__file__)),
        "protocol_sha256": file_sha(Path(__file__).with_name("ARCH_SPECTRAL_PROTOCOL.md")),
        "scored_manifest_sha256": summary["training_sha256"],
        "held_seed_sha256": summary["seed_sha256"],
        "cells": {},
    }
    for key, fit in manifest["fits"].items():
        arm = fit["arm"]
        if ARMS[arm] != 3:
            continue
        path = directory / f"{key}.pt"
        if file_sha(path) != fit["checkpoint_sha256"]:
            raise ValueError("changed frozen checkpoint")
        model = MiniAleph(width, arm).eval()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        rows = []
        for i, pair in enumerate(HELD):
            # Same ordinal and task array as the frozen fresh-panel generator.
            f = features[(64 + 16) * i : (64 + 16) * i + 1]
            source = model.source(f)
            foil = f.clone()
            foil[:, :, :2] = f[:, :, 3:5]
            foil[:, :, 2] *= -1
            v = model.source(foil) - source
            unit = v / token_rms(v)
            for strength in (0.0, 1.25):
                x = source
                for visit in range(3):
                    values = norms(model, x, source, arm, visit, 0.01)
                    if i == 0:
                        fine = norms(model, x, source, arm, visit, 0.005)
                        values["half_step_relative_difference"] = max(
                            abs(fine[k] - v) / max(abs(v), 1e-12) for k, v in values.items()
                        )
                    rows.append({"family": list(pair), "strength": strength, "visit": visit + 1, **values})
                    raw = model.core(x, source, arm, visit)[0]
                    alpha = model.alpha(arm, visit, raw, x)
                    end = raw if arm == "direct3" or visit == 2 else x + alpha * (raw - x)
                    x = end + strength * token_rms(end) * unit if visit == 0 else end
        result["cells"][key] = rows
        print("local spectra", key, flush=True)
    # Exclude hidden probe states and any labels from the exported artifact.
    atomic_json(directory / "spectra.json", result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(args.out)
