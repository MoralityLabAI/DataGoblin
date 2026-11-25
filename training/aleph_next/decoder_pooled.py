"""Exploratory ALEPH-03 oracle with exactly the public features visible to the receiver."""

import argparse

import numpy as np
import torch
from decoder_lab import CONDITIONS, build_sender, cells, entropy_bits, fresh_panel, raw_hash, read, uploads, write


def pooled_oracle(frames):
    x, mass, yes = (np.concatenate([frame[i] for frame in frames]) for i in range(3))
    _, ii = np.unique(x, return_inverse=True, axis=0)
    ii = ii.reshape(-1)
    mass = np.bincount(ii, weights=mass) / len(frames)
    yes = np.bincount(ii, weights=yes) / len(frames)
    support = mass > 0
    mass, yes = mass[support], yes[support]
    post = yes / mass
    return {
        "brier": float(np.sum(mass * post * (1 - post))),
        "entropy_bits": float(np.sum(mass * entropy_bits(post))),
        "occupied_public_cells": len(mass),
    }


def evaluate(v2, prior, out):
    training, result = read(out / "training.json"), read(out / "results.json")
    if not (out / "private.seed").exists():
        raise ValueError("this diagnostic requires the completed draw")
    games, salt_hash = fresh_panel(out, v2, prior, training["training_frames_sha256"])
    if salt_hash != result["private_seed_sha256"] or any(
        raw_hash(g) != result["frames"][n]["sha256"] for n, g in games.items()
    ):
        raise ValueError("evaluation frames differ from the published primary assay")
    summary, raw = {}, list(games.values())
    source = read(v2 / "training.json")
    fits = read(out / "fit-scores.json")
    for sender_name, expected in training["senders"].items():
        arm, seed = sender_name.rsplit("-", 1)
        sender, stamp = build_sender(v2, arm, int(seed), source)
        if stamp != expected:
            raise ValueError("sender changed")
        codes = uploads(sender, raw)
        conditions = {}
        for condition in CONDITIONS:
            public = pooled_oracle([cells(g, c, condition) for g, c in zip(raw, codes, strict=True)])
            by_frame = fits[f"{sender_name}-to-C-{condition}"]["mean"]["bayes"]
            if public["brier"] - by_frame < -1e-12:
                raise AssertionError("conditioning on frame increased ideal Brier")
            conditions[condition] = {
                **public,
                "per_frame_brier": by_frame,
                "frame_identity_gap": public["brier"] - by_frame,
            }
        side, valid = conditions["side"], conditions["valid"]
        summary[sender_name] = {
            "conditions": conditions,
            "authorized_input_oracle_value": side["brier"] - valid["brier"],
            "per_frame_oracle_value": side["per_frame_brier"] - valid["per_frame_brier"],
            "authorized_input_cmi_bits": side["entropy_bits"] - valid["entropy_bits"],
        }
    write(
        out / "pooled-oracle.json",
        {
            "interpretation": "Retrospective exact finite-panel oracle within public feature equivalence classes. "
            "Uses all evaluation cell frequencies only for post-scoring analysis; never a deployable learned reader.",
            "private_seed_sha256": salt_hash,
            "senders": summary,
        },
        immutable=True,
    )
    for a in ("C", "D", "MLP"):
        rows = [d for k, d in summary.items() if k.split("-")[0] == a]
        print(
            a,
            {
                k: float(np.mean([d[k] for d in rows]))
                for k in (
                    "per_frame_oracle_value",
                    "authorized_input_oracle_value",
                    "authorized_input_cmi_bits",
                )
            },
            flush=True,
        )


if __name__ == "__main__":
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("v2", "prior-readout", "out"):
        p.add_argument("--" + arg, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    evaluate(args.v2, args.prior_readout, args.out)
