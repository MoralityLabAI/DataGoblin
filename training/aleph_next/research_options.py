"""Generate train-only, costed receiver choices from actual frozen Model Atlas participants."""

import itertools

import numpy as np
from decoder_lab import READERS, cells, ckpt_path, digest, model_id, predictions


def options(fits, decoder, name, seed, frames, keys, codes, sender_sha):
    risks, members = {}, {}
    for reader, condition in itertools.product(READERS, ("side", "valid")):
        label = model_id(name, reader, condition)
        path = ckpt_path(decoder, label, reader)
        record = fits["fits"][label]
        if digest(path) != record["sha256"]:
            raise ValueError("option checkpoint hash mismatch")
        members[label] = record["sha256"]
        data = [cells(frame, bit, condition) for frame, bit in zip(frames, codes, strict=True)]
        x, mass, positive = (np.stack([r[i] for r in data]) for i in range(3))
        pred = predictions(reader, seed, path, x)
        # E[(Y-p)^2] = P(Y=1)(1-p)^2 + P(Y=0)p^2, with exact finite event masses.
        risks[label] = ((positive * (1 - pred) ** 2 + (mass - positive) * pred**2).sum(1) * 1e6).round().astype(int)
    for i, key in enumerate(keys):
        # Rotate the bidding slate so the structured prior does not settle every example.
        names = [READERS[(i + seed + j) % len(READERS)] for j in (0, 1, 3)]
        slate = {r: {c: int(risks[model_id(name, r, c)][i]) for c in ("side", "valid")} for r in names}
        gap = min(v["side"] for v in slate.values()) - min(v["valid"] for v in slate.values())
        for cost in sorted({0, max(1, gap - 100), max(2, gap + 100)}):
            available = {r + "/" + c: val + cost * (c == "valid") for r, row in slate.items() for c, val in row.items()}
            best = min(available, key=lambda k: (available[k], k))
            table = " ".join(f"{r}={v['side']},{v['valid']}" for r, v in sorted(slate.items()))
            guess = names[0] + "/valid"
            prompt = (
                f"Choose a reader for frozen sender {name}. Posted side-only and valid-bit Brier losses "
                f"in millionths (side,valid): {table}. Reading the bit costs {cost} millionths. "
                "Minimize total loss. Resolve ties by alphabetical reader/channel. "
                f"A researcher proposes {guess}. Return the best reader/channel: "
            )
            yield dict(
                split="train",
                family="ldt-train-" + key.split("/")[0],
                prompt=prompt,
                answer=best,
                sender=sender_sha,
                receivers={
                    k: members[k]
                    for k in members
                    if any(k == model_id(name, r, c) for r in names for c in ("side", "valid"))
                },
                proposal=guess,
                proposed_excess_microloss=available[guess] - available[best],
                objective="costed-reader-choice",
                verified_costs=available,
                verifier="enumerated binary Brier, nearest micro-unit then integer bit cost",
            )
