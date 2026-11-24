"""Train and score dense one-bit receipt policies on delayed parity and independent LDT games."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from receipt_ldt import make_games
from receipt_models import ReceiptModel

SEEDS = (137, 163, 191)
ARMS = ("C", "D", "MLP")
TRAIN_PRIORS = (0, 0.025, 0.05, 0.10, 0.15, 0.35, 0.75, 1)
TEST_PRIORS = (0, 0.03125, 0.0625, 0.07, 4 / 53, 0.08, 0.09375, 0.125, 0.25, 1)
CODES = (np.arange(0, 256, 2)[:, None] >> np.arange(8)) & 1
LINEAR = np.array([[int(x & d).bit_count() % 2 for x in range(8)] for d in range(1, 8)])


def parity(prior):
    x = np.arange(8)
    producer = np.zeros((8, 12))
    producer[:, :3] = 2 * ((x[:, None] >> np.arange(3)) & 1) - 1
    producer[:, 3] = prior
    ts = [(t, s) for t in range(1, 8) for s in range(1, 8)]
    receiver = np.zeros((49, 12))
    for q, (t, s) in enumerate(ts):
        receiver[q, :3] = 2 * ((t >> np.arange(3)) & 1) - 1
        receiver[q, 3:6] = 2 * ((s >> np.arange(3)) & 1) - 1
        receiver[q, 6] = prior
    y = np.array([LINEAR[t - 1] for t, s in ts]).T
    side = np.array([LINEAR[s - 1] for t, s in ts]).T
    weights = np.array([((1 - prior) / 7 + prior * (t == 1)) / 7 for t, s in ts])
    return producer, receiver, y, side, weights


class Game:
    def __init__(self, raw):
        p, r, y, s, w = [torch.tensor(np.array(v), dtype=torch.float32) for v in raw]
        q = len(w)
        if not (p.shape == (8, 12) and r.shape == (q, 12) and y.shape == s.shape == (8, q)):
            raise ValueError("unexpected game shape")
        if not (
            torch.all((s == 0) | (s == 1))
            and torch.all((y == 0) | (y == 1))
            and torch.all(w >= 0)
            and abs(w.sum().item() - 1) < 1e-5
        ):
            raise ValueError("invalid finite game")
        self.p, self.y, self.s, self.w = p, y, s, w
        self.r = r.repeat(4, 1)
        # Public contexts ordered by (message, verified side, query).
        self.msg = torch.arange(2).repeat_interleave(2 * q).float()
        self.side = torch.arange(2).repeat_interleave(q).repeat(2).float()
        self.side_mask = torch.stack((1 - s, s))

    def masses(self, prob):
        # Send precisely one of two symbols. Its probability is private to the sender.
        distribution = torch.stack((1 - prob, prob))[:, None, :, None] / 8
        combined = distribution * self.side_mask.unsqueeze(0)
        return combined.sum(2), (combined * self.y).sum(2)

    def posterior(self, prob):
        mass, yes = self.masses(prob)
        return yes / mass.clamp_min(1e-15)

    def risks(self, prob, guesses):
        mass, yes = self.masses(prob)
        # E[(Y-a)^2] in each observable cell.
        real = yes * (1 - guesses).square() + (mass - yes) * guesses.square()
        ideal = yes * (mass - yes) / mass.clamp_min(1e-15)
        return (real * self.w).sum(), (ideal * self.w).sum()

    def guesses(self, model):
        return model.decode(self.r, self.side, self.msg).sigmoid().reshape(2, 2, -1)


def train_phase(model, games, steps, seed):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    losses = []
    for step in range(steps):
        game = games[rng.integers(len(games))]
        prob = model.encode(game.p).sigmoid()
        real, ideal = game.risks(prob, game.guesses(model))
        loss = (real + ideal) * 0.5
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            losses.append([step + 1, float(real.detach()), float(ideal.detach())])
    return losses


def controls(raw):
    _, _, y, s, w = raw
    # Every deterministic partition of eight worlds up to swapping its two messages.
    risk = np.zeros(len(CODES))
    for msg in (0, 1):
        for side in (0, 1):
            cell = (msg == CODES)[:, :, None] & (s == side)[None, :, :]
            count = cell.sum(1)
            count_yes = (cell * y).sum(1)
            risk += ((count_yes * (count - count_yes) / np.maximum(count, 1)) * w).sum(1) / 8
    index = LINEAR @ (1 << np.arange(8)) // 2
    return risk, float(min(risk[index]))


@torch.no_grad()
def score(model, raw, reference):
    game = Game(raw)
    code = (model.encode(game.p) >= 0).float()
    guesses = game.guesses(model)
    real, ideal = game.risks(code, guesses)
    risk, linear = reference
    bits = code.int().tolist()
    canonical = int(sum((v ^ bits[0]) << x for x, v in enumerate(bits)))
    if abs(ideal.item() - risk[canonical // 2]) > 2e-6:
        raise AssertionError("neural and enumerated hard-channel risks disagree")
    si = game.s.long()
    qi = torch.arange(len(game.w))
    predicted = guesses[code.long()[:, None], si, qi] >= 0.5
    side_only = game.posterior(torch.zeros(8))[0, si, qi] >= 0.5
    right, base_right = predicted == game.y.bool(), side_only == game.y.bool()

    def rate(mask):
        return float((mask.float() * game.w).sum() / 8)

    return dict(
        code=canonical,
        ones=sum(bits),
        brier=real.item(),
        code_bayes=float(risk[canonical // 2]),
        side_only=float(risk[0]),
        best_linear=linear,
        majority=float(risk[232 // 2]),
        oracle=float(min(risk)),
        optimal=bool(risk[canonical // 2] <= min(risk) + 1e-12),
        corrected=rate(right & ~base_right),
        spoiled=rate(~right & base_right),
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-seed-file", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "score"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out = args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "training.json"
    if args.phase == "train":
        if manifest.exists() or any(out.glob("*.pt")):
            parser.error("choose an empty run directory for training")
        pg = [Game(parity(v)) for v in TRAIN_PRIORS]
        lg = [Game(v) for v in make_games("train", 5521).values()]
        receipt = {"torch": str(torch.__version__), "python": platform.python_version(), "runs": {}}
        for seed in SEEDS:
            for arm in ARMS:
                torch.manual_seed(seed)
                model = ReceiptModel(arm)
                row = dict(parameters=sum(p.numel() for p in model.parameters()))
                for phase, games, steps in (("parity", pg, 1200), ("ldt", lg, 600)):
                    start = time.monotonic()
                    trace = train_phase(model, games, steps, seed + int(phase == "ldt"))
                    path = out / f"{arm}-{seed}-{phase}.pt"
                    torch.save(model.state_dict(), path)
                    row[phase] = dict(steps=steps, seconds=time.monotonic() - start, sha256=digest(path), trace=trace)
                    print(arm, seed, phase, round(row[phase]["seconds"], 1), flush=True)
                receipt["runs"][f"{arm}-{seed}"] = row
        manifest.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return
    trained = json.loads(manifest.read_text(encoding="utf-8"))
    # Check all checkpoints before generating a single transfer game.
    for key, row in trained["runs"].items():
        for phase in ("parity", "ldt"):
            if digest(out / f"{key}-{phase}.pt") != row[phase]["sha256"]:
                raise ValueError("checkpoint mismatch")
    if set(trained["runs"]) != {f"{a}-{s}" for a in ARMS for s in SEEDS}:
        raise ValueError("incomplete training manifest")
    test_seed = int(args.eval_seed_file.read_text(encoding="utf-8").strip())
    test = make_games("test", test_seed)
    pg = {str(v): parity(v) for v in TEST_PRIORS}
    panels = {k: {n: (v, controls(v)) for n, v in games.items()} for k, games in (("parity", pg), ("ldt", test))}
    familiar = make_games("train", test_seed ^ 0x55A1)
    panels["familiar"] = {n: (v, controls(v)) for n, v in familiar.items()}
    fingerprints = {
        k: hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in v)).hexdigest() for k, v in test.items()
    }
    results = {
        "training": trained,
        "test_seed_sha256": digest(args.eval_seed_file),
        "games": fingerprints,
        "scores": {},
    }
    for key in trained["runs"]:
        arm = key.split("-")[0]
        model = ReceiptModel(arm).eval()
        row = {}
        for phase in ("parity", "ldt"):
            model.load_state_dict(torch.load(out / f"{key}-{phase}.pt", weights_only=True, map_location="cpu"))
            for name in ("parity", "ldt") if phase == "parity" else ("ldt", "familiar"):
                row[f"{phase}_to_{name}"] = {n: score(model, raw, ref) for n, (raw, ref) in panels[name].items()}
        results["scores"][key] = row
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
