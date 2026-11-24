"""Matched CPU follow-up on whole-code supervision and held rival relations."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from receipt_coordination import fit_coordinated
from receipt_exposure import fit_matched_outcome
from receipt_gym import ARMS, SEEDS, Game, controls, digest, parity, score, train_phase
from receipt_models import ReceiptModel
from receipt_relations import make_relations
from torch.nn import functional as F

PRIORS = (0.0, 0.015, 0.045, 0.068, 0.07, 0.072, 4 / 53, 0.079, 0.08, 0.085, 0.175, 0.5, 1.0)
BRANCHES = ("coordinated", "matched_outcome", "primary", "original", "majority")
TRAIN_SEED = 23017
MAJORITY = torch.tensor([float(x.bit_count() >= 2) for x in range(8)])


def save_json(path, value):
    content = json.dumps(value, indent=2, allow_nan=False)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def load_state(model, path, sha):
    if digest(path) != sha:
        raise ValueError(f"hash mismatch for {path.name}")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def fit_majority(model, games, seed):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    trace = []
    for step in range(600):
        game = games[rng.integers(len(games))]
        logits = model.encode(game.p)
        teach = F.binary_cross_entropy_with_logits(logits, MAJORITY)
        brier, _ = game.risks(MAJORITY, game.guesses(model))
        optimizer.zero_grad(set_to_none=True)
        (teach + brier).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == 599:
            trace.append([step + 1, float(teach.detach()), float(brier.detach())])
    return trace


def source_contract(base):
    here = Path(__file__).parent
    scripts = (
        "receipt_followup.py",
        "receipt_gym.py",
        "receipt_models.py",
        "receipt_ldt.py",
        "receipt_relations.py",
        "receipt_coordination.py",
        "receipt_exposure.py",
        "RECEIPT_FOLLOWUP_PROTOCOL.md",
    )
    return {
        "sources": {
            name: hashlib.sha256((here / name).read_text(encoding="utf-8").encode()).hexdigest() for name in scripts
        },
        "baseline_manifest": digest(base / "training.json"),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
    }


def train(base, out, baseline):
    manifest = out / "training.json"
    contract = source_contract(base)
    record = json.loads(manifest.read_text()) if manifest.exists() else {"contract": contract, "runs": {}}
    if record["contract"] != contract:
        raise ValueError("training inputs changed since partial run")
    games = [Game(raw) for raw in make_relations("train", TRAIN_SEED).values()]
    for seed in SEEDS:
        for arm in ARMS:
            key = f"{arm}-{seed}"
            cells = record["runs"].setdefault(key, {})
            for branch in BRANCHES:
                name = f"{key}-{branch}.pt"
                if branch in cells:
                    if digest(out / name) != cells[branch]["sha256"]:
                        raise ValueError(f"damaged checkpoint {name}")
                    continue
                torch.manual_seed(seed)
                model = ReceiptModel(arm)
                parent = "matched random initialization"
                start = time.perf_counter()
                if branch == "coordinated":
                    trace = fit_coordinated(model, seed, 1200)
                elif branch == "matched_outcome":
                    trace = fit_matched_outcome(model, seed, 1200)
                else:
                    if branch == "original":
                        parent = baseline[key]["parity"]["sha256"]
                        load_state(model, base / f"{key}-parity.pt", parent)
                    else:
                        parent = cells["coordinated"]["sha256"]
                        load_state(model, out / f"{key}-coordinated.pt", parent)
                    trace = (
                        fit_majority(model, games, seed + 1)
                        if branch == "majority"
                        else train_phase(model, games, 600, seed + 1)
                    )
                tmp = out / (name + ".tmp")
                torch.save(model.state_dict(), tmp)
                tmp.replace(out / name)
                cells[branch] = {
                    "parent": parent,
                    "sha256": digest(out / name),
                    "trace": trace,
                    "seconds": time.perf_counter() - start,
                    "steps": 1200 if branch in ("coordinated", "matched_outcome") else 600,
                }
                save_json(manifest, record)
                print(key, branch, round(cells[branch]["seconds"], 1), flush=True)


def panel(games):
    return {key: (raw, controls(raw)) for key, raw in games.items()}


@torch.no_grad()
def score_panel(model, prepared, majority_diagnostic=False):
    rows = {}
    for key, (raw, ref) in prepared.items():
        row = score(model, raw, ref)
        if majority_diagnostic:
            g = Game(raw)
            risk, _ = g.risks(MAJORITY, g.guesses(model))
            row["clamped_majority_brier"] = float(risk)
            row["majority_bit_accuracy"] = float(((model.encode(g.p) >= 0) == MAJORITY.bool()).float().mean())
        rows[key] = row
    return rows


def evaluate(base, out, baseline, seed_file):
    record = json.loads((out / "training.json").read_text())
    if record["contract"] != source_contract(base):
        raise ValueError("scorer and training sources differ")
    keys = [f"{arm}-{seed}" for seed in SEEDS for arm in ARMS]
    for key in keys:
        for branch in BRANCHES:
            if digest(out / f"{key}-{branch}.pt") != record["runs"][key][branch]["sha256"]:
                raise ValueError("incomplete or corrupt checkpoint set")
        if digest(base / f"{key}-parity.pt") != baseline[key]["parity"]["sha256"]:
            raise ValueError("invalid baseline checkpoint")
    # The full training bundle must exist before the private draw enters the process.
    salt = int(seed_file.read_text().strip())
    held = make_relations("test", salt)
    familiar = make_relations("train", salt ^ 0xB70E)
    pg = panel({str(v): parity(v) for v in PRIORS})
    held_panel, familiar_panel = panel(held), panel(familiar)
    game_hashes = {name: hashlib.sha256(b"".join(v.tobytes() for v in raw)).hexdigest() for name, raw in held.items()}
    end0, end1 = pg["0.0"][1][0], pg["1.0"][1][0]
    solver = {}
    for value in PRIORS:
        interpolated = (1 - value) * end0 + value * end1
        code = int(np.argmin(interpolated))
        risks = pg[str(value)][1][0]
        solver[str(value)] = {"code": 2 * code, "regret": float(risks[code] - risks.min())}
    result = {
        "manifest_sha256": digest(out / "training.json"),
        "seed_sha256": digest(seed_file),
        "games": game_hashes,
        "affine_solver": solver,
        "scores": {},
    }
    for key in keys:
        arm = key.split("-")[0]
        model = ReceiptModel(arm).eval()
        load_state(model, base / f"{key}-parity.pt", baseline[key]["parity"]["sha256"])
        rows = {"baseline": {"parity": score_panel(model, pg)}}
        for branch in BRANCHES:
            load_state(model, out / f"{key}-{branch}.pt", record["runs"][key][branch]["sha256"])
            entry = {"parity": score_panel(model, pg)}
            if branch not in ("coordinated", "matched_outcome"):
                entry["held"] = score_panel(model, held_panel, branch == "majority")
                entry["familiar"] = score_panel(model, familiar_panel)
            rows[branch] = entry
        result["scores"][key] = rows
    path = out / "results.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("different evaluation already present in this directory")
    save_json(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "score"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path)
    args = parser.parse_args()
    if args.phase == "score" and not args.seed_file:
        parser.error("score requires a local --seed-file")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.out.resolve() == args.base.resolve():
        parser.error("baseline and new output directories must differ")
    args.out.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((args.base / "training.json").read_text())["runs"]
    if args.phase == "train":
        train(args.base, args.out, baseline)
    else:
        evaluate(args.base, args.out, baseline, args.seed_file)


if __name__ == "__main__":
    main()
