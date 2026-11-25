"""Cross fresh receivers with frozen v2 senders on independent containment frames."""

import argparse
import hashlib
import json
import secrets
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from receipt_followup import PRIORS, TRAIN_SEED, load_state, save_json
from receipt_gym import ARMS, SEEDS, Game, controls, digest, parity
from receipt_models import ReceiptModel
from receipt_relations import make_relations, verified_side_sha256

STEPS = 500
BRACKET = (0.07, 0.08, 0.072, 0.079)
SOURCES = (
    "receipt_readout.py",
    "RECEIPT_READOUT_PROTOCOL.md",
    "receipt_models.py",
    "receipt_relations.py",
    "receipt_ldt.py",
    "receipt_gym.py",
    "receipt_followup.py",
    "cpu_worker.py",
)


def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def weight_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.named_parameters():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pack(bits):
    return sum(int(bit) << i for i, bit in enumerate(bits))


def frozen(arm, path, expected):
    model = ReceiptModel(arm).eval().requires_grad_(False)
    load_state(model, path, expected)
    return model


def new_receiver(arm, seed):
    torch.manual_seed(100000 + seed)
    return ReceiptModel(arm)


def source_contract(v1, v2):
    here = Path(__file__).resolve().parent
    return {
        "scripts": {f: hashlib.sha256((here / f).read_text(encoding="utf-8").encode()).hexdigest() for f in SOURCES},
        "v1_manifest_sha256": digest(v1 / "training.json"),
        "v2_manifest_sha256": digest(v2 / "training.json"),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
    }


def prepare(sender, games):
    """Cache exact train-label frequencies, weights and public receiver observations."""
    frames = []
    with torch.no_grad():
        for game in games:
            code = (sender.encode(game.p) >= 0).float()
            mass, yes = game.masses(code)
            post = yes / mass.clamp_min(1e-15)
            frames.append((game, post, mass * game.w))
    return frames


def fit(receiver, sender, frames, probe, seed, steps=STEPS):
    if any(p.requires_grad for p in sender.parameters()):
        raise ValueError("the upload producer must be frozen")
    if {id(p) for p in sender.parameters()} & {id(p) for p in receiver.parameters()}:
        raise ValueError("sender and receiver parameters overlap")
    optimizer = torch.optim.Adam(receiver.parameters(), lr=0.003)
    rng = np.random.default_rng(seed)
    initial_hash = weight_hash(sender)
    with torch.no_grad():
        initial = sender.encode(probe)
    trace = []
    for step in range(steps + 1):
        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                current = sender.encode(probe)
            if not torch.equal(initial, current):
                raise AssertionError("frozen sender logits changed")
            trace.append(
                {
                    "update": step,
                    "sha256": fingerprint(current),
                    "max_abs_drift": float((current - initial).abs().max()),
                }
            )
        if step == steps:
            break
        if step % len(frames) == 0:
            order = rng.permutation(len(frames))
        game, target, weight = frames[order[step % len(frames)]]
        loss = (weight * (game.guesses(receiver) - target).square()).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(receiver.parameters(), 1.0)
        optimizer.step()
    if weight_hash(sender) != initial_hash or any(p.grad is not None for p in sender.parameters()):
        raise AssertionError("frozen sender weights or gradients changed")
    return {"updates": steps, "sender_weight_sha256": initial_hash, "retention": trace}


@torch.no_grad()
def parity_receipt(sender, probe, risks):
    logits = sender.encode(probe).reshape(len(PRIORS), 8)
    rows = {}
    for prior, values in zip(PRIORS, logits, strict=True):
        bits = (values >= 0).int()
        raw = pack(bits)
        canonical = (raw ^ 255) if raw & 1 else raw
        table = risks[prior]
        rows[str(prior)] = {
            "raw_mask": raw,
            "canonical": canonical,
            "optimal": bool(table[canonical // 2] <= table.min() + 1e-12),
        }
    return {
        "sha256": fingerprint(logits),
        "priors": rows,
        "bracket_codes_optimal": all(rows[str(p)]["optimal"] for p in BRACKET),
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def train(v1, v2, out):
    contract = source_contract(v1, v2)
    target = out / "training.json"
    record = read_json(target) if target.exists() else {"contract": contract, "senders": {}, "fits": {}}
    if record["contract"] != contract:
        raise ValueError("inputs changed during the assay")
    v1_rows, v2_rows = (read_json(p / "training.json")["runs"] for p in (v1, v2))
    frames = [Game(v) for v in make_relations("train", TRAIN_SEED).values()]
    probe = torch.cat([Game(parity(p)).p for p in PRIORS])
    risks = {p: controls(parity(p))[0] for p in PRIORS}
    for seed in SEEDS:
        for arm in ARMS:
            key = f"{arm}-{seed}"
            row = v2_rows[key]
            sha = row["primary"]["sha256"]
            if row["primary"]["parent"] != row["coordinated"]["sha256"]:
                raise ValueError("LDT checkpoint does not descend from its declared parity parent")
            sender = frozen(arm, v2 / f"{key}-primary.pt", sha)
            parent = frozen(arm, v2 / f"{key}-coordinated.pt", row["coordinated"]["sha256"])
            initial = frozen(arm, v1 / f"{key}-parity.pt", v1_rows[key]["parity"]["sha256"])
            stamps = {
                "primary_checkpoint": sha,
                "parity_parent_checkpoint": row["coordinated"]["sha256"],
                "v1_parity": parity_receipt(initial, probe, risks),
                "coordinated_parity": parity_receipt(parent, probe, risks),
                "primary_ldt": parity_receipt(sender, probe, risks),
            }
            if key in record["senders"] and record["senders"][key] != stamps:
                raise ValueError("sender provenance changed")
            record["senders"][key] = stamps
            prepared = prepare(sender, frames)
            for receiver_arm in ARMS:
                label = f"{key}-to-{receiver_arm}"
                ckpt = out / f"{label}.pt"
                if label in record["fits"]:
                    if digest(ckpt) != record["fits"][label]["sha256"]:
                        raise ValueError("receiver checkpoint mismatch")
                    continue
                receiver = new_receiver(receiver_arm, seed)
                track = fit(receiver, sender, prepared, probe, seed)
                tmp = ckpt.with_suffix(".tmp")
                torch.save(receiver.state_dict(), tmp)
                tmp.replace(ckpt)
                record["fits"][label] = {**track, "sha256": digest(ckpt)}
                save_json(target, record)
            print("frozen sender complete", key, flush=True)


@torch.no_grad()
def frame_scores(sender, receivers, frames):
    rows = {}
    for name, (g, ref) in frames.items():
        bit = (sender.encode(g.p) >= 0).float()
        raw = pack(bit)
        canonical = raw ^ 255 if raw & 1 else raw
        risk, linear = ref
        realized = {k: float(g.risks(bit, g.guesses(receiver))[0]) for k, receiver in receivers.items()}
        ideal = float(risk[canonical // 2])
        if abs(float(g.risks(bit, g.posterior(bit))[0]) - ideal) > 1e-6:
            raise AssertionError("posterior and enumerated source ceilings differ")
        rows[name] = {
            **realized,
            "side_only": float(risk[0]),
            "ideal_bit": ideal,
            "best_linear": linear,
            "oracle": float(risk.min()),
            "raw_mask": raw,
        }
    return rows


def balanced(rows):
    groups = defaultdict(list)
    for frame, row in rows.items():
        groups[frame.rsplit("/", 1)[0]].append(row)
    families = {
        family: {k: mean(r[k] for r in group) for k in group[0] if k != "raw_mask"} for family, group in groups.items()
    }
    totals = {k: mean(row[k] for row in families.values()) for k in next(iter(families.values()))}
    return {"families": families, "overall": totals}


def score(v1, v2, out):
    target = out / "training.json"
    record = read_json(target)
    if record["contract"] != source_contract(v1, v2):
        raise ValueError("source or baseline manifest mismatch")
    keys = [f"{a}-{s}" for s in SEEDS for a in ARMS]
    labels = [f"{key}-to-{a}" for key in keys for a in ARMS]
    if set(labels) != set(record["fits"]) or set(keys) != set(record["senders"]):
        raise ValueError("freeze and verify all receivers before drawing new frames")
    for label in labels:
        if digest(out / f"{label}.pt") != record["fits"][label]["sha256"]:
            raise ValueError("receiver file changed")
    for key in keys:
        if digest(v2 / f"{key}-primary.pt") != record["senders"][key]["primary_checkpoint"]:
            raise ValueError("sender file changed")
    seed_path = out / "private.seed"
    if not seed_path.exists():
        with seed_path.open("x", encoding="ascii") as fp:
            fp.write(str(secrets.randbits(128)) + "\n")
    salt = int(seed_path.read_text(encoding="ascii"))
    raw = make_relations("test", salt)
    frames = {name: (Game(v), controls(v)) for name, v in raw.items()}
    game_hashes = {
        k: {"frame": hashlib.sha256(b"".join(a.tobytes() for a in v)).hexdigest(), "side": verified_side_sha256(v)}
        for k, v in raw.items()
    }
    result = {
        "manifest_sha256": digest(target),
        "private_seed_sha256": digest(seed_path),
        "games": game_hashes,
        "frames": {},
        "summary": {},
    }
    for key in keys:
        arm, text_seed = key.split("-")
        seed = int(text_seed)
        sender = frozen(arm, v2 / f"{key}-primary.pt", record["senders"][key]["primary_checkpoint"])
        receivers = {"intact": sender}
        for a in ARMS:
            label = f"{key}-to-{a}"
            receivers[f"random_{a}"] = new_receiver(a, seed).eval()
            receivers[f"trained_{a}"] = frozen(a, out / f"{label}.pt", record["fits"][label]["sha256"])
        rows = frame_scores(sender, receivers, frames)
        result["frames"][key] = rows
        result["summary"][key] = balanced(rows)
    destination = out / "results.json"
    if destination.exists() and result != read_json(destination):
        raise ValueError("refusing to replace a different evaluation")
    save_json(destination, result)
    for key in keys:
        row = result["summary"][key]["overall"]
        print(
            key,
            *(
                f"{k}={row[k]:.5f}"
                for k in ("intact", "trained_C", "trained_D", "trained_MLP", "side_only", "ideal_bit")
            ),
        )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("train", "score"))
    for flag in ("v1", "v2", "out"):
        cli.add_argument("--" + flag, type=Path, required=True)
    args = cli.parse_args()
    if args.out.resolve() in (args.v1.resolve(), args.v2.resolve()):
        cli.error("choose an output directory distinct from the checkpoint inputs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    (train if args.phase == "train" else score)(args.v1, args.v2, args.out)
