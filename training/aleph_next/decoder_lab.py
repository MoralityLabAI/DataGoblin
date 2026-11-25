"""ALEPH-03. Compare learned side and one-bit decoders on the same finite games."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, vmap

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "aleph_rlt_placement"
sys.path.insert(0, str(LEGACY))
from receipt_followup import TRAIN_SEED  # noqa: E402
from receipt_gym import ARMS, SEEDS, controls, digest  # noqa: E402
from receipt_models import ReceiptModel  # noqa: E402
from receipt_readout import frozen, weight_hash  # noqa: E402
from receipt_relations import make_relations, verified_side_sha256  # noqa: E402

READERS = (*ARMS, "logistic", "empirical", "structured")
CONDITIONS = ("side", "constant", "shuffle", "valid")
STEPS = 500
PANELS = 4
WORLDS = (np.arange(8)[:, None] >> np.arange(3)) & 1
CAP = 0.005


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value, *, immutable=False):
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f"refusing to replace {path.name}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def raw_hash(raw):
    return hashlib.sha256(b"".join(v.tobytes() for v in raw)).hexdigest()


def contract(v2):
    legacy = (
        "receipt_followup",
        "receipt_gym",
        "receipt_ldt",
        "receipt_relations",
        "receipt_models",
        "receipt_readout",
    )
    files = [Path(__file__), HERE / "DECODER_PROTOCOL.md", *(LEGACY / f"{name}.py" for name in legacy)]
    return {
        "source_sha256": {p.name: hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest() for p in files},
        "v2_manifest_sha256": digest(v2 / "training.json"),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
    }


def world_labels(query):
    return (WORLDS @ query[:, :3].T > 0) & (WORLDS @ query[:, 3:6].T > 0)


class Reader(nn.Module):
    """Every neural predictor consumes only the public 12-vector, side and message."""

    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        if architecture in ARMS:
            self.actor = ReceiptModel(architecture)
        elif architecture == "logistic":
            self.linear = nn.Linear(14, 1)
        elif architecture == "structured":
            self.upload_logits = nn.Parameter(torch.zeros(2, 8))
            self.register_buffer("worlds", torch.tensor(WORLDS, dtype=torch.float32), persistent=False)
        else:
            raise ValueError(architecture)

    def forward(self, inputs):
        query, side, msg = inputs[..., :12], inputs[..., 12], inputs[..., 13]
        if self.architecture in ARMS:
            return self.actor._logit(query, side, msg, 1.0).sigmoid()
        if self.architecture == "logistic":
            return self.linear(inputs).squeeze(-1).sigmoid()
        # Enumerate every possible world. No current private realization enters here.
        possible = ((2 * query[:, 6:9] @ self.worlds.T - 1) == side[:, None]).float()
        yes = ((query[:, :3] @ self.worlds.T > 0) & (query[:, 3:6] @ self.worlds.T > 0)).float()
        message_p = self.upload_logits.sigmoid().index_select(0, query[:, 9].long())
        likelihood = 0.5 + msg[:, None] * (message_p - 0.5)
        weighted = possible * likelihood
        return (weighted * yes).sum(-1) / weighted.sum(-1).clamp_min(1e-12)


def observations(query, condition):
    q = len(query)
    msg = np.repeat([-1.0, 1.0], 2 * q)
    if condition == "side":
        msg.fill(0)
    side = np.tile(np.repeat([-1.0, 1.0], q), 2)
    return np.column_stack((np.tile(query, (4, 1)), side, msg)).astype(np.float32)


def cells(raw, code, condition):
    """Joint P(M,S,Q), P(Y=1,M,S,Q), over all worlds and native late queries."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    _, query, y, side, q_mass = raw
    code = np.asarray(code)
    if code.shape != (8,) or not np.isin(code, (0, 1)).all():
        raise ValueError("only eight binary uploads may cross the sender boundary")
    transmitted = code if condition in ("shuffle", "valid") else np.zeros(8, dtype=int)
    cell = (transmitted[None, None, :, None] == np.arange(2)[:, None, None, None]) & (
        side[None, None, :, :] == np.arange(2)[None, :, None, None]
    )
    mass = cell.sum(axis=2) * (q_mass / 8)
    positive = (cell * y[None, None]).sum(axis=2) * (q_mass / 8)
    if condition == "shuffle":
        side_post = positive.sum(axis=0) / np.maximum(mass.sum(axis=0), 1e-15)
        positive = mass * side_post
    return observations(query, condition), mass.ravel(), positive.ravel()


def entropy_bits(p):
    v = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log2(v) - (1 - p) * np.log2(1 - v)


def metric_vectors(mass, positive, predictions):
    """Vectorize over frames. Last dimension enumerates public cells; zero mass vanishes."""
    post = np.divide(positive, mass, out=np.zeros_like(positive), where=mass > 0)
    error = predictions - post
    bayes = (mass * post * (1 - post)).sum(axis=-1)
    regret = (mass * error**2).sum(axis=-1)
    bin_id = np.minimum((predictions * 10).astype(int), 9)
    ece = sum(np.abs((mass * error * (bin_id == i)).sum(axis=-1)) for i in range(10))
    return {
        "brier": bayes + regret,
        "bayes": bayes,
        "regret": regret,
        "ece10": ece,
        "bias": (mass * error).sum(axis=-1),
        "conditional_entropy_bits": (mass * entropy_bits(post)).sum(axis=-1),
    }


def table_keys(row):
    a, b, cal = row[:3], row[3:6], row[6:9]
    coarse = tuple(int(x) for x in row[9:])
    relations = tuple(int(x) for x in (sum(a), sum(b), a @ b, a @ cal, b @ cal))
    return ["global", str(coarse), str(coarse + relations), str(tuple(int(x) for x in row))]


def fit_table(frames):
    counts = {}
    for obs, mass, yes in frames:
        for x, w, y in zip(obs, mass, yes, strict=True):
            if w <= 0:
                continue
            for k in table_keys(x):
                n, p = counts.get(k, (0.0, 0.0))
                counts[k] = (n + 48 * w, p + 48 * y)
    return counts


def lookup(table, x):
    prob = 0.5
    for i, k in enumerate(table_keys(x)):
        n, p = table.get(k, (0.0, 0.0))
        shrink = 1 if i == 0 else 8
        prob = (p + shrink * prob) / (n + shrink)
    return prob


def build_sender(v2, arm, seed, baseline):
    key = f"{arm}-{seed}"
    row = baseline["runs"][key]
    primary, parent = row["primary"], row["coordinated"]
    if primary["parent"] != parent["sha256"] or digest(v2 / f"{key}-coordinated.pt") != parent["sha256"]:
        raise ValueError("unexpected coordinated parent")
    sender = frozen(arm, v2 / f"{key}-primary.pt", primary["sha256"])
    return sender, {
        "sha256": primary["sha256"],
        "parent_sha256": parent["sha256"],
        "frozen_weight_sha256": weight_hash(sender),
        "params": sum(p.numel() for p in sender.parameters()),
    }


@torch.inference_mode()
def uploads(sender, frames):
    producer = torch.from_numpy(np.concatenate([g[0] for g in frames]).astype(np.float32))
    result = (sender.encode(producer) >= 0).to(torch.int64).numpy().reshape(len(frames), 8)
    # The only data allowed out of this function is the thresholded binary value.
    return result


def labels(seed):
    return [(f"{arm}-{seed}", c) for arm in ARMS for c in CONDITIONS]


def model_id(sender, reader, condition):
    return f"{sender}-to-{reader}-{condition}"


def make_reader(architecture, seed):
    torch.manual_seed(100000 + seed)
    return Reader(architecture)


def train_ensemble(arch, seed, channels):
    """Vectorized independent fits. No gradient, momentum, clipping or weights cross cells."""
    model = make_reader(arch, seed)
    init = {k: v.detach().clone() for k, v in model.named_parameters()}
    keys = labels(seed)
    weights = {k: nn.Parameter(v.expand(len(keys), *v.shape).clone()) for k, v in init.items()}
    x, mass, yes = (
        torch.as_tensor(np.asarray([[cell[i] for cell in channels[key]] for key in keys]), dtype=torch.float32)
        for i in range(3)
    )
    target = yes / mass.clamp_min(1e-15)
    optimizer = torch.optim.Adam(weights.values(), lr=0.003)
    predict = vmap(lambda w, obs: functional_call(model, w, (obs,)), in_dims=(0, 0))
    rng = np.random.default_rng(seed)
    for step in range(STEPS):
        if step % x.size(1) == 0:
            order = rng.permutation(x.size(1))
        j = order[step % x.size(1)]
        pred = predict(weights, x[:, j])
        loss = (mass[:, j] * (pred - target[:, j]).square()).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            norms2 = sum(w.grad.reshape(len(keys), -1).square().sum(1) for w in weights.values())
            factor = (1 / (norms2.sqrt() + 1e-6)).clamp(max=1)
            for w in weights.values():
                w.grad.mul_(factor.reshape(len(keys), *([1] * (w.ndim - 1))))
        optimizer.step()
    return [{k: w[i].detach().clone() for k, w in weights.items()} for i in range(len(keys))]


def ckpt_path(out, label, architecture):
    return out / (label + (".json" if architecture == "empirical" else ".pt"))


def train(v2, out):
    manifest = out / "training.json"
    signature = contract(v2)
    record = read(manifest) if manifest.exists() else {"contract": signature, "fits": {}, "senders": {}}
    if record["contract"] != signature:
        raise ValueError("training source or inputs changed")
    raw = list(make_relations("train", TRAIN_SEED).values())
    if any(not np.array_equal(world_labels(g[1]), g[2]) for g in raw):
        raise ValueError("registered geometry is inconsistent with native training semantics")
    record["training_frames_sha256"] = [raw_hash(g) for g in raw]
    baseline = read(v2 / "training.json")
    for seed in SEEDS:
        channels = {}
        for arm in ARMS:
            name = f"{arm}-{seed}"
            sender, stamp = build_sender(v2, arm, seed, baseline)
            bits = uploads(sender, raw)
            if weight_hash(sender) != stamp["frozen_weight_sha256"]:
                raise ValueError("sender mutated")
            record["senders"][name] = stamp
            for condition in CONDITIONS:
                channels[name, condition] = [cells(g, c, condition) for g, c in zip(raw, bits, strict=True)]
        keys = labels(seed)
        for architecture in READERS:
            ids = [model_id(name, architecture, condition) for name, condition in keys]
            if all(k in record["fits"] for k in ids):
                for k in ids:
                    if digest(ckpt_path(out, k, architecture)) != record["fits"][k]["sha256"]:
                        raise ValueError(f"receiver checksum mismatch: {k}")
                continue
            started = time.perf_counter()
            empirical = architecture == "empirical"
            states = (
                [fit_table(channels[key]) for key in keys]
                if empirical
                else train_ensemble(architecture, seed, channels)
            )
            for key, label, state in zip(keys, ids, states, strict=True):
                path = ckpt_path(out, label, architecture)
                if empirical:
                    write(path, state)
                else:
                    temp = path.with_suffix(".tmp")
                    torch.save(state, temp)
                    temp.replace(path)
                record["fits"][label] = {
                    "sender": key[0],
                    "reader": architecture,
                    "condition": key[1],
                    "seed": seed,
                    "steps": 1 if empirical else STEPS,
                    "params": len(state) if empirical else sum(t.numel() for t in state.values()),
                    "parameter_unit": "occupied rate-table entries" if empirical else "trainable scalars",
                    "recurrence": 2 if architecture in ("C", "D") else 0,
                    "sha256": digest(path),
                }
            write(manifest, record)
            print(f"frozen {architecture} seed {seed}, 12 cells, {time.perf_counter() - started:.1f}s", flush=True)


def fresh_panel(out, v2, prior, training_hashes):
    # Refuse to generate a seed unless both previous held panels are identifiable.
    old = set(read(v2 / "results.json")["games"].values())
    old.update(v["frame"] for v in read(prior)["games"].values())
    if len(old) != 32:
        raise ValueError("expected 32 distinct full-frame digests from both earlier experiments")
    old.update(training_hashes)
    seed_file = out / "private.seed"
    if not seed_file.exists():
        with seed_file.open("x", encoding="ascii") as fp:
            fp.write(str(secrets.randbits(256)) + "\n")
    master = seed_file.read_text(encoding="ascii").strip()
    games = {}
    for panel in range(PANELS):
        salt = int.from_bytes(hashlib.sha256(f"aleph03/fair/{master}/{panel}".encode()).digest(), "big")
        for name, raw in make_relations("test", salt).items():
            fingerprint = raw_hash(raw)
            if fingerprint in old:
                raise ValueError("overlap with training, prior held panel or another new frame")
            old.add(fingerprint)
            games[f"{panel}/{name}"] = raw
    return games, digest(seed_file)


@torch.inference_mode()
def predictions(architecture, seed, path, all_x):
    if architecture == "empirical":
        table = read(path)
        flat, index = np.unique(all_x.reshape(-1, 14), return_inverse=True, axis=0)
        probs = np.asarray([lookup(table, x) for x in flat])[index]
        return probs.reshape(all_x.shape[:-1])
    reader = make_reader(architecture, seed).eval()
    reader.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    values = reader(torch.from_numpy(all_x.reshape(-1, 14)))
    return values.reshape(all_x.shape[:-1]).double().numpy()


def group_metrics(metrics, names):
    families = sorted({name.split("/")[1] for name in names})
    panels = sorted({name.split("/")[0] for name in names})
    result = {k: float(np.mean(v)) for k, v in metrics.items()}
    by_family = {}
    for f in families:
        ix = [i for i, n in enumerate(names) if n.split("/")[1] == f]
        by_family[f] = {k: float(np.mean(v[ix])) for k, v in metrics.items()}
    by_panel = {}
    for p in panels:
        ix = [i for i, n in enumerate(names) if n.split("/")[0] == p]
        by_panel[p] = {k: float(np.mean(v[ix])) for k, v in metrics.items()}
    return {"mean": result, "families": by_family, "panels": by_panel}


def summarize(fits):
    comparisons = {}
    rng = np.random.default_rng(357190)
    seed_draws = rng.integers(len(SEEDS), size=(4000, len(SEEDS), 1))
    panel_draws = rng.integers(PANELS, size=(4000, 1, PANELS))
    for sender_arm in ARMS:
        for reader in READERS:
            seed_gains, family_gains, values = [], {}, []
            panel_gains = []
            for seed in SEEDS:
                cond = {c: fits[model_id(f"{sender_arm}-{seed}", reader, c)] for c in CONDITIONS}
                avg = {c: d["mean"] for c, d in cond.items()}
                side, valid = avg["side"], avg["valid"]
                gain = side["brier"] - valid["brier"]
                oracle = side["bayes"] - valid["bayes"]
                comparison = {
                    **{f"{c}_brier": avg[c]["brier"] for c in CONDITIONS},
                    **{f"{c}_ece10": avg[c]["ece10"] for c in CONDITIONS},
                    "side_bayes": side["bayes"],
                    "valid_bayes": valid["bayes"],
                    "oracle_value": oracle,
                    "learned_value": gain,
                    "valid_minus_side_regret": valid["regret"] - side["regret"],
                    "cmi_bits": valid["cmi_bits"],
                    "constant_minus_valid": avg["constant"]["brier"] - valid["brier"],
                    "shuffle_minus_valid": avg["shuffle"]["brier"] - valid["brier"],
                }
                values.append(comparison)
                seed_gains.append(gain)
                for family in cond["side"]["families"]:
                    fs = cond["side"]["families"][family]["brier"] - cond["valid"]["families"][family]["brier"]
                    family_gains.setdefault(family, []).append(fs)
                panel_gains.append(
                    [
                        cond["side"]["panels"][str(p)]["brier"] - cond["valid"]["panels"][str(p)]["brier"]
                        for p in range(PANELS)
                    ]
                )
            pooled = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
            family_means = {k: float(np.mean(v)) for k, v in family_gains.items()}
            bootstrap = np.asarray(panel_gains)[seed_draws, panel_draws].mean(axis=(1, 2))
            ci = [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]
            gates = {
                "gain_at_least_005": pooled["learned_value"] >= CAP,
                "all_seed_gains_positive": min(seed_gains) > 0,
                "all_family_gains_positive": min(family_means.values()) > 0,
                "better_than_constant": pooled["constant_minus_valid"] > 0,
                "better_than_shuffled": pooled["shuffle_minus_valid"] > 0,
                "paired_interval_lower_positive": ci[0] > 0,
            }
            comparisons[f"{sender_arm}-to-{reader}"] = {
                **pooled,
                "fraction_recovered": pooled["learned_value"] / pooled["oracle_value"]
                if pooled["oracle_value"] > CAP
                else None,
                "seed_learned_values": seed_gains,
                "family_learned_values": family_means,
                "paired_seed_panel_bootstrap_95": ci,
                "gates": gates,
                "passed": all(gates.values()),
                "registered_primary": sender_arm == "C" and reader == "C",
            }
    return comparisons


def score(v2, prior, out):
    manifest = out / "training.json"
    record = read(manifest)
    if record["contract"] != contract(v2):
        raise ValueError("source or v2 contract changed since training")
    expected = {model_id(name, a, c) for seed in SEEDS for name, c in labels(seed) for a in READERS}
    if set(record["fits"]) != expected:
        raise ValueError("all 216 independent receivers must be frozen before scoring")
    for lab, row in record["fits"].items():
        if digest(ckpt_path(out, lab, row["reader"])) != row["sha256"]:
            raise ValueError(f"receiver digest mismatch: {lab}")
    baseline = read(v2 / "training.json")
    senders = {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f"{arm}-{seed}"
            sender, stamp = build_sender(v2, arm, seed, baseline)
            if stamp != record["senders"][key]:
                raise ValueError("frozen sender identity changed")
            senders[key] = sender
    games, private_digest = fresh_panel(out, v2, prior, record["training_frames_sha256"])
    names, raw = list(games), list(games.values())
    references = [controls(g)[0] for g in raw]
    frame_ids = {n: {"sha256": raw_hash(g), "side_sha256": verified_side_sha256(g)} for n, g in games.items()}
    fits, frame_scores = {}, {}
    for key, sender in senders.items():
        seed = int(key.rsplit("-", 1)[1])
        bits = uploads(sender, raw)
        codes = ((bits ^ bits[:, :1]) @ (1 << np.arange(8))) // 2
        side_entropy = None
        for condition in CONDITIONS:
            all_cells = [cells(g, c, condition) for g, c in zip(raw, bits, strict=True)]
            x, mass, yes = (np.stack([cell[i] for cell in all_cells]) for i in range(3))
            independent = np.asarray(
                [ref[int(index) if condition == "valid" else 0] for ref, index in zip(references, codes, strict=True)]
            )
            for architecture in READERS:
                lab = model_id(key, architecture, condition)
                pred = predictions(architecture, seed, ckpt_path(out, lab, architecture), x)
                metrics = metric_vectors(mass, yes, pred)
                if not np.allclose(metrics["bayes"], independent, rtol=0, atol=1e-12):
                    raise AssertionError("per-frame finite oracle disagrees with independent native enumeration")
                if condition == "side":
                    side_entropy = metrics["conditional_entropy_bits"]
                metrics["cmi_bits"] = side_entropy - metrics.pop("conditional_entropy_bits")
                if condition != "valid" and max(abs(metrics["cmi_bits"])) > 1e-12:
                    raise AssertionError("null channel retains conditional target information")
                fits[lab] = group_metrics(metrics, names)
                frame_scores[lab] = {k: v.tolist() for k, v in metrics.items()}
        if weight_hash(sender) != record["senders"][key]["frozen_weight_sha256"]:
            raise ValueError("sender drift at inference")
    results = {
        "schema": "aleph03-receivers-v1",
        "training_manifest_sha256": digest(manifest),
        "private_seed_sha256": private_digest,
        "frames": frame_ids,
        "independent_panels": PANELS,
        "compared_previous_held_frame_sha256": 32,
        "training_frames": 24,
        "held_frames": len(games),
        "seed_order": list(SEEDS),
        "checkpoints": len(fits),
        "comparisons": summarize(fits),
    }
    write(out / "per-frame.json", {"frame_order": names, "metrics": frame_scores}, immutable=True)
    write(out / "fit-scores.json", fits, immutable=True)
    write(out / "results.json", results, immutable=True)
    print(
        f"scored {len(fits)} fits on {len(raw)} distinct frames; primary={results['comparisons']['C-to-C']}", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "score"))
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--prior-readout", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "score" and not args.prior_readout:
        parser.error("score requires --prior-readout containing previous frozen-readout frame digests")
    if args.out.resolve() == args.v2.resolve() or HERE.parent.parent in (
        args.out.resolve(),
        *args.out.resolve().parents,
    ):
        parser.error("use a local private output directory outside the source repository")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.phase == "train":
        train(args.v2, args.out)
    else:
        score(args.v2, args.prior_readout, args.out)


if __name__ == "__main__":
    main()
